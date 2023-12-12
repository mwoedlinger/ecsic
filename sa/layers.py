import torch
import torch.nn as nn
from einops import reduce, rearrange, repeat
from torch.nn import functional as F
import math
from .utils import aij

__all__ = ['MaskedConv2d', 'PositionalEncoding2D', 'PositionalEncoding1D', 'Stereo', 
           'StereoAttentionModule', 'HorizontalMaskedConv2d', 'StereoSequential']

class HorizontalMaskedConv2d(nn.Conv2d):
    """
    Mask for a convolution with 3x3 kernel for mask_type=A (typically for the initial layer):
    [[1, 0, 0],
     [1, 0, 0],
     [1, 0, 0]]
    
    mask_type=B (typically for every following layer):
    [[1, 1, 0],
     [1, 1, 0],
     [1, 1, 0]]
    """

    def __init__(self, *args, mask_type: str = "A", **kwargs):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, h, w = self.mask.size()
        self.mask[:, :, :, w // 2 + (mask_type == "B") :] = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data *= self.mask
        return super().forward(x)

# Taken from https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/layers/layers.py
class MaskedConv2d(nn.Conv2d):
    r"""Masked 2D convolution implementation, mask future "unseen" pixels.
    Useful for building auto-regressive network components.
    Introduced in `"Conditional Image Generation with PixelCNN Decoders"
    <https://arxiv.org/abs/1606.05328>`_.
    Inherits the same arguments as a `nn.Conv2d`. Use `mask_type='A'` for the
    first layer (which also masks the "current pixel"), `mask_type='B'` for the
    following layers.
    """

    def __init__(self, *args, mask_type: str = "A", **kwargs):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, h, w = self.mask.size()
        self.mask[:, :, h // 2, w // 2 + (mask_type == "B") :] = 0
        self.mask[:, :, h // 2 + 1 :] = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data *= self.mask
        return super().forward(x)

    
# Performs positional embedding for a 1d sequence. Expects a (b, l, d) tensor.
class PositionalEncoding1D(nn.Module):

    def __init__(self, d: int, max_len: int=10000, concat=True):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe = torch.zeros(1, max_len, d)
        
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        self.concat = concat

    def forward(self, x: torch.Tensor, factor=1) -> torch.Tensor:
        """
        Applies positional encoding in the L dimension
        
        Args:
            x: Tensor, shape [B, L, C]
        """
        if self.concat:
            B, L, _ = x.shape
            return torch.concat([x, factor*self.pe[:, :L].expand(B, -1, -1)], dim=-1)
        else:
            return x + factor*self.pe[:, :x.size(1)]

# Performs positional embedding in the width dimension. Expects a (b, c, h, w) tensor.
class PositionalEncoding2D(PositionalEncoding1D):

    def forward(self, x: torch.Tensor, factor=1) -> torch.Tensor:
        """
        Applies positional encoding in the width dimension
        
        Args:
            x: Tensor, shape [B, C, H, W]
        """
        B = x.shape[0]
        
        x = rearrange(x, 'b c h w -> (b h) w c')
        x = x + factor*self.pe[:, :x.size(1)]
        x = rearrange(x, '(b h) w c -> b c h w', b=B)
        
        return x

class MultiHeadConvAttentionBlock(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, dim_out, dim_embedd=64, num_heads=1, rel_pos_enc=False):
        super(MultiHeadConvAttentionBlock, self).__init__()
        assert dim_out % num_heads == 0, f'dim_out must be divisible by num_heads! {dim_out=}, {num_heads=}'

        self.num_heads = num_heads
        self.dim_embedd = dim_embedd
        self.dim_out = dim_out
        self.rel_pos_enc = rel_pos_enc
        
        self.conv_q = nn.Conv1d(dim_q, num_heads*dim_embedd, 3, 1, 1)
        self.conv_k = nn.Conv1d(dim_k, num_heads*dim_embedd, 3, 1, 1)
        self.convc_v = nn.Conv1d(dim_v, dim_out, 3, 1, 1)
        self.fc_mh = nn.Linear(dim_out, dim_out)

        if rel_pos_enc:
            self.max_pos = 32
            self.w_pos_enc =  nn.Parameter(torch.empty((2*self.max_pos + 1), dim_embedd))
            self.w_pos_enc.data.normal_(0.0, 1.0)
                       
    def forward(self, q, k, v, attn_mask=None, valid_mask=None):
        b_q, _, _ = q.shape
        b_k, n_k, _ = k.shape
        b_v, n_v, _ = v.shape
        
        assert b_q == b_k == b_v, f'batch sizes not identical ({b_q}, {b_k}, {b_v})'
        assert n_k == n_v, f'number of keys ({n_k}) != number of values ({n_v})'         
        
        q = rearrange(q, 'b l c -> b c l')
        k = rearrange(k, 'b l c -> b c l')
        v = rearrange(v, 'b l c -> b c l')
        Q = self.conv_q(q) # Q \in R^{n   x d_q}
        K = self.conv_k(k) # K \in R^{n_v x d_q}
        V = self.convc_v(v) # V \in R^{n_v x d_v}
                
        # Split matrices for multihead attention
        Q_ = rearrange(Q, 'b (h d) l -> b h l d', h=self.num_heads)
        K_ = rearrange(K, 'b (h d) l -> b h l d', h=self.num_heads)
        V_ = rearrange(V, 'b (h d) l -> b h l d', h=self.num_heads)

        # Compute Softmax attention
        A = torch.einsum('b h q d, b h k d -> b h q k', [Q_, K_]) / math.sqrt(self.dim_embedd)

        if self.rel_pos_enc:
            A = A + torch.einsum('b h q d, q k d -> b h q k', [Q_, aij(self.w_pos_enc, n_k, self.max_pos)])
        if attn_mask is not None:
            A = A.masked_fill(attn_mask, -torch.inf)
        A = torch.softmax(A, dim=-1)
        if valid_mask is not None:
            A = A*(A > valid_mask)
        O = torch.einsum('b h q k, b h k d -> b h q d', [A, V_])
        return self.fc_mh(rearrange(O, 'b h n d -> b n (h d)')), A # combine heads  


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, dim_out, dim_embedd=64, num_heads=1):
        super(MultiHeadAttentionBlock, self).__init__()
        assert dim_out % num_heads == 0, f'dim_out must be divisible by num_heads! {dim_out=}, {num_heads=}'

        self.num_heads = num_heads
        self.dim_embedd = dim_embedd
        self.dim_out = dim_out
        
        self.fc_q = nn.Linear(dim_q, num_heads*dim_embedd)
        self.fc_k = nn.Linear(dim_k, num_heads*dim_embedd)
        self.fc_v = nn.Linear(dim_v, dim_out)
        self.fc_mh = nn.Linear(dim_out, dim_out)
               
    def forward(self, q, k, v, attn_mask=None, valid_mask=None):
        b_q, _, _ = q.shape
        b_k, n_k, _ = k.shape
        b_v, n_v, _ = v.shape
        
        assert b_q == b_k == b_v, f'batch sizes not identical ({b_q}, {b_k}, {b_v})'
        assert n_k == n_v, f'number of keys ({n_k}) != number of values ({n_v})'         
        
        Q = self.fc_q(q) # Q \in R^{n   x d_q}
        K = self.fc_k(k) # K \in R^{n_v x d_q}
        V = self.fc_v(v) # V \in R^{n_v x d_v}
                
        # Split matrices for multihead attention
        Q_ = rearrange(Q, 'b l (h d) -> b h l d', h=self.num_heads)
        K_ = rearrange(K, 'b l (h d) -> b h l d', h=self.num_heads)
        V_ = rearrange(V, 'b l (h d) -> b h l d', h=self.num_heads)

        # Compute Softmax attention
        A = torch.einsum('bhqd, bhkd -> bhqk', [Q_, K_]) / math.sqrt(self.dim_embedd)
        if attn_mask is not None:
            A = A.masked_fill(attn_mask, -torch.inf)
        A = torch.softmax(A, dim=-1)
        if valid_mask is not None:
            A = A*(A > valid_mask)
        O = torch.einsum('b h q k, b h k d -> b h q d', [A, V_])

        # combine heads 
        O = self.fc_mh(rearrange(O, 'b h n d -> b n (h d)'))

        return O, A  


class StereoAttentionModule(nn.Module):

    def __init__(self, in_dim, out_dim, embed_dim, num_heads, max_len=2048, max_masking=2048, valid_mask=None,
                 attn_mask=False, pos_encoding=False, conv_attn=True, ln=False, ff=True, rel_pos_enc=False):

        super().__init__()
        self.attn_mask = attn_mask
        self.ln = ln
        self.ff = ff
        self.valid_mask = valid_mask
        self.pos_encoding = pos_encoding

        # Generate attention masks
        if self.attn_mask:
            attn_mask_left = (torch.triu(torch.ones((max_len, max_len)), diagonal=0) < 0.5)
            attn_mask_right = (torch.triu(torch.ones((max_len, max_len)), diagonal=1) > 0.5)
            
            t1 = torch.linspace(0, max_len-1, max_len, dtype=int).repeat((max_len, 1))
            t2 = torch.linspace(max_len-1, 0, max_len, dtype=int).repeat((max_len, 1)).transpose(0,1)
            m = t1 + t2

            self.attn_mask_left = attn_mask_left.logical_or(m > ((max_len-1) + max_masking))
            self.attn_mask_right = attn_mask_right.logical_or(m < ((max_len-1) - max_masking))

        if ln:
            self.ln_l = nn.LayerNorm(in_dim)
            self.ln_r = nn.LayerNorm(in_dim)
        
        feature_dim = in_dim
        qk_dim = feature_dim + 64 if pos_encoding else feature_dim
        AttentionBlockClass = MultiHeadConvAttentionBlock if conv_attn else MultiHeadAttentionBlock

        self.mha_ltr = AttentionBlockClass(qk_dim, qk_dim, in_dim, feature_dim, embed_dim, num_heads, rel_pos_enc)
        self.mha_rtl = AttentionBlockClass(qk_dim, qk_dim, in_dim, feature_dim, embed_dim, num_heads, rel_pos_enc)
                
        self.ff_l = nn.Linear(feature_dim, out_dim) if self.ff else None
        self.ff_r = nn.Linear(feature_dim, out_dim) if self.ff else None
        
    def forward(self, l, r, pos=None):
        B, _, H, W = l.shape
        
        # Split lines
        l = rearrange(l, 'b c h w -> (b h) w c')
        r = rearrange(r, 'b c h w -> (b h) w c')

        # Apply layernorm
        ln = self.ln_l(l) if self.ln else l
        rn = self.ln_r(r) if self.ln else r

        # Apply positional embedding
        if self.pos_encoding:
            pos = F.interpolate(pos, (H, W))
            pos = rearrange(pos, 'b c h w -> (b h) w c')

            ln = torch.cat((ln, pos), dim=-1)
            rn = torch.cat((rn, pos), dim=-1)
        
        if self.attn_mask:  
            if self.attn_mask_left.device != l.device:
                self.attn_mask_left = self.attn_mask_left.to(l.device)
                self.attn_mask_right = self.attn_mask_right.to(l.device)

        attn_mask_left = self.attn_mask_left[:W, :W] if self.attn_mask else None
        attn_mask_right = self.attn_mask_right[:W, :W] if self.attn_mask else None
        
        l_mha, A_rtl = self.mha_rtl(ln, rn, r, attn_mask_left, valid_mask=self.valid_mask)
        r_mha, A_ltr = self.mha_ltr(rn, ln, l, attn_mask_right, valid_mask=self.valid_mask)
        l = l + l_mha
        r = r + r_mha

        l = l + self.ff_l(l) if self.ff else l
        r = r + self.ff_r(r) if self.ff else r

        # Un-split lines
        l = rearrange(l, '(b h) w c -> b c h w', b=B)
        r = rearrange(r, '(b h) w c -> b c h w', b=B)
        
        return l, r, A_rtl, A_ltr

class StereoAttention(StereoAttentionModule):

    def forward(self, x):
        l, r = x.chunk(2, 1)
        return torch.cat(super().forward(l, r), dim=1)

class Stereo(nn.Module):

    def __init__(self, base_class, *args, shared=False, **kwargs,):
        super().__init__()
        
        if shared:
            self.layer_left = self.layer_right = base_class(*args, **kwargs)
        else:
            self.layer_left = base_class(*args, **kwargs)
            self.layer_right = base_class(*args, **kwargs)
        
    def forward(self, l, r):
        return self.layer_left(l), self.layer_right(r) 

class StereoSequential(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, l, r, return_attn=False, *args, **kwargs):
        A_rtl = []
        A_ltr = []

        for module in self:
            if isinstance(module, StereoAttentionModule):
                module_out = module(l, r, *args, **kwargs)

                l, r = module_out[:2]
                A_rtl.append(module_out[2])
                A_ltr.append(module_out[3])
            else:
                l, r = module(l, r)
        
        if return_attn:
            return l, r, A_rtl, A_ltr
        else:
            return l, r

