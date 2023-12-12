import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import Dict, tictoc, quantise
from .layers import *
from .metrics import laplace_cdf, calc_rate
from einops import rearrange

__all__ = ['DebugModel', 'StereoBaseline', 'StereoAttentionModelPlus']


class StereoBaseline(nn.Module):
    """
    Baseline model for stereo image compression. Encoder/decoder with hyperprior entropy model. 
    Left and right image are compressed separately. Is used as base class for ECSIC.
    """

    def __init__(self, in_channels=3, N=192, M=12):
        super().__init__()
        
        self.E = StereoSequential(
            Stereo(nn.Conv2d, in_channels, N, 3, 2, 1),					                # 1
            Stereo(nn.PReLU, N, init=0.2),
            Stereo(nn.Conv2d, N, N, 3, 2, 1),							                # 2
            Stereo(nn.PReLU, N, init=0.2),
            Stereo(nn.Conv2d, N, N, 3, 2, 1),							                # 4
            Stereo(nn.PReLU, N, init=0.2),
            Stereo(nn.Conv2d, N, M, 3, 1, 1)							                # 8
        )

        self.D = StereoSequential(
            Stereo(nn.ConvTranspose2d, M, N, 3, 2, 1, 1),				                # 8
            Stereo(nn.PReLU, N, init=0.2),
            Stereo(nn.ConvTranspose2d, N, N, 3, 2, 1, 1),				                # 4
            Stereo(nn.PReLU, N, init=0.2),
            Stereo(nn.ConvTranspose2d, N, N, 3, 2, 1, 1),				                # 2
            Stereo(nn.PReLU, N, init=0.2),
            Stereo(nn.ConvTranspose2d, N, in_channels, 3, 1, 1, 0)                      # 1
        )
    
        self.HE = StereoSequential(
            Stereo(nn.Conv2d, M, N, 3, 2, 1),							                # 8
            Stereo(nn.PReLU, N, init=0.2),
            Stereo(nn.Conv2d, N, N, 3, 2, 1),							                # 16
            Stereo(nn.PReLU, N, init=0.2),
            Stereo(nn.Conv2d, N, M, 3, 1, 1)							                # 32
        )

        self.HD = StereoSequential(
            Stereo(nn.ConvTranspose2d, M, N, 3, 2, 1, 1),				                # 32
            Stereo(nn.PReLU, N, init=0.2),
            Stereo(nn.ConvTranspose2d, N, N, 3, 2, 1, 1),				                # 16
            Stereo(nn.PReLU, N, init=0.2),
            Stereo(nn.Conv2d, N, 2*M, 3, 1, 1),				                            # 8
        )

        self.zl_loc = nn.Parameter(torch.empty((1, M, 1, 1)))
        self.zl_loc.data.normal_(0.0, 1.0)

        self.zl_scale = nn.Parameter(torch.empty((1, M, 1, 1)))
        self.zl_scale.data.uniform_(1.0, 1.5)

        self.zr_loc = nn.Parameter(torch.empty((1, M, 1, 1)))
        self.zr_loc.data.normal_(0.0, 1.0)

        self.zr_scale = nn.Parameter(torch.empty((1, M, 1, 1)))
        self.zr_scale.data.uniform_(1.0, 1.5)

    def entropy(self, yl, yr, pos=None):
        zl, zr, ahe_rtl, ahe_ltr = self.HE(yl, yr, pos=pos, return_attn=True)     

        # Quantise z
        zl_hat_ent, zl_hat_dec = quantise(zl, self.zl_loc, training=self.training)
        zr_hat_ent, zr_hat_dec = quantise(zr, self.zr_loc, training=self.training)

        # Compute probability parameters for y
        yl_entropy, yr_entropy, ahd_rtl, ahd_ltr = self.HD(zl_hat_dec, zr_hat_dec, pos=pos, return_attn=True)
        yl_loc, yl_scale = torch.chunk(yl_entropy, chunks=2, dim=1)
        yr_loc, yr_scale = torch.chunk(yr_entropy, chunks=2, dim=1)

        # Quantise y
        yl_hat_ent, yl_hat_dec = quantise(yl, yl_loc, training=self.training)
        yr_hat_ent, yr_hat_dec = quantise(yr, yr_loc, training=self.training)

        latents = Dict(
            left = Dict(
                y_hat_ent=yl_hat_ent,
                y_hat_dec=yl_hat_dec,
                y_loc=yl_loc,
                y_scale=yl_scale,

                z_hat_ent=zl_hat_ent,
                z_hat_dec=zl_hat_dec,
                z_loc=self.zl_loc,
                z_scale=self.zl_scale
            ),
            right = Dict(
                y_hat_ent=yr_hat_ent,
                y_hat_dec=yr_hat_dec,
                y_loc=yr_loc,
                y_scale=yr_scale,

                z_hat_ent=zr_hat_ent,
                z_hat_dec=zr_hat_dec,
                z_loc=self.zr_loc,
                z_scale=self.zr_scale
            )
        )

        return latents

    def forward(self, xl, xr, pos=None):
        # forward pass through model
        yl, yr, ae_rtl, ae_ltr = self.E(xl, xr, pos=pos, return_attn=True)
        latents = self.entropy(yl, yr, pos)
        xl_hat, xr_hat, ad_rtl, ad_ltr = self.D(latents.left.y_hat_dec, latents.right.y_hat_dec, pos=pos, return_attn=True)

        # Calculate rates for z and y
        bpp_zl = calc_rate(latents.left.z_hat_ent, latents.left.z_loc, latents.left.z_scale)
        bpp_zr = calc_rate(latents.right.z_hat_ent, latents.right.z_loc, latents.right.z_scale)
        bpp_yl = calc_rate(latents.left.y_hat_ent, latents.left.y_loc, latents.left.y_scale)
        bpp_yr = calc_rate(latents.right.y_hat_ent, latents.right.y_loc, latents.right.y_scale)

        return Dict(
            latents=latents,
            rate=Dict(
                left = Dict(
                    y=bpp_yl,
                    z=bpp_zl
                ),
                right = Dict(
                    y=bpp_yr,
                    z=bpp_zr
                )
            ),
            pred=Dict(
                left=xl_hat,
                right=xr_hat
            )
        )


class StereoAttentionModelPlus(StereoBaseline):

    def __init__(self, in_channels=3, N=192, M=12, z_context=True, y_context=True, attn_mask=False, pos_encoding=False, 
                 ln=True, shared=True, ff=True, valid_mask=None, rel_pos_enc=False, embed=None, heads=4, only_D=False):
        super().__init__(in_channels=in_channels, N=N, M=M)

        self.z_context = z_context
        self.y_context = y_context
        args = {
            'attn_mask': attn_mask,
            'pos_encoding': pos_encoding,
            'ln': ln,
            'ff': ff,
            'valid_mask': valid_mask,
            'rel_pos_enc': rel_pos_enc
        }
        embed = 2*N if embed is None else embed
        heads = heads

 
        self.E = StereoSequential(
            Stereo(nn.Conv2d, in_channels, N, 3, 2, 1, shared=shared),					# 1
            Stereo(nn.PReLU, N, init=0.2, shared=shared),
            Stereo(nn.Conv2d, N, N, 3, 2, 1, shared=shared),							# 2
            Stereo(nn.PReLU, N, init=0.2, shared=shared),
            Stereo(nn.Conv2d, N, N, 3, 2, 1, shared=shared),							# 4
            Stereo(nn.PReLU, N, init=0.2, shared=shared),
            StereoAttentionModule(N, N, embed, heads, **args),	                        # 8
            Stereo(nn.PReLU, N, init=0.2),
            Stereo(nn.Conv2d, N, M, 3, 1, 1)							                # 8
        )

        self.D = StereoSequential(
            Stereo(nn.ConvTranspose2d, M, N, 3, 2, 1, 1),				                # 8
            Stereo(nn.PReLU, N, init=0.2),
            StereoAttentionModule(N, N, embed, heads, **args),	                        # 8
            Stereo(nn.PReLU, N, init=0.2),
            Stereo(nn.ConvTranspose2d, N, N, 3, 2, 1, 1),				                # 4
            Stereo(nn.PReLU, N, init=0.2),
            Stereo(nn.ConvTranspose2d, N, N, 3, 2, 1, 1),				                # 2
            Stereo(nn.PReLU, N, init=0.2),
            Stereo(nn.ConvTranspose2d, N, in_channels, 3, 1, 1, 0)                      # 1
        )
    
        self.HE = StereoSequential(
            Stereo(nn.Conv2d, M, N, 3, 2, 1),							                # 8
            Stereo(nn.PReLU, N, init=0.2),
            Stereo(nn.Conv2d, N, N, 3, 2, 1),							                # 16
            Stereo(nn.PReLU, N, init=0.2),
            StereoAttentionModule(N, N, embed, heads, **args),		                    # 16
            Stereo(nn.PReLU, N, init=0.2),
            Stereo(nn.Conv2d, N, M, 3, 1, 1)							                # 32
        )

        self.HD = StereoSequential(
            Stereo(nn.ConvTranspose2d, M, N, 3, 2, 1, 1),				                # 32
            Stereo(nn.PReLU, N, init=0.2),
            StereoAttentionModule(N, N, embed, heads, **args),		                    # 16
            Stereo(nn.PReLU, N, init=0.2),
            Stereo(nn.ConvTranspose2d, N, N, 3, 2, 1, 1),				                # 16
            Stereo(nn.PReLU, N, init=0.2),
        )
        self.hd_out_left = nn.Conv2d(N, 2*M, 3, 1, 1)

        if self.z_context:
            self.zC = nn.Sequential(
                nn.Conv2d(3*M, 4*M, 3, 1, 1),
                nn.PReLU(4*M, init=0.2),
                nn.Conv2d(4*M, 4*M, 3, 1, 1),
                nn.PReLU(4*M, init=0.2),
                nn.Conv2d(4*M, 2*M, 3, 1, 1)
            )

        if self.y_context:
            self.hd_out_right = nn.Conv2d(N, N, 3, 1, 1)

            self.yle_conv = nn.Conv2d(M, N, 3, 1, 1)
            self.yle_prelu = nn.PReLU(N, init=0.2)
            self.yre_conv = nn.Conv2d(N, N, 3, 1, 1)
            self.yre_prelu = nn.PReLU(N, init=0.2)
            self.ye_sa = StereoAttentionModule(N, N, embed, heads, max_len=256, max_masking=32, **args)
            self.ye_prelu = nn.PReLU(2*N, init=0.2)
            self.ye_conv = nn.Conv2d(2*N, 2*M, 3, 1, 1)
        else:
            self.hd_out_right = nn.Conv2d(N, 2*M, 3, 1, 1)

    def yr_entropy(self, yl_hat, yr_entropy, pos):
        yle = self.yle_prelu(self.yle_conv(yl_hat))
        yre = self.yre_prelu(self.yre_conv(yr_entropy))
        yle, yre, ay_rtl, ay_ltr = self.ye_sa(yle, yre, pos)
        ye = self.ye_prelu(torch.cat([yle, yre], dim=1))
        ye = self.ye_conv(ye)

        yr_loc, yr_scale = ye.chunk(2, 1)

        return yr_loc, yr_scale, ay_rtl, ay_ltr

    def zr_entropy(self, zl_hat):
        s = zl_hat.shape

        zC_in = torch.cat([self.zr_loc.expand(s), self.zr_scale.expand(s), zl_hat], dim=1)
        zC_out = self.zC(zC_in)
        zr_loc, zr_scale = zC_out.chunk(2, 1)
        zr_scale = F.relu(zr_scale)

        return zr_loc, zr_scale

    def entropy(self, yl, yr, pos=None):
        zl, zr, ahe_rtl, ahe_ltr = self.HE(yl, yr, pos=pos, return_attn=True)     

        # Quantise z
        zl_hat_ent, zl_hat_dec = quantise(zl, self.zl_loc, training=self.training)

        if self.z_context:
            zr_loc, zr_scale = self.zr_entropy(zl_hat_dec)
        else:
            zr_loc, zr_scale = self.zr_loc, self.zr_scale
        zr_hat_ent, zr_hat_dec = quantise(zr, zr_loc, training=self.training)

        # Compute probability parameters for y
        yl_entropy, yr_entropy, ahd_rtl, ahd_ltr = self.HD(zl_hat_dec, zr_hat_dec, pos=pos, return_attn=True)
        yl_entropy = self.hd_out_left(yl_entropy)
        yr_entropy = self.hd_out_right(yr_entropy)
        yl_loc, yl_scale = torch.chunk(yl_entropy, chunks=2, dim=1)

        # Quantise y left
        yl_hat_ent, yl_hat_dec = quantise(yl, yl_loc, training=self.training)

        # Compute probability parameters for y right
        if self.y_context:
            yr_loc, yr_scale, ay_rtl, ay_ltr = self.yr_entropy(yl_hat_dec, yr_entropy, pos=pos)
        else:
            yr_loc, yr_scale = torch.chunk(yr_entropy, chunks=2, dim=1)
            ay_rtl, ay_ltr = [], []
        yr_hat_ent, yr_hat_dec = quantise(yr, yr_loc, training=self.training)

        latents = Dict(
            left = Dict(
                y_hat_ent=yl_hat_ent,
                y_hat_dec=yl_hat_dec,
                y_loc=yl_loc,
                y_scale=yl_scale,

                z_hat_ent=zl_hat_ent,
                z_hat_dec=zl_hat_dec,
                z_loc=self.zl_loc,
                z_scale=self.zl_scale
            ),
            right = Dict(
                y_hat_ent=yr_hat_ent,
                y_hat_dec=yr_hat_dec,
                y_loc=yr_loc,
                y_scale=yr_scale,

                z_hat_ent=zr_hat_ent,
                z_hat_dec=zr_hat_dec,
                z_loc=zr_loc,
                z_scale=zr_scale
            )
        )

        return latents


class DebugModel(StereoBaseline):

    def __init__(self, in_channels=3, M=6):
        super().__init__(in_channels=in_channels, M=M)
        s = 'DEBUG MODEL IS USED!'
        print('#'*(len(s) + 4) + f'\n# {s} #\n' + '#'*(len(s) + 4))

        self.E = Stereo(nn.Conv2d, in_channels, M, 3, 1, 1)
        self.D = Stereo(nn.Conv2d, M, in_channels, 3, 1, 1)
        self.HE = Stereo(nn.Conv2d, M, M, 3, 1, 1)
        self.HD = Stereo(nn.Conv2d, M, 2*M, 3, 1, 1)