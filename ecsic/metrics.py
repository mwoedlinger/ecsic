import torch
from torch.autograd import Function
import torch.nn.functional as F

__all__ = ['calc_bpp', 'calc_mse', 'calc_psnr', 'laplace_cdf', 'calc_rate']

def calc_bpp(rate, image):
	H, W = image.shape[-2:]
	return rate.mean() / (H * W)

def calc_mse(target, pred):
	mse = torch.mean((target - pred) ** 2, dim=(-1, -2, -3)) * 255.0 ** 2
	if mse.shape[0] == 1:
		mse = mse[0]
	return mse.mean()

def calc_psnr(mse, eps):
	mse = F.threshold(mse, eps, eps)
	psnr = 10. * torch.log10(255. ** 2 / mse)
	return psnr

class LaplaceCDF(torch.autograd.Function):
	"""
	CDF of the Laplacian distribution.
	"""

	@staticmethod
	def forward(ctx, x):
		s = torch.sign(x)
		expm1 = torch.expm1(-x.abs())
		ctx.save_for_backward(expm1)
		return 0.5 - 0.5 * s * expm1

	@staticmethod
	def backward(ctx, grad_output):
		expm1, = ctx.saved_tensors
		return 0.5 * grad_output * (expm1 + 1)

def _standard_cumulative_laplace(input):
	"""
	CDF of the Laplacian distribution.
	"""
	return LaplaceCDF.apply(input)

def laplace_cdf(input):
	""" 
	Computes CDF of standard Laplace distribution
	"""
	return _standard_cumulative_laplace(input)

class LowerBound(Function):
	""" Applies a lower bounded threshold function on to the inputs
		ensuring all scalars in the input >= bound.
		
		Gradients are propagated for values below the bound (as opposed to
		the built in PyTorch operations such as threshold and clamp)
	"""

	@staticmethod
	def forward(ctx, inputs, bound):
		b = torch.ones(inputs.size(), device=inputs.device, dtype=inputs.dtype) * bound
		ctx.save_for_backward(inputs, b)
		return torch.max(inputs, b)

	@staticmethod
	def backward(ctx, grad_output):
		inputs, b = ctx.saved_tensors

		pass_through_1 = inputs >= b
		pass_through_2 = grad_output < 0

		pass_through = pass_through_1 | pass_through_2
		return pass_through.type(grad_output.dtype) * grad_output, None

def calc_rate(y_q, mean, scale, sigma_lower_bound=0.1, likelihood_lower_bound=1e-9, offset=0.5, per_channel=False):
	"""
	Rate loss estimation of quantised latent variables using the provided CDF function (default = Laplacian CDF)
	Computation is performed per batch (across, channels, height, width), i.e. return shape is [BATCH]
	"""
	scale = LowerBound.apply(scale, sigma_lower_bound)
	y_q0 = y_q - mean
	y_q0 = y_q0.abs()
	upper = laplace_cdf(( offset - y_q0) / scale)
	lower = laplace_cdf((-offset - y_q0) / scale)
	likelihood = upper - lower
	likelihood = LowerBound.apply(likelihood, likelihood_lower_bound)

	if per_channel:
		total_bits = -torch.sum(torch.log2(likelihood), dim=(-1, -2))
	else:
		total_bits = -torch.sum(torch.log2(likelihood), dim=(-1, -2, -3))
	return total_bits