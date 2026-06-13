"""Standard-normal transforms for the probit-scale TLL estimators.

``qnorm`` is the probit map ``Phi^{-1}`` and ``dnorm`` the standard-normal
density ``phi``; both are applied elementwise and underpin the back-transform
from the Gaussian KDE on the probit scale to a copula density on ``[0, 1]^2``.
"""

import math

import torch
from torch import Tensor

SQRT_2PI_INV: float = 1.0 / math.sqrt(2.0 * math.pi)


def qnorm(p: Tensor) -> Tensor:
  """Standard-normal quantile function ``Phi^{-1}``, applied elementwise."""
  return torch.special.ndtri(p)


def dnorm(z: Tensor) -> Tensor:
  """Standard-normal probability density ``phi``, applied elementwise."""
  return torch.exp(-0.5 * z * z) * SQRT_2PI_INV
