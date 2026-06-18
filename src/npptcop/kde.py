"""Probit-transform local-likelihood Gaussian KDE (Geenens TLL, constant method).

Fitting probit-transforms copula-scale data, estimates the density there with a
bivariate Gaussian kernel and a ``(2, 2)`` bandwidth matrix, and back-transforms
to ``[0, 1]^2``. This is the paper's bulk estimator ``c_B`` and the building
block reused by the tail estimator.
"""

import torch
from torch import Tensor

from npptcop.bandwidth import select_bandwidth_constant
from npptcop.transforms import SQRT_2PI_INV, dnorm, qnorm

_CLAMP_EPS = 1e-6


class ProbitTLL:
  """Geenens (2017) probit-transform local-likelihood Gaussian KDE.

  ``density`` returns the raw probit-scale estimate ``f_hat`` (used by the tail
  estimator, which divides by its own ``phi`` factors); ``evaluate`` returns the
  back-transformed bulk copula density ``c_B(u) = f_hat(Phi^{-1} u) / (phi phi)``.
  """

  def __init__(
    self, bandwidth: Tensor | None = None, ridge: float = 0.0
  ) -> None:
    """Optionally fix the ``(2, 2)`` bandwidth; if ``None`` select it at fit.

    ``ridge`` is the non-negative diagonal regularization added to a *selected*
    bandwidth (ignored when an explicit ``bandwidth`` is given); see
    :func:`npptcop.bandwidth.select_bandwidth_constant`.
    """
    self.bandwidth = (
      None
      if bandwidth is None
      else torch.as_tensor(bandwidth, dtype=torch.float64)
    )
    self.ridge = ridge

  def fit(self, u: Tensor) -> "ProbitTLL":
    """Probit-transform ``u`` (shape ``(n, 2)`` in ``(0, 1)``), select/store the
    bandwidth, and cache the whitening factors used by ``density``.
    """
    u = torch.as_tensor(u, dtype=torch.float64)
    z_data = qnorm(u.clamp(_CLAMP_EPS, 1.0 - _CLAMP_EPS))
    bandwidth = (
      select_bandwidth_constant(z_data, ridge=self.ridge)
      if self.bandwidth is None
      else self.bandwidth.to(z_data)
    )
    chol = torch.linalg.cholesky(bandwidth)
    inv_chol = torch.linalg.inv(chol)
    self.bandwidth_ = bandwidth
    self._inv_chol = inv_chol
    self._det_inv_chol = torch.linalg.det(inv_chol)
    self._z_data_std = (inv_chol @ z_data.T).T
    return self

  def density(self, z: Tensor) -> Tensor:
    """Estimated probit-scale density ``f_hat`` at points ``z`` (``(m, 2)``)."""
    z = torch.as_tensor(z, dtype=torch.float64)
    z_eval_std = (self._inv_chol @ z.T).T
    diff = self._z_data_std.unsqueeze(0) - z_eval_std.unsqueeze(1)
    kernel = (
      torch.exp(-0.5 * (diff[..., 0] ** 2 + diff[..., 1] ** 2))
      * SQRT_2PI_INV
      * SQRT_2PI_INV
      * self._det_inv_chol
    )
    return kernel.mean(dim=1)

  def evaluate(self, u: Tensor) -> Tensor:
    """Bulk copula density ``c_B`` at copula points ``u`` (``(m, 2)``)."""
    u = torch.as_tensor(u, dtype=torch.float64)
    z = qnorm(u.clamp(_CLAMP_EPS, 1.0 - _CLAMP_EPS))
    return self.density(z) / (dnorm(z[:, 0]) * dnorm(z[:, 1]))
