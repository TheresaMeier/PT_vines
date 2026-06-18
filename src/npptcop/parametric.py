"""Parametric corner tail copula density estimator.

A drop-in alternative to :class:`~npptcop.tail.TailCopula` that fits a classical
single-parameter extreme-value tail copula family by maximum likelihood instead
of the nonparametric KDE. The four families are taken from the X-Vine companion
code (``LL.BiTC``, Kiriliouk-Lee-Segers 2023, arXiv:2312.15205); each is the
lower tail copula density ``lambda`` on ``(0, 1)^2``, homogeneous of degree
``-1``:

- ``husler_reiss`` (``par > 0``);
- ``neg_logistic`` (``par > 0``);
- ``logistic`` (``par > 1``);
- ``dirichlet`` (``par > 0``).

The estimator shares the corner reflection/rescaling and the ``p = k / n`` tail
mass with ``TailCopula``: the only difference is how the tail-conditional density
``h`` is produced. Here ``h(s) = lambda(s; par) / Z(par)`` with the
unit-square normalizer ``Z(par) = \\int_{(0,1)^2} lambda``, which has a closed
form per family (the homogeneity reduces it to a 1-D integral). The targets then
follow the same relations ``r(s) = (p / q) h(s)`` and ``c(s) = (p / q**2) h(s)``.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from scipy.optimize import minimize_scalar
from scipy.special import betainc
from torch import Tensor

from npptcop.tail import _CLAMP_EPS, _ROTATIONS, _corner_tail_data

# The densities are evaluated in log-space: each is a product of powers whose
# exponents grow with ``par``, so the intermediate factors over/underflow at
# extreme points while their product stays finite. Summing the logs (the powers
# cancel) keeps the singular near-origin values finite.


def _lam_husler_reiss(s: Tensor, par: float) -> Tensor:
  """Husler-Reiss tail copula density on ``(0, 1)^2`` (``par > 0``)."""
  x, y = s[:, 0], s[:, 1]
  z = torch.log(x) - torch.log(y) - par / 2.0
  log_lam = (
    -0.5 * math.log(2.0 * math.pi * par) - z**2 / (2.0 * par) - torch.log(x)
  )
  return torch.exp(log_lam)


def _lam_neg_logistic(s: Tensor, par: float) -> Tensor:
  """Negative-logistic tail copula density on ``(0, 1)^2`` (``par > 0``)."""
  lx, ly = torch.log(s[:, 0]), torch.log(s[:, 1])
  log_sum = torch.logaddexp(-par * lx, -par * ly)
  log_lam = (
    math.log1p(par) + (-par - 1.0) * (lx + ly) + (-1.0 / par - 2.0) * log_sum
  )
  return torch.exp(log_lam)


def _lam_logistic(s: Tensor, par: float) -> Tensor:
  """Logistic tail copula density on ``(0, 1)^2`` (``par > 1``)."""
  lx, ly = torch.log(s[:, 0]), torch.log(s[:, 1])
  log_sum = torch.logaddexp(par * lx, par * ly)
  log_lam = (
    math.log(par - 1.0) + (par - 1.0) * (lx + ly) + (1.0 / par - 2.0) * log_sum
  )
  return torch.exp(log_lam)


def _lam_dirichlet(s: Tensor, par: float) -> Tensor:
  """Dirichlet tail copula density on ``(0, 1)^2`` (``par > 0``)."""
  x, y = s[:, 0], s[:, 1]
  log_b = (
    math.lgamma(par + 1.0) + math.lgamma(par) - math.lgamma(2.0 * par + 1.0)
  )
  log_lam = (
    -log_b
    - (2.0 * par + 1.0) * torch.log(x + y)
    + par * (torch.log(x) + torch.log(y))
  )
  return torch.exp(log_lam)


def _phi(z: float) -> float:
  """Standard-normal CDF ``Phi`` at a scalar ``z`` (stdlib ``erf``)."""
  return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_husler_reiss(par: float) -> float:
  """Unit-square normalizer ``Z = 2 - 2 Phi(sqrt(par) / 2)``."""
  return 2.0 - 2.0 * _phi(math.sqrt(par) / 2.0)


def _norm_neg_logistic(par: float) -> float:
  """Unit-square normalizer ``Z = 2**(-1 / par)``."""
  return 2.0 ** (-1.0 / par)


def _norm_logistic(par: float) -> float:
  """Unit-square normalizer ``Z = 2 - 2**(1 / par)``."""
  return 2.0 - 2.0 ** (1.0 / par)


def _norm_dirichlet(par: float) -> float:
  """Unit-square normalizer ``Z = 2 I_{1/2}(par + 1, par)`` (regularized beta)."""
  return 2.0 * float(betainc(par + 1.0, par, 0.5))


@dataclass(frozen=True)
class _Family:
  """A parametric tail family: density, unit-square normalizer, and bounds.

  ``bounds`` are the open ``par`` range passed to ``minimize_scalar``; the
  logistic lower bound stays strictly above ``1`` so its ``(par - 1)`` prefactor
  never vanishes.
  """

  name: str
  lam: Callable[[Tensor, float], Tensor]
  norm: Callable[[float], float]
  bounds: tuple[float, float]


_FAMILIES: dict[str, _Family] = {
  "husler_reiss": _Family(
    "husler_reiss", _lam_husler_reiss, _norm_husler_reiss, (1e-4, 50.0)
  ),
  "neg_logistic": _Family(
    "neg_logistic", _lam_neg_logistic, _norm_neg_logistic, (1e-4, 50.0)
  ),
  "logistic": _Family(
    "logistic", _lam_logistic, _norm_logistic, (1.0 + 1e-4, 50.0)
  ),
  "dirichlet": _Family(
    "dirichlet", _lam_dirichlet, _norm_dirichlet, (1e-4, 50.0)
  ),
}


@dataclass(frozen=True)
class ParametricTailFit:
  """Fitted parametric tail summary.

  Mirrors :class:`~npptcop.tail.TailFit` (cutoff ``q``, tail mass ``p = k / n``,
  tail count ``k``, sample size ``n``) and adds the chosen ``family``, the fitted
  parameter ``par``, and the final negative log-likelihood ``nll``.
  """

  q: float
  p: float
  k: int
  n: int
  family: str
  par: float
  nll: float


class ParametricTailCopula:
  """Parametric corner tail copula density estimator.

  Args:
    family: tail family, one of ``"husler_reiss"``, ``"neg_logistic"``,
      ``"logistic"``, ``"dirichlet"``.
    q: tail cutoff in ``(0, 1)``; the corner block has size ``q`` per axis.
    rotation: corner selector in ``{0, 90, 180, 270}`` (pyvinecopulib
      convention); ``0`` is the lower-left tail.
  """

  def __init__(self, family: str, q: float, rotation: int = 0) -> None:
    if family not in _FAMILIES:
      raise ValueError(
        f"family must be one of {tuple(_FAMILIES)}, got {family!r}"
      )
    if not 0.0 < q < 1.0:
      raise ValueError(f"q must lie in (0, 1), got {q}")
    if rotation not in _ROTATIONS:
      raise ValueError(f"rotation must be one of {_ROTATIONS}, got {rotation}")
    self.family = family
    self.q = float(q)
    self.rotation = rotation
    self._fam = _FAMILIES[family]

  def fit(self, u: Tensor) -> "ParametricTailCopula":
    """Extract the rescaled corner data and fit ``par`` by maximum likelihood.

    Maximizes the unit-square conditional likelihood of ``h = lambda / Z`` over
    the rescaled tail observations via a bounded scalar search; records the tail
    mass ``p = k / n``.
    """
    s_data, k, n = _corner_tail_data(u, self.q, self.rotation)
    fam = self._fam

    def nll(par: float) -> float:
      return float(
        -torch.log(fam.lam(s_data, par)).sum() + k * math.log(fam.norm(par))
      )

    res = minimize_scalar(nll, bounds=fam.bounds, method="bounded")
    self._par = float(res.x)
    self._nll = float(res.fun)
    self._Z = fam.norm(self._par)
    self._n = n
    self._k = k
    self._p = k / n
    return self

  def h(self, s: Tensor) -> Tensor:
    """Tail-conditional density ``h(s) = lambda(s; par) / Z`` on ``[0, 1]^2``."""
    s = torch.as_tensor(s, dtype=torch.float64).clamp(
      _CLAMP_EPS, 1.0 - _CLAMP_EPS
    )
    return self._fam.lam(s, self._par) / self._Z

  def c(self, s: Tensor) -> Tensor:
    """Copula density ``c(q s) = (p / q**2) h(s)`` at rescaled points ``s``."""
    return (self._p / self.q**2) * self.h(s)

  def r(self, s: Tensor) -> Tensor:
    """Tail copula density ``r(s) = q c(q s) = (p / q) h(s)``."""
    return (self._p / self.q) * self.h(s)

  @property
  def fit_(self) -> ParametricTailFit:
    """Fitted summary ``(q, p, k, n, family, par, nll)``."""
    return ParametricTailFit(
      q=self.q,
      p=self._p,
      k=self._k,
      n=self._n,
      family=self.family,
      par=self._par,
      nll=self._nll,
    )
