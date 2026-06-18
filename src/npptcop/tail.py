"""Nonparametric corner tail copula density estimator (the method's second step).

For observations falling in a corner block of size ``q``, rescale them to
``[0, 1]^2``, fit a probit-transform local-likelihood Gaussian KDE there, and
back-transform. The three targets are the tail-conditional density ``h`` (a
density on ``[0, 1]^2``), the copula density ``c`` on the corner block, and the
tail copula density ``r`` (the paper's ``lambda``), related by
``c(q s) = (p / q**2) h(s)`` and ``r(s) = q c(q s) = (p / q) h(s)`` where
``p = k / n`` is the tail mass. Any of the four corners is selected through
``rotation``, matching pyvinecopulib's density-rotation (axis-reflection)
convention: ``0`` lower-left, ``90`` lower-right, ``180`` upper-right (the
"upper tail"), ``270`` upper-left.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from npptcop.kde import ProbitTLL

_CLAMP_EPS = 1e-6
_ROTATIONS = (0, 90, 180, 270)


def _reflect(u: Tensor, rotation: int) -> Tensor:
  """Map the corner selected by ``rotation`` onto the lower-left ``[0, q]^2``.

  ``rotation`` is validated by ``TailCopula.__init__``; the ``270`` branch is the
  fallthrough.
  """
  if rotation == 0:
    return u
  if rotation == 90:
    return torch.stack((1.0 - u[:, 0], u[:, 1]), dim=1)
  if rotation == 180:
    return 1.0 - u
  return torch.stack((u[:, 0], 1.0 - u[:, 1]), dim=1)


def _corner_tail_data(
  u: Tensor, q: float, rotation: int
) -> tuple[Tensor, int, int]:
  """Extract the rescaled corner observations shared by the tail estimators.

  Reflect ``u`` to the chosen corner, mask to ``[0, q]^2``, rescale by ``1 / q``,
  and clamp to ``(0, 1)``. Returns ``(s_data, k, n)`` where ``s_data`` is the
  ``(k, 2)`` tensor of rescaled tail points, ``k`` the tail count, and ``n`` the
  sample size; raises ``ValueError`` if the tail block is empty.
  """
  u = torch.as_tensor(u, dtype=torch.float64)
  n = u.shape[0]
  corner = _reflect(u, rotation)
  mask = (corner[:, 0] <= q) & (corner[:, 1] <= q)
  tail = corner[mask]
  k = int(tail.shape[0])
  if k == 0:
    raise ValueError(f"no observations fall in the tail block [0, {q}]^2")
  s_data = (tail / q).clamp(_CLAMP_EPS, 1.0 - _CLAMP_EPS)
  return s_data, k, n


@dataclass(frozen=True)
class TailFit:
  """Fitted tail summary: cutoff ``q``, tail mass ``p = k / n``, tail count ``k``,
  sample size ``n``, and the selected ``(2, 2)`` bandwidth matrix.
  """

  q: float
  p: float
  k: int
  n: int
  bandwidth: Tensor


class TailCopula:
  """Nonparametric corner tail copula density estimator.

  Args:
    q: tail cutoff in ``(0, 1)``; the corner block has size ``q`` per axis.
    bandwidth: optional fixed ``(2, 2)`` bandwidth; if ``None`` it is selected
      on the rescaled, probit-transformed tail observations at fit time.
    rotation: corner selector in ``{0, 90, 180, 270}`` (pyvinecopulib
      convention); ``0`` is the lower-left tail.
    ridge: non-negative diagonal regularization for the selected bandwidth,
      guarding against a singular bandwidth on small or weakly dependent tail
      samples; ``0`` (the default) leaves the selection unregularized.
  """

  def __init__(
    self,
    q: float,
    bandwidth: Tensor | None = None,
    rotation: int = 0,
    ridge: float = 0.0,
  ) -> None:
    if not 0.0 < q < 1.0:
      raise ValueError(f"q must lie in (0, 1), got {q}")
    if rotation not in _ROTATIONS:
      raise ValueError(f"rotation must be one of {_ROTATIONS}, got {rotation}")
    self.q = float(q)
    self.rotation = rotation
    self._kde = ProbitTLL(bandwidth, ridge=ridge)

  def fit(self, u: Tensor) -> "TailCopula":
    """Reflect to the chosen corner, mask to ``[0, q]^2``, rescale by ``1 / q``,
    and fit the tail KDE; records the tail mass ``p = k / n``.
    """
    s_data, k, n = _corner_tail_data(u, self.q, self.rotation)
    self._kde.fit(s_data)
    self._n = n
    self._k = k
    self._p = k / n
    return self

  def h(self, s: Tensor) -> Tensor:
    """Tail-conditional density on ``[0, 1]^2`` at rescaled points ``s``."""
    return self._kde.evaluate(s)

  def c(self, s: Tensor) -> Tensor:
    """Copula density ``c(q s) = (p / q**2) h(s)`` at rescaled points ``s``."""
    return (self._p / self.q**2) * self.h(s)

  def r(self, s: Tensor) -> Tensor:
    """Tail copula density ``r(s) = q c(q s) = (p / q) h(s)``."""
    return (self._p / self.q) * self.h(s)

  @property
  def fit_(self) -> TailFit:
    """Fitted summary ``(q, p, k, n, bandwidth)``."""
    return TailFit(
      q=self.q,
      p=self._p,
      k=self._k,
      n=self._n,
      bandwidth=self._kde.bandwidth_,
    )
