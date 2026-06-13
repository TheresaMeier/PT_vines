"""Constant-method TLL bandwidth selection, mirroring pyvinecopulib's TllBicop.

Ports the ``constant`` bandwidth path of ``TllBicop::select_bandwidth`` and its
``tools_stats`` helpers (windowed smoother, conditional expectation function,
alternating conditional expectations) to torch. These are correctness-critical:
the stable-sort tie-breaking and the iteration tolerances determine the maximal
correlation, and hence the bandwidth matrix, so they are kept verbatim.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def _pearson_cor(x: Tensor) -> Tensor:
  """Pearson correlation of the two columns of ``x`` (shape ``(n, 2)``)."""
  x0 = x[:, 0] - x[:, 0].mean()
  x1 = x[:, 1] - x[:, 1].mean()
  return (x0 * x1).sum() / ((x0**2).sum().sqrt() * (x1**2).sum().sqrt())


def _win_smoother(x: Tensor, wl: int) -> Tensor:
  """Centered moving average of half-window ``wl`` with flat edge clamps.

  Mirrors ``tools_stats::win``: zero-pad ``x`` by ``wl`` on each side, convolve
  with a uniform kernel of length ``2 * wl + 1``, then clamp the leading and
  trailing ``wl`` entries so the edges are flat.
  """
  n = x.shape[0]
  weight = torch.full(
    (1, 1, 2 * wl + 1), 1.0 / (2 * wl + 1), dtype=x.dtype, device=x.device
  )
  x_padded = F.pad(x.view(1, 1, n), (wl, wl))
  out = F.conv1d(x_padded, weight).view(n).clone()
  if wl > 0:
    out[:wl] = out[wl]
    out[-wl:] = out[n - wl - 1]
  return out


def _cef(x: Tensor, ind: Tensor, ranks: Tensor, wl: int) -> Tensor:
  """``win(x[ind], wl)[ranks]``: smooth in sorted order, map back (``cef``)."""
  return _win_smoother(x[ind], wl)[ranks]


def _ace(
  data: Tensor,
  *,
  outer_iter_max: int = 100,
  inner_iter_max: int = 10,
  outer_abs_tol: float = 2e-15,
  inner_abs_tol: float = 1e-4,
) -> Tensor:
  """Alternating conditional expectations (unweighted bivariate case).

  Mirrors ``tools_stats::ace`` and returns the ``(n, 2)`` ACE-transformed
  scores ``phi`` used to compute the maximal correlation.
  """
  n = data.shape[0]
  dtype, device = data.dtype, data.device
  wl = int(math.ceil(n / 5.0))

  ind = torch.empty(n, 2, dtype=torch.long, device=device)
  ranks = torch.empty(n, 2, dtype=torch.long, device=device)
  for i in range(2):
    order = data[:, i].argsort(stable=True)
    ind[:, i] = order
    ranks[order, i] = torch.arange(n, device=device)

  phi = ranks.to(dtype).clone()
  phi -= (n - 1) / 2.0 - 1.0
  phi /= math.sqrt(n * (n - 1) / 12.0)

  outer_iter, outer_eps, outer_abs_err = 1, 1.0, 1.0
  while outer_iter <= outer_iter_max and outer_abs_err > outer_abs_tol:
    inner_iter, inner_eps, inner_abs_err = 1, 1.0, 1.0
    while inner_iter <= inner_iter_max and inner_abs_err > inner_abs_tol:
      phi[:, 1] = _cef(phi[:, 0], ind[:, 1], ranks[:, 1], wl)
      phi[:, 1] = phi[:, 1] - phi[:, 1].sum() / n
      phi[:, 1] = phi[:, 1] / ((phi[:, 1] ** 2).sum() / (n - 1)).sqrt()
      prev = inner_eps
      inner_eps = ((phi[:, 1] - phi[:, 0]) ** 2).sum().item() / n
      inner_abs_err = abs(prev - inner_eps)
      inner_iter += 1
    phi[:, 0] = _cef(phi[:, 1], ind[:, 0], ranks[:, 0], wl)
    phi[:, 0] = phi[:, 0] - phi[:, 0].sum() / n
    phi[:, 0] = phi[:, 0] / ((phi[:, 0] ** 2).sum() / (n - 1)).sqrt()
    prev = outer_eps
    outer_eps = ((phi[:, 1] - phi[:, 0]) ** 2).sum().item() / n
    outer_abs_err = abs(prev - outer_eps)
    outer_iter += 1

  return phi


def _pairwise_mcor(x: Tensor) -> float:
  """Maximal correlation of ``x`` via ACE followed by Pearson correlation."""
  phi = _ace(x)
  return _pearson_cor(phi).item()


def select_bandwidth_constant(z: Tensor) -> Tensor:
  """Bandwidth matrix for the constant-method local-likelihood KDE.

  Mirrors ``TllBicop::select_bandwidth`` for ``method == "constant"`` on
  probit-scale data ``z`` (shape ``(n, 2)``). Returns the ``(2, 2)`` matrix
  ``mult * cov * scale`` with ``mult = n**(-1/3)``, ``cov`` the unit-variance
  correlation matrix (Pearson correlation clamped to ``[-0.95, 0.95]``), and
  ``scale = |cor / mcor|**(0.5 * mcor)`` the maximal-correlation adjustment.
  """
  n = z.shape[0]
  cor = _pearson_cor(z).clamp(-0.95, 0.95).item()
  cov = torch.tensor([[1.0, cor], [cor, 1.0]], dtype=z.dtype, device=z.device)
  mult = n ** (-1.0 / 3.0)
  mcor = _pairwise_mcor(z)
  scale = abs(cor / mcor) ** (0.5 * mcor)
  return mult * cov * scale
