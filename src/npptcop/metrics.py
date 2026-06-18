"""Grid-based integrated density-error metrics and a unit-square grid helper."""

import torch
from torch import Tensor


def grid_metrics_density(
  est: Tensor,
  truth: Tensor,
  cell_area: float,
  eps: float = 1e-12,
) -> dict[str, float]:
  """Integrated density errors of ``est`` against ``truth`` on a regular grid.

  Returns ``{"ISE", "IAE", "KL"}`` as Riemann sums weighted by ``cell_area``:
  the integrated squared error, the integrated absolute error, and the
  generalized Kullback-Leibler divergence (I-divergence)
  ``int [truth log(truth / est) - truth + est]``. The generalized form is
  non-negative for any non-negative ``est``/``truth`` (not only normalized
  densities) and reduces to the usual KL when both integrate to one; the
  ``- truth + est`` term is what keeps it non-negative when ``est``
  over-concentrates. Densities are floored at ``eps`` inside the logarithm.
  """
  est = torch.as_tensor(est, dtype=torch.float64)
  truth = torch.as_tensor(truth, dtype=torch.float64)
  est_pos = est.clamp_min(eps)
  truth_pos = truth.clamp_min(eps)
  ise = ((est - truth) ** 2).sum() * cell_area
  iae = (est - truth).abs().sum() * cell_area
  kl = (truth_pos * (truth_pos / est_pos).log() - truth + est).sum() * cell_area
  return {"ISE": float(ise), "IAE": float(iae), "KL": float(kl)}


def unit_grid(size: int = 50, eps: float = 1e-4) -> tuple[Tensor, float]:
  """Cartesian ``(size**2, 2)`` grid on ``[eps, 1 - eps]^2`` and its cell area."""
  u_1d = torch.linspace(eps, 1.0 - eps, size, dtype=torch.float64)
  grid = torch.cartesian_prod(u_1d, u_1d)
  cell_area = float((u_1d[1] - u_1d[0]) ** 2)
  return grid, cell_area
