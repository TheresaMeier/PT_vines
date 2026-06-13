"""Golden regression test: reproduce the canonical notebook run.

Frozen from the torch backend at Clayton ``theta = 3``, ``n = 2000``,
``seed = 97``, ``rotation = 0``, ``q = n**(-1/2)``, ``grid_size = 50`` (captured
2026-06-12). Only the integer tail count ``k`` is asserted exactly; floating
quantities use a tight tolerance because reduction order is not bit-portable
across BLAS/thread counts.
"""

from collections.abc import Callable

import pyvinecopulib as pv
import torch
from torch import Tensor

from npptcop import TailCopula, grid_metrics_density, unit_grid

Q = 2000 ** (-0.5)
PTS = torch.tensor([[0.25, 0.25], [0.5, 0.5], [0.9, 0.1]])

GOLDEN_K = 33
GOLDEN_P = 0.0165
GOLDEN_B = [
  [0.3036575961611723, 0.2055985186840647],
  [0.2055985186840647, 0.3036575961611723],
]
GOLDEN_H = [2.0798152874392266, 1.581874145687043, 0.0025467203354444]
GOLDEN_C = [68.63390448549448, 52.201846807672425, 0.08404177106966519]
GOLDEN_R = [1.5347007599079339, 1.1672687801298594, 0.0018792311306124658]
GOLDEN_ISE = 26063.69178393582
GOLDEN_IAE = 0.08100162037777786
GOLDEN_KL = 0.43130519223270236


def _sample(clayton: pv.Bicop) -> Tensor:
  return torch.as_tensor(
    clayton.simulate(2000, seeds=[97]), dtype=torch.float64
  )


def test_golden_summary(clayton: pv.Bicop, tight: Callable[..., bool]) -> None:
  est = TailCopula(Q).fit(_sample(clayton))
  assert est.fit_.k == GOLDEN_K  # integer-deterministic given the seed
  assert tight(est.fit_.p, GOLDEN_P)
  assert tight(est.fit_.bandwidth, torch.tensor(GOLDEN_B))


def test_golden_pointwise(
  clayton: pv.Bicop, tight: Callable[..., bool]
) -> None:
  est = TailCopula(Q).fit(_sample(clayton))
  assert tight(est.h(PTS), torch.tensor(GOLDEN_H))
  assert tight(est.c(PTS), torch.tensor(GOLDEN_C))
  assert tight(est.r(PTS), torch.tensor(GOLDEN_R))


def test_golden_metrics(clayton: pv.Bicop, tight: Callable[..., bool]) -> None:
  est = TailCopula(Q).fit(_sample(clayton))
  grid, cell_area = unit_grid(50)
  c_true = torch.as_tensor(clayton.pdf((grid * Q).numpy()), dtype=torch.float64)
  metrics = grid_metrics_density(est.c(grid), c_true, (Q**2) * cell_area)
  assert tight(metrics["ISE"], GOLDEN_ISE)
  assert tight(metrics["IAE"], GOLDEN_IAE)
  assert tight(metrics["KL"], GOLDEN_KL)
