"""Cross-cutting invariants: public API surface and vectorization."""

from collections.abc import Callable

import torch
from torch import Tensor

import npptcop
from npptcop import ProbitTLL, TailCopula, unit_grid


def test_public_api_surface() -> None:
  assert set(npptcop.__all__) == {
    "ProbitTLL",
    "TailCopula",
    "TailFit",
    "grid_metrics_density",
    "unit_grid",
  }
  for name in npptcop.__all__:
    assert hasattr(npptcop, name)


def test_evaluation_is_permutation_equivariant(
  small_sample: Tensor, machine: Callable[..., bool]
) -> None:
  est = ProbitTLL().fit(small_sample)
  grid, _ = unit_grid(20)
  perm = torch.randperm(grid.shape[0])
  assert machine(est.evaluate(grid)[perm], est.evaluate(grid[perm]))


def test_tail_evaluation_is_permutation_equivariant(
  clayton_sample: Tensor, machine: Callable[..., bool]
) -> None:
  est = TailCopula(2000 ** (-0.5)).fit(clayton_sample)
  grid, _ = unit_grid(20)
  perm = torch.randperm(grid.shape[0])
  assert machine(est.c(grid)[perm], est.c(grid[perm]))
