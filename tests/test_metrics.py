"""Tests for the grid-based metrics and the unit-grid helper."""

from collections.abc import Callable

import torch

from npptcop import grid_metrics_density, unit_grid


def test_zero_when_estimate_equals_truth(machine: Callable[..., bool]) -> None:
  truth = torch.tensor([0.5, 1.5, 2.0, 0.25])
  out = grid_metrics_density(truth, truth, cell_area=0.1)
  assert machine(out["ISE"], 0.0)
  assert machine(out["IAE"], 0.0)
  assert machine(out["KL"], 0.0)


def test_closed_form_ise_iae(tight: Callable[..., bool]) -> None:
  est = torch.tensor([1.0, 2.0])
  truth = torch.tensor([1.5, 1.0])
  out = grid_metrics_density(est, truth, cell_area=2.0)
  # ISE = (0.25 + 1.0) * 2 ; IAE = (0.5 + 1.0) * 2
  assert tight(out["ISE"], 2.5)
  assert tight(out["IAE"], 3.0)


def test_kl_handles_zero_estimate() -> None:
  est = torch.tensor([0.0, 1.0])
  truth = torch.tensor([1.0, 1.0])
  out = grid_metrics_density(est, truth, cell_area=1.0)
  assert torch.isfinite(torch.tensor(out["KL"]))
  assert out["KL"] > 0.0


def test_return_keys_and_types() -> None:
  out = grid_metrics_density(torch.ones(3), torch.ones(3), cell_area=1.0)
  assert set(out) == {"ISE", "IAE", "KL"}
  assert all(isinstance(v, float) for v in out.values())


def test_unit_grid_shape_and_cell_area(tight: Callable[..., bool]) -> None:
  grid, cell_area = unit_grid(size=10, eps=1e-4)
  assert grid.shape == (100, 2)
  assert grid.min() >= 1e-4 and grid.max() <= 1 - 1e-4
  step = (1 - 2e-4) / 9
  assert tight(cell_area, step**2)
