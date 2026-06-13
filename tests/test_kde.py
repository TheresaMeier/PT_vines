"""Tests for the probit-transform local-likelihood Gaussian KDE (ProbitTLL)."""

from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from npptcop import ProbitTLL, unit_grid
from npptcop.transforms import qnorm


def _mvn_pdf(z: Tensor, mean: Tensor, cov: Tensor) -> Tensor:
  """Bivariate normal density at rows of ``z`` with given mean and covariance."""
  inv = torch.linalg.inv(cov)
  det = torch.linalg.det(cov)
  diff = z - mean
  quad = (diff @ inv * diff).sum(dim=-1)
  return torch.exp(-0.5 * quad) / (2.0 * torch.pi * det.sqrt())


def test_density_single_point_is_mvn(machine: Callable[..., bool]) -> None:
  bandwidth = torch.tensor([[0.4, 0.1], [0.1, 0.3]])
  u = torch.tensor([[0.3, 0.6]])
  est = ProbitTLL(bandwidth).fit(u)
  z_eval = torch.tensor([[0.1, 0.2], [-0.5, 1.0], [0.0, 0.0]])
  expected = _mvn_pdf(z_eval, qnorm(u), bandwidth)
  assert machine(est.density(z_eval), expected)


def test_density_is_mean_over_data(machine: Callable[..., bool]) -> None:
  bandwidth = torch.tensor([[0.4, 0.1], [0.1, 0.3]])
  u = torch.tensor([[0.3, 0.6], [0.8, 0.2]])
  est = ProbitTLL(bandwidth).fit(u)
  z_eval = torch.tensor([[0.1, 0.2], [-0.5, 1.0]])
  z_data = qnorm(u)
  expected = 0.5 * (
    _mvn_pdf(z_eval, z_data[0], bandwidth)
    + _mvn_pdf(z_eval, z_data[1], bandwidth)
  )
  assert machine(est.density(z_eval), expected)


def test_fit_selects_symmetric_psd_bandwidth(
  small_sample: Tensor, machine: Callable[..., bool]
) -> None:
  est = ProbitTLL().fit(small_sample)
  b = est.bandwidth_
  assert b.shape == (2, 2)
  assert machine(b, b.T)
  torch.linalg.cholesky(b)  # raises if not positive definite


def test_evaluate_shape_and_positive(small_sample: Tensor) -> None:
  est = ProbitTLL().fit(small_sample)
  grid, _ = unit_grid(20)
  out = est.evaluate(grid)
  assert out.shape == (grid.shape[0],)
  assert torch.all(out > 0.0)


def test_bulk_density_integrates_to_one(small_sample: Tensor) -> None:
  est = ProbitTLL().fit(small_sample)
  grid, cell_area = unit_grid(80)
  integral = float(est.evaluate(grid).sum() * cell_area)
  assert 0.85 < integral < 1.15


def test_non_pd_bandwidth_raises() -> None:
  bad = torch.tensor([[1.0, 2.0], [2.0, 1.0]])  # indefinite
  with pytest.raises(RuntimeError):
    ProbitTLL(bad).fit(torch.tensor([[0.3, 0.6]]))
