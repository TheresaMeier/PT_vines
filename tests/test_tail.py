"""Tests for the corner tail copula density estimator (TailCopula)."""

from collections.abc import Callable

import numpy as np
import pyvinecopulib as pv
import pytest
import torch
from torch import Tensor

from npptcop import TailCopula, grid_metrics_density, unit_grid

Q = 2000 ** (-0.5)

# Reflections mapping each corner onto the lower-left (involutions), used to
# check the rotation parameter against an explicit external reflection.
_REFLECT: dict[int, Callable[[Tensor], Tensor]] = {
  0: lambda x: x,
  90: lambda x: torch.stack((1.0 - x[:, 0], x[:, 1]), dim=1),
  180: lambda x: 1.0 - x,
  270: lambda x: torch.stack((x[:, 0], 1.0 - x[:, 1]), dim=1),
}


def test_algebraic_identities(
  clayton_sample: Tensor, machine: Callable[..., bool]
) -> None:
  est = TailCopula(Q).fit(clayton_sample)
  grid, _ = unit_grid(30)
  h, c, r = est.h(grid), est.c(grid), est.r(grid)
  assert machine(c, (est.fit_.p / Q**2) * h)
  assert machine(r, Q * c)


def test_tail_mass_is_k_over_n(clayton_sample: Tensor) -> None:
  est = TailCopula(Q).fit(clayton_sample)
  mask = (clayton_sample[:, 0] <= Q) & (clayton_sample[:, 1] <= Q)
  assert est.fit_.k == int(mask.sum())
  assert est.fit_.n == clayton_sample.shape[0]
  assert est.fit_.p == est.fit_.k / est.fit_.n


def test_densities_nonnegative(clayton_sample: Tensor) -> None:
  est = TailCopula(Q).fit(clayton_sample)
  grid, _ = unit_grid(30)
  for values in (est.h(grid), est.c(grid), est.r(grid)):
    assert torch.all(values >= 0.0)


def test_h_integrates_to_one(clayton_sample: Tensor) -> None:
  est = TailCopula(Q).fit(clayton_sample)
  grid, cell_area = unit_grid(80)
  assert 0.9 < float(est.h(grid).sum() * cell_area) < 1.1


def test_tail_mass_recovers_cdf(
  clayton: pv.Bicop, clayton_sample: Tensor
) -> None:
  est = TailCopula(Q).fit(clayton_sample)
  p_true = float(np.asarray(clayton.cdf(np.array([[Q, Q]])))[0])
  assert abs(est.fit_.p - p_true) / p_true < 0.2


def test_copula_density_improves_with_n(clayton: pv.Bicop) -> None:
  grid, cell_area = unit_grid(50)

  def iae(n: int) -> float:
    q = n ** (-0.5)
    u = torch.as_tensor(clayton.simulate(n, seeds=[97]), dtype=torch.float64)
    c_hat = TailCopula(q).fit(u).c(grid)
    c_true = torch.as_tensor(
      clayton.pdf((grid * q).numpy()), dtype=torch.float64
    )
    return grid_metrics_density(c_hat, c_true, (q**2) * cell_area)["IAE"]

  assert iae(2000) < iae(500)


def test_empty_tail_raises() -> None:
  u = torch.full((50, 2), 0.9)
  with pytest.raises(ValueError, match="no observations"):
    TailCopula(0.1).fit(u)


@pytest.mark.parametrize("q", [0.0, 1.0, -0.1, 1.5])
def test_invalid_q_raises(q: float) -> None:
  with pytest.raises(ValueError, match="q must lie"):
    TailCopula(q)


@pytest.mark.parametrize("rotation", [45, 360, -90])
def test_invalid_rotation_raises(rotation: int) -> None:
  with pytest.raises(ValueError, match="rotation must be"):
    TailCopula(Q, rotation=rotation)


def test_ridge_threads_into_bandwidth(
  clayton_sample: Tensor, machine: Callable[..., bool]
) -> None:
  base = TailCopula(Q).fit(clayton_sample).fit_.bandwidth
  ridged = TailCopula(Q, ridge=1e-3).fit(clayton_sample).fit_.bandwidth
  assert machine(ridged, base + 1e-3 * torch.eye(2))


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_rotation_matches_reflection(
  rotation: int,
  rotated_sample: Callable[[int], Tensor],
  machine: Callable[..., bool],
) -> None:
  # Data drawn from the rotation-R Clayton populates the corner R selects.
  u = rotated_sample(rotation)
  grid, _ = unit_grid(30)
  direct = TailCopula(Q, rotation=rotation).fit(u)
  reflected = TailCopula(Q, rotation=0).fit(_REFLECT[rotation](u))
  assert direct.fit_.k > 0
  assert machine(direct.c(grid), reflected.c(grid))
