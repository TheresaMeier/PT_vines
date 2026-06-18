"""Tests for the parametric corner tail copula estimator (ParametricTailCopula).

Mirrors ``test_tail.py``. The closed-form unit-square normalizers are pinned
against the homogeneity 1-D reduction and a coarse 2-D Riemann sum; ``_FAMILIES``
is imported directly (the sanctioned internal-helper exception, as in
``test_bandwidth.py``).
"""

import math
from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from npptcop import ParametricTailCopula, unit_grid
from npptcop.parametric import _FAMILIES

Q = 2000 ** (-0.5)

FAMILIES: tuple[str, ...] = (
  "husler_reiss",
  "neg_logistic",
  "logistic",
  "dirichlet",
)
# A valid parameter per family (logistic needs par > 1).
PARS: dict[str, float] = {
  "husler_reiss": 1.0,
  "neg_logistic": 1.0,
  "logistic": 2.0,
  "dirichlet": 1.0,
}

# Reflections mapping each corner onto the lower-left (involutions).
_REFLECT: dict[int, Callable[[Tensor], Tensor]] = {
  0: lambda x: x,
  90: lambda x: torch.stack((1.0 - x[:, 0], x[:, 1]), dim=1),
  180: lambda x: 1.0 - x,
  270: lambda x: torch.stack((x[:, 0], 1.0 - x[:, 1]), dim=1),
}


def _sample_from_family(
  family: str, par: float, n: int, seed: int, n_ang: int = 4000
) -> Tensor:
  """Draw ``n`` points on ``(0, 1)^2`` exactly from ``h = lambda / Z``.

  Uses the radial-angular decomposition ``s = r (a, 1 - a)``: since ``lambda``
  is homogeneous of degree ``-1``, the joint density in ``(r, a)`` is constant
  in the radius ``r`` (uniform on ``(0, rmax)`` with ``rmax = 1 / max(a, 1-a)``),
  so the angle ``a`` has the singularity-free density ``lambda(a, 1-a) * rmax``.
  """
  fam = _FAMILIES[family]
  a = torch.linspace(1e-6, 1.0 - 1e-6, n_ang, dtype=torch.float64)
  rmax = 1.0 / torch.maximum(a, 1.0 - a)
  weights = fam.lam(torch.stack((a, 1.0 - a), dim=1), par) * rmax
  gen = torch.Generator().manual_seed(seed)
  idx = torch.multinomial(weights, n, replacement=True, generator=gen)
  step = float(a[1] - a[0])
  a_s = (a[idx] + (torch.rand(n, generator=gen) - 0.5) * step).clamp(
    1e-6, 1.0 - 1e-6
  )
  r = torch.rand(n, generator=gen) / torch.maximum(a_s, 1.0 - a_s)
  return torch.stack((r * a_s, r * (1.0 - a_s)), dim=1).clamp(1e-4, 1.0 - 1e-4)


@pytest.mark.parametrize("family", FAMILIES)
def test_density_homogeneous_degree_minus_one(
  family: str, tight: Callable[..., bool]
) -> None:
  lam = _FAMILIES[family].lam
  par = PARS[family]
  gen = torch.Generator().manual_seed(0)
  s = 0.1 + 0.8 * torch.rand((64, 2), generator=gen)  # interior of (0, 1)^2
  t = 0.5
  assert tight(lam(t * s, par), (1.0 / t) * lam(s, par))


@pytest.mark.parametrize("family", FAMILIES)
def test_h_integrates_to_one(family: str, clayton_sample: Tensor) -> None:
  # The parametric h has an integrable 1/r singularity at the origin, so a grid
  # Riemann sum converges only slowly; integrate via the singularity-free
  # radial-angular reduction instead (h is homogeneous of degree -1, so
  # int h ds = int_0^1 h(a, 1 - a) * rmax(a) da).
  est = ParametricTailCopula(family, Q).fit(clayton_sample)
  a = torch.linspace(1e-7, 1.0 - 1e-7, 200001, dtype=torch.float64)
  rmax = 1.0 / torch.maximum(a, 1.0 - a)
  integrand = est.h(torch.stack((a, 1.0 - a), dim=1)) * rmax
  assert 0.99 < float(torch.trapezoid(integrand, a)) < 1.01


@pytest.mark.parametrize("family", FAMILIES)
def test_algebraic_identities(
  family: str, clayton_sample: Tensor, machine: Callable[..., bool]
) -> None:
  est = ParametricTailCopula(family, Q).fit(clayton_sample)
  grid, _ = unit_grid(30)
  h, c, r = est.h(grid), est.c(grid), est.r(grid)
  assert machine(c, (est.fit_.p / Q**2) * h)
  assert machine(r, Q * c)


@pytest.mark.parametrize("family", FAMILIES)
def test_densities_nonnegative(family: str, clayton_sample: Tensor) -> None:
  est = ParametricTailCopula(family, Q).fit(clayton_sample)
  grid, _ = unit_grid(30)
  for values in (est.h(grid), est.c(grid), est.r(grid)):
    assert torch.all(values >= 0.0)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("par", [0.5, 1.5, 3.0])
def test_normalizer_matches_1d_reduction(family: str, par: float) -> None:
  # The logistic is only defined for par > 1; shift the small value up.
  if family == "logistic" and par <= 1.0:
    par = 1.5
  fam = _FAMILIES[family]
  w = torch.linspace(1e-9, 1.0 - 1e-9, 200001, dtype=torch.float64)
  ones = torch.ones_like(w)
  integrand = fam.lam(torch.stack((ones, w), dim=1), par) + fam.lam(
    torch.stack((w, ones), dim=1), par
  )
  z_1d = float(torch.trapezoid(integrand, w))
  assert abs(fam.norm(par) - z_1d) / z_1d < 1e-3


@pytest.mark.parametrize("family", FAMILIES)
def test_normalizer_matches_2d_riemann(family: str) -> None:
  par = PARS[family]
  fam = _FAMILIES[family]
  grid, cell_area = unit_grid(300)
  z_2d = float(fam.lam(grid, par).sum() * cell_area)
  assert abs(fam.norm(par) - z_2d) / z_2d < 0.2


def test_normalizer_closed_forms() -> None:
  for par in (0.5, 1.0, 2.0):
    assert math.isclose(
      _FAMILIES["neg_logistic"].norm(par), 2.0 ** (-1.0 / par)
    )
    phi = 0.5 * (1.0 + math.erf(math.sqrt(par) / 2.0 / math.sqrt(2.0)))
    assert math.isclose(_FAMILIES["husler_reiss"].norm(par), 2.0 - 2.0 * phi)
  for par in (1.5, 2.0, 4.0):
    assert math.isclose(
      _FAMILIES["logistic"].norm(par), 2.0 - 2.0 ** (1.0 / par)
    )


@pytest.mark.parametrize(
  "family,par_true",
  [
    ("husler_reiss", 1.0),
    ("neg_logistic", 1.5),
    ("logistic", 2.5),
    ("dirichlet", 2.0),
  ],
)
def test_mle_recovers_parameter(family: str, par_true: float) -> None:
  s = _sample_from_family(family, par_true, n=8000, seed=0)
  est = ParametricTailCopula(family, Q).fit(s * Q)
  assert abs(est.fit_.par - par_true) / par_true < 0.1


def test_fitted_nll_is_local_minimum() -> None:
  family, par_true = "logistic", 2.5
  s = _sample_from_family(family, par_true, n=4000, seed=1)
  est = ParametricTailCopula(family, Q).fit(s * Q)
  fam = _FAMILIES[family]
  k = s.shape[0]

  def nll(par: float) -> float:
    return float(
      -torch.log(fam.lam(s, par)).sum() + k * math.log(fam.norm(par))
    )

  par_hat = est.fit_.par
  assert nll(par_hat) <= nll(par_hat * 1.05)
  assert nll(par_hat) <= nll(par_hat * 0.95)


def test_fit_summary_is_populated(clayton_sample: Tensor) -> None:
  est = ParametricTailCopula("logistic", Q).fit(clayton_sample)
  fit = est.fit_
  assert fit.family == "logistic"
  assert 1.0 < fit.par < 50.0
  assert math.isfinite(fit.nll)
  assert fit.n == clayton_sample.shape[0]
  assert fit.k > 0
  assert fit.p == fit.k / fit.n


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_rotation_matches_reflection(
  rotation: int,
  rotated_sample: Callable[[int], Tensor],
  machine: Callable[..., bool],
) -> None:
  u = rotated_sample(rotation)
  grid, _ = unit_grid(30)
  direct = ParametricTailCopula("logistic", Q, rotation=rotation).fit(u)
  reflected = ParametricTailCopula("logistic", Q, rotation=0).fit(
    _REFLECT[rotation](u)
  )
  assert direct.fit_.k > 0
  assert machine(direct.c(grid), reflected.c(grid))


def test_invalid_family_raises() -> None:
  with pytest.raises(ValueError, match="family must be"):
    ParametricTailCopula("bogus", Q)


@pytest.mark.parametrize("q", [0.0, 1.0, -0.1, 1.5])
def test_invalid_q_raises(q: float) -> None:
  with pytest.raises(ValueError, match="q must lie"):
    ParametricTailCopula("logistic", q)


@pytest.mark.parametrize("rotation", [45, 360, -90])
def test_invalid_rotation_raises(rotation: int) -> None:
  with pytest.raises(ValueError, match="rotation must be"):
    ParametricTailCopula("logistic", Q, rotation=rotation)


def test_empty_tail_raises() -> None:
  u = torch.full((50, 2), 0.9)
  with pytest.raises(ValueError, match="no observations"):
    ParametricTailCopula("logistic", 0.1).fit(u)
