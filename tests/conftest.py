"""Shared fixtures and tolerance presets for the npptcop test suite.

Ground truth comes from ``pyvinecopulib`` (simulate / pdf / cdf). Tolerance
presets encode the testing philosophy: ``machine`` for algebraic identities that
are exact in float64, ``tight`` for same-backend regression reproduction, and
``stat`` for statistical estimation against ground truth.
"""

from collections.abc import Callable

import numpy as np
import pyvinecopulib as pv
import pytest
import torch
from torch import Tensor

torch.set_default_dtype(torch.float64)

THETA = 3.0
SEED = 97

# A closeness check usable on tensors or scalars; see ``machine`` / ``tight``.
Close = Callable[[Tensor | float, Tensor | float], bool]


def _make_close(rtol: float, atol: float) -> Close:
  def _close(a: Tensor | float, b: Tensor | float) -> bool:
    return bool(
      torch.isclose(
        torch.as_tensor(a, dtype=torch.float64),
        torch.as_tensor(b, dtype=torch.float64),
        rtol=rtol,
        atol=atol,
      ).all()
    )

  return _close


@pytest.fixture
def machine() -> Close:
  """Closeness for algebraic identities exact in float64 (rtol 0, atol 1e-12)."""
  return _make_close(rtol=0.0, atol=1e-12)


@pytest.fixture
def tight() -> Close:
  """Closeness for same-backend regression reproduction (rtol 1e-9)."""
  return _make_close(rtol=1e-9, atol=1e-12)


def clayton_bicop(rotation: int = 0) -> pv.Bicop:
  """Clayton copula with parameter ``theta = 3`` and the given rotation."""
  # pyvinecopulib's stubs do not enumerate the BicopFamily members.
  family = pv.BicopFamily.clayton  # ty: ignore[unresolved-attribute]
  return pv.Bicop(
    family=family,
    rotation=rotation,
    parameters=np.array([[THETA]], dtype=float),
  )


def sample(bicop: pv.Bicop, n: int) -> Tensor:
  """Draw ``n`` reproducible observations (seed 97) as a float64 tensor."""
  return torch.as_tensor(bicop.simulate(n, seeds=[SEED]), dtype=torch.float64)


@pytest.fixture(scope="session")
def clayton() -> pv.Bicop:
  """Lower-left tail-dependent Clayton copula (rotation 0)."""
  return clayton_bicop()


@pytest.fixture(scope="session")
def clayton_sample(clayton: pv.Bicop) -> Tensor:
  """Canonical n = 2000 Clayton sample (seed 97)."""
  return sample(clayton, 2000)


@pytest.fixture(scope="session")
def small_sample(clayton: pv.Bicop) -> Tensor:
  """Small n = 400 Clayton sample for fast unit tests."""
  return sample(clayton, 400)


@pytest.fixture(scope="session")
def independent_sample() -> Tensor:
  """n = 2000 independent uniforms (zero dependence)."""
  rng = np.random.default_rng(SEED)
  return torch.as_tensor(rng.random((2000, 2)), dtype=torch.float64)


@pytest.fixture(scope="session")
def comonotone_sample() -> Tensor:
  """n = 500 perfectly positively dependent sample (both columns equal)."""
  rng = np.random.default_rng(SEED)
  u = rng.random(500)
  return torch.as_tensor(np.column_stack([u, u]), dtype=torch.float64)


@pytest.fixture(scope="session")
def rotated_sample() -> Callable[[int], Tensor]:
  """Factory returning an n = 2000 Clayton sample for a given rotation."""

  def _make(rotation: int) -> Tensor:
    return sample(clayton_bicop(rotation), 2000)

  return _make
