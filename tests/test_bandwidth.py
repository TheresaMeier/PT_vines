"""Tests for the constant-method TLL bandwidth selection.

These pin the ported ``tools_stats`` helpers, so (as the one allowed exception)
they import the internal symbols directly rather than via the public API.
"""

from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from npptcop.bandwidth import (
  _ace,
  _pairwise_mcor,
  _pearson_cor,
  _win_smoother,
  select_bandwidth_constant,
)
from npptcop.transforms import qnorm


def _zero_correlation_sample() -> Tensor:
  """Symmetric data with zero linear correlation but strong dependence.

  ``x`` is symmetric about 0, so ``cor(x, x**2) == 0`` exactly while the
  maximal correlation is positive; ``scale`` then collapses to 0 and the
  unregularized bandwidth is singular.
  """
  x = torch.linspace(-2.0, 2.0, 21)
  return torch.stack((x, x**2), dim=1)


def test_pearson_cor_perfect_dependence(tight: Callable[..., bool]) -> None:
  x0 = torch.linspace(-2.0, 2.0, 50)
  pos = torch.stack((x0, 2.0 * x0 + 3.0), dim=1)
  neg = torch.stack((x0, -x0), dim=1)
  assert tight(_pearson_cor(pos), 1.0)
  assert tight(_pearson_cor(neg), -1.0)


def test_win_smoother_constant_input(machine: Callable[..., bool]) -> None:
  x = torch.full((20,), 2.5)
  assert machine(_win_smoother(x, 3), x)


def test_win_smoother_edge_clamp(machine: Callable[..., bool]) -> None:
  x = torch.arange(20, dtype=torch.float64)
  out = _win_smoother(x, 2)
  assert machine(out[0], out[2])
  assert machine(out[1], out[2])
  assert machine(out[-1], out[-3])
  assert machine(out[-2], out[-3])


def test_win_smoother_wl_zero_is_identity(machine: Callable[..., bool]) -> None:
  x = torch.arange(10, dtype=torch.float64)
  assert machine(_win_smoother(x, 0), x)


def test_ace_mcor_independent_small(independent_sample: Tensor) -> None:
  assert abs(_pairwise_mcor(independent_sample)) < 0.15


def test_ace_mcor_comonotone_near_one(comonotone_sample: Tensor) -> None:
  assert _pairwise_mcor(comonotone_sample) > 0.95


def test_ace_is_deterministic(small_sample: Tensor) -> None:
  z = qnorm(small_sample.clamp(1e-6, 1 - 1e-6))
  assert torch.equal(_ace(z), _ace(z))


def test_select_bandwidth_formula(
  small_sample: Tensor, machine: Callable[..., bool]
) -> None:
  z = qnorm(small_sample.clamp(1e-6, 1 - 1e-6))
  n = z.shape[0]
  cor = _pearson_cor(z).clamp(-0.95, 0.95).item()
  mcor = _pairwise_mcor(z)
  scale = abs(cor / mcor) ** (0.5 * mcor)
  cov = torch.tensor([[1.0, cor], [cor, 1.0]])
  expected = (n ** (-1.0 / 3.0)) * cov * scale
  assert machine(select_bandwidth_constant(z), expected)


def test_select_bandwidth_clamps_correlation(
  comonotone_sample: Tensor, tight: Callable[..., bool]
) -> None:
  z_pos = qnorm(comonotone_sample.clamp(1e-6, 1 - 1e-6))
  b_pos = select_bandwidth_constant(z_pos)
  assert tight(b_pos[0, 1] / b_pos[0, 0], 0.95)

  u = comonotone_sample[:, 0]
  counter = torch.stack((u, 1.0 - u), dim=1)
  z_neg = qnorm(counter.clamp(1e-6, 1 - 1e-6))
  b_neg = select_bandwidth_constant(z_neg)
  assert tight(b_neg[0, 1] / b_neg[0, 0], -0.95)


def test_select_bandwidth_symmetric_psd(
  small_sample: Tensor, machine: Callable[..., bool]
) -> None:
  b = select_bandwidth_constant(qnorm(small_sample.clamp(1e-6, 1 - 1e-6)))
  assert machine(b, b.T)
  torch.linalg.cholesky(b)  # raises if not positive definite


def test_ridge_off_matches_unregularized(
  small_sample: Tensor, machine: Callable[..., bool]
) -> None:
  z = qnorm(small_sample.clamp(1e-6, 1 - 1e-6))
  assert machine(
    select_bandwidth_constant(z, ridge=0.0), select_bandwidth_constant(z)
  )


def test_ridge_adds_to_diagonal(
  small_sample: Tensor, machine: Callable[..., bool]
) -> None:
  z = qnorm(small_sample.clamp(1e-6, 1 - 1e-6))
  base = select_bandwidth_constant(z)
  ridged = select_bandwidth_constant(z, ridge=1e-3)
  assert machine(ridged, base + 1e-3 * torch.eye(2))


def test_ridge_rescues_singular_bandwidth() -> None:
  z = _zero_correlation_sample()
  with pytest.raises(torch.linalg.LinAlgError):
    torch.linalg.cholesky(select_bandwidth_constant(z, ridge=0.0))
  torch.linalg.cholesky(select_bandwidth_constant(z, ridge=1e-3))


def test_negative_ridge_raises() -> None:
  with pytest.raises(ValueError, match="ridge must be non-negative"):
    select_bandwidth_constant(_zero_correlation_sample(), ridge=-1.0)
