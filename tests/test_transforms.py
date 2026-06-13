"""Tests for the standard-normal probit transforms."""

import math
from collections.abc import Callable

import torch

from npptcop.transforms import SQRT_2PI_INV, dnorm, qnorm


def test_qnorm_known_quantiles(tight: Callable[..., bool]) -> None:
  p = torch.tensor([0.5, 0.975, 0.025])
  expected = torch.tensor([0.0, 1.959963984540054, -1.959963984540054])
  assert tight(qnorm(p), expected)


def test_dnorm_known_values(tight: Callable[..., bool]) -> None:
  z = torch.tensor([0.0, 1.0, -1.0])
  expected = torch.tensor(
    [SQRT_2PI_INV, 0.24197072451914337, 0.24197072451914337]
  )
  assert tight(dnorm(z), expected)


def test_sqrt_2pi_inv_constant() -> None:
  assert math.isclose(SQRT_2PI_INV, 1.0 / math.sqrt(2.0 * math.pi))


def test_dnorm_is_even(machine: Callable[..., bool]) -> None:
  z = torch.linspace(-3.0, 3.0, 25)
  assert machine(dnorm(z), dnorm(-z))


def test_probit_round_trip(tight: Callable[..., bool]) -> None:
  u = torch.linspace(0.01, 0.99, 50)
  recovered = torch.distributions.Normal(0.0, 1.0).cdf(qnorm(u))
  assert tight(recovered, u)


def test_qnorm_preserves_shape_and_dtype() -> None:
  u = torch.rand(7, 2)
  out = qnorm(u)
  assert out.shape == (7, 2)
  assert out.dtype == torch.float64
