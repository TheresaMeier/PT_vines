"""Parallel bivariate-tail simulation study for the ``npptcop`` package.

Ports ``old/scripts/sim_study_biv_tail.py`` onto the installed public API
(``ProbitTLL``, ``TailCopula``, ``grid_metrics_density``, ``unit_grid``). For
each ``(seed, family, tau, n)`` cell it simulates a copula, computes the true
tail targets, and compares the bulk ("Ordinary") and tail-adaptive ("Tail")
estimators with integrated ISE/IAE/KL errors plus the tail-mass error.

Each family is evaluated at the corner where its tail dependence concentrates
(see ``FAMILY_ROTATION``): Clayton lower-left, Gumbel upper-right, and the
radially symmetric Gaussian and Student lower-left.

The "Tail" model uses the package's ``select_bandwidth_constant`` rule, not the
old ``k**(-1/3) * cov`` covariance rule, so its numbers differ from the legacy
CSV by design; the "Ordinary" (bulk) numbers match. A small ``--ridge`` (default
``1e-6``) regularizes the selected bandwidth so weakly dependent tail samples
stay positive-definite instead of being skipped; pass ``--ridge 0`` for the
unregularized formula.

Run (``pandas`` lives in the ``interactive`` extra)::

  uv run --extra interactive python examples/sim_study_biv_tail.py
  uv run --extra interactive python examples/sim_study_biv_tail.py --dry-run
"""

import argparse
import logging
import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pyvinecopulib as pv
import torch
from torch import Tensor

from npptcop import ProbitTLL, TailCopula, grid_metrics_density, unit_grid

LOG = logging.getLogger("sim_study_biv_tail")

# All supported families (the --families choices and FAMILY_ROTATION keys).
FAMILIES: tuple[str, ...] = ("clayton", "gumbel", "student", "gaussian")
# Default study families: the three with tail dependence in the studied corner.
# Gaussian is tail-independent, so it is omitted by default (still selectable).
DEFAULT_FAMILIES: tuple[str, ...] = ("clayton", "gumbel", "student")
# Each family is studied at the corner where its tail dependence concentrates
# (pyvinecopulib density-rotation convention): Clayton lower-left, Gumbel
# upper-right, the radially symmetric Student lower-left.
FAMILY_ROTATION: dict[str, int] = {
  "clayton": 0,
  "gumbel": 180,
  "student": 0,
  "gaussian": 0,
}
COLUMNS: tuple[str, ...] = (
  "seed",
  "family",
  "tau",
  "n",
  "k",
  "q",
  "model",
  "target",
  "ISE",
  "IAE",
  "KL",
  "p_hat",
  "p_true",
  "AE",
  "RE",
)


# --- multi-family helpers (not in the package) ------------------------------


def q_of_n(n: int) -> float:
  """Lower-tail cutoff ``q = n**(-1/2)``."""
  return n ** (-0.5)


def copula_params(family: str, tau: float, nu: int = 4) -> dict[str, float]:
  """Map a Kendall tau to copula parameters for the given family."""
  if family == "clayton":
    return {"theta": 2.0 * tau / (1.0 - tau)}
  if family == "gumbel":
    return {"theta": 1.0 / (1.0 - tau)}
  if family in {"gaussian", "student"}:
    params = {"rho": math.sin(math.pi * tau / 2.0)}
    if family == "student":
      params["nu"] = float(nu)
    return params
  raise ValueError(f"Unknown family: {family}")


def set_bicop(family: str, params: dict[str, float]) -> pv.Bicop:
  """Construct a pyvinecopulib ``Bicop`` for the family and parameters."""
  # pyvinecopulib's stubs do not enumerate the BicopFamily members.
  if family == "clayton":
    return pv.Bicop(
      family=pv.BicopFamily.clayton,  # ty: ignore[unresolved-attribute]
      rotation=0,
      parameters=np.array([[params["theta"]]], dtype=float),
    )
  if family == "gumbel":
    return pv.Bicop(
      family=pv.BicopFamily.gumbel,  # ty: ignore[unresolved-attribute]
      rotation=0,
      parameters=np.array([[params["theta"]]], dtype=float),
    )
  if family == "gaussian":
    return pv.Bicop(
      family=pv.BicopFamily.gaussian,  # ty: ignore[unresolved-attribute]
      parameters=np.array([[params["rho"]]], dtype=float),
    )
  if family == "student":
    return pv.Bicop(
      family=pv.BicopFamily.student,  # ty: ignore[unresolved-attribute]
      parameters=np.array([[params["rho"]], [params["nu"]]], dtype=float),
    )
  raise ValueError(f"Unknown family: {family}")


# --- task spec / result containers (picklable, top-level) -------------------


@dataclass(frozen=True)
class ComboSpec:
  """One simulation cell plus the shared run settings it needs."""

  seed: int
  family: str
  tau: float
  n: int
  nu: int
  grid_size: int
  eps: float
  rotation: int
  min_tail_count: int
  ridge: float


@dataclass(frozen=True)
class ComboResult:
  """Worker outcome: the spec, a status string, tail count, and row dicts."""

  spec: ComboSpec
  status: str
  k: int | None
  rows: list[dict[str, object]]


# --- worker (top-level, picklable, never logs, never writes files) ----------


def _init_worker() -> None:
  """Per-worker setup: float64 default and single-threaded torch."""
  torch.set_default_dtype(torch.float64)
  torch.set_num_threads(1)


def _reflect(u: Tensor, rotation: int) -> Tensor:
  """Map the rotation's corner onto the lower-left (pyvinecopulib convention)."""
  if rotation == 90:
    return torch.stack((1.0 - u[:, 0], u[:, 1]), dim=1)
  if rotation == 180:
    return 1.0 - u
  if rotation == 270:
    return torch.stack((u[:, 0], 1.0 - u[:, 1]), dim=1)
  return u


def _corner_count(u: Tensor, q: float, rotation: int) -> int:
  """Number of observations in the rotation's corner block of size ``q``.

  Mirrors ``TailCopula``'s reflection so the tiny-tail skip can run before
  fitting (bandwidth selection is undefined for a handful of points).
  """
  corner = _reflect(u, rotation)
  return int(((corner[:, 0] <= q) & (corner[:, 1] <= q)).sum())


def _corner_mass(bicop: pv.Bicop, q: float, rotation: int) -> float:
  """Exact probability mass in the rotation's corner block of size ``q``."""

  def cdf(a: float, b: float) -> float:
    return float(np.asarray(bicop.cdf(np.array([[a, b]])))[0])

  if rotation == 90:
    return q - cdf(1.0 - q, q)
  if rotation == 180:
    return 2.0 * q - 1.0 + cdf(1.0 - q, 1.0 - q)
  if rotation == 270:
    return q - cdf(q, 1.0 - q)
  return cdf(q, q)


def _truth_targets(
  bicop: pv.Bicop, u_tail_grid: Tensor, q: float, rotation: int
) -> tuple[Tensor, Tensor, Tensor, float]:
  """True ``(c, r, h, p)`` on the rotation's corner block via pyvinecopulib."""
  corner = _reflect(u_tail_grid, rotation)
  c_true = torch.as_tensor(bicop.pdf(corner.numpy()), dtype=torch.float64)
  r_true = q * c_true
  p_true = _corner_mass(bicop, q, rotation)
  h_true = (q**2 / p_true) * c_true
  return c_true, r_true, h_true, p_true


def _ordinary_targets(
  u_data: Tensor,
  u_tail_grid: Tensor,
  q: float,
  cell_area_tail: float,
  ridge: float,
  rotation: int,
) -> tuple[Tensor, Tensor, Tensor, float]:
  """Bulk ``ProbitTLL`` ``(c, r, h, p)`` at the rotation's corner block."""
  corner = _reflect(u_tail_grid, rotation)
  c_body = ProbitTLL(ridge=ridge).fit(u_data).evaluate(corner)
  p_body = float(c_body.sum() * cell_area_tail)
  r_body = q * c_body
  h_body = (q**2 / p_body) * c_body
  return c_body, r_body, h_body, p_body


def _metric_rows(
  spec: ComboSpec,
  q: float,
  k: int,
  ests: dict[str, dict[str, Tensor]],
  truths: dict[str, Tensor],
  cell_areas: dict[str, float],
  p_hats: dict[str, float],
  p_true: float,
) -> list[dict[str, object]]:
  """Long-format rows: r/h/c metric rows plus one ``p`` row per model."""
  base: dict[str, object] = {
    "seed": spec.seed,
    "family": spec.family,
    "tau": spec.tau,
    "n": spec.n,
    "k": k,
    "q": q,
  }
  rows: list[dict[str, object]] = []
  for model, model_ests in ests.items():
    for target, est in model_ests.items():
      metrics = grid_metrics_density(est, truths[target], cell_areas[target])
      rows.append({**base, "model": model, "target": target, **metrics})
  for model, p_hat in p_hats.items():
    rows.append(
      {
        **base,
        "model": model,
        "target": "p",
        "p_hat": p_hat,
        "p_true": p_true,
        "AE": abs(p_hat - p_true),
        "RE": abs(p_hat - p_true) / p_true,
      }
    )
  return rows


def run_combo(spec: ComboSpec) -> ComboResult:
  """Run one ``(seed, family, tau, n)`` cell; return rows or a skip/error."""
  try:
    params = copula_params(spec.family, spec.tau, spec.nu)
    bicop = set_bicop(spec.family, params)
    u_data = torch.as_tensor(
      bicop.simulate(spec.n, seeds=[spec.seed]), dtype=torch.float64
    )
    q = q_of_n(spec.n)
    # Skip before fitting: bandwidth selection (ACE) is undefined for a few
    # points, so a tiny tail must be skipped on count, not via a failed fit.
    k = _corner_count(u_data, q, spec.rotation)
    if k < spec.min_tail_count:
      status = f"skipped (k={k} < {spec.min_tail_count})"
      return ComboResult(spec, status, k, [])

    grid, cell_area = unit_grid(spec.grid_size, spec.eps)
    u_tail_grid = grid * q
    cell_area_tail = (q**2) * cell_area

    tail = TailCopula(q, rotation=spec.rotation, ridge=spec.ridge).fit(u_data)
    c_true, r_true, h_true, p_true = _truth_targets(
      bicop, u_tail_grid, q, spec.rotation
    )
    c_body, r_body, h_body, p_body = _ordinary_targets(
      u_data, u_tail_grid, q, cell_area_tail, spec.ridge, spec.rotation
    )

    ests = {
      "Ordinary": {"r": r_body, "h": h_body, "c": c_body},
      "Tail": {"r": tail.r(grid), "h": tail.h(grid), "c": tail.c(grid)},
    }
    truths = {"r": r_true, "h": h_true, "c": c_true}
    cell_areas = {"r": cell_area, "h": cell_area, "c": cell_area_tail}
    p_hats = {"Ordinary": p_body, "Tail": tail.fit_.p}

    rows = _metric_rows(spec, q, k, ests, truths, cell_areas, p_hats, p_true)
    return ComboResult(spec, "ok", k, rows)
  except torch.linalg.LinAlgError:
    # Residual non-positive-definite bandwidth that --ridge cannot repair: a
    # small tail sample whose ACE maximal correlation is undefined (NaN), so the
    # selected bandwidth is non-finite. Such a cell is not estimable.
    return ComboResult(spec, "skipped (singular bandwidth)", None, [])
  except Exception as exc:
    return ComboResult(spec, f"error: {exc!r}", None, [])


# --- CLI / orchestration ----------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
  """Argparse parser for the simulation grid and run settings."""
  parser = argparse.ArgumentParser(
    description="Parallel bivariate-tail simulation study for npptcop."
  )
  parser.add_argument(
    "--families", nargs="+", choices=FAMILIES, default=list(DEFAULT_FAMILIES)
  )
  parser.add_argument("--taus", nargs="+", type=float, default=[0.4, 0.8])
  parser.add_argument(
    "--ns", nargs="+", type=int, default=[200, 500, 1000, 2000, 5000]
  )
  parser.add_argument("--seeds", nargs="+", type=int, default=list(range(30)))
  parser.add_argument("--nu", type=int, default=4)
  parser.add_argument("--grid-size", type=int, default=100)
  parser.add_argument("--eps", type=float, default=1e-4)
  parser.add_argument("--min-tail-count", type=int, default=5)
  parser.add_argument("--ridge", type=float, default=1e-6)
  parser.add_argument(
    "--output", type=Path, default=Path("results/sim_study_biv_tail.csv")
  )
  parser.add_argument("--workers", type=int, default=None)
  parser.add_argument(
    "--log-level",
    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    default="INFO",
  )
  parser.add_argument("--dry-run", action="store_true")
  return parser


def enumerate_combos(args: argparse.Namespace) -> list[ComboSpec]:
  """Cartesian product of seeds x families x taus x ns as ``ComboSpec``s."""
  return [
    ComboSpec(
      seed=seed,
      family=family,
      tau=tau,
      n=n,
      nu=args.nu,
      grid_size=args.grid_size,
      eps=args.eps,
      rotation=FAMILY_ROTATION[family],
      min_tail_count=args.min_tail_count,
      ridge=args.ridge,
    )
    for seed, family, tau, n in product(
      args.seeds, args.families, args.taus, args.ns
    )
  ]


def run_study(specs: list[ComboSpec], workers: int) -> list[ComboResult]:
  """Run all combos in a spawn process pool, logging progress as they finish."""
  ctx = multiprocessing.get_context("spawn")
  results: list[ComboResult] = []
  total = len(specs)
  with ProcessPoolExecutor(
    max_workers=workers, mp_context=ctx, initializer=_init_worker
  ) as executor:
    futures = {executor.submit(run_combo, spec): spec for spec in specs}
    for i, future in enumerate(as_completed(futures), start=1):
      result = future.result()
      spec = result.spec
      k_str = "-" if result.k is None else str(result.k)
      LOG.info(
        "[%d/%d] %-8s | tau=%.2f | n=%5d | seed=%d -> k=%s, %s",
        i,
        total,
        spec.family,
        spec.tau,
        spec.n,
        spec.seed,
        k_str,
        result.status,
      )
      results.append(result)
  return results


def results_to_frame(results: list[ComboResult]) -> pd.DataFrame:
  """Flatten combo rows into a long-format DataFrame with fixed columns."""
  rows = [row for result in results for row in result.rows]
  return pd.DataFrame(rows, columns=list(COLUMNS))


def main() -> None:
  """Parse args, configure logging, run the study, and write the CSV."""
  args = build_parser().parse_args()
  logging.basicConfig(
    level=args.log_level,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
  )
  # Keep BLAS single-threaded from worker start (set before the pool spawns).
  os.environ.setdefault("OMP_NUM_THREADS", "1")
  os.environ.setdefault("MKL_NUM_THREADS", "1")

  specs = enumerate_combos(args)
  workers = args.workers or min(os.cpu_count() or 1, 8)
  LOG.info(
    "%d combos | %d workers -> %s",
    len(specs),
    workers,
    args.output,
  )
  if args.dry_run:
    LOG.info("dry run: no computation or output")
    return

  results = run_study(specs, workers)
  n_ok = sum(1 for r in results if r.status == "ok")
  LOG.info("done: %d/%d combos produced rows", n_ok, len(results))

  frame = results_to_frame(results)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  frame.to_csv(args.output, index=False)
  LOG.info("wrote %d rows -> %s", len(frame), args.output)


if __name__ == "__main__":
  main()
