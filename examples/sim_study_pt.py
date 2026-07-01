"""Parallel piecewise-copula simulation study for ``npptcop``.

This script extends the lower-tail study to compare full-copula, tail-only, and
piecewise/tail-blended estimators on the same simulation grid. It keeps the same
family/seed/tau/n structure as the lower-tail study, and adds a sweep over the
blending width ``smooth_eps`` for the smooth PT estimators.

For each ``(seed, family, tau, n, smooth_eps)`` cell it:

* simulates pseudo-observations from the chosen copula,
* fits a parametric body, a TLL body, a nonparametric lower-tail estimator,
  and a parametric lower-tail estimator,
* forms hard PT and smooth PT composites,
* computes ISE / IAE / KL on the full grid, the lower-tail block, and the
  transition band, and
* records the lower-tail mass error ``p_hat`` vs ``p_true``.

The Tawn family is included with ``delta1 = 0.8``, ``delta2 = 0.5``, and
rotation ``180``; ``theta`` is obtained from the same Kendall's tau grid used
for the other families.

Run (``pandas`` lives in the ``interactive`` extra)::_

  uv run --extra interactive python examples/sim_study_piecewise_pt.py
  uv run --extra interactive python examples/sim_study_piecewise_pt.py --dry-run
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

from npptcop import (
  ParametricTailCopula,
  ProbitTLL,
  TailCopula,
  grid_metrics_density,
  unit_grid,
)

LOG = logging.getLogger("sim_study_piecewise_pt")

# Families supported by the simulation grid.
FAMILIES: tuple[str, ...] = ("clayton", "gumbel", "student", "gaussian", "tawn")
DEFAULT_FAMILIES: tuple[str, ...] = ("clayton", "gumbel", "student", "tawn")

# Family rotations: the studied corner is reflected to the lower-left.
FAMILY_ROTATION: dict[str, int] = {
  "clayton": 0,
  "gumbel": 180,
  "student": 0,
  "gaussian": 0,
  "tawn": 180,
}

# Tail family used inside the PT composites.
DEFAULT_TAIL_FAMILY = "neg_logistic"

# Model labels used in the result table.
MODEL_ORDER: tuple[str, ...] = (
  "ParametricBody",
  "TLLBody",
  "TailTLL",
  "TailNegLog",
  "PT_TLL_TLL",
  "PT_TLL_NegLog",
  "PT_Param_NegLog",
  "PT_TLL_TLL_Smooth",
  "PT_TLL_NegLog_Smooth",
  "PT_Param_NegLog_Smooth",
)

# Result columns.
COLUMNS: tuple[str, ...] = (
  "seed",
  "family",
  "tau",
  "n",
  "smooth_eps",
  "grid_eps",
  "rotation",
  "k_obs",
  "k_optim",
  "q",
  "tail_family",
  "model",
  "region",
  "target",
  "ISE",
  "IAE",
  "KL",
  "p_hat",
  "p_true",
  "AE",
  "RE",
)


# --- multi-family helpers ---------------------------------------------------


def empirical_lower_tail_dependence(psobs, k=None):
  """Empirical estimator of the lower-tail dependence coefficient."""

  n = psobs.shape[0]
  if k is None:
    k = int(torch.sqrt(torch.tensor(n)))

  threshold = k / n
  joint_exceedances = torch.sum(
    (psobs[:, 0] < threshold) & (psobs[:, 1] < threshold)
  )
  return joint_exceedances / k


def select_k(u_data, rotation, k_grid, smooth_window=30, std_tol=0.01):
  """Heuristic selector for the lower-tail count k."""
  u_sel = _reflect(u_data, rotation)
  vals = np.array([empirical_lower_tail_dependence(u_sel, k) for k in k_grid])

  for i in range(len(vals) - smooth_window + 1):
    if np.std(vals[i : i + smooth_window]) < std_tol:
      return int(k_grid[i]), vals

  scores = np.array(
    [
      np.std(vals[i : i + smooth_window])
      for i in range(len(vals) - smooth_window + 1)
    ]
  )
  i_star = int(np.argmin(scores))
  return int(k_grid[i_star]), vals


def copula_params(family: str, tau: float, nu: int = 4) -> dict[str, float]:
  """Map Kendall's tau to copula parameters for the chosen family."""
  if family == "clayton":
    return {"theta": 2.0 * tau / (1.0 - tau)}
  if family in {"gumbel", "tawn"}:
    params = {"theta": 1.0 / (1.0 - tau)}
    if family == "tawn":
      params["delta1"] = 0.8
      params["delta2"] = 0.5
    return params
  if family in {"gaussian", "student"}:
    params = {"rho": math.sin(math.pi * tau / 2.0)}
    if family == "student":
      params["nu"] = float(nu)
    return params
  raise ValueError(f"Unknown family: {family}")


def set_bicop(family: str, params: dict[str, float]) -> pv.Bicop:
  """Construct a pyvinecopulib ``Bicop`` for the family and parameters."""
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
  if family == "tawn":
    return pv.Bicop(
      family=pv.BicopFamily.tawn,  # ty: ignore[unresolved-attribute]
      rotation=0,
      parameters=np.array(
        [[params["delta1"]], [params["delta2"]], [params["theta"]]],
        dtype=float,
      ),
    )
  raise ValueError(f"Unknown family: {family}")


# --- task spec / result containers -----------------------------------------


@dataclass(frozen=True)
class ComboSpec:
  """One simulation cell plus the shared run settings it needs."""

  seed: int
  family: str
  tau: float
  n: int
  nu: int
  grid_size: int
  grid_eps: float
  smooth_eps: float
  rotation: int
  min_tail_count: int
  ridge: float
  tail_family: str


@dataclass(frozen=True)
class ComboResult:
  """Worker outcome: the spec, a status string, tail count, and rows."""

  spec: ComboSpec
  status: str
  k_obs: int | None
  k_optim: int | None
  rows: list[dict[str, object]]


# --- worker helpers ---------------------------------------------------------


def _init_worker() -> None:
  """Per-worker setup: float64 default and single-threaded torch."""
  torch.set_default_dtype(torch.float64)
  torch.set_num_threads(1)


def _reflect(u: Tensor, rotation: int) -> Tensor:
  """Map the studied corner onto the lower-left (pyvinecopulib convention)."""
  if rotation == 90:
    return torch.stack((1.0 - u[:, 0], u[:, 1]), dim=1)
  if rotation == 180:
    return 1.0 - u
  if rotation == 270:
    return torch.stack((u[:, 0], 1.0 - u[:, 1]), dim=1)
  return u


def _corner_count(u: Tensor, q: float, rotation: int) -> int:
  """Number of observations in the studied corner block of size ``q``."""
  corner = _reflect(u, rotation)
  return int(((corner[:, 0] <= q) & (corner[:, 1] <= q)).sum())


def _corner_mass(bicop: pv.Bicop, q: float, rotation: int) -> float:
  """Exact probability mass in the studied corner block of size ``q``."""

  def cdf(a: float, b: float) -> float:
    return float(np.asarray(bicop.cdf(np.array([[a, b]])))[0])

  if rotation == 90:
    return q - cdf(1.0 - q, q)
  if rotation == 180:
    return 2.0 * q - 1.0 + cdf(1.0 - q, 1.0 - q)
  if rotation == 270:
    return q - cdf(q, 1.0 - q)
  return cdf(q, q)


def _truth_density(
  bicop: pv.Bicop, u_corner: Tensor, q: float, rotation: int
) -> tuple[Tensor, float]:
  """True density on the full grid plus the exact lower-tail mass."""
  c_true = torch.as_tensor(bicop.pdf(u_corner.numpy()), dtype=torch.float64)
  p_true = _corner_mass(bicop, q, rotation)
  return c_true, p_true


def _fit_parametric_body(u_data: Tensor) -> pv.Bicop:
  """Fit the best parametric body copula to the full sample."""
  family_set = [
    fam
    for fam in pv.BicopFamily.__members__.values()
    if fam != pv.BicopFamily.tll
  ]
  controls = pv.FitControlsBicop(family_set=family_set)
  return pv.Bicop.from_data(
    data=np.asfortranarray(u_data.numpy()), controls=controls
  )


def _fit_tll_body(u_data: Tensor, ridge: float) -> ProbitTLL:
  """Fit the nonparametric bulk estimator."""
  return ProbitTLL(ridge=ridge).fit(u_data)


def _fit_tail_tll(
  u_data: Tensor, q: float, rotation: int, ridge: float
) -> TailCopula:
  """Fit the nonparametric lower-tail estimator."""
  return TailCopula(q, rotation=rotation, ridge=ridge).fit(u_data)


def _fit_tail_parametric(
  u_data: Tensor, q: float, rotation: int, family: str
) -> ParametricTailCopula:
  """Fit the parametric tail estimator."""
  return ParametricTailCopula(family, q, rotation=rotation).fit(u_data)


def lower_left_linear_alpha(u_eval: Tensor, q: float, eps: float) -> Tensor:
  """Linear transition in the band ``q-eps < max(u,v) < q``."""
  u = u_eval[:, 0]
  v = u_eval[:, 1]
  r = torch.maximum(u, v)
  alpha = torch.zeros_like(r)

  inner = r <= (q - eps)
  band = (r > (q - eps)) & (r < q)

  alpha[inner] = 1.0
  alpha[band] = (q - r[band]) / eps
  return alpha


def _rows_for_model(
  spec: ComboSpec,
  model: str,
  region: str,
  est: Tensor,
  truth: Tensor,
  mask: Tensor,
  cell_area: float,
  p_hat: float | None,
  p_true: float,
  include_p: bool,
) -> list[dict[str, object]]:
  """Create density rows and, optionally, a mass row for a model/region."""
  if mask.numel() == 0 or int(mask.sum()) == 0:
    return []
  if torch.isnan(est[mask]).any():
    return []

  base: dict[str, object] = {
    "seed": spec.seed,
    "family": spec.family,
    "tau": spec.tau,
    "n": spec.n,
    "smooth_eps": spec.smooth_eps,
    "grid_eps": spec.grid_eps,
    "rotation": spec.rotation,
    "k_obs": None,
    "k_optim": None,
    "q": spec.n ** (-0.5),
    "tail_family": spec.tail_family,
    "model": model,
    "region": region,
  }

  metrics = grid_metrics_density(est[mask], truth[mask], cell_area)
  rows = [{**base, "target": "c", **metrics}]
  if include_p and p_hat is not None:
    rows.append(
      {
        **base,
        "target": "p",
        "ISE": np.nan,
        "IAE": np.nan,
        "KL": np.nan,
        "p_hat": p_hat,
        "p_true": p_true,
        "AE": abs(p_hat - p_true),
        "RE": abs(p_hat - p_true) / p_true,
      }
    )
  return rows


# --- core simulation --------------------------------------------------------


def run_combo(spec: ComboSpec) -> ComboResult:
  """Run one ``(seed, family, tau, n, smooth_eps)`` cell."""
  try:
    params = copula_params(spec.family, spec.tau, spec.nu)
    bicop = set_bicop(spec.family, params)
    u_data = torch.as_tensor(
      bicop.simulate(spec.n, seeds=[spec.seed]), dtype=torch.float64
    )

    k_grid = np.arange(5, max(6, spec.n // 5) + 1)
    k_optim, _ = select_k(u_data, spec.rotation, k_grid)

    q = u_data.shape[0] ** (-0.5)
    k_obs = _corner_count(u_data, q, spec.rotation)
    if k_obs < spec.min_tail_count:
      status = f"skipped (k={k_obs} < {spec.min_tail_count})"
      return ComboResult(spec, status, k_obs=k_obs, k_optim=k_optim, rows=[])

    grid, cell_area = unit_grid(spec.grid_size, spec.grid_eps)
    u_corner = _reflect(grid, spec.rotation)
    tail_mask = (u_corner[:, 0] <= q) & (u_corner[:, 1] <= q)
    transition_mask = lower_left_linear_alpha(
      u_corner, q=q, eps=spec.smooth_eps
    )
    transition_mask = (transition_mask > 0) & (transition_mask < 1)

    c_true, p_true = _truth_density(bicop, u_corner, q, spec.rotation)

    # Fit the estimators and assemble their densities.
    body_param = _fit_parametric_body(u_data)
    body_tll = _fit_tll_body(u_data, spec.ridge)
    tail_tll = _fit_tail_tll(u_data, q, spec.rotation, spec.ridge)
    tail_param = _fit_tail_parametric(
      u_data, q, spec.rotation, spec.tail_family
    )

    c_param = torch.as_tensor(
      body_param.pdf(u_corner.numpy()), dtype=torch.float64
    )
    c_tll = body_tll.evaluate(u_corner)

    c_tail_tll = torch.full_like(c_tll, float("nan"))
    c_tail_tll[tail_mask] = tail_tll.c(u_corner[tail_mask] / q)

    c_tail_param = torch.full_like(c_tll, float("nan"))
    c_tail_param[tail_mask] = tail_param.c(u_corner[tail_mask] / q)

    c_pt_tll = c_tll.clone()
    c_pt_tll[tail_mask] = c_tail_tll[tail_mask]

    c_pt_tail_param = c_tll.clone()
    c_pt_tail_param[tail_mask] = c_tail_param[tail_mask]

    c_pt_param = c_param.clone()
    c_pt_param[tail_mask] = c_tail_param[tail_mask]

    alpha = lower_left_linear_alpha(u_corner, q=q, eps=spec.smooth_eps)
    mask_alpha = alpha > 0

    tail_tll_s = c_tll.clone()
    tail_tll_s[mask_alpha] = tail_tll.c(u_corner[mask_alpha] / q)
    c_pt_tll_s = alpha * tail_tll_s + (1.0 - alpha) * c_tll

    tail_param_s = c_tll.clone()
    tail_param_s[mask_alpha] = tail_param.c(u_corner[mask_alpha] / q)
    c_pt_tail_param_s = alpha * tail_param_s + (1.0 - alpha) * c_tll

    tail_param_body_s = c_param.clone()
    tail_param_body_s[mask_alpha] = tail_param.c(u_corner[mask_alpha] / q)
    c_pt_param_s = alpha * tail_param_body_s + (1.0 - alpha) * c_param

    models: dict[str, Tensor] = {
      "ParametricBody": c_param,
      "TLLBody": c_tll,
      "PT_TLL_TLL": c_pt_tll,
      "PT_TLL_NegLog": c_pt_tail_param,
      "PT_Param_NegLog": c_pt_param,
      "PT_TLL_TLL_Smooth": c_pt_tll_s,
      "PT_TLL_NegLog_Smooth": c_pt_tail_param_s,
      "PT_Param_NegLog_Smooth": c_pt_param_s,
    }

    rows: list[dict[str, object]] = []

    for model, c_est in models.items():
      # Full-grid region: only models without NaNs on the full grid.
      if not torch.isnan(c_est).any():
        p_hat = float(c_est[tail_mask].sum() * cell_area)
        rows.extend(
          _rows_for_model(
            spec,
            model,
            "full",
            c_est,
            c_true,
            torch.ones_like(tail_mask, dtype=torch.bool),
            cell_area,
            p_hat,
            p_true,
            include_p=False,
          )
        )
      else:
        p_hat = None

      # Lower-tail block.
      if not torch.isnan(c_est[tail_mask]).any():
        p_hat_tail = float(c_est[tail_mask].sum() * cell_area)
        rows.extend(
          _rows_for_model(
            spec,
            model,
            "lower_tail",
            c_est,
            c_true,
            tail_mask,
            cell_area,
            p_hat_tail,
            p_true,
            include_p=True,
          )
        )

      # Transition band.
      if (
        int(transition_mask.sum()) > 0
        and not torch.isnan(c_est[transition_mask]).any()
      ):
        rows.extend(
          _rows_for_model(
            spec,
            model,
            "transition",
            c_est,
            c_true,
            transition_mask,
            cell_area,
            None,
            p_true,
            include_p=False,
          )
        )

    return ComboResult(spec, "ok", k_obs=k_obs, k_optim=k_optim, rows=rows)

  except torch.linalg.LinAlgError:
    return ComboResult(spec, "skipped (singular bandwidth)", None, None, [])
  except Exception as exc:
    return ComboResult(spec, f"error: {exc!r}", None, None, [])


# --- CLI / orchestration ----------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
  """Argparse parser for the simulation grid and run settings."""
  parser = argparse.ArgumentParser(
    description="Parallel piecewise-copula simulation study for npptcop."
  )
  parser.add_argument(
    "--families", nargs="+", choices=FAMILIES, default=list(DEFAULT_FAMILIES)
  )
  parser.add_argument("--taus", nargs="+", type=float, default=[0.4, 0.8])
  parser.add_argument(
    "--ns", nargs="+", type=int, default=[500, 1000, 2000, 5000, 10000, 20000]
  )
  parser.add_argument("--seeds", nargs="+", type=int, default=list(range(30)))
  parser.add_argument("--nu", type=int, default=4)
  parser.add_argument("--grid-size", type=int, default=100)
  parser.add_argument("--grid-eps", type=float, default=1e-4)
  parser.add_argument(
    "--smooth-eps",
    nargs="+",
    type=float,
    default=[0.01, 0.02, 0.03, 0.04, 0.05],
  )
  parser.add_argument("--min-tail-count", type=int, default=5)
  parser.add_argument("--ridge", type=float, default=1e-6)
  parser.add_argument(
    "--tail-family",
    type=str,
    default=DEFAULT_TAIL_FAMILY,
    choices=["neg_logistic", "logistic", "husler_reiss", "dirichlet"],
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=Path("results/sim_study_piecewise_pt.csv"),
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
  """Cartesian product of seeds x families x taus x ns x smooth_eps."""
  return [
    ComboSpec(
      seed=seed,
      family=family,
      tau=tau,
      n=n,
      nu=args.nu,
      grid_size=args.grid_size,
      grid_eps=args.grid_eps,
      smooth_eps=smooth_eps,
      rotation=FAMILY_ROTATION[family],
      min_tail_count=args.min_tail_count,
      ridge=args.ridge,
      tail_family=args.tail_family,
    )
    for seed, family, tau, n, smooth_eps in product(
      args.seeds, args.families, args.taus, args.ns, args.smooth_eps
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
      k_str = "-" if result.k_obs is None else str(result.k_obs)
      LOG.info(
        "[%d/%d] %-8s | tau=%.2f | n=%5d | eps=%.3f | seed=%d -> k_obs=%s, %s",
        i,
        total,
        spec.family,
        spec.tau,
        spec.n,
        spec.smooth_eps,
        spec.seed,
        k_str,
        result.status,
      )
      results.append(result)
  return results


def results_to_frame(results: list[ComboResult]) -> pd.DataFrame:
  """Flatten combo rows into a long-format DataFrame with fixed columns."""
  rows = [row for result in results for row in result.rows]
  frame = pd.DataFrame(rows, columns=list(COLUMNS))
  if frame.empty:
    return frame
  frame["model"] = pd.Categorical(
    frame["model"], categories=MODEL_ORDER, ordered=True
  )
  frame["region"] = pd.Categorical(
    frame["region"],
    categories=["full", "lower_tail", "transition"],
    ordered=True,
  )
  frame["family"] = pd.Categorical(
    frame["family"].str.capitalize(),
    categories=["Clayton", "Gumbel", "Student", "Gaussian", "Tawn"],
    ordered=True,
  )
  return frame.sort_values(
    ["family", "tau", "n", "smooth_eps", "model", "region", "target"]
  ).reset_index(drop=True)


def main() -> None:
  """Parse args, configure logging, run the study, and write the CSV."""
  args = build_parser().parse_args()
  logging.basicConfig(
    level=args.log_level,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
  )
  os.environ.setdefault("OMP_NUM_THREADS", "1")
  os.environ.setdefault("MKL_NUM_THREADS", "1")

  specs = enumerate_combos(args)
  workers = args.workers or min(os.cpu_count() or 1, 8)
  LOG.info("%d combos | %d workers -> %s", len(specs), workers, args.output)
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
