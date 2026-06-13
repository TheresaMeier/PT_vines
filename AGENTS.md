# AGENTS.md

Normative engineering spec for contributors and coding agents working on this
repository: scope, invariants, module boundaries, public APIs, and acceptance
criteria.

## Project overview

`npptcop` ("Nonparametric Pieced-Together Copulas") estimates a bivariate copula
density `c` nonparametrically by *piecing together* two estimators over the unit
square, following `latex/main.tex`:

- a **bulk** estimator `ĉ_B` on the interior — Geenens (2017) probit-transform
  local-likelihood Gaussian KDE (TLL, "constant" method, mirroring
  pyvinecopulib's `TllBicop`);
- a **tail** estimator `ĉ_T` on a corner block of size `q` — rescale the corner
  observations to `[0, 1]²`, fit the same probit-TLL KDE there, and back-transform;
- the raw piece-together `c̃ = ĉ_B·1{interior} + ĉ_T·1{tail}`, then a
  marginal-scaling projection (iterative proportional fitting) onto valid copula
  densities.

**Current status.** The package implements the method's **second step** — the
corner tail estimator (`TailCopula`) and its building block (`ProbitTLL`). The
lower-left corner is the primitive; the other three corners are obtained by
rotation. The full piece-together pipeline and the marginal-scaling projection
are future milestones. The numerical backend is **PyTorch** (float64), chosen
for GPU acceleration and large samples; the library never mutates global torch
state (dtype/device follow the input).

Two important remarks about the repository's design philosophy:

- This repository is under active development. For new features and ongoing design
    work, internal and public APIs may change when doing so improves correctness,
    clarity, or architecture. Do not preserve backward compatibility by default
    unless the task explicitly requires it.
- This repository is quantitatively sensitive: small changes can produce
  mathematically incorrect behavior even when the code
  looks structurally sound. Treat all conventions, patterns,
  and documented formulas in this repository as correctness-critical.

`CLAUDE.md` and
`.github/copilot-instructions.md` are thin pointers into this file.

## Scope

In scope:

- bivariate copula *density* estimation (not CDF, sampling, or fitting);
- the corner tail estimator on a block of size `q` (lower-left corner as the
  primitive; the other three corners via `rotation`);
- the probit-transform local-likelihood KDE building block (Geenens bulk
  estimator);
- constant-method bandwidth selection (Pearson-correlation covariance with an
  ACE maximal-correlation scaling and an `n^(-1/3)` multiplier), with an optional
  non-negative `ridge` (default `0`, which reproduces the unregularized formula
  exactly) that keeps the bandwidth positive-definite when the scaling collapses;
- arbitrary-point and grid evaluation of the targets `h`, `c`, `r`, the tail
  mass `p`, and integrated error metrics (ISE/IAE/KL).

Out of scope (for now):

- the full piece-together estimator and the marginal-scaling/IPF projection;
- dimensions `> 2` and vine extensions;
- plotting, tabular reporting, and simulation harnesses (these live in
  `examples/`, not the library);
- non-Gaussian kernels and the notebook's "smooth blending" variant.

## Package structure

All package code lives in `src/npptcop/`.

```text
src/npptcop/
  __init__.py     # Curated public API (re-exports + __all__)
  transforms.py   # qnorm (Phi^-1), dnorm (phi), SQRT_2PI_INV
  bandwidth.py    # constant-method TLL bandwidth: _ace/_cef/_win_smoother/
                  #   _pearson_cor/_pairwise_mcor + select_bandwidth_constant
  kde.py          # ProbitTLL: probit-transform local-likelihood Gaussian KDE
  tail.py         # TailCopula (rotation 0/90/180/270) + TailFit; exposes h/c/r/p
  metrics.py      # grid_metrics_density (ISE/IAE/KL) + unit_grid helper
```

Test suite location: `tests/` (one test module per source module, plus
`test_invariants.py` for cross-cutting properties and `test_golden.py` for the
frozen canonical-run regression).

### Module boundaries and public API

- `transforms` owns scale conversions only; depended on by `kde` and `tail`.
- `bandwidth` owns the bandwidth-matrix selection and its `tools_stats` helpers
  (private `_`-prefixed); `select_bandwidth_constant` is the package-internal
  entry point.
- `kde.ProbitTLL` owns the local-likelihood fit and evaluation; `evaluate`
  returns the bulk density `ĉ_B`, `density` the raw probit-scale `f̂`.
- `tail.TailCopula` composes a `ProbitTLL` on the rescaled corner data and owns
  the `h`/`c`/`r` relations, the tail mass `p`, and the rotation handling.
- `metrics` owns evaluation-only diagnostics and is never imported by estimators.

The curated public API (`__init__.__all__`) is `ProbitTLL`, `TailCopula`,
`TailFit`, `grid_metrics_density`, `unit_grid`. `transforms` and `bandwidth` are
internal. Tests import from the public API, except `test_bandwidth.py`, which
imports the `tools_stats` helpers directly to pin their C++-equivalent behavior.

### Mathematical conventions and notation

These relations are correctness-critical (see the "quantitatively sensitive"
remark above) and mirror the paper:

- `q` — tail cutoff in `(0, 1)`; the corner block has size `q` per axis. The
  notebook/demo use `q = n^(-1/2)`, a modeling choice kept out of the estimator.
- `p = k / n` — tail mass; `k` = number of observations in the corner block.
- `Phi^-1 = qnorm`, `phi = dnorm` — probit map and standard-normal density.
- `h` — tail-conditional density on `[0, 1]²` (the KDE back-transform
  `h(s) = f̂(Phi^-1 s) / (phi · phi)`); integrates to 1.
- `c` — copula density; `ĉ_B` bulk, `ĉ_T` tail.
- `r` — tail copula density (the paper's `lambda`).
- Targets relate by `c(q·s) = (p / q²) · h(s)` and `r(s) = q·c(q·s) = (p / q)·h(s)`.

The paper writes the tail formulas for the *upper* corner, whereas the code
primitive is the *lower-left* corner; `rotation` (0 lower-left, 90 lower-right,
180 upper-right, 270 upper-left, matching pyvinecopulib's density-rotation
convention) maps the chosen corner onto the lower-left before fitting. Account
for this reflection when cross-checking formulas against the paper.

## Tooling

- Python 3.12+, environment via `uv`.
- Lint + format: `ruff`.
- Type check: `ty`.
- Unit test: `pytest` with `pytest-cov` and `pytest-xdist` for parallelism.

Always prefix Python tools with `uv run`.

Validation sequence for any behavior change (run in this order):

```bash
uv run ruff format .
uv run ruff check . --fix
uv run ruff check . --select ANN --fix
uv run ty check .
uv run pytest tests/ --cov=src/npptcop --cov-report=term-missing -v -n auto
```

Coverage must stay at or above the current level (≈100% on each `src/npptcop/`
file); new code must come with focused tests, not blanket exclusions.

Other useful commands:

```bash
uv venv
uv sync
```

For performance work, profile first and optimize only demonstrated hotspots;
preserve all quantitative semantics and documented invariants.

## Working on this repo

### Inspection order

Before making changes, inspect in this order:

1. `AGENTS.md` — project overview, scope, package structure, working guidelines, module boundaries.
2. `latex/main.tex` — the paper, for the high-level motivation, design rationale, and mathematical specification.
3. `src/npptcop/...` — implementation and local patterns.
4. `tests/...` — expected behavior and edge cases.

Prefer matching existing local patterns over introducing new ones.

### Definition of done

For behavior changes or new features:

- keep code compact, and diffs minimal and scoped to the task;
- avoid unnecessary API churn, but do not preserve backward compatibility by
  default unless the task explicitly requires it;
- add or update focused tests, but keep them compact; prefer extending/refactoring existing helpers,
  fixtures, and parametrized tests over duplicating logic;
- update docstrings for public behavior changes;
- run the validation sequence from [Tooling](#tooling);
- do not introduce undocumented conventions.

### Coding conventions

- Use modern Python features and idioms (3.12+); prefer readability and correctness over cleverness or terseness.
- Public functionality must include meaningful docstrings (purpose, params,
  returns, raised errors).
- Tests import from package public APIs rather than deep internals.
- Source modules (`src/npptcop/**`) import explicit symbols (avoid module-object
  imports for local paths).
- Match existing local naming, typing, dataclass, and testing patterns unless there is a clear reason to change them.

### Maintaining this file

If a coding agent repeatedly misses a durable repository convention, or if
code review repeatedly corrects the same kind of mistake, update `AGENTS.md`
rather than relying on undocumented tribal knowledge. Do not add ephemeral,
user-specific, or machine-local preferences here.
