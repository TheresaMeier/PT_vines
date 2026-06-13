# npptcop

Nonparametric pieced-together copula density estimation.

`npptcop` estimates a bivariate copula density by piecing together a **bulk**
estimator on the interior of the unit square and a **tail** estimator on its
corners, following the method in [`latex/main.tex`](latex/main.tex).
Both pieces are built on Geenens' probit-transform local-likelihood Gaussian KDE
(TLL), and the numerical backend is PyTorch (float64).

This is research software under active development. The current release
implements the method's **second step** — the corner tail estimator
(`TailCopula`) and its building block (`ProbitTLL`). Any of the four corners is
selected through `rotation` (matching pyvinecopulib's convention).

## Install

```bash
uv venv
uv sync
```

## Usage

```python
import numpy as np
import torch
import pyvinecopulib as pv
from npptcop import TailCopula, unit_grid

# Simulate a Clayton copula with lower-left tail dependence.
n = 2000
bicop = pv.Bicop(
    family=pv.BicopFamily.clayton,
    parameters=np.array([[3.0]], dtype=float),
)
u = torch.as_tensor(bicop.simulate(n, seeds=[97]), dtype=torch.float64)

# Fit the lower-left tail estimator (rotation=0); use rotation=180 for the
# upper-right corner, 90 / 270 for the off-diagonal corners.
q = n ** (-0.5)                      # tail cutoff
est = TailCopula(q, rotation=0).fit(u)

grid, _ = unit_grid(size=50)         # rescaled points s in [0, 1]^2
h = est.h(grid)                      # tail-conditional density on [0, 1]^2
c = est.c(grid)                      # copula density at (q*s)
r = est.r(grid)                      # tail copula density (lambda)
print(est.fit_.p, est.fit_.k)        # tail mass p = k / n and tail count k
```

See [`examples/`](examples/) for a worked demo notebook
(`tail_copula_clayton.ipynb`) and a parallel simulation study
(`sim_study_biv_tail.py`, run with `uv run --extra interactive python
examples/sim_study_biv_tail.py`).

## Development

The normative engineering spec is [`AGENTS.md`](AGENTS.md). Run the validation
sequence with `uv`:

```bash
uv run ruff format .
uv run ruff check . --fix
uv run ruff check . --select ANN --fix
uv run ty check .
uv run pytest tests/ --cov=src/npptcop --cov-report=term-missing -v -n auto
```
