"""npptcop: Nonparametric Pieced-Together Copulas.

Curated public API. The nonparametric tail copula density estimator
(``TailCopula``) is the method's second step; ``ParametricTailCopula`` is a
drop-in parametric alternative that fits a classical extreme-value tail family by
maximum likelihood; ``ProbitTLL`` is the underlying probit-transform
local-likelihood KDE (the bulk estimator); ``grid_metrics_density`` and
``unit_grid`` support grid-based evaluation and comparison.
"""

from npptcop.kde import ProbitTLL
from npptcop.metrics import grid_metrics_density, unit_grid
from npptcop.parametric import ParametricTailCopula, ParametricTailFit
from npptcop.tail import TailCopula, TailFit

__all__ = [
  "ParametricTailCopula",
  "ParametricTailFit",
  "ProbitTLL",
  "TailCopula",
  "TailFit",
  "grid_metrics_density",
  "unit_grid",
]
