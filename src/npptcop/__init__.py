"""npptcop: Nonparametric Pieced-Together Copulas.

Curated public API. The nonparametric tail copula density estimator
(``TailCopula``) is the method's second step; ``ProbitTLL`` is the underlying
probit-transform local-likelihood KDE (the bulk estimator); ``grid_metrics_density``
and ``unit_grid`` support grid-based evaluation and comparison.
"""

from npptcop.kde import ProbitTLL
from npptcop.metrics import grid_metrics_density, unit_grid
from npptcop.tail import TailCopula, TailFit

__all__ = [
  "ProbitTLL",
  "TailCopula",
  "TailFit",
  "grid_metrics_density",
  "unit_grid",
]
