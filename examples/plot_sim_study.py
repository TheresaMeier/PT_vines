"""Median-error line plot of the simulation-study metrics (plotnine).

Reads the long-format CSV written by ``sim_study_biv_tail.py`` and draws, for the
bulk and tail estimators, the median error of each metric against the sample
size. Families are the facet rows and metrics the facet columns; the estimator
is shown by line colour and the dependence level (Kendall's tau) by line type,
so all dimensions sit in one figure. Both axes are on a log scale.

Run (``plotnine`` lives in the ``interactive`` extra)::

  uv run --extra interactive python examples/plot_sim_study.py
"""

import argparse
from pathlib import Path

import pandas as pd
from plotnine import (
  aes,
  element_text,
  facet_grid,
  geom_line,
  geom_point,
  ggplot,
  labs,
  scale_color_manual,
  scale_x_log10,
  scale_y_log10,
  theme,
  theme_bw,
)

N_BREAKS: tuple[int, ...] = (200, 500, 1000, 2000, 5000)
FAMILY_ORDER: tuple[str, ...] = ("Clayton", "Gumbel", "Student")
METRIC_ORDER: tuple[str, ...] = ("KL(c)", "KL(h)", "RE(p)")
# Display model -> estimator label (line colour); the parametric tail families
# join the nonparametric bulk/tail estimators on the same axes.
ESTIMATOR_LABELS: dict[str, str] = {
  "Ordinary": "bulk",
  "Tail": "tail",
  "HR": "HR",
  "NegLogistic": "neg-logistic",
  "Logistic": "logistic",
  "Dirichlet": "Dirichlet",
}
# Okabe-Ito colourblind-safe palette.
ESTIMATOR_COLORS: dict[str, str] = {
  "bulk": "#999999",
  "tail": "#E69F00",
  "HR": "#56B4E9",
  "neg-logistic": "#009E73",
  "logistic": "#0072B2",
  "Dirichlet": "#CC79A7",
}
# (target, metric column, display label)
_METRICS: tuple[tuple[str, str, str], ...] = (
  ("c", "KL", "KL(c)"),
  ("h", "KL", "KL(h)"),
  ("p", "RE", "RE(p)"),
)


def load_long(path: Path) -> pd.DataFrame:
  """Reshape the wide results CSV into one row per (cell, metric) value."""
  df = pd.read_csv(path)
  frames = []
  for target, column, label in _METRICS:
    part = df[df["target"] == target][
      ["family", "tau", "n", "seed", "model", column]
    ].rename(columns={column: "value"})
    part["metric"] = label
    frames.append(part)
  long = pd.concat(frames, ignore_index=True)
  long = long[long["value"] > 0.0]  # log scale needs strictly positive values
  long["estimator"] = long["model"].map(ESTIMATOR_LABELS)
  long = long.dropna(subset=["estimator"])  # drop any unmapped model
  long = long[long["family"].isin([f.lower() for f in FAMILY_ORDER])]
  long["family"] = pd.Categorical(
    long["family"].str.capitalize(), categories=FAMILY_ORDER, ordered=True
  )
  long["metric"] = pd.Categorical(
    long["metric"], categories=METRIC_ORDER, ordered=True
  )
  long["tau"] = long["tau"].map(lambda t: f"{t:g}")
  return long


def make_plot(long: pd.DataFrame) -> ggplot:
  """Median error vs n: colour by estimator, line type by Kendall's tau."""
  keys = ["family", "metric", "n", "tau", "estimator"]
  med = long.groupby(keys, observed=True)["value"].median().reset_index()
  med["series"] = med["estimator"] + " / " + med["tau"]
  return (
    ggplot(
      med,
      aes(x="n", y="value", color="estimator", linetype="tau", group="series"),
    )
    + geom_line()
    + geom_point(size=1.3)
    + facet_grid("family ~ metric")
    + scale_x_log10(breaks=N_BREAKS)
    + scale_y_log10()
    + scale_color_manual(values=ESTIMATOR_COLORS)
    + labs(
      x="sample size n (log scale)",
      y="median error (log scale; lower is better)",
      color="estimator",
      linetype="Kendall's tau",
    )
    + theme_bw()
    + theme(
      figure_size=(11, 8),
      axis_text_x=element_text(rotation=45, size=7),
    )
  )


def main() -> None:
  """Parse args, build the plot, and save it."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", type=Path, default=Path("results.csv"))
  parser.add_argument(
    "--output", type=Path, default=Path("latex/figures/sim_study.png")
  )
  parser.add_argument("--dpi", type=int, default=130)
  args = parser.parse_args()

  plot = make_plot(load_long(args.input))
  args.output.parent.mkdir(parents=True, exist_ok=True)
  plot.save(args.output, dpi=args.dpi, verbose=False)
  print(f"wrote {args.output}")


if __name__ == "__main__":
  main()
