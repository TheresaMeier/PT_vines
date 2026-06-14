"""Median-error line plot of the simulation-study metrics (plotnine).

Reads the long-format CSV written by ``sim_study_biv_tail.py`` and draws, for the
bulk and tail estimators, the median error of each metric against the *tail
count* ``k`` -- the effective sample size of the tail estimator, which grows
only like ``sqrt(n)`` under ``q_n = n^{-1/2}``. Families are the facet rows and
metrics the facet columns; the estimator is shown by line colour and the
dependence level (Kendall's tau) by line type. Both axes are on a log scale.

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

FAMILY_ORDER: tuple[str, ...] = ("Clayton", "Gumbel", "Student")
METRIC_ORDER: tuple[str, ...] = ("KL(c)", "KL(h)", "RE(p)")
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
      ["family", "tau", "n", "k", "seed", "model", column]
    ].rename(columns={column: "value"})
    part["metric"] = label
    frames.append(part)
  long = pd.concat(frames, ignore_index=True)
  long = long[long["value"] > 0.0]  # log scale needs strictly positive values
  long["estimator"] = long["model"].map({"Ordinary": "bulk", "Tail": "tail"})
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
  """Median error vs tail count k: colour by estimator, line type by tau."""
  med = (
    long.groupby(["family", "metric", "n", "tau", "estimator"], observed=True)[
      "value"
    ]
    .median()
    .reset_index()
  )
  # Effective sample size: median tail count per (family, tau, n) cell.
  kk = (
    long.drop_duplicates(["family", "tau", "n", "seed"])
    .groupby(["family", "tau", "n"], observed=True)["k"]
    .median()
    .reset_index()
  )
  med = med.merge(kk, on=["family", "tau", "n"])
  med["series"] = med["estimator"] + " / " + med["tau"]
  return (
    ggplot(
      med,
      aes(x="k", y="value", color="estimator", linetype="tau", group="series"),
    )
    + geom_line()
    + geom_point(size=1.3)
    + facet_grid("family ~ metric")
    + scale_x_log10()
    + scale_y_log10()
    + scale_color_manual(values={"bulk": "#999999", "tail": "#E69F00"})
    + labs(
      x="tail count k (log scale)",
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
