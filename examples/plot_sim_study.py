"""Faceted boxplots of the simulation-study metrics (plotnine).

Reads the long-format CSV written by ``sim_study_biv_tail.py`` and draws, for the
bulk and tail estimators, boxplots of each error metric across sample sizes.
Families are the facet rows; the columns are the metrics, sub-divided by the
dependence level (Kendall's tau) -- this is how the tau dimension is folded into
a single plot, with method shown by fill and n on the x-axis. The y-axis is on a
log scale because the three metrics differ by orders of magnitude.

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
  geom_boxplot,
  ggplot,
  labs,
  scale_fill_manual,
  scale_y_log10,
  theme,
  theme_bw,
)

NS_ORDER: tuple[str, ...] = ("100", "500", "1000", "5000")
FAMILY_ORDER: tuple[str, ...] = ("Clayton", "Gumbel", "Student", "Gaussian")
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
      ["family", "tau", "n", "seed", "model", column]
    ].rename(columns={column: "value"})
    part["metric"] = label
    frames.append(part)
  long = pd.concat(frames, ignore_index=True)
  long = long[long["value"] > 0.0]  # log scale needs strictly positive values
  long["estimator"] = long["model"].map({"Ordinary": "bulk", "Tail": "tail"})
  long["family"] = pd.Categorical(
    long["family"].str.capitalize(), categories=FAMILY_ORDER, ordered=True
  )
  long["metric"] = pd.Categorical(
    long["metric"], categories=METRIC_ORDER, ordered=True
  )
  long["n"] = pd.Categorical(
    long["n"].astype(str), categories=NS_ORDER, ordered=True
  )
  long["tau"] = pd.Categorical(long["tau"].map(lambda t: f"tau={t}"))
  return long


def make_plot(long: pd.DataFrame) -> ggplot:
  """Build the family x (metric, tau) faceted boxplot."""
  return (
    ggplot(long, aes(x="n", y="value", fill="estimator"))
    + geom_boxplot(outlier_size=0.3, size=0.3)
    + facet_grid("family ~ metric + tau", scales="free_y")
    + scale_y_log10()
    + scale_fill_manual(values={"bulk": "#999999", "tail": "#E69F00"})
    + labs(
      x="sample size n",
      y="error (log scale; lower is better)",
      fill="estimator",
    )
    + theme_bw()
    + theme(
      figure_size=(20, 9),
      axis_text_x=element_text(rotation=45, size=7),
      strip_text=element_text(size=8),
    )
  )


def main() -> None:
  """Parse args, build the plot, and save it."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", type=Path, default=Path("results.csv"))
  parser.add_argument(
    "--output", type=Path, default=Path("results/sim_study.png")
  )
  parser.add_argument("--dpi", type=int, default=120)
  args = parser.parse_args()

  plot = make_plot(load_long(args.input))
  args.output.parent.mkdir(parents=True, exist_ok=True)
  plot.save(args.output, dpi=args.dpi, verbose=False)
  print(f"wrote {args.output}")


if __name__ == "__main__":
  main()
