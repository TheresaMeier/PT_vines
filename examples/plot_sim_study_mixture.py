"""Median-error line plot for the simulation-study metrics (plotnine).

Reads the long-format CSV written by ``sim_study_biv_tail.py`` and draws the
median error of each metric against the sample size.

- Single-family mode: facet rows are families, line type is Kendall's tau.
- Patchwork mode: facet rows are patchwork settings, line type is the
  ``tau_a / tau_b`` setting label.

Both axes are on a log scale.
"""

from __future__ import annotations

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

N_BREAKS: tuple[int, ...] = (200, 500, 1000, 2000, 5000, 10000, 20000, 50000)
METRIC_ORDER: tuple[str, ...] = ("KL(c)", "KL(h)", "RE(p)")

ESTIMATOR_LABELS: dict[str, str] = {
  "Ordinary": "bulk",
  "Tail": "tail",
  "HR": "HR",
  "NegLogistic": "neg-logistic",
  "Logistic": "logistic",
  "Dirichlet": "Dirichlet",
}
ESTIMATOR_COLORS: dict[str, str] = {
  "bulk": "#999999",
  "tail": "#E69F00",
  "HR": "#56B4E9",
  "neg-logistic": "#009E73",
  "logistic": "#0072B2",
  "Dirichlet": "#CC79A7",
}

_METRICS: tuple[tuple[str, str, str], ...] = (
  ("c", "KL", "KL(c)"),
  ("h", "KL", "KL(h)"),
  ("p", "RE", "RE(p)"),
)


def _scenario_label(row: pd.Series) -> str:
  """Human-readable facet label."""
  if row["mode"] == "single":
    return str(row["family"]).capitalize()
  if row["mode"] == "patchwork":
    return f"{row['tau_a']:g}/{row['tau_b']:g}"
  return str(row.get("family", "") or row.get("mixture", "") or row["mode"])


def _line_key(row: pd.Series) -> str:
  """Line-type key."""
  if row["mode"] == "single":
    return f"{row['tau']:g}"
  if row["mode"] == "patchwork":
    return f"{row['tau_a']:g}/{row['tau_b']:g}"
  return str(row.get("mixture", "") or row["mode"])


def load_long(path: Path, mode: str = "both") -> pd.DataFrame:
  """Reshape the CSV into one row per (cell, metric) value."""
  df = pd.read_csv(path)

  if mode != "both":
    df = df[df["mode"] == mode].copy()

  frames = []
  for target, column, label in _METRICS:
    cols = [
      "mode",
      "family",
      "tau",
      "tau_a",
      "tau_b",
      "n",
      "seed",
      "model",
      column,
    ]
    part = df[df["target"] == target][cols].rename(columns={column: "value"})
    part["metric"] = label
    part["estimator"] = part["model"].map(ESTIMATOR_LABELS)

    part["scenario"] = part.apply(_scenario_label, axis=1)
    part["line_key"] = part.apply(_line_key, axis=1)

    frames.append(part)

  long = pd.concat(frames, ignore_index=True)
  long = long[long["value"] > 0.0].copy()
  long = long.dropna(subset=["estimator"])
  long["metric"] = pd.Categorical(
    long["metric"], categories=METRIC_ORDER, ordered=True
  )
  return long


def make_plot(long: pd.DataFrame) -> ggplot:
  """Median error vs n: colour by estimator, line type by setting label."""
  keys = ["scenario", "metric", "n", "line_key", "estimator"]
  med = long.groupby(keys, observed=True)["value"].median().reset_index()
  med["series"] = med["estimator"] + " / " + med["line_key"].astype(str)

  return (
    ggplot(
      med,
      aes(
        x="n",
        y="value",
        color="estimator",
        linetype="line_key",
        group="series",
      ),
    )
    + geom_line()
    + geom_point(size=1.3)
    + facet_grid("scenario ~ metric")
    + scale_x_log10(breaks=N_BREAKS)
    + scale_y_log10()
    + scale_color_manual(values=ESTIMATOR_COLORS)
    + labs(
      x="sample size n (log scale)",
      y="median error (log scale; lower is better)",
      color="estimator",
      linetype="tau / setting",
    )
    + theme_bw()
    + theme(
      figure_size=(11, 8),
      axis_text_x=element_text(rotation=45, size=7),
    )
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--input",
    type=Path,
    default=Path("results/sim_study_biv_tail_patchwork.csv"),
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=Path("latex/figures/sim_study_patchwork.png"),
  )
  parser.add_argument(
    "--mode",
    choices=["single", "patchwork", "both"],
    default="both",
    help="Which rows to plot from the CSV.",
  )
  parser.add_argument("--dpi", type=int, default=130)
  args = parser.parse_args()

  plot = make_plot(load_long(args.input, mode=args.mode))
  args.output.parent.mkdir(parents=True, exist_ok=True)
  plot.save(args.output, dpi=args.dpi, verbose=False)
  print(f"wrote {args.output}")


if __name__ == "__main__":
  main()
