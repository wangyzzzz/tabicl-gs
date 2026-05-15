from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tabicl_gs.eval.summary import summarize_metrics_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare direct TabICLv2 with two-stage variants.")
    parser.add_argument("--results-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for result_dir in args.results_dir:
        metrics = pd.read_csv(Path(result_dir) / "fold_metrics.csv")
        metrics["source_dir"] = str(Path(result_dir))
        frames.append(metrics)
    all_metrics = pd.concat(frames, ignore_index=True)
    summary = summarize_metrics_frame(all_metrics)

    runtime_cols = [column for column in ["fit_seconds", "predict_seconds", "total_seconds"] if column in all_metrics.columns]
    if runtime_cols:
        runtime_summary = all_metrics.groupby(["strategy", "model"], as_index=False)[runtime_cols].mean()
        summary = summary.merge(runtime_summary, on=["strategy", "model"], how="left")

    summary.to_csv(output_dir / "comparison_metrics.csv", index=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=summary, x="model", y="pearson", hue="strategy")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_pearson.png", dpi=200)
    print(summary)


if __name__ == "__main__":
    main()
