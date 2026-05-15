from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize 5-fold GS results.")
    parser.add_argument("--results-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for result_dir in args.results_dir:
        result_path = Path(result_dir) / "fold_metrics.csv"
        frame = pd.read_csv(result_path)
        frame["source_dir"] = str(Path(result_dir))
        frames.append(frame)
    metrics = pd.concat(frames, ignore_index=True)
    summary = metrics.groupby(["strategy", "model"], as_index=False)[["pearson", "rmse", "mae", "r2"]].mean()
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=summary, x="model", y="pearson", hue="strategy")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "summary_pearson.png", dpi=200)
    print(summary)


if __name__ == "__main__":
    main()
