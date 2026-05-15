from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize baseline-only 10 trait results.")
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def summarize(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    for metrics_path in sorted(output_root.glob("*/fold_metrics.csv")):
        trait_slug = metrics_path.parent.name
        frame = pd.read_csv(metrics_path)
        if frame.empty:
            continue
        frame["trait_slug"] = trait_slug
        rows.append(frame)

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    fold_df = pd.concat(rows, ignore_index=True)
    summary_df = (
        fold_df.groupby(["trait_slug", "model"], as_index=False)
        .agg(
            folds_completed=("fold", "nunique"),
            pearson_mean=("pearson", "mean"),
            pearson_std=("pearson", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
        )
        .sort_values(["trait_slug", "model"])
    )
    overall_df = (
        summary_df.groupby("model", as_index=False)
        .agg(
            traits_completed=("trait_slug", "nunique"),
            pearson_mean=("pearson_mean", "mean"),
            r2_mean=("r2_mean", "mean"),
        )
        .sort_values("pearson_mean", ascending=False)
    )
    return summary_df, overall_df


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary_df, overall_df = summarize(output_root)
    if summary_df.empty:
        print("No completed baseline experiments found.")
        return

    summary_df.to_csv(output_root / "trait_model_summary.csv", index=False)
    overall_df.to_csv(output_root / "overall_summary.csv", index=False)
    print(overall_df.to_string(index=False))


if __name__ == "__main__":
    main()
