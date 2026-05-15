from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize rice529 dynamic-0.99 group-size/norm sweeps.")
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def _parse_experiment_path(output_root: Path, metrics_path: Path) -> dict[str, object]:
    relative_parts = metrics_path.relative_to(output_root).parts
    if len(relative_parts) < 4:
        raise ValueError(f"Unexpected metrics path layout: {metrics_path}")
    setting = relative_parts[0]
    group_part = relative_parts[1]
    trait_slug = relative_parts[2]
    if not group_part.startswith("group_size_"):
        raise ValueError(f"Unexpected group directory: {group_part}")
    group_size = int(group_part.split("_")[-1])
    return {
        "setting": setting,
        "group_size": group_size,
        "trait_slug": trait_slug,
    }


def _fold_block_stats(experiment_dir: Path) -> dict[int, dict[str, float | int | None]]:
    stats: dict[int, dict[str, float | int | None]] = {}
    for fold_dir in sorted(experiment_dir.glob("fold_*")):
        metadata_path = fold_dir / "fold_metadata.json"
        if not metadata_path.exists():
            continue
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        model_runs = payload.get("model_runs", {})
        tabicl_run = model_runs.get("TabICLv2-2stage", {})
        block_summaries = tabicl_run.get("block_summaries") or []
        reduced_dims = [
            int(block["reduced_embedding_dim"])
            for block in block_summaries
            if block.get("reduced_embedding_dim") is not None
        ]
        explained = [
            float(block["explained_variance_ratio_sum"])
            for block in block_summaries
            if block.get("explained_variance_ratio_sum") is not None
        ]
        explained = [value for value in explained if np.isfinite(value)]
        stats[int(payload["fold"])] = {
            "mean_reduced_block_embedding_dim_from_meta": None if not reduced_dims else float(np.mean(reduced_dims)),
            "min_reduced_block_embedding_dim_from_meta": None if not reduced_dims else int(np.min(reduced_dims)),
            "max_reduced_block_embedding_dim_from_meta": None if not reduced_dims else int(np.max(reduced_dims)),
            "mean_explained_variance_ratio_sum_from_meta": None if not explained else float(np.mean(explained)),
        }
    return stats


def summarize(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, object]] = []
    for metrics_path in sorted(output_root.glob("*/*/*/fold_metrics.csv")):
        parsed = _parse_experiment_path(output_root, metrics_path)
        experiment_dir = metrics_path.parent
        metrics_df = pd.read_csv(metrics_path)
        metrics_df = metrics_df[metrics_df["model"] == "TabICLv2-2stage"].copy()
        if metrics_df.empty:
            continue
        fold_stats = _fold_block_stats(experiment_dir)
        for row in metrics_df.to_dict(orient="records"):
            fold = int(row["fold"])
            merged = dict(parsed)
            merged.update(row)
            merged.update(fold_stats.get(fold, {}))
            if merged.get("mean_reduced_block_embedding_dim") is None:
                merged["mean_reduced_block_embedding_dim"] = merged.get("mean_reduced_block_embedding_dim_from_meta")
            if merged.get("min_reduced_block_embedding_dim") is None:
                merged["min_reduced_block_embedding_dim"] = merged.get("min_reduced_block_embedding_dim_from_meta")
            if merged.get("max_reduced_block_embedding_dim") is None:
                merged["max_reduced_block_embedding_dim"] = merged.get("max_reduced_block_embedding_dim_from_meta")
            if merged.get("mean_explained_variance_ratio_sum") is None:
                merged["mean_explained_variance_ratio_sum"] = merged.get("mean_explained_variance_ratio_sum_from_meta")
            fold_rows.append(merged)

    fold_df = pd.DataFrame(fold_rows)
    if fold_df.empty:
        return fold_df, pd.DataFrame(), pd.DataFrame()

    experiment_df = (
        fold_df.groupby(["setting", "group_size", "trait_slug"], as_index=False)
        .agg(
            folds_completed=("fold", "nunique"),
            pearson_mean=("pearson", "mean"),
            pearson_std=("pearson", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            num_blocks_mean=("num_blocks", "mean"),
            stage2_input_dim_mean=("stage2_input_dim", "mean"),
            mean_reduced_block_embedding_dim_mean=("mean_reduced_block_embedding_dim", "mean"),
            min_reduced_block_embedding_dim_min=("min_reduced_block_embedding_dim", "min"),
            max_reduced_block_embedding_dim_max=("max_reduced_block_embedding_dim", "max"),
            mean_explained_variance_ratio_sum_mean=("mean_explained_variance_ratio_sum", "mean"),
        )
        .sort_values(["setting", "group_size", "trait_slug"])
    )

    overall_df = (
        experiment_df.groupby(["setting", "group_size"], as_index=False)
        .agg(
            traits_completed=("trait_slug", "nunique"),
            pearson_mean=("pearson_mean", "mean"),
            r2_mean=("r2_mean", "mean"),
            num_blocks_mean=("num_blocks_mean", "mean"),
            stage2_input_dim_mean=("stage2_input_dim_mean", "mean"),
            mean_reduced_block_embedding_dim_mean=("mean_reduced_block_embedding_dim_mean", "mean"),
            mean_explained_variance_ratio_sum_mean=("mean_explained_variance_ratio_sum_mean", "mean"),
        )
        .sort_values(["setting", "group_size"])
    )
    return fold_df, experiment_df, overall_df


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    fold_df, experiment_df, overall_df = summarize(output_root)
    if fold_df.empty:
        print("No completed experiments found.")
        return

    fold_df.to_csv(output_root / "fold_summary.csv", index=False)
    experiment_df.to_csv(output_root / "experiment_summary.csv", index=False)
    overall_df.to_csv(output_root / "overall_summary.csv", index=False)

    print(overall_df.to_string(index=False))


if __name__ == "__main__":
    main()
