from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pandas as pd

from tabicl_gs.pipeline.experiment import run_experiment


def build_pca_sweep_config(
    base_config: dict[str, Any],
    trait_col: str,
    group_size: int,
    include_block_scalar: bool,
    pca_dim: int,
    output_dir: str,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    config["trait_col"] = trait_col
    config["grouping_strategy"] = "window"
    config["group_size"] = int(group_size)
    config["include_block_scalar"] = bool(include_block_scalar)
    config["output_dir"] = output_dir
    config["main_models"] = [
        {"name": "TabICLv2-2stage", "stage1_backend": "tabicl", "stage2_backend": "tabicl"}
    ]
    if "tuning" in config:
        config["tuning"]["enabled"] = False
    if "stage1" in config and "tabicl" in config["stage1"]:
        config["stage1"]["tabicl"]["embedding_reduce_dim"] = int(pca_dim)
    else:
        config["stage1"]["embedding_reduce_dim"] = int(pca_dim)
    if "baselines" in config:
        for key in ["gblup", "bayesA", "bayesB", "bayesLasso"]:
            config["baselines"][key] = False
    config["save_block_summaries"] = True
    return config


def summarize_pca_sweep(root_dir: str | Path) -> pd.DataFrame:
    root_dir = Path(root_dir)
    rows = []
    window_rows = []
    for metrics_path in sorted(root_dir.glob("pca_*/fold_metrics.csv")):
        pca_dir = metrics_path.parent
        pca_dim = int(pca_dir.name.split("_", 1)[1])
        metrics = pd.read_csv(metrics_path)
        model_metrics = metrics[metrics["model"] == "TabICLv2-2stage"]
        if model_metrics.empty:
            continue
        rows.append(
            {
                "pca_dim": pca_dim,
                "pearson_mean": model_metrics["pearson"].mean(),
                "r2_mean": model_metrics["r2"].mean(),
                "pearson_std": model_metrics["pearson"].std(ddof=0),
                "r2_std": model_metrics["r2"].std(ddof=0),
            }
        )
        for meta_path in sorted(pca_dir.glob("fold_*/fold_metadata.json")):
            with meta_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            model_runs = payload.get("model_runs", {})
            tabicl_run = model_runs.get("TabICLv2-2stage", {})
            for block in tabicl_run.get("block_summaries", []) or []:
                window_rows.append(
                    {
                        "fold": payload["fold"],
                        "pca_dim": pca_dim,
                        "block_id": block["block_id"],
                        "num_snps": block["num_snps"],
                        "raw_embedding_dim": block["raw_embedding_dim"],
                        "reduced_embedding_dim": block["reduced_embedding_dim"],
                        "explained_variance_ratio_sum": block.get("explained_variance_ratio_sum"),
                    }
                )
    summary = pd.DataFrame(rows).sort_values("pca_dim")
    summary.to_csv(root_dir / "pca_summary.csv", index=False)
    if window_rows:
        window_df = pd.DataFrame(window_rows).sort_values(["pca_dim", "fold", "block_id"])
        window_df.to_csv(root_dir / "window_explained_variance.csv", index=False)
        window_summary = (
            window_df.groupby(["pca_dim", "block_id"], as_index=False)["explained_variance_ratio_sum"]
            .mean()
            .rename(columns={"explained_variance_ratio_sum": "mean_explained_variance_ratio_sum"})
        )
        window_summary.to_csv(root_dir / "window_explained_variance_summary.csv", index=False)
    return summary


def run_pca_sweep(
    base_config: dict[str, Any],
    trait_col: str,
    group_size: int,
    include_block_scalar: bool,
    pca_dims: list[int],
    output_root: str,
) -> pd.DataFrame:
    for pca_dim in pca_dims:
        config = build_pca_sweep_config(
            base_config=base_config,
            trait_col=trait_col,
            group_size=group_size,
            include_block_scalar=include_block_scalar,
            pca_dim=pca_dim,
            output_dir=f"{output_root}/pca_{pca_dim}",
        )
        run_experiment(config)
    return summarize_pca_sweep(output_root)
