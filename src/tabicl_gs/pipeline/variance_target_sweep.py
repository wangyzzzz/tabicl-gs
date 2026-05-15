from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pandas as pd

from tabicl_gs.pipeline.experiment import run_experiment


def build_variance_target_config(
    base_config: dict[str, Any],
    trait_col: str,
    group_size: int,
    include_block_scalar: bool,
    variance_target: float,
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
    config["save_block_summaries"] = True
    stage1_cfg = config["stage1"]["tabicl"] if "tabicl" in config.get("stage1", {}) else config["stage1"]
    stage1_cfg["embedding_reduce_dim"] = None
    stage1_cfg["embedding_explained_variance_target"] = float(variance_target)
    stage1_cfg["track_full_explained_variance"] = True
    if "baselines" in config:
        for key in ["gblup", "bayesA", "bayesB", "bayesLasso"]:
            config["baselines"][key] = False
    return config


def summarize_variance_target_sweep(root_dir: str | Path) -> pd.DataFrame:
    root_dir = Path(root_dir)
    rows = []
    window_rows = []
    for metrics_path in sorted(root_dir.glob("target_*/fold_metrics.csv")):
        target_dir = metrics_path.parent
        target_value = float(target_dir.name.split("_", 1)[1])
        metrics = pd.read_csv(metrics_path)
        model_metrics = metrics[metrics["model"] == "TabICLv2-2stage"]
        if model_metrics.empty:
            continue
        rows.append(
            {
                "variance_target": target_value,
                "pearson_mean": model_metrics["pearson"].mean(),
                "r2_mean": model_metrics["r2"].mean(),
                "chosen_dim_mean": model_metrics["reduced_block_embedding_dim"].mean(),
                "chosen_dim_min": model_metrics["reduced_block_embedding_dim"].min(),
                "chosen_dim_max": model_metrics["reduced_block_embedding_dim"].max(),
            }
        )
        for meta_path in sorted(target_dir.glob("fold_*/fold_metadata.json")):
            with meta_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            model_runs = payload.get("model_runs", {})
            tabicl_run = model_runs.get("TabICLv2-2stage", {})
            for block in tabicl_run.get("block_summaries", []) or []:
                for idx, cum_var in enumerate(block.get("explained_variance_curve", []) or [], start=1):
                    window_rows.append(
                        {
                            "fold": payload["fold"],
                            "variance_target": target_value,
                            "block_id": block["block_id"],
                            "pc_dim": idx,
                            "cumulative_explained_variance_ratio": cum_var,
                            "chosen_dim": block["reduced_embedding_dim"],
                        }
                    )
    summary = pd.DataFrame(rows).sort_values("variance_target")
    summary.to_csv(root_dir / "variance_target_summary.csv", index=False)
    if window_rows:
        window_df = pd.DataFrame(window_rows).sort_values(["variance_target", "fold", "block_id", "pc_dim"])
        window_df.to_csv(root_dir / "window_explained_variance_full.csv", index=False)
    return summary


def run_variance_target_sweep(
    base_config: dict[str, Any],
    trait_col: str,
    group_size: int,
    include_block_scalar: bool,
    variance_targets: list[float],
    output_root: str,
) -> pd.DataFrame:
    for target in variance_targets:
        config = build_variance_target_config(
            base_config=base_config,
            trait_col=trait_col,
            group_size=group_size,
            include_block_scalar=include_block_scalar,
            variance_target=target,
            output_dir=f"{output_root}/target_{target:.2f}",
        )
        run_experiment(config)
    return summarize_variance_target_sweep(output_root)
