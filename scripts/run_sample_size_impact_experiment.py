from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tabicl_gs.config import deep_update, load_experiment_config
from tabicl_gs.eval.metrics import regression_metrics
from tabicl_gs.eval.splits import make_outer_cv_splits
from tabicl_gs.pipeline.sample_size_impact import (
    build_nested_train_subsets,
    compute_trait_level_sample_size_summary,
    run_tabicl_block_search_on_subset,
    should_run_block_search,
    summarize_sample_size_metrics,
)
from tabicl_gs.pipeline.dual_prior_fold_search import _load_fold_data
from tabicl_gs.pipeline.experiment import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sample-size impact experiment with fixed block size.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--trait-col", required=True)
    parser.add_argument("--plink-prefix", required=True)
    parser.add_argument("--phenotype-csv", required=True)
    parser.add_argument("--phenotype-sample-id-col", required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--fold-ids", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--proportions", nargs="+", type=float, default=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--selection-tag", default="unspecified")
    parser.add_argument("--block-search-proportions", nargs="*", type=float, default=[])
    parser.add_argument("--block-search-min", type=int, default=200)
    parser.add_argument("--block-search-max", type=int, default=1500)
    parser.add_argument("--block-search-trials", type=int, default=10)
    parser.add_argument("--block-search-inner-folds", type=int, default=3)
    return parser.parse_args()


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _slice_train_subset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    subset_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return X_train[subset_indices].astype(np.float32), y_train[subset_indices].astype(np.float32)


def _run_baselines_on_subset(
    *,
    config: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    work_dir: Path,
) -> list[dict]:
    from tabicl_gs.data.plink import impute_by_train_mean
    from tabicl_gs.models.baselines import run_r_baseline

    X_train_base, X_test_base = impute_by_train_mean(X_train, X_test)
    rows = []
    for model_name in ("GBLUP", "BayesB"):
        result = run_r_baseline(
            model_name=model_name,
            X_train=X_train_base,
            y_train=y_train,
            X_test=X_test_base,
            output_dir=work_dir / model_name,
            rscript_path=config["baselines"]["rscript_path"],
            seed=int(config["seed"]),
            sommer_method=config["baselines"].get("sommer_method"),
            keep_artifacts=False,
        )
        metric = regression_metrics(y_test, np.asarray(result.predictions, dtype=np.float32))
        rows.append(
            {
                "model": model_name,
                "test_pearson": float(metric["pearson"]),
                "test_r2": float(metric["r2"]),
            }
        )
    return rows


def _build_sample_override_config(
    config: dict,
    *,
    fold_id: int,
    subset_indices: np.ndarray,
    gate_prior_train_source: str | None = None,
) -> dict:
    override = {
        "_sample_size_override": {
            "fold_id": int(fold_id),
            "train_subset_indices": np.asarray(subset_indices, dtype=np.int64).tolist(),
        }
    }
    if gate_prior_train_source is not None:
        override["gate_prior_train_source"] = str(gate_prior_train_source)
    return deep_update(config, override)


def _estimate_prior_only_alpha_from_cache(cached: dict) -> float:
    from tabicl_gs.pipeline.dual_prior_utils import resolve_gate_prior_train_predictions

    alpha_bayesb, alpha_gblup, _ = resolve_gate_prior_train_predictions(cached, prior_train_source="inner_oof")
    gap = alpha_bayesb - alpha_gblup
    alpha_targets = np.zeros_like(cached["y_outer_train"], dtype=np.float32)
    stable = np.abs(gap) > 1e-6
    alpha_targets[stable] = (cached["y_outer_train"][stable] - alpha_gblup[stable]) / gap[stable]
    return float(np.clip(alpha_targets, 0.0, 1.0).mean())


def main() -> None:
    args = parse_args()
    base_config = load_experiment_config(args.config)
    runtime_override = {
        "group_size": int(args.group_size),
        "trait_col": args.trait_col,
        "plink_prefix": args.plink_prefix,
        "phenotype_csv": args.phenotype_csv,
        "phenotype_sample_id_col": args.phenotype_sample_id_col,
    }
    config = deep_update(base_config, runtime_override)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    _save_json(
        output_root / "selection.json",
        {
            "selection_tag": args.selection_tag,
            "group_size": int(args.group_size),
            "proportions": [float(p) for p in args.proportions],
            "repeats": int(args.repeats),
            "fold_ids": [int(f) for f in args.fold_ids],
            "block_search_proportions": [float(p) for p in args.block_search_proportions],
            "block_search_trials": int(args.block_search_trials),
        },
    )

    from tabicl_gs.pipeline.dual_prior_fold_search import (
        precompute_dual_prior_cache,
        run_dual_prior_fixed_block_on_fold,
        run_dual_prior_fixed_block_with_frozen_gate_on_fold,
    )
    from tabicl_gs.models.model_specs import resolve_two_stage_model_specs
    from tabicl_gs.pipeline.experiment import (
        _build_stage_features,
        _prepare_stage2_config,
        _resolve_second_stage_adjustment,
        _resolve_stage2_feature_mode,
    )
    from tabicl_gs.models.factory import fit_stage2_model

    fold1_train, fold1_y, _, _ = _load_fold_data(config, fold_id=1)
    shared_subset_map = build_nested_train_subsets(
        train_indices=np.arange(fold1_train.shape[0], dtype=np.int64),
        proportions=[float(p) for p in args.proportions],
        repeats=int(args.repeats),
        seed=int(config["seed"]),
    )

    selection_rows: list[dict] = []
    for (repeat, proportion), subset_indices in shared_subset_map.items():
        subset_root = output_root / "sample_subsets" / f"repeat_{repeat}" / f"p_{proportion:.2f}"
        subset_root.mkdir(parents=True, exist_ok=True)
        np.save(subset_root / "fold1_train_indices.npy", np.asarray(subset_indices, dtype=np.int64))
        X_train = fold1_train[subset_indices].astype(np.float32)
        y_train = fold1_y[subset_indices].astype(np.float32)
        effective_group_size = int(args.group_size)
        if should_run_block_search(float(proportion), args.block_search_proportions):
            search_cfg = _build_sample_override_config(
                config,
                fold_id=1,
                subset_indices=np.asarray(subset_indices, dtype=np.int64),
            )
            search_root = subset_root / "block_search"
            search_summary = run_tabicl_block_search_on_subset(
                base_config=search_cfg,
                X_train=X_train,
                y_train=y_train,
                output_root=search_root,
                min_block=int(args.block_search_min),
                max_block=int(args.block_search_max),
                n_trials=int(args.block_search_trials),
                inner_folds=int(args.block_search_inner_folds),
                seed=int(config["seed"]) + int(repeat) * 100 + int(round(float(proportion) * 100)),
            )
            if not search_summary.empty:
                effective_group_size = int(search_summary.iloc[0]["group_size"])

        fold1_config = _build_sample_override_config(
            config,
            fold_id=1,
            subset_indices=np.asarray(subset_indices, dtype=np.int64),
            gate_prior_train_source="inner_oof",
        )
        fold1_dual_dir = subset_root / "fold_1" / "dual_prior"
        dual_row = run_dual_prior_fixed_block_on_fold(
            base_config=fold1_config,
            fold_id=1,
            group_size=int(effective_group_size),
            output_dir=fold1_dual_dir,
        )
        gate_summary_path = fold1_dual_dir / "group_shared_gate_group_summary.json"
        prior_cache_root = fold1_dual_dir / "prior_cache" / "_prior_only_reuse"
        cached_fold1 = precompute_dual_prior_cache(
            fold1_config,
            fold_id=1,
            cache_root=prior_cache_root,
            inner_folds=3,
        )
        alpha = _estimate_prior_only_alpha_from_cache(cached_fold1)
        _save_json(
            subset_root / "selection.json",
            {
                "repeat": int(repeat),
                "proportion": float(proportion),
                "selected_group_size": int(effective_group_size),
                "fold1_gate_summary_path": str(gate_summary_path),
                "prior_only_alpha": float(alpha),
                "fold1_dual_pearson": float(dual_row["pearson"]),
                "fold1_dual_r2": float(dual_row["r2"]),
            },
        )
        selection_rows.append(
            {
                "repeat": int(repeat),
                "proportion": float(proportion),
                "group_size": int(effective_group_size),
                "prior_only_alpha": float(alpha),
                "fold1_gate_summary_path": str(gate_summary_path),
            }
        )

    selection_df = pd.DataFrame(selection_rows).sort_values(["repeat", "proportion"]).reset_index(drop=True)
    selection_df.to_csv(output_root / "selection_by_repeat_proportion.csv", index=False)

    selection_lookup = {
        (int(row["repeat"]), float(row["proportion"])): row for row in selection_rows
    }

    fold_rows: list[dict] = []
    for fold_id in [int(f) for f in args.fold_ids]:
        X_train_full, y_train_full, X_test, y_test = _load_fold_data(config, fold_id=fold_id)
        train_size = X_train_full.shape[0]
        for (repeat, proportion), fold1_subset_indices in shared_subset_map.items():
            selection = selection_lookup[(int(repeat), float(proportion))]
            n_keep = len(np.asarray(fold1_subset_indices, dtype=np.int64))
            if float(proportion) >= 1.0:
                n_keep = int(train_size)
            fold_subset_map = build_nested_train_subsets(
                train_indices=np.arange(train_size, dtype=np.int64),
                proportions=[float(proportion)],
                repeats=int(repeat),
                seed=int(config["seed"]) + int(fold_id) * 1000,
            )
            subset_indices = np.asarray(fold_subset_map[(int(repeat), float(proportion))], dtype=np.int64)
            if subset_indices.shape[0] != n_keep:
                subset_indices = subset_indices[:n_keep].copy()
                subset_indices.sort()
            subset_dir = output_root / "sample_subsets" / f"fold_{fold_id}" / f"repeat_{repeat}" / f"p_{proportion:.2f}"
            subset_dir.mkdir(parents=True, exist_ok=True)
            np.save(subset_dir / "train_indices.npy", subset_indices.astype(np.int64))

            X_train, y_train = _slice_train_subset(X_train_full, y_train_full, subset_indices)
            effective_group_size = int(selection["group_size"])

            # 1. no-prior
            no_prior_cfg = deep_update(
                config,
                {
                    "output_dir": str(subset_dir / "no_prior"),
                    "main_models": [{"name": "TabICLv2-2stage", "stage1_backend": "tabicl", "stage2_backend": "tabicl"}],
                    "stage2": {
                        "tabicl": {
                            "n_estimators": int(config.get("stage1", {}).get("tabicl", {}).get("n_estimators", 1)),
                            "norm_methods": list(config.get("stage1", {}).get("tabicl", {}).get("norm_methods", ["none"])),
                            "feat_shuffle_method": str(config.get("stage1", {}).get("tabicl", {}).get("feat_shuffle_method", "none")),
                            "batch_size": config.get("stage1", {}).get("tabicl", {}).get("batch_size", 1),
                            "checkpoint_version": str(config.get("stage1", {}).get("tabicl", {}).get("checkpoint_version", "tabicl-regressor-v2-20260212.ckpt")),
                            "device": str(config.get("stage1", {}).get("tabicl", {}).get("device", "cuda")),
                        },
                    },
                    "baselines": {**config["baselines"], "gblup": False, "bayesA": False, "bayesB": False, "bayesLasso": False},
                },
            )

            sampled_snp_ids = [f"snp_{i}" for i in range(X_train.shape[1])]
            model_spec = resolve_two_stage_model_specs(no_prior_cfg)[0]
            X_train_stage2, X_test_stage2, block_summaries, num_blocks = _build_stage_features(
                model_spec=model_spec,
                X_train=X_train,
                y_train=y_train,
                sampled_snp_ids=sampled_snp_ids,
                strategy=no_prior_cfg["grouping_strategy"],
                group_size=int(effective_group_size),
                seed=int(config["seed"]) + fold_id * 1000 + repeat,
                pad_incomplete_last_block=bool(no_prior_cfg.get("pad_incomplete_last_block", True)),
                embedding_reduce_dim=None,
                include_block_scalar=bool(no_prior_cfg.get("include_block_scalar", False)),
                second_stage_adjustment_config=_resolve_second_stage_adjustment(no_prior_cfg, model_spec.name),
                collect_block_diagnostics=False,
                embedding_extraction_mode=str(no_prior_cfg.get("embedding_extraction_mode", "legacy")),
                stage2_feature_mode=_resolve_stage2_feature_mode(model_spec),
                X_eval=X_test,
            )
            stage2_cfg = _prepare_stage2_config(model_spec, block_summaries, include_block_scalar=False)
            _, no_prior_pred, _ = fit_stage2_model(
                model_spec.stage2_backend,
                X_train_stage2,
                y_train,
                X_test_stage2,
                stage2_cfg,
                int(config["seed"]) + fold_id * 100 + repeat,
            )
            no_prior_metric = regression_metrics(y_test, np.asarray(no_prior_pred, dtype=np.float32))
            fold_rows.append(
                {
                    "fold": fold_id,
                    "repeat": int(repeat),
                    "proportion": float(proportion),
                    "model": "no_prior",
                    "group_size": int(effective_group_size),
                    "n_train": int(X_train.shape[0]),
                    "test_pearson": float(no_prior_metric["pearson"]),
                    "test_r2": float(no_prior_metric["r2"]),
                }
            )

            # 2. baselines
            for row in _run_baselines_on_subset(
                config=config,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                work_dir=subset_dir / "baselines",
            ):
                fold_rows.append(
                    {
                        "fold": fold_id,
                        "repeat": int(repeat),
                        "proportion": float(proportion),
                        "group_size": int(effective_group_size),
                        "n_train": int(X_train.shape[0]),
                        **row,
                    }
                )

            # 3. dual/prior-only 复用缩样版临时配置，重新走 OOF prior/gate
            sample_config = _build_sample_override_config(
                config,
                fold_id=int(fold_id),
                subset_indices=subset_indices,
                gate_prior_train_source="inner_oof",
            )
            if int(fold_id) == 1:
                dual_row = run_dual_prior_fixed_block_on_fold(
                    base_config=sample_config,
                    fold_id=int(fold_id),
                    group_size=int(effective_group_size),
                    output_dir=subset_dir / "dual_prior",
                )
            else:
                dual_row = run_dual_prior_fixed_block_with_frozen_gate_on_fold(
                    base_config=sample_config,
                    fold_id=int(fold_id),
                    group_size=int(effective_group_size),
                    gate_summary_path=selection["fold1_gate_summary_path"],
                    output_dir=subset_dir / "dual_prior",
                )
            fold_rows.append(
                {
                    "fold": fold_id,
                    "repeat": int(repeat),
                    "proportion": float(proportion),
                    "model": "dual_prior",
                    "group_size": int(effective_group_size),
                    "n_train": int(X_train.shape[0]),
                    "test_pearson": float(dual_row["pearson"]),
                    "test_r2": float(dual_row["r2"]),
                }
            )

            prior_cache_root = subset_dir / "dual_prior" / "prior_cache"
            cached = precompute_dual_prior_cache(
                sample_config,
                fold_id=int(fold_id),
                cache_root=prior_cache_root / "_prior_only_reuse",
                inner_folds=0,
            )
            alpha = float(selection["prior_only_alpha"])
            prior_pred = alpha * np.asarray(cached["bayesb_test"], dtype=np.float32) + (1.0 - alpha) * np.asarray(cached["gblup_test"], dtype=np.float32)
            prior_metric = regression_metrics(y_test, prior_pred)
            fold_rows.append(
                {
                    "fold": fold_id,
                    "repeat": int(repeat),
                    "proportion": float(proportion),
                    "model": "prior_only",
                    "group_size": int(effective_group_size),
                    "n_train": int(X_train.shape[0]),
                    "test_pearson": float(prior_metric["pearson"]),
                    "test_r2": float(prior_metric["r2"]),
                }
            )

    fold_df = pd.DataFrame(fold_rows).sort_values(["model", "fold", "repeat", "proportion"]).reset_index(drop=True)
    fold_df.to_csv(output_root / "fold_metrics.csv", index=False)
    summary_df = summarize_sample_size_metrics(fold_df)
    summary_df.to_csv(output_root / "summary_by_model_proportion.csv", index=False)
    trait_summary = compute_trait_level_sample_size_summary(fold_df)
    _save_json(output_root / "trait_summary.json", trait_summary)
    print(fold_df)
    print(summary_df)
    print(json.dumps(trait_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
