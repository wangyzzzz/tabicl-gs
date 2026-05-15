from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tabicl_gs.config import load_experiment_config
from tabicl_gs.data.grouping import subsample_snp_indices
from tabicl_gs.data.plink import align_phenotype_to_sample_ids, load_plink_matrix, plink_num_snps, read_phenotype_table
from tabicl_gs.eval.metrics import regression_metrics
from tabicl_gs.eval.splits import make_outer_cv_splits
from tabicl_gs.models.factory import fit_stage2_model
from tabicl_gs.models.model_specs import resolve_two_stage_model_specs
from tabicl_gs.pipeline.experiment import (
    _build_stage_features,
    _prepare_stage2_config,
    _resolve_second_stage_adjustment,
    _resolve_stage2_feature_mode,
)


def build_block_search_space(group_sizes: list[int]) -> list[int]:
    return [int(value) for value in group_sizes]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_tabicl_oof_bundle(output_dir: Path, y_true: np.ndarray, y_pred: np.ndarray, metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "tabicl_inner_oof_targets.npy", np.asarray(y_true, dtype=np.float32))
    np.save(output_dir / "tabicl_inner_oof_predictions.npy", np.asarray(y_pred, dtype=np.float32))
    _write_json(output_dir / "tabicl_inner_oof_summary.json", metadata)


def _load_fold1_data(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    phenotype = read_phenotype_table(
        config["phenotype_csv"],
        sample_id_col=config.get("phenotype_sample_id_col", "sample_id"),
    )
    total_snps = plink_num_snps(config["plink_prefix"])
    selected_snp_indices = subsample_snp_indices(total_snps, config["max_snps"], config["seed"])
    plink_data = load_plink_matrix(config["plink_prefix"], snp_indices=selected_snp_indices)
    sample_id_col = config.get("phenotype_sample_id_col", "sample_id")
    phenotype, keep_indices = align_phenotype_to_sample_ids(
        phenotype,
        plink_data.sample_ids,
        sample_id_col=sample_id_col,
    )
    genotype = plink_data.matrix[np.asarray(keep_indices, dtype=np.int64)].astype(np.float32)
    target = phenotype[config["trait_col"]].to_numpy(dtype=np.float32)
    valid_mask = np.isfinite(target)
    genotype = genotype[valid_mask]
    target = target[valid_mask]
    outer_splits = make_outer_cv_splits(genotype, config["outer_cv_folds"], config["seed"])
    train_idx, _ = outer_splits[0]
    return genotype[train_idx], target[train_idx]


def _build_config(base_config: dict[str, Any], group_size: int, output_dir: str) -> dict[str, Any]:
    config = deepcopy(base_config)
    config["group_size"] = int(group_size)
    config["output_dir"] = output_dir
    config["grouping_strategy"] = "window"
    config["include_block_scalar"] = bool(config.get("include_block_scalar", False))
    config["collect_block_diagnostics"] = False
    config["save_block_summaries"] = True
    config["embedding_extraction_mode"] = str(config.get("embedding_extraction_mode", "legacy"))
    main_models = config.get("main_models")
    if not main_models:
        config["main_models"] = [{"name": "TabICLv2-2stage", "stage1_backend": "tabicl", "stage2_backend": "tabicl"}]
        main_models = config["main_models"]
    first_model = dict(main_models[0])
    config["main_models"] = [first_model]

    stage1_backend = str(first_model["stage1_backend"]).lower()
    if stage1_backend == "tabicl":
        stage1_cfg = config["stage1"]["tabicl"]
        stage1_cfg["embedding_reduce_dim"] = None
        stage1_cfg["embedding_explained_variance_target"] = 0.99
        stage1_cfg["track_full_explained_variance"] = False
    if "baselines" in config:
        for key in ["gblup", "bayesA", "bayesB", "bayesLasso"]:
            config["baselines"][key] = False
    return config


def run_inner_oof_tabicl_block_search(
    base_config: dict[str, Any],
    output_root: str | Path,
    group_sizes: list[int],
    inner_folds: int = 3,
) -> pd.DataFrame:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    X_outer_train, y_outer_train = _load_fold1_data(base_config)
    inner_splits = make_outer_cv_splits(X_outer_train, inner_folds, base_config["seed"] + 1)
    rows: list[dict[str, Any]] = []

    for group_size in build_block_search_space(group_sizes):
        oof_pred = np.zeros_like(np.asarray(y_outer_train, dtype=np.float32), dtype=np.float32)
        feature_dims: list[int] = []
        reduced_dims: list[float] = []
        combo_dir = output_root / f"block_{int(group_size)}"
        config = _build_config(base_config, group_size=int(group_size), output_dir=str(combo_dir))
        model_spec = resolve_two_stage_model_specs(config)[0]
        stage2_feature_mode = _resolve_stage2_feature_mode(model_spec)

        for inner_id, (inner_train_idx, inner_valid_idx) in enumerate(inner_splits, start=1):
            inner_dir = combo_dir / f"inner_{inner_id}"
            inner_dir.mkdir(parents=True, exist_ok=True)
            X_inner_train = X_outer_train[inner_train_idx]
            y_inner_train = y_outer_train[inner_train_idx]
            X_inner_valid = X_outer_train[inner_valid_idx]

            X_train_stage2, X_valid_stage2, block_summaries, _ = _build_stage_features(
                model_spec=model_spec,
                X_train=X_inner_train,
                y_train=y_inner_train,
                sampled_snp_ids=[f"snp_{i}" for i in range(X_inner_train.shape[1])],
                strategy=config["grouping_strategy"],
                group_size=int(group_size),
                seed=config["seed"] + 1000 + int(inner_id),
                pad_incomplete_last_block=bool(config.get("pad_incomplete_last_block", True)),
                embedding_reduce_dim=None,
                include_block_scalar=bool(config.get("include_block_scalar", False)),
                second_stage_adjustment_config=_resolve_second_stage_adjustment(config, model_spec.name),
                collect_block_diagnostics=False,
                embedding_extraction_mode=str(config.get("embedding_extraction_mode", "legacy")),
                stage2_feature_mode=stage2_feature_mode,
                X_eval=X_inner_valid,
            )
            stage2_config = _prepare_stage2_config(
                model_spec,
                block_summaries,
                include_block_scalar=bool(config.get("include_block_scalar", False)),
            )
            _, valid_pred, _ = fit_stage2_model(
                model_spec.stage2_backend,
                X_train_stage2,
                y_inner_train,
                X_valid_stage2,
                stage2_config,
                config["seed"] + 10000 + int(inner_id),
            )
            oof_pred[inner_valid_idx] = np.asarray(valid_pred, dtype=np.float32)
            feature_dims.append(int(X_train_stage2.shape[1]))
            reduced_dims.append(float(np.mean([b["reduced_embedding_dim"] for b in block_summaries])))

        metric = regression_metrics(y_outer_train, oof_pred)
        rows.append(
            {
                "group_size": int(group_size),
                "variance_target_pct": 99,
                "inner_oof_pearson": float(metric["pearson"]),
                "inner_oof_r2": float(metric["r2"]),
                "feature_dim_min": int(min(feature_dims)) if feature_dims else 0,
                "feature_dim_max": int(max(feature_dims)) if feature_dims else 0,
                "mean_reduced_block_embedding_dim": float(np.mean(reduced_dims)) if reduced_dims else np.nan,
                "output_dir": str(combo_dir),
            }
        )
        pd.DataFrame(rows).sort_values("inner_oof_pearson", ascending=False).to_csv(
            output_root / "fold1_inner_oof_block_search.csv",
            index=False,
        )

    summary = pd.DataFrame(rows).sort_values("inner_oof_pearson", ascending=False)
    summary.to_csv(output_root / "fold1_inner_oof_block_search.csv", index=False)
    if not summary.empty:
        best_row = summary.iloc[0].to_dict()
        _write_json(
            output_root / "best_block.json",
            {
                "group_size": int(best_row["group_size"]),
                "best_value": float(best_row["inner_oof_pearson"]),
                "inner_oof_r2": float(best_row["inner_oof_r2"]),
            },
        )
        _evaluate_single_group_size(
            base_config=base_config,
            group_size=int(best_row["group_size"]),
            inner_folds=int(inner_folds),
            output_dir=output_root / "best_block_oof",
            save_oof_bundle=True,
        )
    return summary


def _compute_tabicl_inner_oof(
    base_config: dict[str, Any],
    group_size: int,
    inner_folds: int,
    output_root: Path,
    save_inner_fold_predictions: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    X_outer_train, y_outer_train = _load_fold1_data(base_config)
    inner_splits = make_outer_cv_splits(X_outer_train, inner_folds, base_config["seed"] + 1)
    oof_pred = np.zeros_like(np.asarray(y_outer_train, dtype=np.float32), dtype=np.float32)
    feature_dims: list[int] = []
    reduced_dims: list[float] = []
    config = _build_config(base_config, group_size=int(group_size), output_dir=str(output_root))
    model_spec = resolve_two_stage_model_specs(config)[0]
    stage2_feature_mode = _resolve_stage2_feature_mode(model_spec)

    for inner_id, (inner_train_idx, inner_valid_idx) in enumerate(inner_splits, start=1):
        inner_dir = output_root / f"inner_{inner_id}"
        inner_dir.mkdir(parents=True, exist_ok=True)
        X_inner_train = X_outer_train[inner_train_idx]
        y_inner_train = y_outer_train[inner_train_idx]
        X_inner_valid = X_outer_train[inner_valid_idx]

        X_train_stage2, X_valid_stage2, block_summaries, _ = _build_stage_features(
            model_spec=model_spec,
            X_train=X_inner_train,
            y_train=y_inner_train,
            sampled_snp_ids=[f"snp_{i}" for i in range(X_inner_train.shape[1])],
            strategy=config["grouping_strategy"],
            group_size=int(group_size),
            seed=config["seed"] + 1000 + int(inner_id),
            pad_incomplete_last_block=bool(config.get("pad_incomplete_last_block", True)),
            embedding_reduce_dim=None,
            include_block_scalar=bool(config.get("include_block_scalar", False)),
            second_stage_adjustment_config=_resolve_second_stage_adjustment(config, model_spec.name),
            collect_block_diagnostics=False,
            embedding_extraction_mode=str(config.get("embedding_extraction_mode", "legacy")),
            stage2_feature_mode=stage2_feature_mode,
            X_eval=X_inner_valid,
        )
        stage2_config = _prepare_stage2_config(
            model_spec,
            block_summaries,
            include_block_scalar=bool(config.get("include_block_scalar", False)),
        )
        _, valid_pred, _ = fit_stage2_model(
            model_spec.stage2_backend,
            X_train_stage2,
            y_inner_train,
            X_valid_stage2,
            stage2_config,
            config["seed"] + 10000 + int(inner_id),
        )
        valid_pred = np.asarray(valid_pred, dtype=np.float32)
        oof_pred[inner_valid_idx] = valid_pred
        if save_inner_fold_predictions:
            np.save(inner_dir / "tabicl_valid_predictions.npy", valid_pred)
            np.save(inner_dir / "valid_idx.npy", np.asarray(inner_valid_idx, dtype=np.int64))
        feature_dims.append(int(X_train_stage2.shape[1]))
        reduced_dims.append(float(np.mean([b["reduced_embedding_dim"] for b in block_summaries])))

    metric = regression_metrics(y_outer_train, oof_pred)
    summary = {
        "group_size": int(group_size),
        "inner_folds": int(inner_folds),
        "variance_target_pct": 99,
        "inner_oof_pearson": float(metric["pearson"]),
        "inner_oof_r2": float(metric["r2"]),
        "feature_dim_min": int(min(feature_dims)) if feature_dims else 0,
        "feature_dim_max": int(max(feature_dims)) if feature_dims else 0,
        "mean_reduced_block_embedding_dim": float(np.mean(reduced_dims)) if reduced_dims else np.nan,
        "output_dir": str(output_root),
    }
    return np.asarray(y_outer_train, dtype=np.float32), oof_pred.astype(np.float32), summary


def _evaluate_single_group_size(
    base_config: dict[str, Any],
    group_size: int,
    inner_folds: int,
    output_dir: Path,
    save_oof_bundle: bool = False,
    write_best_block: bool = True,
) -> dict[str, Any]:
    y_true, oof_pred, row = _compute_tabicl_inner_oof(
        base_config=base_config,
        group_size=int(group_size),
        inner_folds=int(inner_folds),
        output_root=output_dir / f"block_{int(group_size)}",
        save_inner_fold_predictions=bool(save_oof_bundle),
    )
    pd.DataFrame([row]).to_csv(output_dir / "fold1_inner_oof_block_search.csv", index=False)
    if write_best_block:
        _write_json(
            output_dir / "best_block.json",
            {
                "group_size": int(row["group_size"]),
                "best_value": float(row["inner_oof_pearson"]),
                "inner_oof_r2": float(row["inner_oof_r2"]),
            },
        )
    if save_oof_bundle:
        _save_tabicl_oof_bundle(
            output_dir,
            y_true=y_true,
            y_pred=oof_pred,
            metadata={
                "source": "fixed_block_inner_oof",
                "group_size": int(row["group_size"]),
                "inner_oof_pearson": float(row["inner_oof_pearson"]),
                "inner_oof_r2": float(row["inner_oof_r2"]),
                "oof_prediction_path": str(output_dir / "tabicl_inner_oof_predictions.npy"),
                "oof_target_path": str(output_dir / "tabicl_inner_oof_targets.npy"),
            },
        )
    return {
        "group_size": int(row["group_size"]),
        "variance_target_pct": int(row["variance_target_pct"]),
        "inner_oof_pearson": float(row["inner_oof_pearson"]),
        "inner_oof_r2": float(row["inner_oof_r2"]),
        "feature_dim_min": int(row["feature_dim_min"]),
        "feature_dim_max": int(row["feature_dim_max"]),
        "mean_reduced_block_embedding_dim": float(row["mean_reduced_block_embedding_dim"]),
        "output_dir": str(output_dir / f"block_{int(group_size)}"),
    }


def run_fold1_tabicl2stage_block_search(
    base_config: dict[str, Any],
    output_root: str | Path,
    min_block: int,
    max_block: int,
    n_trials: int,
    seed: int,
    inner_folds: int = 3,
) -> pd.DataFrame:
    import optuna

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        group_size = int(trial.suggest_int("group_size", int(min_block), int(max_block)))
        row = _evaluate_single_group_size(
            base_config=base_config,
            group_size=int(group_size),
            inner_folds=int(inner_folds),
            output_dir=output_root / "trials" / f"trial_{int(trial.number):03d}_block_{int(group_size)}",
        )
        row["trial"] = int(trial.number)
        rows.append(row)
        pd.DataFrame(rows).sort_values("inner_oof_pearson", ascending=False).to_csv(
            output_root / "fold1_block_search_tabicl.csv",
            index=False,
        )
        return float(row["inner_oof_pearson"])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    summary = pd.DataFrame(rows).sort_values("inner_oof_pearson", ascending=False)
    summary.to_csv(output_root / "fold1_block_search_tabicl.csv", index=False)
    if not summary.empty:
        best_row = summary.iloc[0].to_dict()
        _write_json(
            output_root / "best_block.json",
            {
                "group_size": int(best_row["group_size"]),
                "best_value": float(best_row["inner_oof_pearson"]),
                "inner_oof_r2": float(best_row["inner_oof_r2"]),
                "trial": int(best_row["trial"]),
            },
        )
        best_group_size = int(best_row["group_size"])
        best_output_dir = output_root / "best_block_oof"
        _evaluate_single_group_size(
            base_config=base_config,
            group_size=best_group_size,
            inner_folds=int(inner_folds),
            output_dir=best_output_dir,
            save_oof_bundle=True,
        )
    return summary


def load_base_config(path: str) -> dict[str, Any]:
    return load_experiment_config(path)
