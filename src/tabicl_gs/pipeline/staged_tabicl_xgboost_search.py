from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from tabicl_gs.config import load_experiment_config
from tabicl_gs.data.grouping import subsample_snp_indices
from tabicl_gs.data.plink import (
    align_phenotype_to_sample_ids,
    impute_by_train_mean,
    load_plink_matrix,
    plink_num_snps,
    read_phenotype_table,
)
from tabicl_gs.eval.metrics import regression_metrics
from tabicl_gs.eval.splits import make_outer_cv_splits
from tabicl_gs.models.group_shared_gate import GroupSharedGateRegressor
from tabicl_gs.models.model_specs import resolve_two_stage_model_specs
from tabicl_gs.models.xgboost_model import build_xgboost_regressor
from tabicl_gs.pipeline.block_prior import aggregate_beta_to_block_prior
from tabicl_gs.pipeline.dual_prior_fold_search import (
    _build_group_features_from_stage2,
    precompute_dual_prior_cache,
)
from tabicl_gs.pipeline.experiment import (
    _build_stage_features,
    _resolve_second_stage_adjustment,
    _resolve_stage2_feature_mode,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_outer_fold_data(config: dict[str, Any], fold_id: int) -> tuple[np.ndarray, np.ndarray]:
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
    train_idx, _ = outer_splits[int(fold_id) - 1]
    return genotype[train_idx], target[train_idx]


def _load_outer_fold_train_test_data(config: dict[str, Any], fold_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    train_idx, test_idx = outer_splits[int(fold_id) - 1]
    return genotype[train_idx], target[train_idx], genotype[test_idx], target[test_idx]


def _base_stage1_config(base_config: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(base_config)
    config["grouping_strategy"] = str(config.get("grouping_strategy", "window"))
    config["include_block_scalar"] = bool(config.get("include_block_scalar", False))
    config["collect_block_diagnostics"] = False
    config["save_block_summaries"] = True
    config["embedding_extraction_mode"] = str(config.get("embedding_extraction_mode", "legacy"))
    return config


def _load_cached_feature_folds(cache_root: Path) -> dict[str, Any]:
    meta = json.loads((cache_root / "metadata.json").read_text(encoding="utf-8"))
    folds = []
    for inner in meta["inner_folds"]:
        inner_dir = cache_root / f"inner_{int(inner['inner_id'])}"
        folds.append(
            {
                "inner_id": int(inner["inner_id"]),
                "train_features": np.load(inner_dir / "train_features.npy").astype(np.float32),
                "valid_features": np.load(inner_dir / "valid_features.npy").astype(np.float32),
                "y_train": np.load(inner_dir / "y_train.npy").astype(np.float32),
                "valid_idx": np.load(inner_dir / "valid_idx.npy").astype(np.int64),
            }
        )
    return {
        "y_outer_train": np.load(cache_root / "y_outer_train.npy").astype(np.float32),
        "oof_features": (
            np.load(cache_root / "oof_features.npy").astype(np.float32)
            if (cache_root / "oof_features.npy").exists()
            else None
        ),
        "metadata": meta,
        "folds": folds,
    }


def _load_block_summaries(cache_root: Path, inner_id: int) -> list[dict[str, Any]]:
    summary_path = cache_root / f"inner_{int(inner_id)}" / "block_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return list(payload["block_summaries"])


def _reconstruct_prior_oof(prior_cache: dict[str, Any], y_outer_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bayesb_oof = np.zeros_like(np.asarray(y_outer_train, dtype=np.float32), dtype=np.float32)
    gblup_oof = np.zeros_like(np.asarray(y_outer_train, dtype=np.float32), dtype=np.float32)
    inner_cache = prior_cache.get("inner_cache", [])
    if not inner_cache:
        raise ValueError("prior_cache.inner_cache is required to reconstruct strict OOF priors.")
    for inner_meta in inner_cache:
        valid_idx = np.asarray(inner_meta["inner_valid_idx"], dtype=np.int64)
        bayesb_oof[valid_idx] = np.asarray(inner_meta["bayesb_valid"], dtype=np.float32)
        gblup_oof[valid_idx] = np.asarray(inner_meta["gblup_valid"], dtype=np.float32)
    return bayesb_oof.astype(np.float32), gblup_oof.astype(np.float32)


def build_pure_tabicl_oof_feature_cache(
    base_config: dict[str, Any],
    fold_id: int,
    group_size: int,
    cache_root: str | Path,
    inner_folds: int = 3,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    cache_root = Path(cache_root)
    if not force_rebuild and (cache_root / "metadata.json").exists():
        return _load_cached_feature_folds(cache_root)

    cache_root.mkdir(parents=True, exist_ok=True)
    config = _base_stage1_config(base_config)
    model_spec = resolve_two_stage_model_specs(config)[0]
    stage2_feature_mode = _resolve_stage2_feature_mode(model_spec)
    X_outer_train, y_outer_train = _load_outer_fold_data(config, fold_id=fold_id)
    sampled_snp_ids = [f"snp_{i}" for i in range(X_outer_train.shape[1])]
    inner_splits = make_outer_cv_splits(X_outer_train, inner_folds, config["seed"] + int(fold_id))
    metadata_folds: list[dict[str, Any]] = []

    for inner_id, (inner_train_idx, inner_valid_idx) in enumerate(inner_splits, start=1):
        inner_dir = cache_root / f"inner_{inner_id}"
        inner_dir.mkdir(parents=True, exist_ok=True)
        X_inner_train = X_outer_train[inner_train_idx]
        y_inner_train = y_outer_train[inner_train_idx]
        X_inner_valid = X_outer_train[inner_valid_idx]

        X_train_stage2, X_valid_stage2, block_summaries, num_blocks = _build_stage_features(
            model_spec=model_spec,
            X_train=X_inner_train,
            y_train=y_inner_train,
            sampled_snp_ids=sampled_snp_ids,
            strategy=config["grouping_strategy"],
            group_size=int(group_size),
            seed=config["seed"] + int(fold_id) * 1000 + int(inner_id),
            pad_incomplete_last_block=bool(config.get("pad_incomplete_last_block", True)),
            embedding_reduce_dim=None,
            include_block_scalar=bool(config.get("include_block_scalar", False)),
            second_stage_adjustment_config=_resolve_second_stage_adjustment(config, model_spec.name),
            collect_block_diagnostics=False,
            embedding_extraction_mode=str(config.get("embedding_extraction_mode", "legacy")),
            stage2_feature_mode=stage2_feature_mode,
            X_eval=X_inner_valid,
        )

        np.save(inner_dir / "train_features.npy", np.asarray(X_train_stage2, dtype=np.float32))
        np.save(inner_dir / "valid_features.npy", np.asarray(X_valid_stage2, dtype=np.float32))
        np.save(inner_dir / "y_train.npy", np.asarray(y_inner_train, dtype=np.float32))
        np.save(inner_dir / "valid_idx.npy", np.asarray(inner_valid_idx, dtype=np.int64))
        _write_json(
            inner_dir / "block_summary.json",
            {
                "num_blocks": int(num_blocks),
                "stage2_input_dim": int(X_train_stage2.shape[1]),
                "mean_reduced_block_embedding_dim": float(
                    np.mean([summary["reduced_embedding_dim"] for summary in block_summaries])
                ),
                "block_summaries": block_summaries,
            },
        )
        metadata_folds.append(
            {
                "inner_id": int(inner_id),
                "inner_train_size": int(len(inner_train_idx)),
                "inner_valid_size": int(len(inner_valid_idx)),
                "num_blocks": int(num_blocks),
                "stage2_input_dim": int(X_train_stage2.shape[1]),
                "mean_reduced_block_embedding_dim": float(
                    np.mean([summary["reduced_embedding_dim"] for summary in block_summaries])
                ),
            }
        )

    if not metadata_folds:
        raise RuntimeError("Failed to build OOF stage2 features.")

    np.save(cache_root / "y_outer_train.npy", np.asarray(y_outer_train, dtype=np.float32))
    _write_json(
        cache_root / "metadata.json",
        {
            "fold_id": int(fold_id),
            "group_size": int(group_size),
            "inner_folds": metadata_folds,
            "oof_feature_dim_min": int(min(meta["stage2_input_dim"] for meta in metadata_folds)),
            "oof_feature_dim_max": int(max(meta["stage2_input_dim"] for meta in metadata_folds)),
            "num_samples": int(X_outer_train.shape[0]),
        },
    )
    return _load_cached_feature_folds(cache_root)


def fit_ridge_oof_from_cached_folds(
    feature_folds: list[dict[str, Any]],
    y_outer_train: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    oof_pred = np.zeros_like(np.asarray(y_outer_train, dtype=np.float32), dtype=np.float32)
    for fold in feature_folds:
        model = Ridge(alpha=float(alpha), random_state=None)
        model.fit(fold["train_features"], fold["y_train"])
        pred = np.asarray(model.predict(fold["valid_features"]), dtype=np.float32)
        oof_pred[fold["valid_idx"]] = pred
    return oof_pred.astype(np.float32)


def fit_xgboost_oof_from_cached_folds(
    feature_folds: list[dict[str, Any]],
    y_outer_train: np.ndarray,
    xgb_params: dict[str, Any],
) -> np.ndarray:
    oof_pred = np.zeros_like(np.asarray(y_outer_train, dtype=np.float32), dtype=np.float32)
    for fold in feature_folds:
        model = build_xgboost_regressor(**xgb_params)
        model.fit(fold["train_features"], fold["y_train"])
        pred = np.asarray(model.predict(fold["valid_features"]), dtype=np.float32)
        oof_pred[fold["valid_idx"]] = pred
    return oof_pred.astype(np.float32)


def run_fold1_block_search_with_ridge(
    base_config: dict[str, Any],
    output_root: str | Path,
    min_block: int,
    max_block: int,
    n_trials: int,
    seed: int,
    inner_folds: int = 3,
    ridge_alpha: float = 1.0,
) -> pd.DataFrame:
    import optuna

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    X_outer_train, y_outer_train = _load_outer_fold_data(base_config, fold_id=1)
    config = _base_stage1_config(base_config)
    model_spec = resolve_two_stage_model_specs(config)[0]
    stage2_feature_mode = _resolve_stage2_feature_mode(model_spec)
    sampled_snp_ids = [f"snp_{i}" for i in range(X_outer_train.shape[1])]
    inner_splits = make_outer_cv_splits(X_outer_train, inner_folds, config["seed"] + 1)
    rows: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        group_size = int(trial.suggest_int("group_size", int(min_block), int(max_block)))
        oof_pred = np.zeros_like(y_outer_train, dtype=np.float32)
        feature_dims = []
        for inner_id, (inner_train_idx, inner_valid_idx) in enumerate(inner_splits, start=1):
            X_inner_train = X_outer_train[inner_train_idx]
            y_inner_train = y_outer_train[inner_train_idx]
            X_inner_valid = X_outer_train[inner_valid_idx]
            X_train_stage2, X_valid_stage2, _, _ = _build_stage_features(
                model_spec=model_spec,
                X_train=X_inner_train,
                y_train=y_inner_train,
                sampled_snp_ids=sampled_snp_ids,
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
            model = Ridge(alpha=float(ridge_alpha))
            model.fit(X_train_stage2, y_inner_train)
            oof_pred[inner_valid_idx] = np.asarray(model.predict(X_valid_stage2), dtype=np.float32)
            feature_dims.append(int(X_train_stage2.shape[1]))
        metric = regression_metrics(y_outer_train, oof_pred)
        row = {
            "trial": int(trial.number),
            "group_size": int(group_size),
            "ridge_alpha": float(ridge_alpha),
            "inner_oof_pearson": float(metric["pearson"]),
            "inner_oof_r2": float(metric["r2"]),
            "feature_dim_min": int(min(feature_dims)) if feature_dims else 0,
            "feature_dim_max": int(max(feature_dims)) if feature_dims else 0,
        }
        rows.append(row)
        pd.DataFrame(rows).sort_values("inner_oof_pearson", ascending=False).to_csv(
            output_root / "fold1_block_search_ridge.csv", index=False
        )
        return float(row["inner_oof_pearson"])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    summary = pd.DataFrame(rows).sort_values("inner_oof_pearson", ascending=False)
    summary.to_csv(output_root / "fold1_block_search_ridge.csv", index=False)
    _write_json(
        output_root / "best_block.json",
        {
            "group_size": int(study.best_params["group_size"]),
            "best_value": float(study.best_value),
            "ridge_alpha": float(ridge_alpha),
        },
    )
    return summary


def run_fold1_xgboost_search_on_cached_block(
    cache_root: str | Path,
    output_root: str | Path,
    n_trials: int,
    seed: int,
) -> pd.DataFrame:
    import optuna

    cache_root = Path(cache_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cache = _load_cached_feature_folds(cache_root)
    y_outer_train = cache["y_outer_train"]
    rows: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": int(trial.suggest_categorical("n_estimators", [200, 400, 800])),
            "max_depth": int(trial.suggest_categorical("max_depth", [2, 3, 4])),
            "learning_rate": float(trial.suggest_categorical("learning_rate", [0.03, 0.05])),
            "min_child_weight": int(trial.suggest_categorical("min_child_weight", [1, 5, 10])),
            "subsample": float(trial.suggest_categorical("subsample", [0.8, 0.9])),
            "colsample_bytree": float(trial.suggest_categorical("colsample_bytree", [0.8, 0.9])),
            "reg_lambda": float(trial.suggest_categorical("reg_lambda", [5.0, 10.0])),
            "reg_alpha": float(trial.suggest_categorical("reg_alpha", [0.0, 0.5])),
            "tree_method": "hist",
            "objective": "reg:squarederror",
            "random_state": int(seed + trial.number),
            "n_jobs": 4,
            "device": "cpu",
        }
        oof_pred = fit_xgboost_oof_from_cached_folds(cache["folds"], y_outer_train, params)
        metric = regression_metrics(y_outer_train, oof_pred)
        trial_dir = output_root / "trials" / f"trial_{trial.number:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        np.save(trial_dir / "oof_predictions.npy", oof_pred.astype(np.float32))
        row = {
            "trial": int(trial.number),
            "inner_oof_pearson": float(metric["pearson"]),
            "inner_oof_r2": float(metric["r2"]),
            **params,
        }
        rows.append(row)
        pd.DataFrame(rows).sort_values("inner_oof_pearson", ascending=False).to_csv(
            output_root / "fold1_xgboost_search.csv", index=False
        )
        return float(row["inner_oof_pearson"])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    summary = pd.DataFrame(rows).sort_values("inner_oof_pearson", ascending=False)
    summary.to_csv(output_root / "fold1_xgboost_search.csv", index=False)
    best_trial = next(row for row in rows if int(row["trial"]) == int(study.best_trial.number))
    _write_json(output_root / "best_xgboost.json", best_trial)
    return summary


def fit_group_gate_from_best_xgboost_oof(
    base_config: dict[str, Any],
    fold_id: int,
    feature_cache_root: str | Path,
    xgboost_search_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    feature_cache_root = Path(feature_cache_root)
    xgboost_search_root = Path(xgboost_search_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    feature_cache = _load_cached_feature_folds(feature_cache_root)
    y_outer_train = np.asarray(feature_cache["y_outer_train"], dtype=np.float32)
    best_xgb = json.loads((xgboost_search_root / "best_xgboost.json").read_text(encoding="utf-8"))
    trial_id = int(best_xgb["trial"])
    xgb_oof = np.load(xgboost_search_root / "trials" / f"trial_{trial_id:03d}" / "oof_predictions.npy").astype(np.float32)
    if xgb_oof.shape[0] != y_outer_train.shape[0]:
        raise ValueError(
            f"Best XGBoost OOF length {xgb_oof.shape[0]} does not match outer-train length {y_outer_train.shape[0]}."
        )

    prior_cache = precompute_dual_prior_cache(
        base_config=base_config,
        fold_id=int(fold_id),
        cache_root=output_root / "prior_cache",
        inner_folds=len(feature_cache["folds"]),
    )
    bayesb_oof, gblup_oof = _reconstruct_prior_oof(prior_cache, y_outer_train)
    model_spec = resolve_two_stage_model_specs(base_config)[0]
    stage2_cfg = dict(model_spec.stage2_config)
    group_mode = str(stage2_cfg.get("group_mode", "prior_guided_group"))
    num_groups = int(stage2_cfg.get("num_groups", 3))
    include_block_scalar = bool(base_config.get("include_block_scalar", False))

    representative_block_summaries = _load_block_summaries(feature_cache_root, inner_id=1)
    prior_scores = aggregate_beta_to_block_prior(
        beta=prior_cache["bayesb_beta"],
        block_summaries=representative_block_summaries,
        method="l2",
    ).astype(np.float32)

    oof_group_features = None
    representative_block_group_ids = None
    for fold in feature_cache["folds"]:
        block_summaries = _load_block_summaries(feature_cache_root, inner_id=int(fold["inner_id"]))
        valid_group_features, block_group_ids = _build_group_features_from_stage2(
            fold["valid_features"],
            block_summaries,
            group_mode=group_mode,
            num_groups=num_groups,
            prior_scores=prior_scores,
            include_block_scalar=include_block_scalar,
        )
        if oof_group_features is None:
            oof_group_features = np.zeros((y_outer_train.shape[0], valid_group_features.shape[1]), dtype=np.float32)
        oof_group_features[fold["valid_idx"]] = valid_group_features.astype(np.float32)
        if representative_block_group_ids is None and block_group_ids is not None:
            representative_block_group_ids = np.asarray(block_group_ids, dtype=np.int64)

    if oof_group_features is None:
        raise RuntimeError("Failed to build OOF group features from cached block embeddings.")

    gate_model = GroupSharedGateRegressor(
        block_input_dims=[int(oof_group_features.shape[1])],
        prior_scores=prior_scores.tolist(),
        num_groups=num_groups,
        group_mode=group_mode,
        assignment_mode=str(stage2_cfg.get("assignment_mode", "nearest_centroid")),
        temperature=float(stage2_cfg.get("temperature", 1.0)),
        hidden_dim=int(stage2_cfg.get("hidden_dim", 16)),
        lr=float(stage2_cfg.get("lr", 1e-3)),
        weight_decay=float(stage2_cfg.get("weight_decay", 1e-4)),
        max_epochs=int(stage2_cfg.get("max_epochs", 200)),
        device=str(stage2_cfg.get("device", "cpu")),
        random_state=int(base_config["seed"]) + int(fold_id) * 100,
    )
    if representative_block_group_ids is not None:
        gate_model.block_prior_group_ids_ = representative_block_group_ids

    gate_model.fit_from_group_features(
        oof_group_features,
        y_outer_train,
        xgb_oof,
        bayesb_oof,
        gblup_oof,
    )
    gate_oof_pred = gate_model.predict_from_group_features(
        oof_group_features,
        xgb_oof,
        bayesb_oof,
        gblup_oof,
    )
    metric = regression_metrics(y_outer_train, gate_oof_pred)
    np.save(output_root / "xgboost_best_oof.npy", xgb_oof.astype(np.float32))
    np.save(output_root / "bayesb_oof.npy", bayesb_oof.astype(np.float32))
    np.save(output_root / "gblup_oof.npy", gblup_oof.astype(np.float32))
    np.save(output_root / "gate_oof_predictions.npy", gate_oof_pred.astype(np.float32))
    summary = {
        "fold_id": int(fold_id),
        "group_size": int(feature_cache["metadata"]["group_size"]),
        "xgboost_best_trial": int(trial_id),
        "xgboost_best_params": best_xgb,
        "group_mode": group_mode,
        "num_groups": int(num_groups),
        "inner_oof_pearson": float(metric["pearson"]),
        "inner_oof_r2": float(metric["r2"]),
        "group_summary": gate_model.get_group_summary(),
        "feature_cache_root": str(feature_cache_root),
        "xgboost_search_root": str(xgboost_search_root),
    }
    _write_json(output_root / "group_gate_summary.json", summary)
    pd.DataFrame(
        [
            {
                "fold": int(fold_id),
                "group_size": int(feature_cache["metadata"]["group_size"]),
                "xgboost_best_trial": int(trial_id),
                "inner_oof_pearson": float(metric["pearson"]),
                "inner_oof_r2": float(metric["r2"]),
                "group_counts": str(summary["group_summary"].get("group_counts", [])),
            }
        ]
    ).to_csv(output_root / "group_gate_metrics.csv", index=False)
    return summary


def run_staged_tabicl_xgboost_fixed_from_fold1_on_fold(
    base_config: dict[str, Any],
    fold_id: int,
    fold1_artifact_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    fold1_artifact_root = Path(fold1_artifact_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_block_path = fold1_artifact_root / "fold1_block_search_ridge" / "best_block.json"
    if not best_block_path.exists():
        best_block_path = fold1_artifact_root / "fold1_tabicl_block_search" / "best_block.json"
    if not best_block_path.exists():
        raise FileNotFoundError(f"Could not find best_block.json under {fold1_artifact_root}")
    best_block = json.loads(best_block_path.read_text(encoding="utf-8"))
    best_xgb = json.loads((fold1_artifact_root / "fold1_xgboost_search" / "best_xgboost.json").read_text(encoding="utf-8"))

    config = _base_stage1_config(base_config)
    model_spec = resolve_two_stage_model_specs(config)[0]
    stage2_feature_mode = _resolve_stage2_feature_mode(model_spec)
    group_size = int(best_block["group_size"])
    X_train, y_train, X_test, y_test = _load_outer_fold_train_test_data(config, fold_id=int(fold_id))
    X_train, X_test = impute_by_train_mean(X_train, X_test)
    sampled_snp_ids = [f"snp_{i}" for i in range(X_train.shape[1])]

    X_train_stage2, X_test_stage2, block_summaries, num_blocks = _build_stage_features(
        model_spec=model_spec,
        X_train=X_train,
        y_train=y_train,
        sampled_snp_ids=sampled_snp_ids,
        strategy=config["grouping_strategy"],
        group_size=group_size,
        seed=config["seed"] + int(fold_id) * 1000,
        pad_incomplete_last_block=bool(config.get("pad_incomplete_last_block", True)),
        embedding_reduce_dim=None,
        include_block_scalar=bool(config.get("include_block_scalar", False)),
        second_stage_adjustment_config=_resolve_second_stage_adjustment(config, model_spec.name),
        collect_block_diagnostics=False,
        embedding_extraction_mode=str(config.get("embedding_extraction_mode", "legacy")),
        stage2_feature_mode=stage2_feature_mode,
        X_eval=X_test,
    )

    xgb_params = {
        key: best_xgb[key]
        for key in [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "reg_lambda",
            "reg_alpha",
            "tree_method",
            "objective",
            "random_state",
            "n_jobs",
            "device",
        ]
        if key in best_xgb
    }
    xgb_model = build_xgboost_regressor(**xgb_params)
    xgb_model.fit(X_train_stage2, y_train)
    xgb_test_pred = np.asarray(xgb_model.predict(X_test_stage2), dtype=np.float32)
    metric = regression_metrics(y_test, xgb_test_pred)

    np.save(output_dir / "xgboost_test_predictions.npy", xgb_test_pred.astype(np.float32))
    np.save(output_dir / "predictions_test.npy", xgb_test_pred.astype(np.float32))
    np.save(output_dir / "targets_test.npy", y_test.astype(np.float32))
    row = {
        "fold": int(fold_id),
        "group_size": int(group_size),
        "xgboost_trial": int(best_xgb.get("trial", -1)),
        "xgb_pearson": float(metric["pearson"]),
        "xgb_r2": float(metric["r2"]),
        "pearson": float(metric["pearson"]),
        "r2": float(metric["r2"]),
        "num_blocks": int(num_blocks),
        "mean_reduced_block_embedding_dim": float(
            np.mean([summary["reduced_embedding_dim"] for summary in block_summaries])
        ),
        "stage2_input_dim": int(X_train_stage2.shape[1]),
        "output_dir": str(output_dir),
    }
    pd.DataFrame([row]).to_csv(output_dir / "fold_metrics.csv", index=False)
    return row


def load_base_config(path: str) -> dict[str, Any]:
    return load_experiment_config(path)
