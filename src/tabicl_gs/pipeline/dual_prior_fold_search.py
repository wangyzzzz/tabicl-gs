from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tabicl_gs.config import load_experiment_config
from tabicl_gs.data.grouping import subsample_snp_indices
from tabicl_gs.data.plink import align_phenotype_to_sample_ids, impute_by_train_mean, load_plink_matrix, plink_num_snps, read_phenotype_table
from tabicl_gs.eval.metrics import regression_metrics
from tabicl_gs.eval.splits import make_outer_cv_splits
from tabicl_gs.models.baselines import run_r_baseline
from tabicl_gs.models.factory import _fit_expert_regressor, fit_stage2_model
from tabicl_gs.models.group_shared_gate import GroupSharedGateRegressor, _build_group_membership_from_prior
from tabicl_gs.models.model_specs import resolve_two_stage_model_specs
from tabicl_gs.models.vi_sparse_prior import VISparsePriorRegressor
from tabicl_gs.pipeline.block_prior import aggregate_beta_to_block_prior
from tabicl_gs.pipeline.dual_prior_utils import (
    reconstruct_train_predictions_from_inner_cache as _reconstruct_train_predictions_from_inner_cache,
    resolve_gate_prior_train_predictions as _resolve_gate_prior_train_predictions,
)
from tabicl_gs.pipeline.experiment import (
    _append_stage2_prior_feature,
    _build_stage_features,
    _compute_residual_target_predictions,
    _prepare_stage2_config,
    _resolve_second_stage_adjustment,
    _resolve_stage2_feature_mode,
    _save_tabicl_inner_oof_bundle,
)


def build_block_search_bounds(min_block: int, max_block: int) -> tuple[int, int]:
    return int(min_block), int(max_block)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_tabicl_oof_bundle(
    output_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    _save_tabicl_inner_oof_bundle(output_dir, y_true=y_true, y_pred=y_pred, metadata=metadata)


def save_prior_cache(cache_dir: Path, payload: dict[str, np.ndarray]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for key, value in payload.items():
        np.save(cache_dir / f"{key}.npy", np.asarray(value, dtype=np.float32))


def load_prior_cache(cache_dir: Path) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    for path in cache_dir.glob("*.npy"):
        payload[path.stem] = np.load(path).astype(np.float32)
    return payload


def _load_fold_data(config: dict[str, Any], fold_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    X_train = genotype[train_idx]
    y_train = target[train_idx]
    sample_override = config.get("_sample_size_override")
    if isinstance(sample_override, dict):
        override_fold_id = sample_override.get("fold_id")
        subset_indices = sample_override.get("train_subset_indices")
        if override_fold_id is not None and int(override_fold_id) == int(fold_id):
            if subset_indices is None:
                raise ValueError("_sample_size_override requires train_subset_indices when fold_id matches.")
            subset_indices = np.asarray(subset_indices, dtype=np.int64)
            X_train = X_train[subset_indices]
            y_train = y_train[subset_indices]
    return X_train, y_train, genotype[test_idx], target[test_idx]


def _build_config(base_config: dict[str, Any], group_size: int, output_dir: str) -> dict[str, Any]:
    config = deepcopy(base_config)
    config["group_size"] = int(group_size)
    config["output_dir"] = output_dir
    stage1_cfg = config["stage1"]["tabicl"]
    stage1_cfg["embedding_reduce_dim"] = None
    stage1_cfg["embedding_explained_variance_target"] = 0.99
    stage1_cfg["track_full_explained_variance"] = False
    return config


def _compute_bayesb_predictions_with_beta(
    *,
    base_config: dict[str, Any],
    X_train_base: np.ndarray,
    y_train: np.ndarray,
    X_eval_base: np.ndarray,
    output_dir: Path,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_result = run_r_baseline(
        model_name="BayesB",
        X_train=X_train_base,
        y_train=y_train,
        X_test=X_train_base,
        output_dir=output_dir / "train_fit",
        rscript_path=base_config["baselines"]["rscript_path"],
        seed=seed,
        sommer_method=base_config["baselines"].get("sommer_method"),
        keep_artifacts=True,
        return_beta=True,
    )
    eval_result = run_r_baseline(
        model_name="BayesB",
        X_train=X_train_base,
        y_train=y_train,
        X_test=X_eval_base,
        output_dir=output_dir / "eval_fit",
        rscript_path=base_config["baselines"]["rscript_path"],
        seed=seed,
        sommer_method=base_config["baselines"].get("sommer_method"),
        keep_artifacts=True,
        return_beta=False,
    )
    if train_result.beta is None:
        raise ValueError("BayesB prior initialization requires beta coefficients, but beta was not returned.")
    return (
        np.asarray(train_result.predictions, dtype=np.float32),
        np.asarray(eval_result.predictions, dtype=np.float32),
        np.asarray(train_result.beta, dtype=np.float32),
    )


def _resolve_prior_backend(base_config: dict[str, Any]) -> str:
    specs = resolve_two_stage_model_specs(base_config)
    if not specs:
        return "bayesb"
    stage2_cfg = specs[0].stage2_config
    return str(stage2_cfg.get("prior_backend", "bayesb")).lower()


def _build_vi_prior_regressor(base_config: dict[str, Any], seed: int) -> VISparsePriorRegressor:
    specs = resolve_two_stage_model_specs(base_config)
    stage2_cfg = specs[0].stage2_config if specs else {}
    vi_cfg = dict(stage2_cfg.get("vi_prior_config", {}))
    vi_cfg.setdefault("random_state", int(seed))
    return VISparsePriorRegressor(**vi_cfg)


def _compute_vi_predictions_with_beta(
    *,
    base_config: dict[str, Any],
    X_train_base: np.ndarray,
    y_train: np.ndarray,
    X_eval_base: np.ndarray,
    output_dir: Path,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = _build_vi_prior_regressor(base_config, seed=seed)
    model.fit(X_train_base, y_train)
    train_pred = np.asarray(model.predict(X_train_base), dtype=np.float32)
    eval_pred = np.asarray(model.predict(X_eval_base), dtype=np.float32)
    np.save(output_dir / "train_pred.npy", train_pred)
    np.save(output_dir / "eval_pred.npy", eval_pred)
    np.save(output_dir / "coef_mean.npy", np.asarray(model.coef_, dtype=np.float32))
    np.save(output_dir / "coef_var.npy", np.asarray(model.coef_var_, dtype=np.float32))
    _write_json(
        output_dir / "metadata.json",
        {
            "model": "VI",
            "n_train": int(X_train_base.shape[0]),
            "n_test": int(X_eval_base.shape[0]),
            "random_state": int(seed),
            "loss": float(getattr(model, "loss_", float("nan"))),
        },
    )
    return train_pred, eval_pred, np.asarray(model.coef_, dtype=np.float32), np.asarray(model.coef_var_, dtype=np.float32)


def _compute_baseline_train_eval_predictions(
    *,
    model_name: str,
    base_config: dict[str, Any],
    X_train_base: np.ndarray,
    y_train: np.ndarray,
    X_eval_base: np.ndarray,
    output_dir: Path,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_result = run_r_baseline(
        model_name=model_name,
        X_train=X_train_base,
        y_train=y_train,
        X_test=X_train_base,
        output_dir=output_dir / "train_fit",
        rscript_path=base_config["baselines"]["rscript_path"],
        seed=seed,
        sommer_method=base_config["baselines"].get("sommer_method"),
        keep_artifacts=True,
        return_beta=False,
    )
    eval_result = run_r_baseline(
        model_name=model_name,
        X_train=X_train_base,
        y_train=y_train,
        X_test=X_eval_base,
        output_dir=output_dir / "eval_fit",
        rscript_path=base_config["baselines"]["rscript_path"],
        seed=seed,
        sommer_method=base_config["baselines"].get("sommer_method"),
        keep_artifacts=True,
        return_beta=False,
    )
    return (
        np.asarray(train_result.predictions, dtype=np.float32),
        np.asarray(eval_result.predictions, dtype=np.float32),
    )


def _use_bayes_family_selector(base_config: dict[str, Any]) -> bool:
    if _use_vi_prior(base_config):
        return False
    specs = resolve_two_stage_model_specs(base_config)
    if not specs:
        return False
    stage2_cfg = specs[0].stage2_config
    return bool(stage2_cfg.get("use_bayes_family_selector", False))


def _use_vi_prior(base_config: dict[str, Any]) -> bool:
    return _resolve_prior_backend(base_config) == "vi"


def _use_rkhs_prior(base_config: dict[str, Any]) -> bool:
    specs = resolve_two_stage_model_specs(base_config)
    if not specs:
        return False
    stage2_cfg = specs[0].stage2_config
    return bool(stage2_cfg.get("use_rkhs_prior", False))


def _maybe_attach_block_prior_scores(
    stage2_config: dict[str, Any],
    stage2_backend: str,
    block_summaries: list[dict[str, Any]],
    bayesb_beta: np.ndarray | None,
) -> dict[str, Any]:
    config = dict(stage2_config)
    if stage2_backend.lower() not in {"static_block_weight", "group_weight_pooling", "group_shared_gate"}:
        return config
    if bayesb_beta is None:
        raise ValueError(f"{stage2_backend} requires cached BayesB beta coefficients for prior initialization.")
    config["prior_scores"] = aggregate_beta_to_block_prior(
        beta=bayesb_beta,
        block_summaries=block_summaries,
        method="l2",
    ).tolist()
    return config


def _get_block_feature_slices(
    block_summaries: list[dict[str, Any]],
    include_block_scalar: bool = False,
) -> list[tuple[int, int]]:
    slices: list[tuple[int, int]] = []
    start = 0
    for summary in block_summaries:
        width = int(summary["reduced_embedding_dim"]) + (1 if include_block_scalar else 0)
        end = start + width
        slices.append((start, end))
        start = end
    return slices


def _summarize_stage2_blocks(
    X_stage2_core: np.ndarray,
    block_summaries: list[dict[str, Any]],
    include_block_scalar: bool = False,
) -> np.ndarray:
    X_stage2_core = np.asarray(X_stage2_core, dtype=np.float32)
    outputs = []
    for start, end in _get_block_feature_slices(block_summaries, include_block_scalar=include_block_scalar):
        block_chunk = X_stage2_core[:, start:end]
        if block_chunk.shape[1] == 0:
            outputs.append(np.zeros((X_stage2_core.shape[0], 1), dtype=np.float32))
            continue
        outputs.append(np.sqrt(np.mean(np.square(block_chunk), axis=1, keepdims=True)).astype(np.float32))
    if not outputs:
        return np.zeros((X_stage2_core.shape[0], 0), dtype=np.float32)
    return np.concatenate(outputs, axis=1).astype(np.float32)


def _build_group_features_from_stage2(
    X_stage2_core: np.ndarray,
    block_summaries: list[dict[str, Any]],
    group_mode: str,
    num_groups: int,
    prior_scores: np.ndarray | None,
    include_block_scalar: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    block_summary_matrix = _summarize_stage2_blocks(
        X_stage2_core,
        block_summaries,
        include_block_scalar=include_block_scalar,
    )
    if group_mode == "embedding_group":
        return block_summary_matrix.astype(np.float32), None
    if group_mode != "prior_guided_group":
        raise ValueError(f"Unsupported group_mode: {group_mode}")
    if prior_scores is None:
        raise ValueError("prior_guided_group requires prior_scores.")
    block_group_ids = _build_group_membership_from_prior(np.asarray(prior_scores, dtype=np.float32), int(num_groups))
    grouped = []
    for group_id in range(int(num_groups)):
        mask = block_group_ids == group_id
        if not np.any(mask):
            grouped.append(np.zeros((block_summary_matrix.shape[0], 1), dtype=np.float32))
        else:
            grouped.append(block_summary_matrix[:, mask].mean(axis=1, keepdims=True).astype(np.float32))
    return np.concatenate(grouped, axis=1).astype(np.float32), block_group_ids.astype(np.int64)


def _resolve_group_shared_gate_expert(model_spec) -> tuple[str, dict[str, Any]]:
    stage2_cfg = dict(model_spec.stage2_config)
    return str(stage2_cfg.get("expert_backend", "tabicl")).lower(), dict(stage2_cfg.get("expert_config", {}))


def _init_group_shared_gate_model(
    stage2_config: dict[str, Any],
    feature_dim: int,
    prior_scores: np.ndarray,
    block_prior_group_ids: np.ndarray | None,
    seed: int,
) -> GroupSharedGateRegressor:
    model = GroupSharedGateRegressor(
        block_input_dims=[int(feature_dim)],
        prior_scores=np.asarray(prior_scores, dtype=np.float32).tolist(),
        num_groups=int(stage2_config.get("num_groups", 3)),
        group_mode=str(stage2_config.get("group_mode", "embedding_group")),
        assignment_mode=str(stage2_config.get("assignment_mode", "nearest_centroid")),
        temperature=float(stage2_config.get("temperature", 1.0)),
        hidden_dim=int(stage2_config.get("hidden_dim", 16)),
        lr=float(stage2_config.get("lr", 1e-3)),
        weight_decay=float(stage2_config.get("weight_decay", 1e-4)),
        max_epochs=int(stage2_config.get("max_epochs", 200)),
        device=str(stage2_config.get("device", "cpu")),
        random_state=int(seed),
    )
    if block_prior_group_ids is not None:
        model.block_prior_group_ids_ = np.asarray(block_prior_group_ids, dtype=np.int64)
    return model


def _compute_inner_fold_prior_cache(
    base_config: dict[str, Any],
    X_outer_train: np.ndarray,
    y_outer_train: np.ndarray,
    inner_train_idx: np.ndarray,
    inner_valid_idx: np.ndarray,
    cache_root: Path,
    seed_offset: int,
    inner_folds: int = 3,
) -> dict[str, np.ndarray]:
    X_inner_train = X_outer_train[inner_train_idx]
    y_inner_train = y_outer_train[inner_train_idx]
    X_inner_valid = X_outer_train[inner_valid_idx]
    X_inner_train_base, X_inner_valid_base = impute_by_train_mean(X_inner_train, X_inner_valid)

    if _use_vi_prior(base_config):
        bayesb_train, bayesb_valid, bayesb_beta, vi_var = _compute_vi_predictions_with_beta(
            base_config=base_config,
            X_train_base=X_inner_train_base,
            y_train=y_inner_train,
            X_eval_base=X_inner_valid_base,
            output_dir=cache_root / "vi_prior",
            seed=int(base_config["seed"]) + int(seed_offset),
        )
    else:
        bayesb_train, bayesb_valid, bayesb_beta = _compute_bayesb_predictions_with_beta(
            base_config=base_config,
            X_train_base=X_inner_train_base,
            y_train=y_inner_train,
            X_eval_base=X_inner_valid_base,
            output_dir=cache_root / "bayesb",
            seed=int(base_config["seed"]) + int(seed_offset),
        )
        vi_var = None
    gblup_train, gblup_valid, _ = _compute_residual_target_predictions(
        residual_target_config={"baseline_model": "GBLUP"},
        fold_dir=cache_root / "gblup",
        X_train_base=X_inner_train_base,
        y_train=y_inner_train,
        X_test_base=X_inner_valid_base,
        config=base_config,
        fold_id=seed_offset,
    )
    payload = {
        "bayesb_train": bayesb_train,
        "gblup_train": gblup_train,
        "bayesb_valid": bayesb_valid,
        "gblup_valid": gblup_valid,
        "bayesb_beta": bayesb_beta,
    }
    if _use_rkhs_prior(base_config):
        rkhs_train, rkhs_valid = _compute_baseline_train_eval_predictions(
            model_name="RKHS",
            base_config=base_config,
            X_train_base=X_inner_train_base,
            y_train=y_inner_train,
            X_eval_base=X_inner_valid_base,
            output_dir=cache_root / "rkhs",
            seed=int(base_config["seed"]) + int(seed_offset) + 7000,
        )
        payload["rkhs_train"] = rkhs_train
        payload["rkhs_valid"] = rkhs_valid
    if vi_var is not None:
        payload["vi_coef_var"] = vi_var
    if _use_bayes_family_selector(base_config):
        bayeslasso_train, bayeslasso_valid = _compute_baseline_train_eval_predictions(
            model_name="BayesLasso",
            base_config=base_config,
            X_train_base=X_inner_train_base,
            y_train=y_inner_train,
            X_eval_base=X_inner_valid_base,
            output_dir=cache_root / "bayeslasso",
            seed=int(base_config["seed"]) + int(seed_offset) + 5000,
        )
        payload["bayeslasso_train"] = bayeslasso_train
        payload["bayeslasso_valid"] = bayeslasso_valid

    save_prior_cache(
        cache_root,
        payload,
    )
    return load_prior_cache(cache_root)


def precompute_dual_prior_cache(
    base_config: dict[str, Any],
    fold_id: int,
    cache_root: str | Path,
    inner_folds: int = 3,
) -> dict[str, Any]:
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    X_outer_train, y_outer_train, X_outer_test, _ = _load_fold_data(base_config, fold_id=fold_id)
    X_outer_train_base, X_outer_test_base = impute_by_train_mean(X_outer_train, X_outer_test)
    if _use_vi_prior(base_config):
        bayesb_train, bayesb_test, bayesb_beta, vi_var = _compute_vi_predictions_with_beta(
            base_config=base_config,
            X_train_base=X_outer_train_base,
            y_train=y_outer_train,
            X_eval_base=X_outer_test_base,
            output_dir=cache_root / "vi_outer",
            seed=int(base_config["seed"]) + int(fold_id),
        )
    else:
        bayesb_train, bayesb_test, bayesb_beta = _compute_bayesb_predictions_with_beta(
            base_config=base_config,
            X_train_base=X_outer_train_base,
            y_train=y_outer_train,
            X_eval_base=X_outer_test_base,
            output_dir=cache_root / "bayesb_outer",
            seed=int(base_config["seed"]) + int(fold_id),
        )
        vi_var = None
    gblup_train, gblup_test, _ = _compute_residual_target_predictions(
        residual_target_config={"baseline_model": "GBLUP"},
        fold_dir=cache_root / "gblup_outer",
        X_train_base=X_outer_train_base,
        y_train=y_outer_train,
        X_test_base=X_outer_test_base,
        config=base_config,
        fold_id=fold_id,
    )
    outer_payload: dict[str, np.ndarray] = {
        "bayesb_train": np.asarray(bayesb_train, dtype=np.float32),
        "gblup_train": np.asarray(gblup_train, dtype=np.float32),
        "bayesb_test": np.asarray(bayesb_test, dtype=np.float32),
        "gblup_test": np.asarray(gblup_test, dtype=np.float32),
        "bayesb_beta": np.asarray(bayesb_beta, dtype=np.float32),
    }
    if _use_rkhs_prior(base_config):
        rkhs_train, rkhs_test = _compute_baseline_train_eval_predictions(
            model_name="RKHS",
            base_config=base_config,
            X_train_base=X_outer_train_base,
            y_train=y_outer_train,
            X_eval_base=X_outer_test_base,
            output_dir=cache_root / "rkhs_outer",
            seed=int(base_config["seed"]) + int(fold_id) + 7000,
        )
        outer_payload["rkhs_train"] = np.asarray(rkhs_train, dtype=np.float32)
        outer_payload["rkhs_test"] = np.asarray(rkhs_test, dtype=np.float32)
    if vi_var is not None:
        outer_payload["vi_coef_var"] = np.asarray(vi_var, dtype=np.float32)
    if _use_bayes_family_selector(base_config):
        bayeslasso_train, bayeslasso_test = _compute_baseline_train_eval_predictions(
            model_name="BayesLasso",
            base_config=base_config,
            X_train_base=X_outer_train_base,
            y_train=y_outer_train,
            X_eval_base=X_outer_test_base,
            output_dir=cache_root / "bayeslasso_outer",
            seed=int(base_config["seed"]) + int(fold_id) + 5000,
        )
        outer_payload["bayeslasso_train"] = np.asarray(bayeslasso_train, dtype=np.float32)
        outer_payload["bayeslasso_test"] = np.asarray(bayeslasso_test, dtype=np.float32)
    inner_cache = []
    if int(inner_folds) > 1:
        inner_splits = make_outer_cv_splits(X_outer_train, inner_folds, base_config["seed"] + fold_id)
        for inner_id, (inner_train_idx, inner_valid_idx) in enumerate(inner_splits, start=1):
            cache = _compute_inner_fold_prior_cache(
                base_config=base_config,
                X_outer_train=X_outer_train,
                y_outer_train=y_outer_train,
                inner_train_idx=inner_train_idx,
                inner_valid_idx=inner_valid_idx,
                cache_root=cache_root / f"inner_{inner_id}",
                seed_offset=fold_id * 100 + inner_id,
                inner_folds=inner_folds,
            )
            inner_cache.append(
                {
                    "inner_id": inner_id,
                    "inner_train_idx": inner_train_idx,
                    "inner_valid_idx": inner_valid_idx,
                    **cache,
                }
            )

    return {
        "X_outer_train": X_outer_train,
        "y_outer_train": y_outer_train,
        "X_outer_test": X_outer_test,
        "y_outer_test": _load_fold_data(base_config, fold_id=fold_id)[3],
        "inner_cache": inner_cache,
        **outer_payload,
    }


def _collect_group_shared_gate_oof_payload(
    *,
    base_config: dict[str, Any],
    cached: dict[str, Any],
    group_size: int,
    output_dir: Path,
    prior_train_source: str = "full_train",
) -> dict[str, Any]:
    X_outer_train = cached["X_outer_train"]
    y_outer_train = cached["y_outer_train"]
    sampled_snp_ids = [f"snp_{i}" for i in range(X_outer_train.shape[1])]
    model_spec = resolve_two_stage_model_specs(base_config)[0]
    config = _build_config(base_config, group_size=group_size, output_dir=str(output_dir))
    stage2_config = dict(model_spec.stage2_config)
    group_mode = str(stage2_config.get("group_mode", "embedding_group"))
    num_groups = int(stage2_config.get("num_groups", 3))
    expert_backend, expert_config = _resolve_group_shared_gate_expert(model_spec)
    include_block_scalar = bool(config.get("include_block_scalar", False))

    oof_tabicl = np.zeros_like(y_outer_train, dtype=np.float32)
    oof_group_features = None
    representative_prior_scores = None
    representative_block_group_ids = None
    reduced_dims = []
    stage2_dims = []
    num_blocks = None

    for inner_meta in cached["inner_cache"]:
        X_inner_train = X_outer_train[inner_meta["inner_train_idx"]]
        y_inner_train = y_outer_train[inner_meta["inner_train_idx"]]
        X_inner_valid = X_outer_train[inner_meta["inner_valid_idx"]]
        X_train_stage2, X_valid_stage2, block_summaries, inner_num_blocks = _build_stage_features(
            model_spec=model_spec,
            X_train=X_inner_train,
            y_train=y_inner_train,
            sampled_snp_ids=sampled_snp_ids,
            strategy=config["grouping_strategy"],
            group_size=group_size,
            seed=config["seed"] + 1000 + int(inner_meta["inner_id"]),
            pad_incomplete_last_block=config.get("pad_incomplete_last_block", True),
            embedding_reduce_dim=None,
            include_block_scalar=include_block_scalar,
            second_stage_adjustment_config=_resolve_second_stage_adjustment(config, model_spec.name),
            collect_block_diagnostics=bool(config.get("collect_block_diagnostics", False)),
            embedding_extraction_mode=str(config.get("embedding_extraction_mode", "legacy")),
            stage2_feature_mode=_resolve_stage2_feature_mode(model_spec),
            X_eval=X_inner_valid,
        )

        if representative_prior_scores is None:
            representative_prior_scores = aggregate_beta_to_block_prior(
                beta=cached["bayesb_beta"],
                block_summaries=block_summaries,
                method="l2",
            ).astype(np.float32)

        valid_group_features, block_group_ids = _build_group_features_from_stage2(
            X_valid_stage2,
            block_summaries,
            group_mode=group_mode,
            num_groups=num_groups,
            prior_scores=representative_prior_scores,
            include_block_scalar=include_block_scalar,
        )
        if representative_block_group_ids is None and block_group_ids is not None:
            representative_block_group_ids = np.asarray(block_group_ids, dtype=np.int64)
        if oof_group_features is None:
            oof_group_features = np.zeros((X_outer_train.shape[0], valid_group_features.shape[1]), dtype=np.float32)
        oof_group_features[inner_meta["inner_valid_idx"]] = valid_group_features.astype(np.float32)

        _, valid_tabicl_pred, _ = _fit_expert_regressor(
            expert_backend,
            expert_config,
            X_train_stage2,
            y_inner_train,
            X_valid_stage2,
            seed=config["seed"] + 10000 + int(inner_meta["inner_id"]),
        )
        oof_tabicl[inner_meta["inner_valid_idx"]] = np.asarray(valid_tabicl_pred, dtype=np.float32)

        reduced_dims.append(float(np.mean([b["reduced_embedding_dim"] for b in block_summaries])))
        stage2_dims.append(float(X_train_stage2.shape[1] + 2))
        num_blocks = inner_num_blocks

    if oof_group_features is None or representative_prior_scores is None:
        raise RuntimeError("Failed to construct global OOF group payload for group_shared_gate.")

    gate_model = _init_group_shared_gate_model(
        stage2_config=stage2_config,
        feature_dim=int(oof_group_features.shape[1]),
        prior_scores=representative_prior_scores,
        block_prior_group_ids=representative_block_group_ids,
        seed=int(config["seed"]) + 20000 + int(group_size),
    )
    gate_bayesb_train, gate_gblup_train, gate_bayes_candidates = _resolve_gate_prior_train_predictions(
        cached,
        prior_train_source=prior_train_source,
    )
    gate_model.fit_from_group_features(
        oof_group_features,
        y_outer_train,
        oof_tabicl,
        gate_bayesb_train,
        gate_gblup_train,
        y_bayes_candidates=gate_bayes_candidates,
    )
    oof_pred = gate_model.predict_from_group_features(
        oof_group_features,
        oof_tabicl,
        gate_bayesb_train,
        gate_gblup_train,
        y_bayes_candidates=gate_bayes_candidates,
    )
    return {
        "oof_pred": oof_pred.astype(np.float32),
        "oof_tabicl": oof_tabicl.astype(np.float32),
        "y_true": y_outer_train.astype(np.float32),
        "gate_model": gate_model,
        "group_summary": gate_model.get_group_summary(),
        "prior_scores": representative_prior_scores.astype(np.float32),
        "block_prior_group_ids": None if representative_block_group_ids is None else representative_block_group_ids.astype(np.int64),
        "mean_reduced_block_embedding_dim": float(np.mean(reduced_dims)) if reduced_dims else np.nan,
        "stage2_input_dim": float(np.mean(stage2_dims)) if stage2_dims else np.nan,
        "num_blocks": int(num_blocks or 0),
    }


def _evaluate_block_with_cached_priors(
    base_config: dict[str, Any],
    cached: dict[str, Any],
    group_size: int,
    output_dir: Path,
    save_tabicl_oof_bundle: bool = True,
) -> dict[str, Any]:
    X_outer_train = cached["X_outer_train"]
    y_outer_train = cached["y_outer_train"]
    sampled_snp_ids = [f"snp_{i}" for i in range(X_outer_train.shape[1])]
    model_spec = resolve_two_stage_model_specs(base_config)[0]
    config = _build_config(base_config, group_size=group_size, output_dir=str(output_dir))
    if model_spec.stage2_backend.lower() == "group_shared_gate":
        output_dir.mkdir(parents=True, exist_ok=True)
        prior_train_source = str(base_config.get("gate_prior_train_source", "inner_oof"))
        payload = _collect_group_shared_gate_oof_payload(
            base_config=base_config,
            cached=cached,
            group_size=group_size,
            output_dir=output_dir,
            prior_train_source=prior_train_source,
        )
        metric = regression_metrics(y_outer_train, payload["oof_pred"])
        row = {
            "group_size": int(group_size),
            "variance_target_pct": 99,
            "inner_oof_pearson": float(metric["pearson"]),
            "inner_oof_r2": float(metric["r2"]),
            "num_blocks": int(payload["num_blocks"]),
            "mean_reduced_block_embedding_dim": float(payload["mean_reduced_block_embedding_dim"]),
            "stage2_input_dim": float(payload["stage2_input_dim"]),
            "output_dir": str(output_dir),
            "group_counts": str(payload["group_summary"].get("group_counts", [])),
        }
        pd.DataFrame([row]).to_csv(output_dir / "group_shared_gate_oof_summary.csv", index=False)
        _write_json(output_dir / "group_shared_gate_group_summary.json", payload["group_summary"])
        if save_tabicl_oof_bundle:
            tabicl_metric = regression_metrics(payload["y_true"], payload["oof_tabicl"])
            _save_tabicl_oof_bundle(
                output_dir,
                y_true=payload["y_true"],
                y_pred=payload["oof_tabicl"],
                metadata={
                    "source": "dual_or_triple_fixed_block_inner_oof",
                    "group_size": int(group_size),
                    "inner_folds": int(len(cached.get("inner_cache", []))),
                    "tabicl_inner_oof_pearson": float(tabicl_metric["pearson"]),
                    "tabicl_inner_oof_r2": float(tabicl_metric["r2"]),
                    "stage2_backend": str(model_spec.stage2_backend),
                    "output_dir": str(output_dir),
                },
            )
        return row

    inner_oof_pred = np.zeros_like(y_outer_train, dtype=np.float32)
    reduced_dims = []
    stage2_dims = []
    num_blocks = None
    for inner_meta in cached["inner_cache"]:
        X_inner_train = X_outer_train[inner_meta["inner_train_idx"]]
        y_inner_train = y_outer_train[inner_meta["inner_train_idx"]]
        X_inner_valid = X_outer_train[inner_meta["inner_valid_idx"]]
        X_train_stage2, X_valid_stage2, block_summaries, inner_num_blocks = _build_stage_features(
            model_spec=model_spec,
            X_train=X_inner_train,
            y_train=y_inner_train,
            sampled_snp_ids=sampled_snp_ids,
            strategy=config["grouping_strategy"],
            group_size=group_size,
            seed=config["seed"] + 1000 + int(inner_meta["inner_id"]),
            pad_incomplete_last_block=config.get("pad_incomplete_last_block", True),
            embedding_reduce_dim=None,
            include_block_scalar=bool(config.get("include_block_scalar", False)),
            second_stage_adjustment_config=_resolve_second_stage_adjustment(config, model_spec.name),
            collect_block_diagnostics=bool(config.get("collect_block_diagnostics", False)),
            embedding_extraction_mode=str(config.get("embedding_extraction_mode", "legacy")),
            stage2_feature_mode=_resolve_stage2_feature_mode(model_spec),
            X_eval=X_inner_valid,
        )
        prior_train = np.column_stack([inner_meta["bayesb_train"], inner_meta["gblup_train"]]).astype(np.float32)
        prior_valid = np.column_stack([inner_meta["bayesb_valid"], inner_meta["gblup_valid"]]).astype(np.float32)
        X_train_stage2, X_valid_stage2 = _append_stage2_prior_feature(
            model_spec=model_spec,
            X_train_stage2=X_train_stage2,
            X_test_stage2=X_valid_stage2,
            prior_train=prior_train,
            prior_test=prior_valid,
        )
        stage2_config = _prepare_stage2_config(model_spec, block_summaries, include_block_scalar=False)
        stage2_config = _maybe_attach_block_prior_scores(
            stage2_config=stage2_config,
            stage2_backend=model_spec.stage2_backend,
            block_summaries=block_summaries,
            bayesb_beta=inner_meta.get("bayesb_beta"),
        )
        _, valid_pred, _ = fit_stage2_model(
            model_spec.stage2_backend,
            X_train_stage2,
            y_inner_train,
            X_valid_stage2,
            stage2_config,
            config["seed"] + 10000 + int(inner_meta["inner_id"]),
        )
        inner_oof_pred[inner_meta["inner_valid_idx"]] = valid_pred.astype(np.float32)
        reduced_dims.append(float(np.mean([b["reduced_embedding_dim"] for b in block_summaries])))
        stage2_dims.append(int(X_train_stage2.shape[1]))
        num_blocks = inner_num_blocks

    metric = regression_metrics(y_outer_train, inner_oof_pred)
    return {
        "group_size": int(group_size),
        "variance_target_pct": 99,
        "inner_oof_pearson": float(metric["pearson"]),
        "inner_oof_r2": float(metric["r2"]),
        "num_blocks": int(num_blocks or 0),
        "mean_reduced_block_embedding_dim": float(np.mean(reduced_dims)) if reduced_dims else np.nan,
        "stage2_input_dim": float(np.mean(stage2_dims)) if stage2_dims else np.nan,
        "output_dir": str(output_dir),
    }


def run_fold1_dual_prior_block_search(
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
    cached = precompute_dual_prior_cache(base_config, fold_id=1, cache_root=output_root / "prior_cache", inner_folds=inner_folds)
    rows: list[dict[str, Any]] = []
    lower, upper = build_block_search_bounds(min_block, max_block)

    def objective(trial: optuna.Trial) -> float:
        group_size = int(trial.suggest_int("group_size", lower, upper))
        combo_dir = output_root / "trials" / f"trial_{trial.number:03d}_block_{group_size}"
        row = _evaluate_block_with_cached_priors(
            base_config=base_config,
            cached=cached,
            group_size=group_size,
            output_dir=combo_dir,
            save_tabicl_oof_bundle=False,
        )
        row["trial"] = int(trial.number)
        rows.append(row)
        pd.DataFrame(rows).sort_values("inner_oof_pearson", ascending=False).to_csv(
            output_root / "fold1_dual_prior_block_search.csv", index=False
        )
        return float(row["inner_oof_pearson"])

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)

    summary = pd.DataFrame(rows).sort_values("inner_oof_pearson", ascending=False)
    summary.to_csv(output_root / "fold1_dual_prior_block_search.csv", index=False)
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
        payload = _collect_group_shared_gate_oof_payload(
            base_config=base_config,
            cached=cached,
            group_size=best_group_size,
            output_dir=best_output_dir,
            prior_train_source=str(base_config.get("gate_prior_train_source", "inner_oof")),
        )
        metric = regression_metrics(payload["y_true"], payload["oof_tabicl"])
        _save_tabicl_oof_bundle(
            best_output_dir,
            y_true=payload["y_true"],
            y_pred=payload["oof_tabicl"],
            metadata={
                "source": "best_block_oof",
                "group_size": int(best_group_size),
                "inner_folds": int(len(cached.get("inner_cache", []))),
                "tabicl_inner_oof_pearson": float(metric["pearson"]),
                "tabicl_inner_oof_r2": float(metric["r2"]),
                "stage2_backend": str(resolve_two_stage_model_specs(base_config)[0].stage2_backend),
                "output_dir": str(best_output_dir),
            },
        )
    return summary


def run_dual_prior_fixed_block_on_fold(
    base_config: dict[str, Any],
    fold_id: int,
    group_size: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cached = precompute_dual_prior_cache(base_config, fold_id=fold_id, cache_root=output_dir / "prior_cache", inner_folds=3)
    X_train = cached["X_outer_train"]
    y_train = cached["y_outer_train"]
    X_test = cached["X_outer_test"]
    y_test = cached["y_outer_test"]
    config = _build_config(base_config, group_size=group_size, output_dir=str(output_dir))
    sampled_snp_ids = [f"snp_{i}" for i in range(X_train.shape[1])]
    model_spec = resolve_two_stage_model_specs(base_config)[0]
    if model_spec.stage2_backend.lower() == "group_shared_gate":
        prior_train_source = str(base_config.get("gate_prior_train_source", "inner_oof"))
        payload = _collect_group_shared_gate_oof_payload(
            base_config=base_config,
            cached=cached,
            group_size=group_size,
            output_dir=output_dir / "global_oof_group",
            prior_train_source=prior_train_source,
        )
        stage2_config = dict(model_spec.stage2_config)
        group_mode = str(stage2_config.get("group_mode", "embedding_group"))
        num_groups = int(stage2_config.get("num_groups", 3))
        include_block_scalar = bool(config.get("include_block_scalar", False))
        expert_backend, expert_config = _resolve_group_shared_gate_expert(model_spec)

        X_train_stage2, X_test_stage2, block_summaries, num_blocks = _build_stage_features(
            model_spec=model_spec,
            X_train=X_train,
            y_train=y_train,
            sampled_snp_ids=sampled_snp_ids,
            strategy=config["grouping_strategy"],
            group_size=group_size,
            seed=config["seed"] + fold_id * 1000,
            pad_incomplete_last_block=config.get("pad_incomplete_last_block", True),
            embedding_reduce_dim=None,
            include_block_scalar=include_block_scalar,
            second_stage_adjustment_config=_resolve_second_stage_adjustment(config, model_spec.name),
            collect_block_diagnostics=bool(config.get("collect_block_diagnostics", False)),
            embedding_extraction_mode=str(config.get("embedding_extraction_mode", "legacy")),
            stage2_feature_mode=_resolve_stage2_feature_mode(model_spec),
            X_eval=X_test,
        )
        _, y_tabicl_test, _ = _fit_expert_regressor(
            expert_backend,
            expert_config,
            X_train_stage2,
            y_train,
            X_test_stage2,
            seed=config["seed"] + fold_id,
        )
        test_group_features, _ = _build_group_features_from_stage2(
            X_test_stage2,
            block_summaries,
            group_mode=group_mode,
            num_groups=num_groups,
            prior_scores=payload["prior_scores"],
            include_block_scalar=include_block_scalar,
        )
        y_test_pred = payload["gate_model"].predict_from_group_features(
            test_group_features,
            np.asarray(y_tabicl_test, dtype=np.float32),
            cached["bayesb_test"],
            cached["gblup_test"],
            y_bayes_candidates=(
                {
                    key: value
                    for key, value in {
                        "BayesLasso": cached.get("bayeslasso_test"),
                        "RKHS": cached.get("rkhs_test"),
                    }.items()
                    if value is not None
                }
                or None
            ),
        )
        metric = regression_metrics(y_test, y_test_pred)
        out = {
            "fold": int(fold_id),
            "group_size": int(group_size),
            "variance_target_pct": 99,
            "pearson": float(metric["pearson"]),
            "r2": float(metric["r2"]),
            "num_blocks": int(num_blocks),
            "mean_reduced_block_embedding_dim": float(np.mean([b["reduced_embedding_dim"] for b in block_summaries])),
            "stage2_input_dim": int(X_train_stage2.shape[1] + 2),
            "output_dir": str(output_dir),
            "group_counts": str(payload["group_summary"].get("group_counts", [])),
        }
        tabicl_metric = regression_metrics(payload["y_true"], payload["oof_tabicl"])
        _save_tabicl_oof_bundle(
            output_dir,
            y_true=payload["y_true"],
            y_pred=payload["oof_tabicl"],
            metadata={
                "source": "dual_or_triple_fixed_block_inner_oof",
                "fold": int(fold_id),
                "group_size": int(group_size),
                "inner_folds": int(len(cached.get("inner_cache", []))),
                "tabicl_inner_oof_pearson": float(tabicl_metric["pearson"]),
                "tabicl_inner_oof_r2": float(tabicl_metric["r2"]),
                "stage2_backend": str(model_spec.stage2_backend),
                "output_dir": str(output_dir),
            },
        )
        _write_json(output_dir / "group_shared_gate_group_summary.json", payload["group_summary"])
        pd.DataFrame([out]).to_csv(output_dir / "fold_metrics.csv", index=False)
        return out

    X_train_stage2, X_test_stage2, block_summaries, num_blocks = _build_stage_features(
        model_spec=model_spec,
        X_train=X_train,
        y_train=y_train,
        sampled_snp_ids=sampled_snp_ids,
        strategy=config["grouping_strategy"],
        group_size=group_size,
        seed=config["seed"] + fold_id * 1000,
        pad_incomplete_last_block=config.get("pad_incomplete_last_block", True),
        embedding_reduce_dim=None,
        include_block_scalar=bool(config.get("include_block_scalar", False)),
        second_stage_adjustment_config=_resolve_second_stage_adjustment(config, model_spec.name),
        collect_block_diagnostics=bool(config.get("collect_block_diagnostics", False)),
        embedding_extraction_mode=str(config.get("embedding_extraction_mode", "legacy")),
        stage2_feature_mode=_resolve_stage2_feature_mode(model_spec),
        X_eval=X_test,
    )
    prior_train = np.column_stack([cached["bayesb_train"], cached["gblup_train"]]).astype(np.float32)
    prior_test = np.column_stack([cached["bayesb_test"], cached["gblup_test"]]).astype(np.float32)
    X_train_stage2, X_test_stage2 = _append_stage2_prior_feature(
        model_spec=model_spec,
        X_train_stage2=X_train_stage2,
        X_test_stage2=X_test_stage2,
        prior_train=prior_train,
        prior_test=prior_test,
    )
    stage2_config = _prepare_stage2_config(model_spec, block_summaries, include_block_scalar=False)
    stage2_config = _maybe_attach_block_prior_scores(
        stage2_config=stage2_config,
        stage2_backend=model_spec.stage2_backend,
        block_summaries=block_summaries,
        bayesb_beta=cached.get("bayesb_beta"),
    )
    _, y_test_pred, _ = fit_stage2_model(
        model_spec.stage2_backend,
        X_train_stage2,
        y_train,
        X_test_stage2,
        stage2_config,
        config["seed"] + fold_id,
    )
    metric = regression_metrics(y_test, y_test_pred)
    out = {
        "fold": int(fold_id),
        "group_size": int(group_size),
        "variance_target_pct": 99,
        "pearson": float(metric["pearson"]),
        "r2": float(metric["r2"]),
        "num_blocks": int(num_blocks),
        "mean_reduced_block_embedding_dim": float(np.mean([b["reduced_embedding_dim"] for b in block_summaries])),
        "stage2_input_dim": int(X_train_stage2.shape[1]),
        "output_dir": str(output_dir),
    }
    pd.DataFrame([out]).to_csv(output_dir / "fold_metrics.csv", index=False)
    return out


def run_dual_prior_fixed_block_with_frozen_gate_on_fold(
    base_config: dict[str, Any],
    fold_id: int,
    group_size: int,
    gate_summary_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cached = precompute_dual_prior_cache(base_config, fold_id=fold_id, cache_root=output_dir / "prior_cache", inner_folds=0)
    X_train = cached["X_outer_train"]
    y_train = cached["y_outer_train"]
    X_test = cached["X_outer_test"]
    y_test = cached["y_outer_test"]
    config = _build_config(base_config, group_size=group_size, output_dir=str(output_dir))
    sampled_snp_ids = [f"snp_{i}" for i in range(X_train.shape[1])]
    model_spec = resolve_two_stage_model_specs(base_config)[0]
    if model_spec.stage2_backend.lower() != "group_shared_gate":
        raise ValueError("Frozen gate evaluation is only supported for group_shared_gate.")

    stage2_config = dict(model_spec.stage2_config)
    group_mode = str(stage2_config.get("group_mode", "embedding_group"))
    num_groups = int(stage2_config.get("num_groups", 3))
    include_block_scalar = bool(config.get("include_block_scalar", False))
    expert_backend, expert_config = _resolve_group_shared_gate_expert(model_spec)

    X_train_stage2, X_test_stage2, block_summaries, num_blocks = _build_stage_features(
        model_spec=model_spec,
        X_train=X_train,
        y_train=y_train,
        sampled_snp_ids=sampled_snp_ids,
        strategy=config["grouping_strategy"],
        group_size=group_size,
        seed=config["seed"] + fold_id * 1000,
        pad_incomplete_last_block=config.get("pad_incomplete_last_block", True),
        embedding_reduce_dim=None,
        include_block_scalar=include_block_scalar,
        second_stage_adjustment_config=_resolve_second_stage_adjustment(config, model_spec.name),
        collect_block_diagnostics=bool(config.get("collect_block_diagnostics", False)),
        embedding_extraction_mode=str(config.get("embedding_extraction_mode", "legacy")),
        stage2_feature_mode=_resolve_stage2_feature_mode(model_spec),
        X_eval=X_test,
    )
    _, y_tabicl_test, _ = _fit_expert_regressor(
        expert_backend,
        expert_config,
        X_train_stage2,
        y_train,
        X_test_stage2,
        seed=config["seed"] + fold_id,
    )
    prior_scores = aggregate_beta_to_block_prior(
        beta=cached["bayesb_beta"],
        block_summaries=block_summaries,
        method="l2",
    ).astype(np.float32)
    test_group_features, _ = _build_group_features_from_stage2(
        X_test_stage2,
        block_summaries,
        group_mode=group_mode,
        num_groups=num_groups,
        prior_scores=prior_scores,
        include_block_scalar=include_block_scalar,
    )
    gate_summary = json.loads(Path(gate_summary_path).read_text(encoding="utf-8"))
    frozen_gate = GroupSharedGateRegressor.from_summary(gate_summary, random_state=int(config["seed"]) + 30000 + int(group_size))
    y_test_pred = frozen_gate.predict_from_group_features(
        test_group_features,
        np.asarray(y_tabicl_test, dtype=np.float32),
        cached["bayesb_test"],
        cached["gblup_test"],
        y_bayes_candidates=(
            {
                key: value
                for key, value in {
                    "BayesLasso": cached.get("bayeslasso_test"),
                    "RKHS": cached.get("rkhs_test"),
                }.items()
                if value is not None
            }
            or None
        ),
    )
    metric = regression_metrics(y_test, y_test_pred)
    out = {
        "fold": int(fold_id),
        "group_size": int(group_size),
        "variance_target_pct": 99,
        "pearson": float(metric["pearson"]),
        "r2": float(metric["r2"]),
        "num_blocks": int(num_blocks),
        "mean_reduced_block_embedding_dim": float(np.mean([b["reduced_embedding_dim"] for b in block_summaries])),
        "stage2_input_dim": int(X_train_stage2.shape[1] + 2),
        "output_dir": str(output_dir),
        "group_counts": str(gate_summary.get("group_counts", [])),
        "frozen_gate_summary_path": str(gate_summary_path),
    }
    _write_json(output_dir / "group_shared_gate_group_summary.json", gate_summary)
    pd.DataFrame([out]).to_csv(output_dir / "fold_metrics.csv", index=False)
    return out


def load_base_config(path: str) -> dict[str, Any]:
    return load_experiment_config(path)
