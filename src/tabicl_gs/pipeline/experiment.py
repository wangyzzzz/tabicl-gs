from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tabicl_gs.data.block_matrix import extract_block_matrix
from tabicl_gs.data.grouping import build_blocks, subsample_snp_indices
from tabicl_gs.data.plink import (
    align_phenotype_to_sample_ids,
    impute_by_train_mean,
    load_plink_matrix,
    plink_num_snps,
    read_phenotype_table,
)
from tabicl_gs.eval.metrics import regression_metrics
from tabicl_gs.eval.splits import make_outer_cv_splits
from tabicl_gs.models.baselines import run_r_baseline
from tabicl_gs.models.factory import _fit_expert_regressor, create_stage1_encoder, fit_stage2_model
from tabicl_gs.models.model_specs import TwoStageModelSpec, resolve_two_stage_model_specs
from tabicl_gs.pipeline.block_quality import apply_block_weights, compute_block_weights
from tabicl_gs.pipeline.block_prior import aggregate_beta_to_block_prior, interleave_block_prior_features
from tabicl_gs.pipeline.block_prior import build_block_level_prior_matrix
from tabicl_gs.pipeline.block_prior import compute_sample_block_prior_predictions

PAD_MARKER = -1


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_tabicl_inner_oof_bundle(
    output_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "tabicl_inner_oof_targets.npy", np.asarray(y_true, dtype=np.float32))
    np.save(output_dir / "tabicl_inner_oof_predictions.npy", np.asarray(y_pred, dtype=np.float32))
    _save_json(output_dir / "tabicl_inner_oof_summary.json", metadata)


def _save_baseline_inner_oof_bundle(
    output_dir: Path,
    baseline_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "inner_oof_targets.npy", np.asarray(y_true, dtype=np.float32))
    np.save(output_dir / "inner_oof_predictions.npy", np.asarray(y_pred, dtype=np.float32))
    _save_json(
        output_dir / "inner_oof_summary.json",
        {
            "baseline_model": str(baseline_name),
            **metadata,
        },
    )


def _select_split_items(
    splits: list[tuple[np.ndarray, np.ndarray]],
    fold_ids: list[int] | None,
    max_folds: int | None,
) -> list[tuple[int, tuple[np.ndarray, np.ndarray]]]:
    split_items = list(enumerate(splits, start=1))
    if fold_ids:
        max_fold = len(split_items)
        invalid = [fold_id for fold_id in fold_ids if fold_id < 1 or fold_id > max_fold]
        if invalid:
            raise ValueError(f"Requested fold ids {invalid} are out of range 1..{max_fold}.")
        keep = set(int(fold_id) for fold_id in fold_ids)
        split_items = [(fold_id, split) for fold_id, split in split_items if fold_id in keep]
    if max_folds is not None:
        split_items = split_items[:max_folds]
    return split_items


def _resolve_sample_size_override(
    config: dict[str, Any],
    *,
    fold_id: int,
    original_n_train: int,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    sample_override = config.get("_sample_size_override")
    if sample_override is None:
        return None, None
    if not isinstance(sample_override, dict):
        raise ValueError("_sample_size_override must be a dict when provided.")

    subset_indices = None
    override_summary: dict[str, Any] = {
        "requested": True,
        "fold_id": int(fold_id),
    }
    fold_subsets = sample_override.get("fold_subsets")
    if fold_subsets is not None:
        if not isinstance(fold_subsets, dict):
            raise ValueError("_sample_size_override.fold_subsets must be a dict when provided.")
        subset_indices = fold_subsets.get(str(fold_id), fold_subsets.get(int(fold_id)))
        override_summary["mode"] = "fold_subsets"
    else:
        override_fold_id = sample_override.get("fold_id")
        if override_fold_id is not None and int(override_fold_id) == int(fold_id):
            subset_indices = sample_override.get("train_subset_indices")
        override_summary["mode"] = "single_fold"

    passthrough_keys = ["proportion", "repeat", "selection_seed", "selection_tag", "note"]
    for key in passthrough_keys:
        if key in sample_override:
            override_summary[key] = sample_override[key]

    if subset_indices is None:
        override_summary["applied"] = False
        return None, override_summary

    subset_array = np.asarray(subset_indices, dtype=np.int64).reshape(-1)
    if subset_array.size == 0:
        raise ValueError("_sample_size_override selected an empty training subset.")
    if np.any(subset_array < 0) or np.any(subset_array >= int(original_n_train)):
        raise ValueError(
            f"_sample_size_override indices for fold {fold_id} are out of range 0..{int(original_n_train) - 1}."
        )
    override_summary["applied"] = True
    override_summary["subset_size"] = int(subset_array.shape[0])
    override_summary["train_subset_indices"] = subset_array.astype(int).tolist()
    return subset_array, override_summary


def _runtime_device_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {"cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", "")}
    try:
        import torch

        metadata["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            metadata["process_cuda_device_index"] = int(torch.cuda.current_device())
            metadata["process_cuda_device_name"] = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    except Exception as exc:  # pragma: no cover - defensive metadata path
        metadata["torch_cuda_metadata_error"] = str(exc)
    return metadata


def _metric_value(metrics: dict[str, float], metric_name: str) -> float:
    if metric_name not in metrics:
        raise ValueError(f"Unsupported tuning metric: {metric_name}")
    return float(metrics[metric_name])


def _summarize_block_dimensions(block_summaries: list[dict[str, Any]]) -> dict[str, float | int | None]:
    if not block_summaries:
        return {
            "raw_block_embedding_dim": None,
            "reduced_block_embedding_dim": None,
            "mean_reduced_block_embedding_dim": None,
            "min_reduced_block_embedding_dim": None,
            "max_reduced_block_embedding_dim": None,
            "mean_explained_variance_ratio_sum": None,
        }

    raw_dim = block_summaries[0].get("raw_embedding_dim")
    reduced_dims = [
        int(summary["reduced_embedding_dim"])
        for summary in block_summaries
        if summary.get("reduced_embedding_dim") is not None
    ]
    explained = [
        float(summary["explained_variance_ratio_sum"])
        for summary in block_summaries
        if summary.get("explained_variance_ratio_sum") is not None
    ]
    explained = [value for value in explained if np.isfinite(value)]

    return {
        "raw_block_embedding_dim": None if raw_dim is None else int(raw_dim),
        "reduced_block_embedding_dim": None if not reduced_dims else int(reduced_dims[0]),
        "mean_reduced_block_embedding_dim": None if not reduced_dims else float(np.mean(reduced_dims)),
        "min_reduced_block_embedding_dim": None if not reduced_dims else int(np.min(reduced_dims)),
        "max_reduced_block_embedding_dim": None if not reduced_dims else int(np.max(reduced_dims)),
        "mean_explained_variance_ratio_sum": None if not explained else float(np.mean(explained)),
    }


def _resolve_stage2_feature_mode(model_spec: TwoStageModelSpec) -> str:
    if model_spec.stage2_backend.lower() != "block_attention":
        return "reduced"
    if bool(model_spec.stage2_config.get("use_raw_block_embeddings", False)):
        return "raw"
    return "reduced"


def _has_tabicl_inner_oof_source(model_spec: TwoStageModelSpec) -> bool:
    backend = model_spec.stage2_backend.lower()
    if backend in {"tabicl", "tabpfn"}:
        return True
    if backend in {"calibrated_correction", "group_shared_gate"}:
        return str(model_spec.stage2_config.get("expert_backend", "tabicl")).lower() in {"tabicl", "tabpfn"}
    return False


def _prepare_stage2_config(
    model_spec: TwoStageModelSpec,
    block_summaries: list[dict[str, Any]],
    include_block_scalar: bool,
) -> dict[str, Any]:
    config = dict(model_spec.stage2_config)
    backend = model_spec.stage2_backend.lower()
    if backend == "sample_mixture":
        config["expert_config"] = dict(config.get("expert_config", {}))
        return config
    if backend in {"static_block_weight", "group_weight_pooling", "group_shared_gate"}:
        config["block_input_dims"] = [
            int(summary["reduced_embedding_dim"]) + (1 if include_block_scalar else 0) for summary in block_summaries
        ]
        return config
    if backend != "block_attention":
        return config
    feature_mode = _resolve_stage2_feature_mode(model_spec)
    config["block_input_dims"] = [
        int(summary["raw_embedding_dim"] if feature_mode == "raw" else summary["reduced_embedding_dim"])
        + (1 if include_block_scalar else 0)
        for summary in block_summaries
    ]
    if bool(model_spec.stage2_config.get("use_prior_token", False)):
        config["block_input_dims"].append(1)
    config.pop("use_raw_block_embeddings", None)
    return config


def _append_stage2_prior_feature(
    model_spec: TwoStageModelSpec,
    X_train_stage2: np.ndarray,
    X_test_stage2: np.ndarray | None,
    prior_train: np.ndarray | None,
    prior_test: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    backend = model_spec.stage2_backend.lower()
    if backend in {"sample_mixture", "calibrated_correction", "group_shared_gate"}:
        if not bool(model_spec.stage2_config.get("use_prior_prediction", False)):
            return X_train_stage2, X_test_stage2
        if prior_train is None:
            raise ValueError("use_prior_prediction=True requires prior_train predictions.")
        prior_train_arr = np.asarray(prior_train, dtype=np.float32)
        if prior_train_arr.ndim == 1:
            prior_train_arr = prior_train_arr.reshape(-1, 1)
        X_train_out = np.concatenate([X_train_stage2.astype(np.float32), prior_train_arr], axis=1).astype(np.float32)
        if X_test_stage2 is None:
            return X_train_out, None
        if prior_test is None:
            raise ValueError("use_prior_prediction=True requires prior_test predictions when X_test_stage2 is provided.")
        prior_test_arr = np.asarray(prior_test, dtype=np.float32)
        if prior_test_arr.ndim == 1:
            prior_test_arr = prior_test_arr.reshape(-1, 1)
        X_test_out = np.concatenate([X_test_stage2.astype(np.float32), prior_test_arr], axis=1).astype(np.float32)
        return X_train_out, X_test_out
    if backend != "block_attention":
        return X_train_stage2, X_test_stage2
    if not bool(model_spec.stage2_config.get("use_prior_token", False)):
        return X_train_stage2, X_test_stage2
    if prior_train is None:
        raise ValueError("use_prior_token=True requires prior_train predictions.")

    prior_train_col = np.asarray(prior_train, dtype=np.float32).reshape(-1, 1)
    X_train_out = np.concatenate([X_train_stage2.astype(np.float32), prior_train_col], axis=1).astype(np.float32)
    if X_test_stage2 is None:
        return X_train_out, None
    if prior_test is None:
        raise ValueError("use_prior_token=True requires prior_test predictions when X_test_stage2 is provided.")
    prior_test_col = np.asarray(prior_test, dtype=np.float32).reshape(-1, 1)
    X_test_out = np.concatenate([X_test_stage2.astype(np.float32), prior_test_col], axis=1).astype(np.float32)
    return X_train_out, X_test_out


def _append_stage2_block_prior_feature(
    model_spec: TwoStageModelSpec,
    X_train_stage2: np.ndarray,
    X_test_stage2: np.ndarray | None,
    block_prior: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if model_spec.stage2_backend.lower() != "block_attention":
        return X_train_stage2, X_test_stage2
    if not bool(model_spec.stage2_config.get("use_block_prior", False)):
        return X_train_stage2, X_test_stage2
    if block_prior is None:
        raise ValueError("use_block_prior=True requires block_prior features.")
    prior = np.asarray(block_prior, dtype=np.float32).reshape(1, -1)
    prior_train = np.repeat(prior, X_train_stage2.shape[0], axis=0)
    X_train_out = np.concatenate([X_train_stage2.astype(np.float32), prior_train], axis=1).astype(np.float32)
    if X_test_stage2 is None:
        return X_train_out, None
    prior_test = np.repeat(prior, X_test_stage2.shape[0], axis=0)
    X_test_out = np.concatenate([X_test_stage2.astype(np.float32), prior_test], axis=1).astype(np.float32)
    return X_train_out, X_test_out


def _build_stage_features(
    model_spec: TwoStageModelSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sampled_snp_ids: list[str],
    strategy: str,
    group_size: int,
    seed: int,
    pad_incomplete_last_block: bool,
    embedding_reduce_dim: int | None,
    include_block_scalar: bool,
    second_stage_adjustment_config: dict[str, Any] | None = None,
    collect_block_diagnostics: bool = True,
    embedding_extraction_mode: str = "current",
    stage2_feature_mode: str = "reduced",
    X_eval: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, list[dict[str, Any]], int]:
    blocks = build_blocks(
        list(range(X_train.shape[1])),
        strategy=strategy,
        group_size=group_size,
        seed=seed,
        pad_incomplete=pad_incomplete_last_block,
        pad_value=PAD_MARKER,
    )
    train_block_embeddings = []
    eval_block_embeddings = []
    block_summaries = []

    for block_id, block_columns in enumerate(blocks, start=1):
        block_start = time.perf_counter()
        block_seed = seed + block_id
        encoder_config = dict(model_spec.stage1_config)
        if embedding_reduce_dim is not None:
            encoder_config["embedding_reduce_dim"] = embedding_reduce_dim
        encoder = create_stage1_encoder(model_spec.stage1_backend, encoder_config, block_seed)

        X_train_block = extract_block_matrix(X_train, block_columns, pad_marker=PAD_MARKER, pad_value=0.0)
        X_train_block, _ = impute_by_train_mean(X_train_block, X_train_block)
        fit_start = time.perf_counter()
        encoder.fit(X_train_block, y_train)
        fit_seconds = time.perf_counter() - fit_start
        transform_train_start = time.perf_counter()
        use_legacy_extraction = embedding_extraction_mode == "legacy"
        if collect_block_diagnostics or include_block_scalar or use_legacy_extraction:
            train_raw, train_scalar = encoder.transform_with_scalar(X_train_block)
        else:
            train_raw = encoder.transform(X_train_block)
            train_scalar = None
        transform_train_seconds = time.perf_counter() - transform_train_start
        reducer_start = time.perf_counter()
        encoder.fit_reducer(train_raw)
        reducer_seconds = time.perf_counter() - reducer_start
        train_feature_base = train_raw.astype(np.float32) if stage2_feature_mode == "raw" else encoder.reduce_embeddings(train_raw)
        train_features = [train_feature_base]

        eval_features = None
        transform_eval_seconds = 0.0
        if X_eval is not None:
            X_eval_block = extract_block_matrix(X_eval, block_columns, pad_marker=PAD_MARKER, pad_value=0.0)
            _, X_eval_block = impute_by_train_mean(X_train_block, X_eval_block)
            transform_eval_start = time.perf_counter()
            if collect_block_diagnostics or include_block_scalar or use_legacy_extraction:
                eval_raw, eval_scalar = encoder.transform_with_scalar(X_eval_block)
            else:
                eval_raw = encoder.transform(X_eval_block)
                eval_scalar = None
            transform_eval_seconds = time.perf_counter() - transform_eval_start
            eval_feature_base = eval_raw.astype(np.float32) if stage2_feature_mode == "raw" else encoder.reduce_embeddings(eval_raw)
            eval_features = [eval_feature_base]

        if include_block_scalar:
            train_features.append(train_scalar.reshape(-1, 1).astype(np.float32))
            if X_eval is not None:
                eval_features.append(eval_scalar.reshape(-1, 1).astype(np.float32))

        train_block_embeddings.append(np.concatenate(train_features, axis=1).astype(np.float32))
        if X_eval is not None:
            eval_block_embeddings.append(np.concatenate(eval_features, axis=1).astype(np.float32))

        meta = encoder.metadata()
        scalar_train_pearson = None
        if collect_block_diagnostics and train_scalar is not None:
            if np.std(y_train) > 0 and np.std(train_scalar) > 0:
                scalar_train_pearson = float(np.corrcoef(y_train, train_scalar)[0, 1])
            else:
                scalar_train_pearson = 0.0
        block_summaries.append(
            {
                "block_id": block_id,
                "num_snps": len(block_columns),
                "raw_embedding_dim": meta.raw_embedding_dim,
                "reduced_embedding_dim": meta.reduced_embedding_dim,
                "explained_variance_ratio_sum": getattr(meta, "explained_variance_ratio_sum", float("nan")) if collect_block_diagnostics else None,
                "scalar_train_pearson": scalar_train_pearson,
                "include_block_scalar": include_block_scalar,
                "device": meta.device,
                "fit_seconds": fit_seconds,
                "transform_train_seconds": transform_train_seconds,
                "reducer_seconds": reducer_seconds,
                "transform_eval_seconds": transform_eval_seconds,
                "block_total_seconds": time.perf_counter() - block_start,
                "snp_indices": [int(index) for index in block_columns if index != PAD_MARKER],
                "snp_ids": [sampled_snp_ids[index] for index in block_columns if index != PAD_MARKER],
                "padding_count": int(sum(1 for index in block_columns if index == PAD_MARKER)),
            }
        )

    train_block_embeddings, eval_block_embeddings, block_summaries = _apply_second_stage_adjustment(
        train_block_embeddings,
        eval_block_embeddings if X_eval is not None else None,
        block_summaries,
        second_stage_adjustment_config,
    )

    X_train_stage2 = np.concatenate(train_block_embeddings, axis=1).astype(np.float32)
    X_eval_stage2 = None if X_eval is None else np.concatenate(eval_block_embeddings, axis=1).astype(np.float32)
    return X_train_stage2, X_eval_stage2, block_summaries, len(blocks)


def _resolve_second_stage_adjustment(config: dict[str, Any], model_name: str) -> dict[str, Any] | None:
    settings = config.get("second_stage_adjustment")
    if not settings or not settings.get("enabled", False):
        return None
    model_names = settings.get("model_names")
    if model_names and model_name not in model_names:
        return None
    return settings


def _apply_second_stage_adjustment(
    train_block_embeddings: list[np.ndarray],
    eval_block_embeddings: list[np.ndarray] | None,
    block_summaries: list[dict[str, Any]],
    adjustment_config: dict[str, Any] | None,
) -> tuple[list[np.ndarray], list[np.ndarray] | None, list[dict[str, Any]]]:
    if not adjustment_config or not adjustment_config.get("enabled", False):
        return train_block_embeddings, eval_block_embeddings, block_summaries
    scores, weights = compute_block_weights(block_summaries, adjustment_config)
    weighted_train = apply_block_weights(train_block_embeddings, weights)
    weighted_eval = None if eval_block_embeddings is None else apply_block_weights(eval_block_embeddings, weights)
    updated = []
    for summary, score, weight in zip(block_summaries, scores, weights):
        record = dict(summary)
        record["quality_score"] = float(score)
        record["block_weight"] = float(weight)
        updated.append(record)
    return weighted_train, weighted_eval, updated


def _resolve_tuning_settings(config: dict[str, Any], model_name: str) -> dict[str, Any] | None:
    tuning = config.get("tuning")
    if not tuning or not tuning.get("enabled", False):
        return None
    model_names = tuning.get("model_names")
    if model_names and model_name not in model_names:
        return None
    return tuning


def _resolve_residual_target(config: dict[str, Any]) -> dict[str, Any] | None:
    settings = config.get("residual_target")
    if not settings or not settings.get("enabled", False):
        return None
    baseline_model = str(settings.get("baseline_model", "GBLUP"))
    if baseline_model not in {"GBLUP", "BayesA", "BayesB", "BayesLasso"}:
        raise ValueError(f"Unsupported residual baseline model: {baseline_model}")
    return {
        "enabled": True,
        "baseline_model": baseline_model,
    }


def _resolve_stage2_prior(config: dict[str, Any], model_spec: TwoStageModelSpec) -> dict[str, Any] | None:
    backend = model_spec.stage2_backend.lower()
    if backend in {"sample_mixture", "calibrated_correction", "group_shared_gate"}:
        if backend in {"calibrated_correction", "group_shared_gate"} and bool(model_spec.stage2_config.get("use_dual_priors", False)):
            return None
        if not bool(model_spec.stage2_config.get("use_prior_prediction", False)):
            return None
    elif backend != "block_attention":
        return None
    if backend == "block_attention" and not bool(model_spec.stage2_config.get("use_prior_token", False)):
        return None
    settings = config.get("stage2_prior")
    baseline_model = None
    if settings and settings.get("enabled", False):
        baseline_model = str(settings.get("baseline_model", "BayesB"))
    else:
        residual_target = _resolve_residual_target(config)
        if residual_target is not None:
            baseline_model = str(residual_target["baseline_model"])
    if baseline_model is None:
        raise ValueError("use_prior_token=True requires either stage2_prior or residual_target configuration.")
    if baseline_model not in {"GBLUP", "BayesA", "BayesB", "BayesLasso"}:
        raise ValueError(f"Unsupported stage2 prior baseline model: {baseline_model}")
    return {"baseline_model": baseline_model}


def _resolve_stage2_dual_prior(config: dict[str, Any], model_spec: TwoStageModelSpec) -> dict[str, Any] | None:
    if model_spec.stage2_backend.lower() not in {"calibrated_correction", "group_shared_gate"}:
        return None
    if not bool(model_spec.stage2_config.get("use_dual_priors", False)):
        return None
    settings = config.get("stage2_dual_prior")
    if not settings or not settings.get("enabled", False):
        raise ValueError("use_dual_priors=True requires stage2_dual_prior configuration.")
    primary_model = str(settings.get("primary_model", "BayesB"))
    secondary_model = str(settings.get("secondary_model", "GBLUP"))
    valid = {"GBLUP", "BayesA", "BayesB", "BayesLasso"}
    if primary_model not in valid or secondary_model not in valid:
        raise ValueError(
            f"Unsupported stage2 dual prior models: primary={primary_model}, secondary={secondary_model}"
        )
    return {
        "primary_model": primary_model,
        "secondary_model": secondary_model,
    }


def _resolve_baseline_inner_oof(config: dict[str, Any]) -> dict[str, Any] | None:
    settings = config.get("baseline_inner_oof")
    if not settings or not settings.get("enabled", False):
        return None
    valid_models = {"GBLUP", "BayesA", "BayesB", "BayesLasso", "RKHS"}
    models = settings.get("models", ["GBLUP", "BayesB", "RKHS"])
    if not isinstance(models, list) or not models:
        raise ValueError("baseline_inner_oof.models must be a non-empty list when enabled.")
    normalized_models: list[str] = []
    for model_name in models:
        name = str(model_name)
        if name not in valid_models:
            raise ValueError(f"Unsupported baseline_inner_oof model: {name}")
        if name not in normalized_models:
            normalized_models.append(name)
    fold_id = int(settings.get("fold", 1))
    n_splits = int(settings.get("n_splits", 3))
    if fold_id < 1:
        raise ValueError("baseline_inner_oof.fold must be >= 1.")
    if n_splits < 2:
        raise ValueError("baseline_inner_oof.n_splits must be >= 2.")
    return {
        "models": normalized_models,
        "fold": fold_id,
        "n_splits": n_splits,
    }


def _resolve_tabicl_inner_oof_settings(config: dict[str, Any]) -> dict[str, Any]:
    enabled = bool(config.get("tabicl_inner_oof_enabled", True))
    fold = config.get("tabicl_inner_oof_fold")
    if fold is None:
        return {"enabled": enabled, "fold": None}
    fold_id = int(fold)
    if fold_id < 1:
        raise ValueError("tabicl_inner_oof_fold must be >= 1 when provided.")
    return {"enabled": enabled, "fold": fold_id}


def _compute_residual_target_predictions(
    residual_target_config: dict[str, Any],
    fold_dir: Path,
    X_train_base: np.ndarray,
    y_train: np.ndarray,
    X_test_base: np.ndarray,
    config: dict[str, Any],
    fold_id: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    baseline_name = str(residual_target_config["baseline_model"])
    residual_dir = fold_dir / "_residual_target" / baseline_name
    train_result = run_r_baseline(
        model_name=baseline_name,
        X_train=X_train_base,
        y_train=y_train,
        X_test=X_train_base,
        output_dir=residual_dir / "train_fit",
        rscript_path=config["baselines"]["rscript_path"],
        seed=config["seed"] + fold_id,
        sommer_method=config["baselines"].get("sommer_method"),
        keep_artifacts=True,
    )
    test_result = run_r_baseline(
        model_name=baseline_name,
        X_train=X_train_base,
        y_train=y_train,
        X_test=X_test_base,
        output_dir=residual_dir / "test_fit",
        rscript_path=config["baselines"]["rscript_path"],
        seed=config["seed"] + fold_id,
        sommer_method=config["baselines"].get("sommer_method"),
        keep_artifacts=True,
    )
    summary = {
        "baseline_model": baseline_name,
        "train_prediction_mean": float(np.mean(train_result.predictions)),
        "test_prediction_mean": float(np.mean(test_result.predictions)),
        "train_command": train_result.command,
        "test_command": test_result.command,
        "metadata": {
            "train": train_result.metadata,
            "test": test_result.metadata,
        },
    }
    return (
        np.asarray(train_result.predictions, dtype=np.float32),
        np.asarray(test_result.predictions, dtype=np.float32),
        summary,
    )


def _compute_oof_baseline_prior_predictions(
    baseline_model: str,
    fold_dir: Path,
    X_train_base: np.ndarray,
    y_train: np.ndarray,
    config: dict[str, Any],
    fold_id: int,
    n_splits: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    inner_splits = make_outer_cv_splits(X_train_base, n_splits=n_splits, seed=config["seed"] + fold_id)
    oof = np.zeros(X_train_base.shape[0], dtype=np.float32)
    call_summaries: list[dict[str, Any]] = []
    for inner_id, (inner_train_idx, inner_valid_idx) in enumerate(inner_splits, start=1):
        inner_dir = fold_dir / "_stage2_prior_oof" / baseline_model / f"inner_fold_{inner_id}"
        result = run_r_baseline(
            model_name=baseline_model,
            X_train=X_train_base[inner_train_idx],
            y_train=y_train[inner_train_idx],
            X_test=X_train_base[inner_valid_idx],
            output_dir=inner_dir,
            rscript_path=config["baselines"]["rscript_path"],
            seed=config["seed"] + fold_id + inner_id,
            sommer_method=config["baselines"].get("sommer_method"),
            keep_artifacts=True,
        )
        oof[inner_valid_idx] = np.asarray(result.predictions, dtype=np.float32)
        call_summaries.append(
            {
                "inner_fold": inner_id,
                "n_train": int(len(inner_train_idx)),
                "n_valid": int(len(inner_valid_idx)),
                "output_dir": str(inner_dir),
            }
        )
    return oof.astype(np.float32), {
        "baseline_model": baseline_model,
        "oof_splits": int(n_splits),
        "calls": call_summaries,
    }


def _compute_fixed_block_tabicl_inner_oof(
    *,
    config: dict[str, Any],
    model_spec: TwoStageModelSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sampled_snp_ids: list[str],
    fold_id: int,
    model_offset: int,
    chosen_group_size: int,
    chosen_embedding_dim: int | None,
    chosen_include_block_scalar: bool,
    second_stage_adjustment: dict[str, Any] | None,
    embedding_extraction_mode: str,
    stage2_feature_mode: str,
    stage2_prior_train: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    inner_splits = make_outer_cv_splits(
        X_train,
        n_splits=int(model_spec.stage2_config.get("oof_splits", 3)),
        seed=int(config["seed"]) + int(fold_id),
    )
    oof_pred = np.zeros(X_train.shape[0], dtype=np.float32)
    call_summaries: list[dict[str, Any]] = []
    for inner_id, (inner_train_idx, inner_valid_idx) in enumerate(inner_splits, start=1):
        X_inner_train = X_train[inner_train_idx]
        y_inner_train = y_train[inner_train_idx]
        X_inner_valid = X_train[inner_valid_idx]
        X_train_stage2, X_valid_stage2, block_summaries, _ = _build_stage_features(
            model_spec=model_spec,
            X_train=X_inner_train,
            y_train=y_inner_train,
            sampled_snp_ids=sampled_snp_ids,
            strategy=config["grouping_strategy"],
            group_size=int(chosen_group_size),
            seed=int(config["seed"]) + int(fold_id) * 1000 + int(model_offset) * 100 + int(inner_id),
            pad_incomplete_last_block=config.get("pad_incomplete_last_block", False),
            embedding_reduce_dim=chosen_embedding_dim,
            include_block_scalar=bool(chosen_include_block_scalar),
            second_stage_adjustment_config=second_stage_adjustment,
            collect_block_diagnostics=bool(config.get("collect_block_diagnostics", False)),
            embedding_extraction_mode=embedding_extraction_mode,
            stage2_feature_mode=stage2_feature_mode,
            X_eval=X_inner_valid,
        )
        prior_train_inner = None
        prior_valid_inner = None
        if stage2_prior_train is not None:
            prior_train_arr = np.asarray(stage2_prior_train, dtype=np.float32)
            prior_train_inner = prior_train_arr[inner_train_idx]
            prior_valid_inner = prior_train_arr[inner_valid_idx]
        X_train_stage2, X_valid_stage2 = _append_stage2_prior_feature(
            model_spec=model_spec,
            X_train_stage2=X_train_stage2,
            X_test_stage2=X_valid_stage2,
            prior_train=prior_train_inner,
            prior_test=prior_valid_inner,
        )
        backend = model_spec.stage2_backend.lower()
        if backend in {"calibrated_correction", "group_shared_gate"}:
            expert_backend = str(model_spec.stage2_config.get("expert_backend", "tabicl")).lower()
            expert_config = dict(model_spec.stage2_config.get("expert_config", {}))
            prior_width = 2 if bool(model_spec.stage2_config.get("use_dual_priors", False)) else 1
            X_train_core = X_train_stage2[:, :-prior_width].astype(np.float32)
            X_valid_core = X_valid_stage2[:, :-prior_width].astype(np.float32)
            _, valid_tabicl_pred, _ = _fit_expert_regressor(
                expert_backend,
                expert_config,
                X_train_core,
                y_inner_train,
                X_valid_core,
                seed=int(config["seed"]) + int(fold_id) * 1000 + int(model_offset) * 100 + 10000 + int(inner_id),
            )
            oof_pred[inner_valid_idx] = np.asarray(valid_tabicl_pred, dtype=np.float32)
        else:
            stage2_config = _prepare_stage2_config(model_spec, block_summaries, bool(chosen_include_block_scalar))
            stage2_model, _, _ = fit_stage2_model(
                model_spec.stage2_backend,
                X_train_stage2,
                y_inner_train,
                X_valid_stage2,
                stage2_config,
                int(config["seed"]) + int(fold_id) * 1000 + int(model_offset) * 100 + 10000 + int(inner_id),
            )
            oof_pred[inner_valid_idx] = np.asarray(_predict_stage2_model(stage2_model, X_valid_stage2), dtype=np.float32)
        call_summaries.append(
            {
                "inner_fold": int(inner_id),
                "n_train": int(len(inner_train_idx)),
                "n_valid": int(len(inner_valid_idx)),
            }
        )
    metrics = regression_metrics(y_train, oof_pred)
    return oof_pred.astype(np.float32), {
        "source": "fixed_block_inner_oof",
        "fold": int(fold_id),
        "group_size": int(chosen_group_size),
        "oof_splits": int(len(inner_splits)),
        "inner_oof_pearson": float(metrics["pearson"]),
        "inner_oof_r2": float(metrics["r2"]),
        "calls": call_summaries,
    }


def _run_optuna(
    tuning_config: dict[str, Any],
    model_spec: TwoStageModelSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sampled_snp_ids: list[str],
    strategy: str,
    seed: int,
    second_stage_adjustment_config: dict[str, Any] | None,
    collect_block_diagnostics: bool,
    embedding_extraction_mode: str,
    stage2_feature_mode: str = "reduced",
    residual_train_target: np.ndarray | None = None,
    train_prediction_offset: np.ndarray | None = None,
    stage2_prior_train: np.ndarray | None = None,
) -> dict[str, Any]:
    import optuna

    objective_metric = tuning_config.get("objective_metric", "pearson")
    direction = "maximize" if objective_metric in {"pearson", "r2"} else "minimize"

    def objective(trial: optuna.Trial) -> float:
        group_size = int(trial.suggest_categorical("group_size", tuning_config["group_size_choices"]))
        embedding_reduce_dim = int(
            trial.suggest_categorical("block_embedding_dim", tuning_config["block_embedding_dim_choices"])
        )
        include_block_scalar = bool(
            trial.suggest_categorical("include_block_scalar", tuning_config["include_block_scalar_choices"])
        )
        X_train_stage2, _, block_summaries, _ = _build_stage_features(
            model_spec=model_spec,
            X_train=X_train,
            y_train=y_train if residual_train_target is None else residual_train_target,
            sampled_snp_ids=sampled_snp_ids,
            strategy=strategy,
            group_size=group_size,
            seed=seed + trial.number * 10000,
            pad_incomplete_last_block=tuning_config.get("pad_incomplete_last_block", True),
            embedding_reduce_dim=embedding_reduce_dim,
            include_block_scalar=include_block_scalar,
            second_stage_adjustment_config=second_stage_adjustment_config,
            collect_block_diagnostics=collect_block_diagnostics,
            embedding_extraction_mode=embedding_extraction_mode,
            stage2_feature_mode=stage2_feature_mode,
            X_eval=None,
        )
        X_train_stage2, _ = _append_stage2_prior_feature(
            model_spec=model_spec,
            X_train_stage2=X_train_stage2,
            X_test_stage2=None,
            prior_train=stage2_prior_train,
            prior_test=None,
        )
        stage2_config = _prepare_stage2_config(model_spec, block_summaries, include_block_scalar)
        _, train_predictions, _ = fit_stage2_model(
            model_spec.stage2_backend,
            X_train_stage2,
            y_train if residual_train_target is None else residual_train_target,
            X_train_stage2,
            stage2_config,
            seed + trial.number,
        )
        if train_prediction_offset is not None:
            train_predictions = train_predictions + train_prediction_offset
        metrics = regression_metrics(y_train, train_predictions)
        return _metric_value(metrics, objective_metric)

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=tuning_config.get("sampler_seed", seed)),
    )
    study.optimize(objective, n_trials=int(tuning_config.get("n_trials", 10)), show_progress_bar=False)
    return study


def _predict_stage2_model(model, X: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(np.asarray(X, dtype=np.float32)), dtype=np.float32)


def _evaluate_trial_params(
    model_spec: TwoStageModelSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sampled_snp_ids: list[str],
    strategy: str,
    seed: int,
    pad_incomplete_last_block: bool,
    params: dict[str, Any],
    second_stage_adjustment_config: dict[str, Any] | None,
    collect_block_diagnostics: bool,
    embedding_extraction_mode: str,
    stage2_feature_mode: str = "reduced",
    residual_train_target: np.ndarray | None = None,
    train_prediction_offset: np.ndarray | None = None,
    test_prediction_offset: np.ndarray | None = None,
    stage2_prior_train: np.ndarray | None = None,
    stage2_prior_test: np.ndarray | None = None,
) -> dict[str, Any]:
    total_start = time.perf_counter()
    group_size = int(params["group_size"])
    embedding_reduce_dim = int(params["block_embedding_dim"])
    include_block_scalar = bool(params["include_block_scalar"])

    stage1_start = time.perf_counter()
    X_train_stage2, X_test_stage2, block_summaries, num_blocks = _build_stage_features(
        model_spec=model_spec,
        X_train=X_train,
        y_train=y_train if residual_train_target is None else residual_train_target,
        X_eval=X_test,
        sampled_snp_ids=sampled_snp_ids,
        strategy=strategy,
        group_size=group_size,
        seed=seed,
        pad_incomplete_last_block=pad_incomplete_last_block,
        embedding_reduce_dim=embedding_reduce_dim,
        include_block_scalar=include_block_scalar,
        second_stage_adjustment_config=second_stage_adjustment_config,
        collect_block_diagnostics=collect_block_diagnostics,
        embedding_extraction_mode=embedding_extraction_mode,
        stage2_feature_mode=stage2_feature_mode,
    )
    X_train_stage2, X_test_stage2 = _append_stage2_prior_feature(
        model_spec=model_spec,
        X_train_stage2=X_train_stage2,
        X_test_stage2=X_test_stage2,
        prior_train=stage2_prior_train,
        prior_test=stage2_prior_test,
    )
    stage1_seconds = time.perf_counter() - stage1_start
    stage2_start = time.perf_counter()
    stage2_config = _prepare_stage2_config(model_spec, block_summaries, include_block_scalar)
    stage2_model, test_predictions, stage2_device = fit_stage2_model(
        model_spec.stage2_backend,
        X_train_stage2,
        y_train if residual_train_target is None else residual_train_target,
        X_test_stage2,
        stage2_config,
        seed,
    )
    stage2_seconds = time.perf_counter() - stage2_start
    train_predictions = _predict_stage2_model(stage2_model, X_train_stage2)
    if train_prediction_offset is not None:
        train_predictions = train_predictions + train_prediction_offset
    if test_prediction_offset is not None:
        test_predictions = test_predictions + test_prediction_offset
    train_metrics = regression_metrics(y_train, train_predictions)
    test_metrics = regression_metrics(y_test, test_predictions)
    block_dim_summary = _summarize_block_dimensions(block_summaries)
    return {
        "group_size": group_size,
        "block_embedding_dim": embedding_reduce_dim,
        "include_block_scalar": include_block_scalar,
        "num_blocks": num_blocks,
        **block_dim_summary,
        "stage2_input_dim": int(X_train_stage2.shape[1]),
        "stage2_device": stage2_device,
        "stage2_model_class": stage2_model.__class__.__name__,
        "stage1_seconds": stage1_seconds,
        "stage2_seconds": stage2_seconds,
        "trial_total_seconds": time.perf_counter() - total_start,
        "train_pearson": float(train_metrics["pearson"]),
        "train_r2": float(train_metrics["r2"]),
        "test_pearson": float(test_metrics["pearson"]),
        "test_r2": float(test_metrics["r2"]),
    }


def run_experiment(config: dict[str, Any], max_folds: int | None = None, fold_ids: list[int] | None = None) -> pd.DataFrame:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

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
    aligned_sample_ids = [plink_data.sample_ids[idx] for idx in keep_indices]
    target = phenotype[config["trait_col"]].to_numpy(dtype=np.float32)
    valid_mask = np.isfinite(target)
    genotype = genotype[valid_mask]
    target = target[valid_mask]
    sample_ids = [sample_id for sample_id, keep in zip(aligned_sample_ids, valid_mask.tolist()) if keep]
    splits = make_outer_cv_splits(genotype, config["outer_cv_folds"], config["seed"])
    split_items = _select_split_items(splits, fold_ids, max_folds)

    metrics_rows: list[dict[str, Any]] = []
    sampled_snp_ids = plink_data.snp_ids
    model_specs = resolve_two_stage_model_specs(config)
    runtime_device = _runtime_device_metadata()
    timing_rows: list[dict[str, Any]] = []
    baseline_inner_oof = _resolve_baseline_inner_oof(config)
    tabicl_inner_oof_settings = _resolve_tabicl_inner_oof_settings(config)

    for fold_id, (train_idx, test_idx) in split_items:
        fold_start = time.perf_counter()
        fold_dir = output_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        X_train = genotype[train_idx]
        X_test = genotype[test_idx]
        y_train = target[train_idx]
        y_test = target[test_idx]
        original_n_train = int(X_train.shape[0])
        sample_override_indices, sample_override_summary = _resolve_sample_size_override(
            config,
            fold_id=int(fold_id),
            original_n_train=original_n_train,
        )
        if sample_override_indices is not None:
            X_train = X_train[sample_override_indices]
            y_train = y_train[sample_override_indices]
        X_train_base, X_test_base = impute_by_train_mean(X_train, X_test)
        residual_target_config = _resolve_residual_target(config)
        residual_train_target = None
        residual_train_offset = None
        residual_test_offset = None
        residual_target_summary = None
        residual_target_seconds = 0.0
        if residual_target_config is not None:
            residual_start = time.perf_counter()
            residual_train_offset, residual_test_offset, residual_target_summary = _compute_residual_target_predictions(
                residual_target_config=residual_target_config,
                fold_dir=fold_dir,
                X_train_base=X_train_base,
                y_train=y_train,
                X_test_base=X_test_base,
                config=config,
                fold_id=fold_id,
            )
            residual_train_target = (y_train - residual_train_offset).astype(np.float32)
            residual_target_seconds = time.perf_counter() - residual_start

        predictions_payload = {
            "sample_id": [sample_ids[index] for index in test_idx],
            "y_true": y_test,
        }
        model_run_summaries: dict[str, Any] = {}
        fold_trial_metric_rows: list[dict[str, Any]] = []
        baseline_fold_metric_rows: list[dict[str, Any]] = []
        baseline_seconds_total = 0.0
        baseline_run_summaries: dict[str, Any] = {}

        for model_offset, model_spec in enumerate(model_specs, start=1):
            second_stage_adjustment = _resolve_second_stage_adjustment(config, model_spec.name)
            collect_block_diagnostics = bool(config.get("collect_block_diagnostics", True)) or bool(second_stage_adjustment)
            embedding_extraction_mode = str(config.get("embedding_extraction_mode", "current"))
            stage2_feature_mode = _resolve_stage2_feature_mode(model_spec)
            stage2_prior_config = _resolve_stage2_prior(config, model_spec)
            stage2_dual_prior_config = _resolve_stage2_dual_prior(config, model_spec)
            stage2_prior_train = residual_train_offset
            stage2_prior_test = residual_test_offset
            stage2_prior_summary = residual_target_summary
            stage2_prior_seconds = residual_target_seconds
            stage2_secondary_prior_summary = None
            if stage2_prior_config is not None and stage2_prior_train is None:
                prior_start = time.perf_counter()
                _, stage2_prior_test, stage2_prior_summary = _compute_residual_target_predictions(
                    residual_target_config=stage2_prior_config,
                    fold_dir=fold_dir,
                    X_train_base=X_train_base,
                    y_train=y_train,
                    X_test_base=X_test_base,
                    config=config,
                    fold_id=fold_id,
                )
                if model_spec.stage2_backend.lower() == "calibrated_correction" and bool(
                    model_spec.stage2_config.get("use_oof_gate_training", False)
                ):
                    stage2_prior_train, oof_summary = _compute_oof_baseline_prior_predictions(
                        baseline_model=str(stage2_prior_config["baseline_model"]),
                        fold_dir=fold_dir,
                        X_train_base=X_train_base,
                        y_train=y_train,
                        config=config,
                        fold_id=fold_id,
                        n_splits=int(model_spec.stage2_config.get("oof_splits", 3)),
                    )
                    stage2_prior_summary = {
                        **stage2_prior_summary,
                        "train_mode": "oof",
                        "oof_summary": oof_summary,
                    }
                else:
                    stage2_prior_train, _, train_summary = _compute_residual_target_predictions(
                        residual_target_config=stage2_prior_config,
                        fold_dir=fold_dir / "_stage2_prior_train_full",
                        X_train_base=X_train_base,
                        y_train=y_train,
                        X_test_base=X_train_base,
                        config=config,
                        fold_id=fold_id,
                    )
                    stage2_prior_summary = {
                        **stage2_prior_summary,
                        "train_mode": "full_train_fit",
                        "train_summary": train_summary,
                    }
                stage2_prior_seconds = time.perf_counter() - prior_start
            if stage2_dual_prior_config is not None:
                dual_prior_start = time.perf_counter()
                _, primary_test, stage2_prior_summary = _compute_residual_target_predictions(
                    residual_target_config={"baseline_model": stage2_dual_prior_config["primary_model"]},
                    fold_dir=fold_dir / "_dual_prior_primary",
                    X_train_base=X_train_base,
                    y_train=y_train,
                    X_test_base=X_test_base,
                    config=config,
                    fold_id=fold_id,
                )
                _, secondary_test, stage2_secondary_prior_summary = _compute_residual_target_predictions(
                    residual_target_config={"baseline_model": stage2_dual_prior_config["secondary_model"]},
                    fold_dir=fold_dir / "_dual_prior_secondary",
                    X_train_base=X_train_base,
                    y_train=y_train,
                    X_test_base=X_test_base,
                    config=config,
                    fold_id=fold_id,
                )
                if bool(model_spec.stage2_config.get("use_oof_gate_training", False)):
                    primary_train, primary_oof_summary = _compute_oof_baseline_prior_predictions(
                        baseline_model=str(stage2_dual_prior_config["primary_model"]),
                        fold_dir=fold_dir / "_dual_prior_primary",
                        X_train_base=X_train_base,
                        y_train=y_train,
                        config=config,
                        fold_id=fold_id,
                        n_splits=int(model_spec.stage2_config.get("oof_splits", 3)),
                    )
                    secondary_train, secondary_oof_summary = _compute_oof_baseline_prior_predictions(
                        baseline_model=str(stage2_dual_prior_config["secondary_model"]),
                        fold_dir=fold_dir / "_dual_prior_secondary",
                        X_train_base=X_train_base,
                        y_train=y_train,
                        config=config,
                        fold_id=fold_id,
                        n_splits=int(model_spec.stage2_config.get("oof_splits", 3)),
                    )
                    stage2_prior_summary = {
                        **stage2_prior_summary,
                        "train_mode": "oof",
                        "oof_summary": primary_oof_summary,
                    }
                    stage2_secondary_prior_summary = {
                        **stage2_secondary_prior_summary,
                        "train_mode": "oof",
                        "oof_summary": secondary_oof_summary,
                    }
                else:
                    primary_train, _, _ = _compute_residual_target_predictions(
                        residual_target_config={"baseline_model": stage2_dual_prior_config["primary_model"]},
                        fold_dir=fold_dir / "_dual_prior_primary_train_full",
                        X_train_base=X_train_base,
                        y_train=y_train,
                        X_test_base=X_train_base,
                        config=config,
                        fold_id=fold_id,
                    )
                    secondary_train, _, _ = _compute_residual_target_predictions(
                        residual_target_config={"baseline_model": stage2_dual_prior_config["secondary_model"]},
                        fold_dir=fold_dir / "_dual_prior_secondary_train_full",
                        X_train_base=X_train_base,
                        y_train=y_train,
                        X_test_base=X_train_base,
                        config=config,
                        fold_id=fold_id,
                    )
                stage2_prior_train = np.column_stack([primary_train, secondary_train]).astype(np.float32)
                stage2_prior_test = np.column_stack([primary_test, secondary_test]).astype(np.float32)
                stage2_prior_seconds = time.perf_counter() - dual_prior_start
            stage2_block_prior = None
            tuning_settings = _resolve_tuning_settings(config, model_spec.name)
            optuna_seconds = 0.0
            if tuning_settings is not None:
                optuna_start = time.perf_counter()
                study = _run_optuna(
                    tuning_config=tuning_settings,
                    model_spec=model_spec,
                    X_train=X_train,
                    y_train=y_train,
                    sampled_snp_ids=sampled_snp_ids,
                    strategy=config["grouping_strategy"],
                    seed=config["seed"] + fold_id * 1000 + model_offset * 100,
                    second_stage_adjustment_config=second_stage_adjustment,
                    collect_block_diagnostics=collect_block_diagnostics,
                    embedding_extraction_mode=embedding_extraction_mode,
                    stage2_feature_mode=stage2_feature_mode,
                    residual_train_target=residual_train_target,
                    train_prediction_offset=residual_train_offset,
                    stage2_prior_train=stage2_prior_train,
                )
                optuna_seconds = time.perf_counter() - optuna_start
                trial_evaluations = []
                for trial in study.trials:
                    evaluated = _evaluate_trial_params(
                        model_spec=model_spec,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        sampled_snp_ids=sampled_snp_ids,
                        strategy=config["grouping_strategy"],
                        seed=config["seed"] + fold_id * 1000 + model_offset * 100 + trial.number * 10000,
                        pad_incomplete_last_block=tuning_settings.get("pad_incomplete_last_block", True),
                        params=trial.params,
                        second_stage_adjustment_config=second_stage_adjustment,
                        collect_block_diagnostics=collect_block_diagnostics,
                        embedding_extraction_mode=embedding_extraction_mode,
                        stage2_feature_mode=stage2_feature_mode,
                        residual_train_target=residual_train_target,
                        train_prediction_offset=residual_train_offset,
                        test_prediction_offset=residual_test_offset,
                        stage2_prior_train=stage2_prior_train,
                        stage2_prior_test=stage2_prior_test,
                    )
                    evaluated.update(
                        {
                            "fold": fold_id,
                            "model": model_spec.name,
                            "trial_number": int(trial.number),
                            "is_best": bool(trial.number == study.best_trial.number),
                        }
                    )
                    trial_evaluations.append(evaluated)
                    fold_trial_metric_rows.append(
                        {
                            "fold": fold_id,
                            "model": model_spec.name,
                            "trial_number": int(trial.number),
                            "group_size": evaluated["group_size"],
                            "block_embedding_dim": evaluated["block_embedding_dim"],
                            "include_block_scalar": evaluated["include_block_scalar"],
                            "train_pearson": evaluated["train_pearson"],
                            "train_r2": evaluated["train_r2"],
                            "test_pearson": evaluated["test_pearson"],
                            "test_r2": evaluated["test_r2"],
                            "trial_total_seconds": evaluated["trial_total_seconds"],
                            "is_best": evaluated["is_best"],
                        }
                    )
                best_eval = next(record for record in trial_evaluations if record["is_best"])
                chosen_group_size = int(best_eval["group_size"])
                chosen_embedding_dim = int(best_eval["block_embedding_dim"])
                chosen_include_block_scalar = bool(best_eval["include_block_scalar"])
                tuning_result = {
                    "best_params": {
                        "group_size": chosen_group_size,
                        "block_embedding_dim": chosen_embedding_dim,
                        "include_block_scalar": chosen_include_block_scalar,
                    },
                    "best_value": float(study.best_value),
                    "objective_metric": tuning_settings.get("objective_metric", "pearson"),
                    "n_trials": len(study.trials),
                    "direction": study.direction.name.lower(),
                }
            else:
                tuning_result = None
                chosen_group_size = int(config["group_size"])
                stage1_embedding = model_spec.stage1_config.get("embedding_reduce_dim")
                chosen_embedding_dim = None if stage1_embedding is None else int(stage1_embedding)
                chosen_include_block_scalar = bool(config.get("include_block_scalar", False))

            if tuning_settings is not None:
                stage2_eval = best_eval
                stage2_device = stage2_eval["stage2_device"]
                raw_block_embedding_dim = stage2_eval["raw_block_embedding_dim"]
                reduced_block_embedding_dim = stage2_eval["reduced_block_embedding_dim"]
                mean_reduced_block_embedding_dim = stage2_eval["mean_reduced_block_embedding_dim"]
                min_reduced_block_embedding_dim = stage2_eval["min_reduced_block_embedding_dim"]
                max_reduced_block_embedding_dim = stage2_eval["max_reduced_block_embedding_dim"]
                mean_explained_variance_ratio_sum = stage2_eval["mean_explained_variance_ratio_sum"]
                stage2_input_dim = stage2_eval["stage2_input_dim"]
                num_blocks = stage2_eval["num_blocks"]
                test_pearson = stage2_eval["test_pearson"]
                test_r2 = stage2_eval["test_r2"]
                stage2_model_class = stage2_eval["stage2_model_class"]
                stage1_seconds = stage2_eval["stage1_seconds"]
                stage2_seconds = stage2_eval["stage2_seconds"]
            else:
                stage1_start = time.perf_counter()
                X_train_stage2, X_test_stage2, block_summaries, num_blocks = _build_stage_features(
                    model_spec=model_spec,
                    X_train=X_train,
                    y_train=y_train if residual_train_target is None else residual_train_target,
                    sampled_snp_ids=sampled_snp_ids,
                    strategy=config["grouping_strategy"],
                    group_size=chosen_group_size,
                    seed=config["seed"] + fold_id * 1000 + model_offset * 100,
                    pad_incomplete_last_block=config.get("pad_incomplete_last_block", False),
                    embedding_reduce_dim=chosen_embedding_dim,
                    include_block_scalar=chosen_include_block_scalar,
                    second_stage_adjustment_config=second_stage_adjustment,
                    collect_block_diagnostics=collect_block_diagnostics,
                    embedding_extraction_mode=embedding_extraction_mode,
                    stage2_feature_mode=stage2_feature_mode,
                    X_eval=X_test,
                )
                if model_spec.stage2_backend.lower() == "block_attention" and bool(model_spec.stage2_config.get("use_block_prior", False)):
                    beta_result = run_r_baseline(
                        model_name=str(stage2_prior_config["baseline_model"]) if stage2_prior_config is not None else "BayesB",
                        X_train=X_train_base,
                        y_train=y_train,
                        X_test=X_test_base,
                        output_dir=fold_dir / "_stage2_block_prior" / "train_fit",
                        rscript_path=config["baselines"]["rscript_path"],
                        seed=config["seed"] + fold_id,
                        sommer_method=config["baselines"].get("sommer_method"),
                        keep_artifacts=True,
                        return_beta=True,
                    )
                    if beta_result.beta is None:
                        raise ValueError("use_block_prior=True requires baseline beta coefficients.")
                    stage2_block_prior = aggregate_beta_to_block_prior(
                        beta=beta_result.beta,
                        block_summaries=block_summaries,
                        method=str(model_spec.stage2_config.get("block_prior_method", "l2")),
                    )
                elif model_spec.stage2_backend.lower() == "tabicl" and bool(model_spec.stage2_config.get("use_block_prior", False)):
                    beta_result = run_r_baseline(
                        model_name=str(stage2_prior_config["baseline_model"]) if stage2_prior_config is not None else "BayesB",
                        X_train=X_train_base,
                        y_train=y_train,
                        X_test=X_test_base,
                        output_dir=fold_dir / "_stage2_block_prior" / "train_fit",
                        rscript_path=config["baselines"]["rscript_path"],
                        seed=config["seed"] + fold_id,
                        sommer_method=config["baselines"].get("sommer_method"),
                        keep_artifacts=True,
                        return_beta=True,
                    )
                    if beta_result.beta is None:
                        raise ValueError("use_block_prior=True requires baseline beta coefficients.")
                    block_prior_mode = str(model_spec.stage2_config.get("block_prior_mode", "sample_prediction"))
                    if block_prior_mode == "sample_prediction":
                        stage2_block_prior = {
                            "train": compute_sample_block_prior_predictions(
                                X_train_base,
                                beta_result.beta,
                                block_summaries,
                                normalize=True,
                            ),
                            "test": compute_sample_block_prior_predictions(
                                X_test_base,
                                beta_result.beta,
                                block_summaries,
                                normalize=True,
                            ),
                        }
                    elif block_prior_mode == "local_bayesb_prediction":
                        train_block_prior_predictions: list[np.ndarray] = []
                        test_block_prior_predictions: list[np.ndarray] = []
                        for block_summary in block_summaries:
                            block_indices = [int(idx) for idx in block_summary.get("snp_indices", [])]
                            X_train_block_prior = X_train_base[:, block_indices]
                            X_test_block_prior = X_test_base[:, block_indices]
                            block_prior_dir = fold_dir / "_stage2_block_prior" / f"block_{int(block_summary['block_id']):03d}"
                            block_train_fit = run_r_baseline(
                                model_name=str(stage2_prior_config["baseline_model"]) if stage2_prior_config is not None else "BayesB",
                                X_train=X_train_block_prior,
                                y_train=y_train,
                                X_test=X_train_block_prior,
                                output_dir=block_prior_dir / "train_fit",
                                rscript_path=config["baselines"]["rscript_path"],
                                seed=config["seed"] + fold_id + int(block_summary["block_id"]),
                                sommer_method=config["baselines"].get("sommer_method"),
                                keep_artifacts=True,
                            )
                            block_test_fit = run_r_baseline(
                                model_name=str(stage2_prior_config["baseline_model"]) if stage2_prior_config is not None else "BayesB",
                                X_train=X_train_block_prior,
                                y_train=y_train,
                                X_test=X_test_block_prior,
                                output_dir=block_prior_dir / "test_fit",
                                rscript_path=config["baselines"]["rscript_path"],
                                seed=config["seed"] + fold_id + int(block_summary["block_id"]),
                                sommer_method=config["baselines"].get("sommer_method"),
                                keep_artifacts=True,
                            )
                            train_block_prior_predictions.append(np.asarray(block_train_fit.predictions, dtype=np.float32))
                            test_block_prior_predictions.append(np.asarray(block_test_fit.predictions, dtype=np.float32))
                        train_block_prior, test_block_prior = build_block_level_prior_matrix(
                            train_block_prior_predictions,
                            test_block_prior_predictions,
                            normalize=True,
                        )
                        stage2_block_prior = {
                            "train": train_block_prior,
                            "test": test_block_prior,
                        }
                    elif block_prior_mode == "global_beta_summary":
                        stage2_block_prior = aggregate_beta_to_block_prior(
                            beta=beta_result.beta,
                            block_summaries=block_summaries,
                            method=str(model_spec.stage2_config.get("block_prior_method", "l2")),
                        )
                    else:
                        raise ValueError(f"Unsupported block_prior_mode: {block_prior_mode}")
                X_train_stage2, X_test_stage2 = _append_stage2_prior_feature(
                    model_spec=model_spec,
                    X_train_stage2=X_train_stage2,
                    X_test_stage2=X_test_stage2,
                    prior_train=stage2_prior_train,
                    prior_test=stage2_prior_test,
                )
                if model_spec.stage2_backend.lower() == "tabicl" and bool(model_spec.stage2_config.get("use_block_prior", False)):
                    train_block_prior = stage2_block_prior["train"] if isinstance(stage2_block_prior, dict) else stage2_block_prior
                    test_block_prior = stage2_block_prior["test"] if isinstance(stage2_block_prior, dict) else stage2_block_prior
                    X_train_stage2 = interleave_block_prior_features(
                        X_train_stage2,
                        block_summaries=block_summaries,
                        block_prior=train_block_prior,
                        feature_mode=stage2_feature_mode,
                    )
                    X_test_stage2 = interleave_block_prior_features(
                        X_test_stage2,
                        block_summaries=block_summaries,
                        block_prior=test_block_prior,
                        feature_mode=stage2_feature_mode,
                    )
                X_train_stage2, X_test_stage2 = _append_stage2_block_prior_feature(
                    model_spec=model_spec,
                    X_train_stage2=X_train_stage2,
                    X_test_stage2=X_test_stage2,
                    block_prior=stage2_block_prior,
                )
                stage1_seconds = time.perf_counter() - stage1_start
                stage2_start = time.perf_counter()
                stage2_config = _prepare_stage2_config(model_spec, block_summaries, chosen_include_block_scalar)
                if model_spec.stage2_backend.lower() in {"static_block_weight", "group_weight_pooling", "group_shared_gate"}:
                    beta_result = run_r_baseline(
                        model_name="BayesB",
                        X_train=X_train_base,
                        y_train=y_train,
                        X_test=X_test_base,
                        output_dir=fold_dir / "_stage2_block_weight_prior" / "train_fit",
                        rscript_path=config["baselines"]["rscript_path"],
                        seed=config["seed"] + fold_id,
                        sommer_method=config["baselines"].get("sommer_method"),
                        keep_artifacts=True,
                        return_beta=True,
                    )
                    if beta_result.beta is None:
                        raise ValueError("block-weight backends require BayesB beta coefficients for prior initialization.")
                    stage2_config["prior_scores"] = aggregate_beta_to_block_prior(
                        beta=beta_result.beta,
                        block_summaries=block_summaries,
                        method="l2",
                    ).tolist()
                stage2_model, stage2_pred, stage2_device = fit_stage2_model(
                    model_spec.stage2_backend,
                    X_train_stage2,
                    y_train if residual_train_target is None else residual_train_target,
                    X_test_stage2,
                    stage2_config,
                    config["seed"] + fold_id + model_offset,
                )
                stage2_seconds = time.perf_counter() - stage2_start
                if residual_test_offset is not None:
                    stage2_pred = stage2_pred + residual_test_offset
                predictions_payload[f"{model_spec.name}_pred"] = stage2_pred
                stage2_metrics = regression_metrics(y_test, stage2_pred)
                block_dim_summary = _summarize_block_dimensions(block_summaries)
                raw_block_embedding_dim = block_dim_summary["raw_block_embedding_dim"]
                reduced_block_embedding_dim = block_dim_summary["reduced_block_embedding_dim"]
                mean_reduced_block_embedding_dim = block_dim_summary["mean_reduced_block_embedding_dim"]
                min_reduced_block_embedding_dim = block_dim_summary["min_reduced_block_embedding_dim"]
                max_reduced_block_embedding_dim = block_dim_summary["max_reduced_block_embedding_dim"]
                mean_explained_variance_ratio_sum = block_dim_summary["mean_explained_variance_ratio_sum"]
                stage2_input_dim = int(X_train_stage2.shape[1])
                test_pearson = float(stage2_metrics["pearson"])
                test_r2 = float(stage2_metrics["r2"])
                stage2_model_class = stage2_model.__class__.__name__
                group_summary = stage2_model.get_group_summary() if hasattr(stage2_model, "get_group_summary") else None
            metrics_rows.append(
                {
                    "fold": fold_id,
                    "model": model_spec.name,
                    "strategy": config["grouping_strategy"],
                    "stage1_backend": model_spec.stage1_backend,
                    "stage2_backend": model_spec.stage2_backend,
                    "group_size": chosen_group_size,
                    "num_blocks": num_blocks,
                    "include_block_scalar": chosen_include_block_scalar,
                    "raw_block_embedding_dim": raw_block_embedding_dim,
                    "reduced_block_embedding_dim": reduced_block_embedding_dim,
                    "mean_reduced_block_embedding_dim": mean_reduced_block_embedding_dim,
                    "min_reduced_block_embedding_dim": min_reduced_block_embedding_dim,
                    "max_reduced_block_embedding_dim": max_reduced_block_embedding_dim,
                    "mean_explained_variance_ratio_sum": mean_explained_variance_ratio_sum,
                    "stage2_input_dim": stage2_input_dim,
                    "device": stage2_device,
                    "cuda_visible_devices": runtime_device.get("cuda_visible_devices", ""),
                    "pearson": test_pearson,
                    "r2": test_r2,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "group_counts": "" if group_summary is None else str(group_summary.get("group_counts", [])),
                }
            )
            model_run_summaries[model_spec.name] = {
                "stage1_backend": model_spec.stage1_backend,
                "stage2_backend": model_spec.stage2_backend,
                "group_size": chosen_group_size,
                "num_blocks": num_blocks,
                "include_block_scalar": chosen_include_block_scalar,
                "raw_block_embedding_dim": raw_block_embedding_dim,
                "reduced_block_embedding_dim": reduced_block_embedding_dim,
                "mean_reduced_block_embedding_dim": mean_reduced_block_embedding_dim,
                "min_reduced_block_embedding_dim": min_reduced_block_embedding_dim,
                "max_reduced_block_embedding_dim": max_reduced_block_embedding_dim,
                "mean_explained_variance_ratio_sum": mean_explained_variance_ratio_sum,
                "stage2_input_dim": stage2_input_dim,
                "stage2_device": stage2_device,
                "stage2_model_class": stage2_model_class,
                "group_summary": group_summary,
                "tuning_result": tuning_result,
                "residual_target": residual_target_summary,
                "residual_target_seconds": residual_target_seconds,
                "stage2_prior": stage2_prior_summary,
                "stage2_prior_seconds": stage2_prior_seconds,
                "block_summaries": block_summaries if (not fold_trial_metric_rows and config.get("save_block_summaries", True)) else None,
                "stage1_seconds": stage1_seconds,
                "stage2_seconds": stage2_seconds,
                "optuna_seconds": optuna_seconds,
            }
            if (
                tuning_settings is None
                and len(model_specs) == 1
                and not fold_trial_metric_rows
                and _has_tabicl_inner_oof_source(model_spec)
                and bool(tabicl_inner_oof_settings["enabled"])
                and (
                    tabicl_inner_oof_settings["fold"] is None
                    or int(fold_id) == int(tabicl_inner_oof_settings["fold"])
                )
            ):
                tabicl_oof_pred, tabicl_oof_summary = _compute_fixed_block_tabicl_inner_oof(
                    config=config,
                    model_spec=model_spec,
                    X_train=X_train,
                    y_train=y_train,
                    sampled_snp_ids=sampled_snp_ids,
                    fold_id=fold_id,
                    model_offset=model_offset,
                    chosen_group_size=chosen_group_size,
                    chosen_embedding_dim=chosen_embedding_dim,
                    chosen_include_block_scalar=chosen_include_block_scalar,
                    second_stage_adjustment=second_stage_adjustment,
                    embedding_extraction_mode=embedding_extraction_mode,
                    stage2_feature_mode=stage2_feature_mode,
                    stage2_prior_train=stage2_prior_train,
                )
                _save_tabicl_inner_oof_bundle(
                    fold_dir,
                    y_true=y_train,
                    y_pred=tabicl_oof_pred,
                    metadata={
                        **tabicl_oof_summary,
                        "model": model_spec.name,
                        "stage1_backend": model_spec.stage1_backend,
                        "stage2_backend": model_spec.stage2_backend,
                    },
                )
                model_run_summaries[model_spec.name]["tabicl_inner_oof"] = {
                    "prediction_path": str(fold_dir / "tabicl_inner_oof_predictions.npy"),
                    "target_path": str(fold_dir / "tabicl_inner_oof_targets.npy"),
                    "summary_path": str(fold_dir / "tabicl_inner_oof_summary.json"),
                }
            timing_rows.append(
                {
                    "fold": fold_id,
                    "model": model_spec.name,
                    "stage1_seconds": stage1_seconds,
                    "stage2_seconds": stage2_seconds,
                    "optuna_seconds": optuna_seconds,
                    "baseline_seconds": np.nan,
                    "total_fold_seconds": np.nan,
                }
            )

        baseline_key_map = {
            "GBLUP": "gblup",
            "BayesA": "bayesA",
            "BayesB": "bayesB",
            "BayesLasso": "bayesLasso",
            "RKHS": "rkhs",
        }
        for baseline_name in ("GBLUP", "BayesA", "BayesB", "BayesLasso", "RKHS"):
            if not config["baselines"].get(baseline_key_map[baseline_name], False):
                continue
            baseline_dir = fold_dir / baseline_name
            baseline_start = time.perf_counter()
            result = run_r_baseline(
                model_name=baseline_name,
                X_train=X_train_base,
                y_train=y_train,
                X_test=X_test_base,
                output_dir=baseline_dir,
                rscript_path=config["baselines"]["rscript_path"],
                seed=config["seed"] + fold_id,
                sommer_method=config["baselines"].get("sommer_method"),
                keep_artifacts=not bool(fold_trial_metric_rows),
            )
            baseline_seconds_total += time.perf_counter() - baseline_start
            baseline_run_summaries[baseline_name] = {
                "output_dir": str(baseline_dir),
            }
            if (
                baseline_inner_oof is not None
                and int(fold_id) == int(baseline_inner_oof["fold"])
                and baseline_name in set(baseline_inner_oof["models"])
            ):
                baseline_oof_pred, baseline_oof_summary = _compute_oof_baseline_prior_predictions(
                    baseline_model=baseline_name,
                    fold_dir=baseline_dir / "_inner_oof",
                    X_train_base=X_train_base,
                    y_train=y_train,
                    config=config,
                    fold_id=fold_id,
                    n_splits=int(baseline_inner_oof["n_splits"]),
                )
                baseline_oof_metrics = regression_metrics(y_train, baseline_oof_pred)
                _save_baseline_inner_oof_bundle(
                    baseline_dir,
                    baseline_name=baseline_name,
                    y_true=y_train,
                    y_pred=baseline_oof_pred,
                    metadata={
                        **baseline_oof_summary,
                        "source": "baseline_inner_oof",
                        "fold": int(fold_id),
                        "inner_oof_pearson": float(baseline_oof_metrics["pearson"]),
                        "inner_oof_r2": float(baseline_oof_metrics["r2"]),
                    },
                )
                baseline_run_summaries[baseline_name]["inner_oof"] = {
                    "prediction_path": str(baseline_dir / "inner_oof_predictions.npy"),
                    "target_path": str(baseline_dir / "inner_oof_targets.npy"),
                    "summary_path": str(baseline_dir / "inner_oof_summary.json"),
                }
            metrics_rows.append(
                {
                    "fold": fold_id,
                    "model": baseline_name,
                    "strategy": config["grouping_strategy"],
                    "stage1_backend": "baseline",
                    "stage2_backend": "baseline",
                    "raw_block_embedding_dim": None,
                    "reduced_block_embedding_dim": None,
                    "mean_reduced_block_embedding_dim": None,
                    "min_reduced_block_embedding_dim": None,
                    "max_reduced_block_embedding_dim": None,
                    "mean_explained_variance_ratio_sum": None,
                    "stage2_input_dim": None,
                    "device": result.metadata.get("device", "R"),
                    "cuda_visible_devices": runtime_device.get("cuda_visible_devices", ""),
                    **regression_metrics(y_test, result.predictions),
                }
            )
            baseline_metrics = regression_metrics(y_test, result.predictions)
            baseline_fold_metric_rows.append(
                {
                    "fold": fold_id,
                    "model": baseline_name,
                    "test_pearson": float(baseline_metrics["pearson"]),
                    "test_r2": float(baseline_metrics["r2"]),
                }
            )

        if fold_trial_metric_rows:
            pd.DataFrame(fold_trial_metric_rows).to_csv(fold_dir / "trial_metrics.csv", index=False)
            pd.DataFrame(baseline_fold_metric_rows).to_csv(fold_dir / "baseline_test_metrics.csv", index=False)
        else:
            predictions_frame = pd.DataFrame(predictions_payload)
            predictions_frame.to_csv(fold_dir / "tabicl_predictions.csv", index=False)
            _save_json(
                fold_dir / "fold_metadata.json",
                {
                    "fold": fold_id,
                    "grouping_strategy": config["grouping_strategy"],
                    "selected_snp_count": len(selected_snp_indices),
                    "selected_snp_indices": selected_snp_indices,
                    "selected_snp_ids": sampled_snp_ids,
                    "original_n_train": original_n_train,
                    "effective_n_train": int(X_train.shape[0]),
                    "sample_size_override": sample_override_summary,
                    "model_runs": model_run_summaries,
                    "baseline_runs": baseline_run_summaries,
                    "runtime_device": runtime_device,
                },
            )

        fold_total_seconds = time.perf_counter() - fold_start
        timing_rows.append(
            {
                "fold": fold_id,
                "model": "baselines",
                "stage1_seconds": np.nan,
                "stage2_seconds": np.nan,
                "optuna_seconds": np.nan,
                "baseline_seconds": baseline_seconds_total,
                "total_fold_seconds": fold_total_seconds,
            }
        )

    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame.to_csv(output_dir / "fold_metrics.csv", index=False)
    if timing_rows:
        pd.DataFrame(timing_rows).to_csv(output_dir / "timing_summary.csv", index=False)
    return metrics_frame
