from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _normalize_proportions(proportions: Iterable[float]) -> list[float]:
    return [float(p) for p in proportions]


def build_nested_train_subsets(
    *,
    train_indices: np.ndarray,
    proportions: Iterable[float],
    repeats: int,
    seed: int,
) -> dict[tuple[int, float], np.ndarray]:
    proportions = sorted(_normalize_proportions(proportions))
    train_indices = np.asarray(train_indices, dtype=np.int64).reshape(-1)
    rng = np.random.default_rng(int(seed))
    subsets: dict[tuple[int, float], np.ndarray] = {}
    for repeat in range(1, int(repeats) + 1):
        permuted = np.asarray(train_indices[rng.permutation(train_indices.shape[0])], dtype=np.int64)
        for proportion in proportions:
            n_keep = max(1, int(round(train_indices.shape[0] * float(proportion))))
            if float(proportion) >= 1.0:
                n_keep = int(train_indices.shape[0])
            chosen = np.sort(permuted[:n_keep].copy())
            subsets[(int(repeat), float(proportion))] = chosen.astype(np.int64)
    return subsets


def should_run_block_search(proportion: float, search_proportions: Iterable[float] | None, atol: float = 1e-8) -> bool:
    if not search_proportions:
        return False
    return any(np.isclose(float(proportion), float(candidate), atol=atol) for candidate in search_proportions)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _trapezoid_area(y: np.ndarray, x: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if y.shape[0] != x.shape[0]:
        raise ValueError("x and y must have the same length for trapezoid integration.")
    if y.shape[0] < 2:
        return float("nan")
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    widths = x[1:] - x[:-1]
    heights = 0.5 * (y[1:] + y[:-1])
    return float(np.sum(widths * heights, dtype=np.float32))


def build_block_search_config(base_config: dict) -> dict:
    config = deepcopy(base_config)
    config["main_models"] = [{"name": "TabICLv2-2stage", "stage1_backend": "tabicl", "stage2_backend": "tabicl"}]
    config["grouping_strategy"] = "window"
    config["include_block_scalar"] = False
    config["collect_block_diagnostics"] = False
    config["embedding_extraction_mode"] = str(config.get("embedding_extraction_mode", "legacy"))
    config["stage2"] = {
        "tabicl": {
            "n_estimators": int(config.get("stage1", {}).get("tabicl", {}).get("n_estimators", 1)),
            "norm_methods": list(config.get("stage1", {}).get("tabicl", {}).get("norm_methods", ["none"])),
            "feat_shuffle_method": str(config.get("stage1", {}).get("tabicl", {}).get("feat_shuffle_method", "none")),
            "batch_size": config.get("stage1", {}).get("tabicl", {}).get("batch_size", 1),
            "checkpoint_version": str(
                config.get("stage1", {}).get("tabicl", {}).get("checkpoint_version", "tabicl-regressor-v2-20260212.ckpt")
            ),
            "device": str(config.get("stage1", {}).get("tabicl", {}).get("device", "cuda")),
        }
    }
    if "baselines" in config:
        for key in ["gblup", "bayesA", "bayesB", "bayesLasso"]:
            config["baselines"][key] = False
    return config


def run_tabicl_block_search_on_subset(
    *,
    base_config: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    output_root: str | Path,
    min_block: int,
    max_block: int,
    n_trials: int,
    inner_folds: int,
    seed: int,
) -> pd.DataFrame:
    import optuna

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

    config = build_block_search_config(base_config)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    sampled_snp_ids = [f"snp_{i}" for i in range(X_train.shape[1])]
    inner_splits = make_outer_cv_splits(X_train, int(inner_folds), int(seed))
    rows: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        group_size = int(trial.suggest_int("group_size", int(min_block), int(max_block)))
        trial_config = deepcopy(config)
        trial_config["group_size"] = int(group_size)
        model_spec = resolve_two_stage_model_specs(trial_config)[0]
        oof_pred = np.zeros_like(y_train, dtype=np.float32)
        reduced_dims = []
        feature_dims = []

        for inner_id, (inner_train_idx, inner_valid_idx) in enumerate(inner_splits, start=1):
            X_inner_train = X_train[inner_train_idx]
            y_inner_train = y_train[inner_train_idx]
            X_inner_valid = X_train[inner_valid_idx]
            X_train_stage2, X_valid_stage2, block_summaries, _ = _build_stage_features(
                model_spec=model_spec,
                X_train=X_inner_train,
                y_train=y_inner_train,
                sampled_snp_ids=sampled_snp_ids,
                strategy=trial_config["grouping_strategy"],
                group_size=group_size,
                seed=int(seed) + int(trial.number) * 1000 + int(inner_id),
                pad_incomplete_last_block=bool(trial_config.get("pad_incomplete_last_block", True)),
                embedding_reduce_dim=None,
                include_block_scalar=bool(trial_config.get("include_block_scalar", False)),
                second_stage_adjustment_config=_resolve_second_stage_adjustment(trial_config, model_spec.name),
                collect_block_diagnostics=False,
                embedding_extraction_mode=str(trial_config.get("embedding_extraction_mode", "legacy")),
                stage2_feature_mode=_resolve_stage2_feature_mode(model_spec),
                X_eval=X_inner_valid,
            )
            stage2_cfg = _prepare_stage2_config(model_spec, block_summaries, include_block_scalar=False)
            _, valid_pred, _ = fit_stage2_model(
                model_spec.stage2_backend,
                X_train_stage2,
                y_inner_train,
                X_valid_stage2,
                stage2_cfg,
                int(seed) + int(trial.number) * 10000 + int(inner_id),
            )
            oof_pred[inner_valid_idx] = np.asarray(valid_pred, dtype=np.float32)
            reduced_dims.append(float(np.mean([b["reduced_embedding_dim"] for b in block_summaries])))
            feature_dims.append(int(X_train_stage2.shape[1]))

        metric = regression_metrics(y_train, oof_pred)
        row = {
            "trial": int(trial.number),
            "group_size": int(group_size),
            "inner_oof_pearson": float(metric["pearson"]),
            "inner_oof_r2": float(metric["r2"]),
            "mean_reduced_block_embedding_dim": float(np.mean(reduced_dims)) if reduced_dims else np.nan,
            "feature_dim_mean": float(np.mean(feature_dims)) if feature_dims else np.nan,
        }
        rows.append(row)
        pd.DataFrame(rows).sort_values("inner_oof_pearson", ascending=False).to_csv(output_root / "fold1_block_search_tabicl.csv", index=False)
        return float(row["inner_oof_pearson"])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=int(seed)))
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    summary = pd.DataFrame(rows).sort_values("inner_oof_pearson", ascending=False).reset_index(drop=True)
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
    return summary


def summarize_sample_size_metrics(
    fold_metrics: pd.DataFrame,
    *,
    reference_by_model: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    frame = fold_metrics.copy()
    grouped = (
        frame.groupby(["model", "proportion"], as_index=False)["test_pearson"]
        .mean()
        .sort_values(["model", "proportion"])
        .reset_index(drop=True)
    )
    rows: list[dict[str, float | str]] = []
    for model, model_frame in grouped.groupby("model"):
        baseline = None
        if reference_by_model is not None and str(model) in reference_by_model:
            baseline = float(reference_by_model[str(model)])
        else:
            baseline_rows = model_frame.loc[np.isclose(model_frame["proportion"], 1.0), "test_pearson"]
            if not baseline_rows.empty:
                baseline = float(baseline_rows.iloc[0])
        for _, row in model_frame.iterrows():
            retention = np.nan
            if baseline is not None and baseline != 0.0:
                retention = float(row["test_pearson"]) / baseline
            rows.append(
                {
                    "model": str(model),
                    "proportion": float(row["proportion"]),
                    "mean_test_pearson": float(row["test_pearson"]),
                    "retention": retention,
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "proportion"]).reset_index(drop=True)


def compute_trait_level_sample_size_summary(fold_metrics: pd.DataFrame) -> dict[str, float]:
    grouped = (
        fold_metrics.groupby(["model", "proportion"], as_index=False)["test_pearson"]
        .mean()
        .sort_values(["model", "proportion"])
        .reset_index(drop=True)
    )
    dual = grouped[grouped["model"] == "dual_prior"].set_index("proportion")["test_pearson"]
    prior = grouped[grouped["model"] == "prior_only"].set_index("proportion")["test_pearson"]
    gap = (dual - prior).sort_index()
    retention = summarize_sample_size_metrics(fold_metrics)
    dual_retention = retention[retention["model"] == "dual_prior"].sort_values("proportion")
    proportions = dual_retention["proportion"].to_numpy(dtype=np.float32)
    ret = dual_retention["retention"].to_numpy(dtype=np.float32)
    available_proportions = [float(p) for p in gap.index.tolist()]
    small = [float(p) for p in available_proportions if float(p) <= 0.4]
    large = [float(p) for p in available_proportions if float(p) >= 0.8]
    summary: dict[str, float | list[float]] = {
        "available_proportions": available_proportions,
        "small_n_amplification": float(np.nanmean([gap.get(p, np.nan) for p in small]) - np.nanmean([gap.get(p, np.nan) for p in large]))
        if small and large
        else float("nan"),
        "dual_auc_retention": _trapezoid_area(ret, proportions),
    }
    for proportion in available_proportions:
        summary[f"gap_{proportion:.2f}"] = float(gap.get(proportion, np.nan))
        normalized = f"{proportion:g}"
        summary[f"gap_{normalized}"] = float(gap.get(proportion, np.nan))
        summary[f"gap_{proportion}"] = float(gap.get(proportion, np.nan))
    return summary
