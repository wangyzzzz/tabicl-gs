from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tabicl_gs.data.plink import (
    align_phenotype_to_sample_ids,
    impute_by_train_mean,
    load_plink_matrix,
    plink_num_snps,
    read_phenotype_table,
)
from tabicl_gs.data.grouping import subsample_snp_indices
from tabicl_gs.eval.metrics import regression_metrics
from tabicl_gs.eval.splits import make_outer_cv_splits
from tabicl_gs.pipeline.experiment import run_experiment


def build_search_combinations(group_sizes: list[int], variance_target_pcts: list[int]) -> list[dict[str, int | float]]:
    return [
        {
            "group_size": int(group_size),
            "variance_target": float(variance_pct) / 100.0,
            "variance_target_pct": int(variance_pct),
        }
        for group_size in group_sizes
        for variance_pct in variance_target_pcts
    ]


def build_fold1_config(
    base_config: dict[str, Any],
    output_dir: str,
    group_size: int,
    variance_target: float,
    use_oof_gate_training: bool,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    config["group_size"] = int(group_size)
    config["output_dir"] = output_dir
    stage1_cfg = config["stage1"]["tabicl"] if "tabicl" in config.get("stage1", {}) else config["stage1"]
    stage1_cfg["embedding_reduce_dim"] = None
    stage1_cfg["embedding_explained_variance_target"] = float(variance_target)
    stage1_cfg["track_full_explained_variance"] = False
    stage2_cfg = config["stage2"]["calibrated_correction"]
    stage2_cfg["use_oof_gate_training"] = bool(use_oof_gate_training)
    return config


def _extract_model_row(metrics: pd.DataFrame, model_name: str) -> dict[str, Any]:
    rows = metrics[metrics["model"] == model_name]
    if rows.empty:
        raise ValueError(f"Could not find model row: {model_name}")
    return rows.iloc[0].to_dict()


def run_fold1_optuna_search(
    base_config: dict[str, Any],
    output_root: str | Path,
    group_sizes: list[int],
    variance_target_pcts: list[int],
    n_trials: int,
    seed: int,
    objective_metric: str = "pearson",
) -> pd.DataFrame:
    import optuna

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    combinations = build_search_combinations(group_sizes, variance_target_pcts)
    records: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        combo = trial.suggest_categorical("combo", list(range(len(combinations))))
        params = combinations[int(combo)]
        trial_dir = output_root / "trials" / f"trial_{trial.number:03d}_block{params['group_size']}_pc{params['variance_target_pct']}"
        config = build_fold1_config(
            base_config=base_config,
            output_dir=str(trial_dir),
            group_size=int(params["group_size"]),
            variance_target=float(params["variance_target"]),
            use_oof_gate_training=True,
        )
        metrics = run_experiment(config, max_folds=1)
        row = _extract_model_row(metrics, str(base_config["main_models"][0]["name"]))
        value = float(row[objective_metric])
        records.append(
            {
                "trial": int(trial.number),
                "group_size": int(params["group_size"]),
                "variance_target_pct": int(params["variance_target_pct"]),
                "variance_target": float(params["variance_target"]),
                "pearson": float(row["pearson"]),
                "r2": float(row["r2"]),
                "stage2_input_dim": int(row["stage2_input_dim"]),
                "mean_reduced_block_embedding_dim": float(row["mean_reduced_block_embedding_dim"]),
                "output_dir": str(trial_dir),
            }
        )
        pd.DataFrame(records).to_csv(output_root / "trial_results.csv", index=False)
        return value

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    results = pd.DataFrame(records)
    if not results.empty:
        results["is_best"] = results["trial"] == int(study.best_trial.number)
        results.to_csv(output_root / "trial_results.csv", index=False)
    return results


def rerun_best_oof_and_non_oof(
    base_config: dict[str, Any],
    output_root: str | Path,
    best_params: dict[str, Any],
) -> pd.DataFrame:
    output_root = Path(output_root)
    rows = []
    for label, use_oof in [("strict_oof", True), ("non_oof", False)]:
        run_dir = output_root / f"best_{label}"
        config = build_fold1_config(
            base_config=base_config,
            output_dir=str(run_dir),
            group_size=int(best_params["group_size"]),
            variance_target=float(best_params["variance_target"]),
            use_oof_gate_training=use_oof,
        )
        metrics = run_experiment(config, max_folds=1)
        row = _extract_model_row(metrics, str(base_config["main_models"][0]["name"]))
        rows.append(
            {
                "run": label,
                "group_size": int(best_params["group_size"]),
                "variance_target_pct": int(best_params["variance_target_pct"]),
                "variance_target": float(best_params["variance_target"]),
                "pearson": float(row["pearson"]),
                "r2": float(row["r2"]),
                "stage2_input_dim": int(row["stage2_input_dim"]),
                "mean_reduced_block_embedding_dim": float(row["mean_reduced_block_embedding_dim"]),
                "output_dir": str(run_dir),
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_root / "best_oof_vs_non_oof.csv", index=False)
    return comparison


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
    splits = make_outer_cv_splits(genotype, config["outer_cv_folds"], config["seed"])
    train_idx, test_idx = splits[int(fold_id) - 1]
    return genotype[train_idx], target[train_idx], genotype[test_idx], target[test_idx]


def run_inner_oof_hparam_search(
    base_config: dict[str, Any],
    output_root: str | Path,
    fold_id: int,
    group_sizes: list[int],
    variance_target_pcts: list[int],
    seed: int,
    inner_folds: int = 3,
) -> pd.DataFrame:
    from tabicl_gs.models.factory import fit_stage2_model
    from tabicl_gs.models.model_specs import resolve_two_stage_model_specs
    from tabicl_gs.pipeline.experiment import (
        _append_stage2_prior_feature,
        _build_stage_features,
        _compute_oof_baseline_prior_predictions,
        _compute_residual_target_predictions,
        _prepare_stage2_config,
        _resolve_second_stage_adjustment,
        _resolve_stage2_feature_mode,
    )

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    X_outer_train, y_outer_train, _, _ = _load_fold_data(base_config, fold_id=fold_id)
    X_outer_train_base, _ = impute_by_train_mean(X_outer_train, X_outer_train)
    model_spec = resolve_two_stage_model_specs(base_config)[0]
    records: list[dict[str, Any]] = []

    for params in build_search_combinations(group_sizes, variance_target_pcts):
        combo_dir = output_root / "inner_oof_trials" / f"block{params['group_size']}_pc{params['variance_target_pct']}"
        trial_config = build_fold1_config(
            base_config,
            output_dir=str(combo_dir),
            group_size=int(params["group_size"]),
            variance_target=float(params["variance_target"]),
            use_oof_gate_training=True,
        )
        stage1_cfg = trial_config["stage1"]["tabicl"]
        trial_model_spec = resolve_two_stage_model_specs(trial_config)[0]
        inner_splits = make_outer_cv_splits(X_outer_train, inner_folds, seed + int(fold_id))
        oof_pred = np.zeros_like(y_outer_train, dtype=np.float32)

        for inner_id, (inner_train_idx, inner_valid_idx) in enumerate(inner_splits, start=1):
            inner_dir = combo_dir / f"inner_{inner_id}"
            X_inner_train = X_outer_train[inner_train_idx]
            y_inner_train = y_outer_train[inner_train_idx]
            X_inner_valid = X_outer_train[inner_valid_idx]
            X_inner_train_base, X_inner_valid_base = impute_by_train_mean(X_inner_train, X_inner_valid)
            _, prior_valid, _ = _compute_residual_target_predictions(
                residual_target_config={"baseline_model": "GBLUP"},
                fold_dir=inner_dir / "prior_valid",
                X_train_base=X_inner_train_base,
                y_train=y_inner_train,
                X_test_base=X_inner_valid_base,
                config=trial_config,
                fold_id=inner_id,
            )
            prior_train_oof, _ = _compute_oof_baseline_prior_predictions(
                baseline_model="GBLUP",
                fold_dir=inner_dir / "prior_train_oof",
                X_train_base=X_inner_train_base,
                y_train=y_inner_train,
                config=trial_config,
                fold_id=inner_id,
                n_splits=inner_folds,
            )
            X_train_stage2, X_valid_stage2, block_summaries, _ = _build_stage_features(
                model_spec=trial_model_spec,
                X_train=X_inner_train,
                y_train=y_inner_train,
                sampled_snp_ids=[str(i) for i in range(X_inner_train.shape[1])],
                strategy=trial_config["grouping_strategy"],
                group_size=int(params["group_size"]),
                seed=trial_config["seed"] + int(fold_id) * 1000 + inner_id,
                pad_incomplete_last_block=trial_config.get("pad_incomplete_last_block", True),
                embedding_reduce_dim=None,
                include_block_scalar=bool(trial_config.get("include_block_scalar", False)),
                second_stage_adjustment_config=_resolve_second_stage_adjustment(trial_config, trial_model_spec.name),
                collect_block_diagnostics=bool(trial_config.get("collect_block_diagnostics", False)),
                embedding_extraction_mode=str(trial_config.get("embedding_extraction_mode", "legacy")),
                stage2_feature_mode=_resolve_stage2_feature_mode(trial_model_spec),
                X_eval=X_inner_valid,
            )
            X_train_stage2, X_valid_stage2 = _append_stage2_prior_feature(
                model_spec=trial_model_spec,
                X_train_stage2=X_train_stage2,
                X_test_stage2=X_valid_stage2,
                prior_train=prior_train_oof,
                prior_test=prior_valid,
            )
            stage2_config = _prepare_stage2_config(
                trial_model_spec,
                block_summaries,
                include_block_scalar=bool(trial_config.get("include_block_scalar", False)),
            )
            _, valid_pred, _ = fit_stage2_model(
                trial_model_spec.stage2_backend,
                X_train_stage2,
                y_inner_train,
                X_valid_stage2,
                stage2_config,
                seed + int(fold_id) * 10000 + inner_id,
            )
            oof_pred[inner_valid_idx] = valid_pred.astype(np.float32)

        metrics = regression_metrics(y_outer_train, oof_pred)
        records.append(
            {
                "fold": int(fold_id),
                "group_size": int(params["group_size"]),
                "variance_target_pct": int(params["variance_target_pct"]),
                "variance_target": float(params["variance_target"]),
                "inner_oof_pearson": float(metrics["pearson"]),
                "inner_oof_r2": float(metrics["r2"]),
                "embedding_explained_variance_target": float(stage1_cfg["embedding_explained_variance_target"]),
                "output_dir": str(combo_dir),
            }
        )
        pd.DataFrame(records).to_csv(output_root / f"fold_{fold_id}_inner_oof_results.csv", index=False)

    results = pd.DataFrame(records).sort_values("inner_oof_pearson", ascending=False)
    results.to_csv(output_root / f"fold_{fold_id}_inner_oof_results.csv", index=False)
    return results
