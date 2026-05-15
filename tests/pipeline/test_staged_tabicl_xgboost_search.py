from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tabicl_gs.pipeline.staged_tabicl_xgboost_search import (
    _load_cached_feature_folds,
    fit_ridge_oof_from_cached_folds,
    fit_xgboost_oof_from_cached_folds,
    fit_group_gate_from_best_xgboost_oof,
    run_staged_tabicl_xgboost_fixed_from_fold1_on_fold,
)


def _write_mock_cache(tmp_path: Path) -> tuple[list[dict], np.ndarray]:
    cache_root = tmp_path / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    y = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    np.save(cache_root / "y_outer_train.npy", y)
    (cache_root / "metadata.json").write_text(
        """
{
  "fold_id": 1,
  "group_size": 500,
  "oof_feature_dim_min": 2,
  "oof_feature_dim_max": 2,
  "num_samples": 4,
  "inner_folds": [
    {"inner_id": 1, "inner_train_size": 2, "inner_valid_size": 2, "num_blocks": 5, "stage2_input_dim": 2, "mean_reduced_block_embedding_dim": 3.0},
    {"inner_id": 2, "inner_train_size": 2, "inner_valid_size": 2, "num_blocks": 5, "stage2_input_dim": 2, "mean_reduced_block_embedding_dim": 3.0}
  ]
}
        """.strip(),
        encoding="utf-8",
    )
    inner1 = cache_root / "inner_1"
    inner2 = cache_root / "inner_2"
    inner1.mkdir()
    inner2.mkdir()
    np.save(inner1 / "train_features.npy", np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32))
    np.save(inner1 / "valid_features.npy", np.array([[2.0, 2.0], [3.0, 3.0]], dtype=np.float32))
    np.save(inner1 / "y_train.npy", np.array([0.0, 1.0], dtype=np.float32))
    np.save(inner1 / "valid_idx.npy", np.array([2, 3], dtype=np.int64))
    np.save(inner2 / "train_features.npy", np.array([[2.0, 2.0], [3.0, 3.0]], dtype=np.float32))
    np.save(inner2 / "valid_features.npy", np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32))
    np.save(inner2 / "y_train.npy", np.array([2.0, 3.0], dtype=np.float32))
    np.save(inner2 / "valid_idx.npy", np.array([0, 1], dtype=np.int64))
    return _load_cached_feature_folds(cache_root)["folds"], y


def test_fit_ridge_oof_from_cached_folds_returns_full_length_predictions(tmp_path: Path):
    folds, y = _write_mock_cache(tmp_path)
    pred = fit_ridge_oof_from_cached_folds(folds, y, alpha=1.0)

    assert pred.shape == y.shape
    assert np.isfinite(pred).all()


def test_fit_xgboost_oof_from_cached_folds_returns_full_length_predictions(tmp_path: Path):
    folds, y = _write_mock_cache(tmp_path)
    # XGBoost is fit independently per inner fold, so cross-fold feature widths may differ.
    folds[1]["train_features"] = np.concatenate(
        [folds[1]["train_features"], np.ones((folds[1]["train_features"].shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    folds[1]["valid_features"] = np.concatenate(
        [folds[1]["valid_features"], np.ones((folds[1]["valid_features"].shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    pred = fit_xgboost_oof_from_cached_folds(
        folds,
        y,
        {
            "n_estimators": 8,
            "max_depth": 2,
            "learning_rate": 0.1,
            "min_child_weight": 1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "tree_method": "hist",
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": 1,
            "device": "cpu",
        },
    )

    assert pred.shape == y.shape
    assert np.isfinite(pred).all()


def test_fit_group_gate_from_best_xgboost_oof_reuses_saved_trial_predictions(monkeypatch, tmp_path: Path):
    cache_root = tmp_path / "feature_cache"
    folds, y = _write_mock_cache(cache_root)
    _ = folds
    xgb_root = tmp_path / "xgboost_search"
    (xgb_root / "trials" / "trial_003").mkdir(parents=True, exist_ok=True)
    best_oof = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    np.save(xgb_root / "trials" / "trial_003" / "oof_predictions.npy", best_oof)
    (xgb_root / "best_xgboost.json").write_text(
        json.dumps({"trial": 3, "max_depth": 3, "n_estimators": 200}),
        encoding="utf-8",
    )

    def fake_precompute_dual_prior_cache(base_config, fold_id, cache_root, inner_folds):
        return {
            "bayesb_beta": np.array([1.0, 0.5, 0.2], dtype=np.float32),
            "inner_cache": [
                {
                    "inner_valid_idx": np.array([2, 3], dtype=np.int64),
                    "bayesb_valid": np.array([0.2, 0.3], dtype=np.float32),
                    "gblup_valid": np.array([0.1, 0.0], dtype=np.float32),
                },
                {
                    "inner_valid_idx": np.array([0, 1], dtype=np.int64),
                    "bayesb_valid": np.array([0.0, 0.1], dtype=np.float32),
                    "gblup_valid": np.array([0.3, 0.2], dtype=np.float32),
                },
            ],
        }

    def fake_build_group_features_from_stage2(X_stage2, block_summaries, group_mode, num_groups, prior_scores, include_block_scalar):
        return np.asarray(X_stage2[:, :1], dtype=np.float32), np.zeros(len(block_summaries), dtype=np.int64)

    monkeypatch.setattr(
        "tabicl_gs.pipeline.staged_tabicl_xgboost_search.precompute_dual_prior_cache",
        fake_precompute_dual_prior_cache,
    )
    monkeypatch.setattr(
        "tabicl_gs.pipeline.staged_tabicl_xgboost_search._build_group_features_from_stage2",
        fake_build_group_features_from_stage2,
    )
    monkeypatch.setattr(
        "tabicl_gs.pipeline.staged_tabicl_xgboost_search.aggregate_beta_to_block_prior",
        lambda beta, block_summaries, method="l2": np.array([0.0], dtype=np.float32),
    )

    block_summary = {
        "num_blocks": 1,
        "stage2_input_dim": 2,
        "mean_reduced_block_embedding_dim": 2.0,
        "block_summaries": [{"block_id": 1, "reduced_embedding_dim": 2, "snp_indices": [0, 1]}],
    }
    for inner_id in [1, 2]:
        inner_dir = cache_root / f"inner_{inner_id}"
        inner_dir.mkdir(parents=True, exist_ok=True)
        (inner_dir / "block_summary.json").write_text(json.dumps(block_summary), encoding="utf-8")
    (cache_root / "metadata.json").write_text(
        json.dumps(
            {
                "fold_id": 1,
                "group_size": 500,
                "oof_feature_dim_min": 2,
                "oof_feature_dim_max": 2,
                "num_samples": 4,
                "inner_folds": [
                    {"inner_id": 1, "inner_train_size": 2, "inner_valid_size": 2, "num_blocks": 1, "stage2_input_dim": 2, "mean_reduced_block_embedding_dim": 2.0},
                    {"inner_id": 2, "inner_train_size": 2, "inner_valid_size": 2, "num_blocks": 1, "stage2_input_dim": 2, "mean_reduced_block_embedding_dim": 2.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    np.save(cache_root / "y_outer_train.npy", y)
    np.save(cache_root / "inner_1" / "train_features.npy", np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32))
    np.save(cache_root / "inner_1" / "valid_features.npy", np.array([[2.0, 2.0], [3.0, 3.0]], dtype=np.float32))
    np.save(cache_root / "inner_1" / "y_train.npy", np.array([0.0, 1.0], dtype=np.float32))
    np.save(cache_root / "inner_1" / "valid_idx.npy", np.array([2, 3], dtype=np.int64))
    np.save(cache_root / "inner_2" / "train_features.npy", np.array([[2.0, 2.0], [3.0, 3.0]], dtype=np.float32))
    np.save(cache_root / "inner_2" / "valid_features.npy", np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32))
    np.save(cache_root / "inner_2" / "y_train.npy", np.array([2.0, 3.0], dtype=np.float32))
    np.save(cache_root / "inner_2" / "valid_idx.npy", np.array([0, 1], dtype=np.int64))

    summary = fit_group_gate_from_best_xgboost_oof(
        base_config={
            "seed": 2026,
            "trait_col": "Trait",
            "main_models": [{"name": "TabICL", "stage1_backend": "tabicl", "stage2_backend": "group_shared_gate"}],
            "stage2": {"group_shared_gate": {"group_mode": "prior_guided_group", "num_groups": 2}},
        },
        fold_id=1,
        feature_cache_root=cache_root,
        xgboost_search_root=xgb_root,
        output_root=tmp_path / "gate_out",
    )

    restored = np.load(tmp_path / "gate_out" / "xgboost_best_oof.npy")
    bayesb_oof = np.load(tmp_path / "gate_out" / "bayesb_oof.npy")
    gblup_oof = np.load(tmp_path / "gate_out" / "gblup_oof.npy")
    assert np.allclose(restored, best_oof)
    assert np.allclose(bayesb_oof, np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32))
    assert np.allclose(gblup_oof, np.array([0.3, 0.2, 0.1, 0.0], dtype=np.float32))
    assert summary["xgboost_best_trial"] == 3


def test_run_staged_tabicl_xgboost_fixed_from_fold1_on_fold_runs_pure_xgboost_without_group_gate(
    monkeypatch,
    tmp_path: Path,
):
    artifact_root = tmp_path / "fold1_artifacts"
    (artifact_root / "fold1_tabicl_block_search").mkdir(parents=True, exist_ok=True)
    (artifact_root / "fold1_xgboost_search").mkdir(parents=True, exist_ok=True)
    (artifact_root / "fold1_tabicl_block_search" / "best_block.json").write_text(
        json.dumps({"group_size": 777, "best_value": 0.7}),
        encoding="utf-8",
    )
    (artifact_root / "fold1_xgboost_search" / "best_xgboost.json").write_text(
        json.dumps(
            {
                "trial": 2,
                "inner_oof_pearson": 0.72,
                "inner_oof_r2": 0.51,
                "n_estimators": 200,
                "max_depth": 3,
                "learning_rate": 0.05,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.9,
                "reg_lambda": 5.0,
                "reg_alpha": 0.0,
                "tree_method": "hist",
                "objective": "reg:squarederror",
                "random_state": 2028,
                "n_jobs": 4,
                "device": "cpu",
            }
        ),
        encoding="utf-8",
    )
    genotype = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ],
        dtype=np.float32,
    )
    phenotype = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4", "s5"],
            "Trait": [0.0, 1.0, 2.0, 3.0, 4.0],
        }
    )

    monkeypatch.setattr(
        "tabicl_gs.pipeline.staged_tabicl_xgboost_search.plink_num_snps",
        lambda prefix: 2,
    )
    monkeypatch.setattr(
        "tabicl_gs.pipeline.staged_tabicl_xgboost_search.subsample_snp_indices",
        lambda total_snps, max_snps, seed: [0, 1],
    )
    monkeypatch.setattr(
        "tabicl_gs.pipeline.staged_tabicl_xgboost_search.load_plink_matrix",
        lambda prefix, snp_indices=None: SimpleNamespace(
            matrix=genotype,
            sample_ids=["s1", "s2", "s3", "s4", "s5"],
            snp_ids=["m1", "m2"],
        ),
    )
    monkeypatch.setattr(
        "tabicl_gs.pipeline.staged_tabicl_xgboost_search.read_phenotype_table",
        lambda phenotype_csv, sample_id_col="sample_id": phenotype.copy(),
    )
    monkeypatch.setattr(
        "tabicl_gs.pipeline.staged_tabicl_xgboost_search.make_outer_cv_splits",
        lambda X, n_splits, seed: [(np.array([0, 1, 2]), np.array([3, 4], dtype=np.int64))],
    )
    monkeypatch.setattr(
        "tabicl_gs.pipeline.staged_tabicl_xgboost_search.impute_by_train_mean",
        lambda X_train, X_test: (np.asarray(X_train, dtype=np.float32), np.asarray(X_test, dtype=np.float32)),
    )
    monkeypatch.setattr(
        "tabicl_gs.pipeline.staged_tabicl_xgboost_search.precompute_dual_prior_cache",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no-prior XGBoost should not call dual-prior cache")),
    )
    monkeypatch.setattr(
        "tabicl_gs.pipeline.staged_tabicl_xgboost_search.GroupSharedGateRegressor.from_summary",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no-prior XGBoost should not use group gate")),
    )

    def fake_build_stage_features(
        model_spec,
        X_train,
        y_train,
        sampled_snp_ids,
        strategy,
        group_size,
        seed,
        pad_incomplete_last_block,
        embedding_reduce_dim,
        include_block_scalar,
        second_stage_adjustment_config,
        collect_block_diagnostics,
        embedding_extraction_mode,
        stage2_feature_mode,
        X_eval,
    ):
        assert group_size == 777
        assert X_train.shape == (3, 2)
        assert X_eval.shape == (2, 2)
        train_features = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
        test_features = np.array([[0.7, 0.8], [0.9, 1.0]], dtype=np.float32)
        block_summaries = [
            {"block_id": 1, "reduced_embedding_dim": 1, "snp_indices": [0]},
            {"block_id": 2, "reduced_embedding_dim": 1, "snp_indices": [1]},
        ]
        return train_features, test_features, block_summaries, 2

    class FakeXGB:
        def __init__(self):
            self.fit_called = False

        def fit(self, X, y):
            self.fit_called = True
            assert X.shape == (3, 2)
            assert y.shape == (3,)
            return self

        def predict(self, X):
            assert self.fit_called
            assert X.shape == (2, 2)
            return np.array([0.6, 0.8], dtype=np.float32)

    monkeypatch.setattr(
        "tabicl_gs.pipeline.staged_tabicl_xgboost_search._build_stage_features",
        fake_build_stage_features,
    )
    monkeypatch.setattr(
        "tabicl_gs.pipeline.staged_tabicl_xgboost_search.build_xgboost_regressor",
        lambda **kwargs: FakeXGB(),
    )

    summary = run_staged_tabicl_xgboost_fixed_from_fold1_on_fold(
        base_config={
            "seed": 2026,
            "outer_cv_folds": 5,
            "max_snps": 10000,
            "trait_col": "Trait",
            "plink_prefix": "dummy/plink",
            "phenotype_csv": "dummy.csv",
            "phenotype_sample_id_col": "sample_id",
            "grouping_strategy": "window",
            "include_block_scalar": False,
            "pad_incomplete_last_block": True,
            "embedding_extraction_mode": "legacy",
            "main_models": [{"name": "TabICL-XGBoost", "stage1_backend": "tabicl", "stage2_backend": "xgboost"}],
            "stage2": {"xgboost": {"device": "cpu", "n_jobs": 4}},
        },
        fold_id=1,
        fold1_artifact_root=artifact_root,
        output_dir=tmp_path / "outer_fold_1",
    )

    assert summary["fold"] == 1
    assert summary["group_size"] == 777
    assert summary["xgboost_trial"] == 2
    assert summary["xgb_pearson"] == summary["pearson"]
    assert summary["xgb_r2"] == summary["r2"]
    assert summary["pearson"] > 0.0
    assert (tmp_path / "outer_fold_1" / "fold_metrics.csv").exists()
    assert (tmp_path / "outer_fold_1" / "predictions_test.npy").exists()
    assert not (tmp_path / "outer_fold_1" / "frozen_gate_summary.json").exists()
