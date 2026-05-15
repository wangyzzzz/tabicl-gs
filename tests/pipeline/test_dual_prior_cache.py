from pathlib import Path

import numpy as np

import tabicl_gs.pipeline.dual_prior_fold_search as dual_prior_fold_search
from tabicl_gs.pipeline.dual_prior_utils import resolve_gate_prior_train_predictions
from tabicl_gs.pipeline.dual_prior_fold_search import build_block_search_bounds, save_prior_cache, load_prior_cache


def test_save_and_load_prior_cache_roundtrip(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    payload = {
        "bayesb_oof": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "gblup_oof": np.array([4.0, 5.0, 6.0], dtype=np.float32),
        "bayesb_test": np.array([7.0], dtype=np.float32),
        "gblup_test": np.array([8.0], dtype=np.float32),
    }

    save_prior_cache(cache_dir, payload)
    loaded = load_prior_cache(cache_dir)

    assert set(loaded.keys()) == set(payload.keys())
    for key in payload:
        assert np.allclose(loaded[key], payload[key])


def test_build_block_search_bounds_returns_int_range():
    assert build_block_search_bounds(300, 2000) == (300, 2000)


def test_reconstruct_fold_train_oof_from_inner_cache():
    y_train = np.zeros(6, dtype=np.float32)
    inner_cache = [
        {
            "inner_valid_idx": np.array([1, 4]),
            "bayesb_train_oof": np.array([0.1, 0.4], dtype=np.float32),
            "gblup_train_oof": np.array([1.1, 1.4], dtype=np.float32),
        },
        {
            "inner_valid_idx": np.array([0, 3]),
            "bayesb_train_oof": np.array([0.0, 0.3], dtype=np.float32),
            "gblup_train_oof": np.array([1.0, 1.3], dtype=np.float32),
        },
        {
            "inner_valid_idx": np.array([2, 5]),
            "bayesb_train_oof": np.array([0.2, 0.5], dtype=np.float32),
            "gblup_train_oof": np.array([1.2, 1.5], dtype=np.float32),
        },
    ]

    bayesb_train_oof = np.zeros_like(y_train)
    gblup_train_oof = np.zeros_like(y_train)
    for inner_meta in inner_cache:
        bayesb_train_oof[inner_meta["inner_valid_idx"]] = inner_meta["bayesb_train_oof"]
        gblup_train_oof[inner_meta["inner_valid_idx"]] = inner_meta["gblup_train_oof"]

    assert np.allclose(bayesb_train_oof, np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32))
    assert np.allclose(gblup_train_oof, np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5], dtype=np.float32))


def test_resolve_gate_prior_train_predictions_can_use_inner_oof_valid_predictions():
    cached = {
        "y_outer_train": np.zeros(6, dtype=np.float32),
        "bayesb_train": np.array([9, 9, 9, 9, 9, 9], dtype=np.float32),
        "gblup_train": np.array([8, 8, 8, 8, 8, 8], dtype=np.float32),
        "inner_cache": [
            {
                "inner_valid_idx": np.array([1, 4]),
                "bayesb_valid": np.array([0.1, 0.4], dtype=np.float32),
                "gblup_valid": np.array([1.1, 1.4], dtype=np.float32),
            },
            {
                "inner_valid_idx": np.array([0, 3]),
                "bayesb_valid": np.array([0.0, 0.3], dtype=np.float32),
                "gblup_valid": np.array([1.0, 1.3], dtype=np.float32),
            },
            {
                "inner_valid_idx": np.array([2, 5]),
                "bayesb_valid": np.array([0.2, 0.5], dtype=np.float32),
                "gblup_valid": np.array([1.2, 1.5], dtype=np.float32),
            },
        ],
    }

    bayesb_train, gblup_train, bayes_candidates = resolve_gate_prior_train_predictions(
        cached,
        prior_train_source="inner_oof",
    )

    assert np.allclose(bayesb_train, np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32))
    assert np.allclose(gblup_train, np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5], dtype=np.float32))
    assert bayes_candidates is None


def test_resolve_gate_prior_train_predictions_supports_triple_prior_sources():
    cached = {
        "y_outer_train": np.zeros(6, dtype=np.float32),
        "bayesb_train": np.array([9, 9, 9, 9, 9, 9], dtype=np.float32),
        "gblup_train": np.array([8, 8, 8, 8, 8, 8], dtype=np.float32),
        "rkhs_train": np.array([7, 7, 7, 7, 7, 7], dtype=np.float32),
        "inner_cache": [
            {
                "inner_valid_idx": np.array([1, 4]),
                "bayesb_valid": np.array([0.1, 0.4], dtype=np.float32),
                "gblup_valid": np.array([1.1, 1.4], dtype=np.float32),
                "rkhs_valid": np.array([2.1, 2.4], dtype=np.float32),
            },
            {
                "inner_valid_idx": np.array([0, 3]),
                "bayesb_valid": np.array([0.0, 0.3], dtype=np.float32),
                "gblup_valid": np.array([1.0, 1.3], dtype=np.float32),
                "rkhs_valid": np.array([2.0, 2.3], dtype=np.float32),
            },
            {
                "inner_valid_idx": np.array([2, 5]),
                "bayesb_valid": np.array([0.2, 0.5], dtype=np.float32),
                "gblup_valid": np.array([1.2, 1.5], dtype=np.float32),
                "rkhs_valid": np.array([2.2, 2.5], dtype=np.float32),
            },
        ],
    }

    bayesb_train, gblup_train, prior_candidates = resolve_gate_prior_train_predictions(
        cached,
        prior_train_source="inner_oof",
    )

    assert np.allclose(bayesb_train, np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32))
    assert np.allclose(gblup_train, np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5], dtype=np.float32))
    assert prior_candidates is not None
    assert "RKHS" in prior_candidates
    assert np.allclose(prior_candidates["RKHS"], np.array([2.0, 2.1, 2.2, 2.3, 2.4, 2.5], dtype=np.float32))


def test_dual_prior_full_train_cache_keys_match_non_strict_oof_path():
    payload = {
        "bayesb_train": np.array([0.1, 0.2], dtype=np.float32),
        "gblup_train": np.array([1.1, 1.2], dtype=np.float32),
        "bayesb_valid": np.array([0.3], dtype=np.float32),
        "gblup_valid": np.array([1.3], dtype=np.float32),
        "bayesb_test": np.array([0.4], dtype=np.float32),
        "gblup_test": np.array([1.4], dtype=np.float32),
    }
    assert set(payload.keys()) == {
        "bayesb_train",
        "gblup_train",
        "bayesb_valid",
        "gblup_valid",
        "bayesb_test",
        "gblup_test",
    }


def test_precompute_dual_prior_cache_includes_rkhs_predictions(monkeypatch, tmp_path: Path):
    X = np.arange(48, dtype=np.float32).reshape(8, 6)
    y = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    calls = {"bayesb": 0, "rkhs": 0, "gblup": 0}

    def fake_load_fold_data(config, fold_id):
        return X[:6], y[:6], X[6:], y[6:]

    def fake_impute_by_train_mean(X_train, X_test):
        return X_train.astype(np.float32), X_test.astype(np.float32)

    def fake_compute_bayesb_predictions_with_beta(**kwargs):
        calls["bayesb"] += 1
        x_train = kwargs["X_train_base"]
        x_eval = kwargs["X_eval_base"]
        n_features = x_train.shape[1]
        return (
            np.full(x_train.shape[0], 0.11, dtype=np.float32),
            np.full(x_eval.shape[0], 0.22, dtype=np.float32),
            np.linspace(1.0, 2.0, n_features, dtype=np.float32),
        )

    def fake_compute_residual_target_predictions(**kwargs):
        calls["gblup"] += 1
        x_train = kwargs["X_train_base"]
        x_test = kwargs["X_test_base"]
        return (
            np.full(x_train.shape[0], 0.33, dtype=np.float32),
            np.full(x_test.shape[0], 0.44, dtype=np.float32),
            {},
        )

    def fake_compute_baseline_train_eval_predictions(**kwargs):
        model_name = kwargs["model_name"]
        x_train = kwargs["X_train_base"]
        x_eval = kwargs["X_eval_base"]
        if model_name == "RKHS":
            calls["rkhs"] += 1
            return (
                np.full(x_train.shape[0], 0.55, dtype=np.float32),
                np.full(x_eval.shape[0], 0.66, dtype=np.float32),
            )
        raise AssertionError(f"Unexpected baseline model: {model_name}")

    monkeypatch.setattr(dual_prior_fold_search, "_load_fold_data", fake_load_fold_data)
    monkeypatch.setattr(dual_prior_fold_search, "impute_by_train_mean", fake_impute_by_train_mean)
    monkeypatch.setattr(
        dual_prior_fold_search,
        "_compute_bayesb_predictions_with_beta",
        fake_compute_bayesb_predictions_with_beta,
    )
    monkeypatch.setattr(
        dual_prior_fold_search,
        "_compute_residual_target_predictions",
        fake_compute_residual_target_predictions,
    )
    monkeypatch.setattr(
        dual_prior_fold_search,
        "_compute_baseline_train_eval_predictions",
        fake_compute_baseline_train_eval_predictions,
    )

    config = {
        "seed": 2026,
        "main_models": [
            {
                "name": "TabICLv2-GroupSharedGate-TriplePrior",
                "stage1_backend": "tabicl",
                "stage2_backend": "group_shared_gate",
            }
        ],
        "stage2": {
            "group_shared_gate": {
                "use_prior_prediction": True,
                "use_dual_priors": True,
                "use_rkhs_prior": True,
            }
        },
        "baselines": {
            "rscript_path": "Rscript",
            "sommer_method": "mmer",
        },
        "phenotype_csv": "unused.csv",
        "plink_prefix": "unused",
        "max_snps": 10000,
        "outer_cv_folds": 5,
        "trait_col": "Trait",
    }

    cached = dual_prior_fold_search.precompute_dual_prior_cache(
        config,
        fold_id=1,
        cache_root=tmp_path / "prior_cache",
        inner_folds=2,
    )

    assert calls["bayesb"] == 3
    assert calls["gblup"] == 3
    assert calls["rkhs"] == 3
    assert "rkhs_train" in cached
    assert "rkhs_test" in cached
    assert cached["rkhs_train"].shape == (6,)
    assert cached["rkhs_test"].shape == (2,)
    assert len(cached["inner_cache"]) == 2
    assert all("rkhs_train" in inner_meta for inner_meta in cached["inner_cache"])
    assert all("rkhs_valid" in inner_meta for inner_meta in cached["inner_cache"])


def test_evaluate_block_with_cached_priors_passes_inner_oof_prior_source(monkeypatch, tmp_path: Path):
    captured = {}

    monkeypatch.setattr(
        dual_prior_fold_search,
        "_build_config",
        lambda base_config, group_size, output_dir: {"seed": 2026},
    )
    monkeypatch.setattr(
        dual_prior_fold_search,
        "resolve_two_stage_model_specs",
        lambda base_config: [type("Spec", (), {"stage2_backend": "group_shared_gate", "stage2_config": {}})()],
    )

    def fake_collect_group_shared_gate_oof_payload(**kwargs):
        captured["prior_train_source"] = kwargs["prior_train_source"]
        return {
            "oof_pred": np.array([0.2, 0.4, 0.6], dtype=np.float32),
            "group_summary": {"group_counts": [3]},
            "num_blocks": 10,
            "mean_reduced_block_embedding_dim": 16.0,
            "stage2_input_dim": 160.0,
        }

    monkeypatch.setattr(
        dual_prior_fold_search,
        "_collect_group_shared_gate_oof_payload",
        fake_collect_group_shared_gate_oof_payload,
    )

    row = dual_prior_fold_search._evaluate_block_with_cached_priors(
        base_config={
            "seed": 2026,
            "gate_prior_train_source": "inner_oof",
        },
        cached={
            "X_outer_train": np.zeros((3, 4), dtype=np.float32),
            "y_outer_train": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        },
        group_size=1000,
        output_dir=tmp_path / "trial",
    )

    assert captured["prior_train_source"] == "inner_oof"
    assert row["group_size"] == 1000


def test_use_bayes_family_selector_is_disabled_for_vi_prior():
    config = {
        "main_models": [
            {
                "name": "TabICLv2-GroupSharedGate-VIPrior",
                "stage1_backend": "tabicl",
                "stage2_backend": "group_shared_gate",
            }
        ],
        "stage2": {
            "group_shared_gate": {
                "prior_backend": "vi",
                "use_bayes_family_selector": True,
            }
        },
    }

    assert dual_prior_fold_search._resolve_prior_backend(config) == "vi"
    assert dual_prior_fold_search._use_bayes_family_selector(config) is False


def test_precompute_dual_prior_cache_with_vi_prior_skips_bayes_selector_and_keeps_compat_fields(monkeypatch, tmp_path: Path):
    X = np.arange(48, dtype=np.float32).reshape(8, 6)
    y = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    calls = {"vi": 0, "gblup": 0, "selector": 0}

    def fake_load_fold_data(config, fold_id):
        return X[:6], y[:6], X[6:], y[6:]

    def fake_impute_by_train_mean(X_train, X_test):
        return X_train.astype(np.float32), X_test.astype(np.float32)

    def fake_compute_vi_predictions_with_beta(**kwargs):
        calls["vi"] += 1
        x_train = kwargs["X_train_base"]
        x_eval = kwargs["X_eval_base"]
        n_features = x_train.shape[1]
        return (
            np.full(x_train.shape[0], 0.11, dtype=np.float32),
            np.full(x_eval.shape[0], 0.22, dtype=np.float32),
            np.linspace(1.0, 2.0, n_features, dtype=np.float32),
            np.linspace(0.1, 0.2, n_features, dtype=np.float32),
        )

    def fake_compute_residual_target_predictions(**kwargs):
        calls["gblup"] += 1
        x_train = kwargs["X_train_base"]
        x_test = kwargs["X_test_base"]
        return (
            np.full(x_train.shape[0], 0.33, dtype=np.float32),
            np.full(x_test.shape[0], 0.44, dtype=np.float32),
            {},
        )

    def fake_compute_baseline_train_eval_predictions(**kwargs):
        calls["selector"] += 1
        raise AssertionError("Bayes family selector should be skipped when prior_backend=vi.")

    monkeypatch.setattr(dual_prior_fold_search, "_load_fold_data", fake_load_fold_data)
    monkeypatch.setattr(dual_prior_fold_search, "impute_by_train_mean", fake_impute_by_train_mean)
    monkeypatch.setattr(
        dual_prior_fold_search,
        "_compute_vi_predictions_with_beta",
        fake_compute_vi_predictions_with_beta,
    )
    monkeypatch.setattr(
        dual_prior_fold_search,
        "_compute_residual_target_predictions",
        fake_compute_residual_target_predictions,
    )
    monkeypatch.setattr(
        dual_prior_fold_search,
        "_compute_baseline_train_eval_predictions",
        fake_compute_baseline_train_eval_predictions,
    )

    config = {
        "seed": 2026,
        "main_models": [
            {
                "name": "TabICLv2-GroupSharedGate-VIPrior",
                "stage1_backend": "tabicl",
                "stage2_backend": "group_shared_gate",
            }
        ],
        "stage2": {
            "group_shared_gate": {
                "prior_backend": "vi",
                "use_bayes_family_selector": True,
                "vi_prior_config": {
                    "max_epochs": 10,
                    "lr": 0.03,
                },
            }
        },
        "baselines": {
            "rscript_path": "Rscript",
            "sommer_method": "mmer",
        },
        "phenotype_csv": "unused.csv",
        "plink_prefix": "unused",
        "max_snps": 10000,
        "outer_cv_folds": 5,
        "trait_col": "Trait",
    }

    cached = dual_prior_fold_search.precompute_dual_prior_cache(
        config,
        fold_id=1,
        cache_root=tmp_path / "prior_cache",
        inner_folds=2,
    )

    assert calls["vi"] == 3
    assert calls["gblup"] == 3
    assert calls["selector"] == 0
    assert "bayeslasso_train" not in cached
    assert "bayeslasso_test" not in cached
    assert "vi_coef_var" in cached
    assert cached["bayesb_train"].shape == (6,)
    assert cached["bayesb_test"].shape == (2,)
    assert cached["gblup_train"].shape == (6,)
    assert cached["gblup_test"].shape == (2,)
    assert cached["bayesb_beta"].shape == (6,)
    assert cached["vi_coef_var"].shape == (6,)
    assert len(cached["inner_cache"]) == 2
    assert all("vi_coef_var" in inner_meta for inner_meta in cached["inner_cache"])


def test_load_fold_data_supports_outer_train_subset_override(monkeypatch):
    phenotype = np.array([0, 1, 2, 3, 4, 5], dtype=np.float32)

    class DummyPlink:
        matrix = np.arange(24, dtype=np.float32).reshape(6, 4)
        sample_ids = [f"s{i}" for i in range(6)]
        snp_ids = [f"m{i}" for i in range(4)]

    import pandas as pd

    monkeypatch.setattr(dual_prior_fold_search, "read_phenotype_table", lambda path, sample_id_col: pd.DataFrame({"sample_id": DummyPlink.sample_ids, "trait": phenotype}))
    monkeypatch.setattr(dual_prior_fold_search, "plink_num_snps", lambda prefix: 4)
    monkeypatch.setattr(dual_prior_fold_search, "subsample_snp_indices", lambda total, max_snps, seed: [0, 1, 2, 3])
    monkeypatch.setattr(dual_prior_fold_search, "load_plink_matrix", lambda prefix, snp_indices=None: DummyPlink())
    monkeypatch.setattr(
        dual_prior_fold_search,
        "align_phenotype_to_sample_ids",
        lambda phenotype, sample_ids, sample_id_col: (phenotype, list(range(len(sample_ids)))),
    )
    monkeypatch.setattr(
        dual_prior_fold_search,
        "make_outer_cv_splits",
        lambda X, n_splits, seed: [(np.array([0, 1, 2, 3]), np.array([4, 5]))],
    )

    cfg = {
        "phenotype_csv": "unused.csv",
        "phenotype_sample_id_col": "sample_id",
        "plink_prefix": "unused",
        "max_snps": 10000,
        "seed": 2026,
        "outer_cv_folds": 5,
        "trait_col": "trait",
        "_sample_size_override": {
            "fold_id": 1,
            "train_subset_indices": [1, 3],
        },
    }

    X_train, y_train, X_test, y_test = dual_prior_fold_search._load_fold_data(cfg, fold_id=1)
    assert X_train.shape == (2, 4)
    assert y_train.tolist() == [1.0, 3.0]
    assert X_test.shape == (2, 4)
    assert y_test.tolist() == [4.0, 5.0]
