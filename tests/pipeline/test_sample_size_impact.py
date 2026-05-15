from __future__ import annotations

import numpy as np
import pandas as pd

import tabicl_gs.pipeline.sample_size_impact as sample_size_impact


def test_build_nested_train_subsets_is_reproducible_and_nested():
    train_indices = np.arange(20, dtype=np.int64)
    proportions = [0.2, 0.5, 1.0]

    left = sample_size_impact.build_nested_train_subsets(
        train_indices=train_indices,
        proportions=proportions,
        repeats=2,
        seed=2026,
    )
    right = sample_size_impact.build_nested_train_subsets(
        train_indices=train_indices,
        proportions=proportions,
        repeats=2,
        seed=2026,
    )

    assert left.keys() == right.keys()
    for key in left:
        assert np.array_equal(left[key], right[key])

    rep1_small = left[(1, 0.2)]
    rep1_mid = left[(1, 0.5)]
    rep1_full = left[(1, 1.0)]
    assert set(rep1_small.tolist()).issubset(set(rep1_mid.tolist()))
    assert set(rep1_mid.tolist()).issubset(set(rep1_full.tolist()))
    assert len(rep1_small) == 4
    assert len(rep1_mid) == 10
    assert len(rep1_full) == 20


def test_summarize_sample_size_metrics_computes_retention_and_amplification():
    fold_metrics = pd.DataFrame(
        [
            {"model": "dual_prior", "proportion": 0.1, "test_pearson": 0.30},
            {"model": "dual_prior", "proportion": 0.2, "test_pearson": 0.40},
            {"model": "dual_prior", "proportion": 0.4, "test_pearson": 0.50},
            {"model": "dual_prior", "proportion": 0.8, "test_pearson": 0.58},
            {"model": "dual_prior", "proportion": 1.0, "test_pearson": 0.60},
            {"model": "prior_only", "proportion": 0.1, "test_pearson": 0.20},
            {"model": "prior_only", "proportion": 0.2, "test_pearson": 0.28},
            {"model": "prior_only", "proportion": 0.4, "test_pearson": 0.38},
            {"model": "prior_only", "proportion": 0.8, "test_pearson": 0.52},
            {"model": "prior_only", "proportion": 1.0, "test_pearson": 0.55},
        ]
    )

    summary = sample_size_impact.summarize_sample_size_metrics(fold_metrics)

    dual = summary[summary["model"] == "dual_prior"].set_index("proportion")
    prior = summary[summary["model"] == "prior_only"].set_index("proportion")

    assert np.isclose(float(dual.loc[0.1, "retention"]), 0.5)
    assert np.isclose(float(prior.loc[0.1, "retention"]), 0.20 / 0.55)

    trait_summary = sample_size_impact.compute_trait_level_sample_size_summary(fold_metrics)
    assert np.isclose(float(trait_summary["gap_0.1"]), 0.10)
    assert np.isclose(float(trait_summary["gap_1.0"]), 0.05)
    assert float(trait_summary["small_n_amplification"]) > 0.0
    expected_auc = sample_size_impact._trapezoid_area(
        dual["retention"].to_numpy(dtype=np.float32),
        dual.index.to_numpy(dtype=np.float32),
    )
    assert np.isclose(float(trait_summary["dual_auc_retention"]), expected_auc)


def test_summarize_sample_size_metrics_allows_missing_full_reference():
    fold_metrics = pd.DataFrame(
        [
            {"model": "dual_prior", "proportion": 0.2, "test_pearson": 0.40},
            {"model": "dual_prior", "proportion": 0.4, "test_pearson": 0.50},
            {"model": "prior_only", "proportion": 0.2, "test_pearson": 0.30},
            {"model": "prior_only", "proportion": 0.4, "test_pearson": 0.35},
        ]
    )

    summary = sample_size_impact.summarize_sample_size_metrics(fold_metrics)
    assert summary["retention"].isna().all()

    referenced = sample_size_impact.summarize_sample_size_metrics(
        fold_metrics,
        reference_by_model={"dual_prior": 0.80, "prior_only": 0.70},
    )
    dual = referenced[referenced["model"] == "dual_prior"].set_index("proportion")
    assert np.isclose(float(dual.loc[0.2, "retention"]), 0.5)


def test_compute_trait_level_sample_size_summary_adapts_to_available_proportions():
    fold_metrics = pd.DataFrame(
        [
            {"model": "dual_prior", "proportion": 0.2, "test_pearson": 0.42},
            {"model": "dual_prior", "proportion": 0.4, "test_pearson": 0.50},
            {"model": "dual_prior", "proportion": 0.8, "test_pearson": 0.57},
            {"model": "dual_prior", "proportion": 1.0, "test_pearson": 0.60},
            {"model": "prior_only", "proportion": 0.2, "test_pearson": 0.31},
            {"model": "prior_only", "proportion": 0.4, "test_pearson": 0.40},
            {"model": "prior_only", "proportion": 0.8, "test_pearson": 0.53},
            {"model": "prior_only", "proportion": 1.0, "test_pearson": 0.56},
        ]
    )

    trait_summary = sample_size_impact.compute_trait_level_sample_size_summary(fold_metrics)
    assert "gap_0.20" in trait_summary
    assert "gap_0.10" not in trait_summary
    assert np.isclose(float(trait_summary["gap_0.20"]), 0.11)
    assert float(trait_summary["small_n_amplification"]) > 0.0


def test_should_run_block_search_uses_float_tolerance():
    assert sample_size_impact.should_run_block_search(0.2, [0.2]) is True
    assert sample_size_impact.should_run_block_search(0.2000000001, [0.2]) is True
    assert sample_size_impact.should_run_block_search(0.4, [0.2]) is False


def test_build_nested_train_subsets_uses_same_subset_for_same_repeat_across_folds_when_seed_matches():
    train_indices = np.arange(15, dtype=np.int64)
    left = sample_size_impact.build_nested_train_subsets(
        train_indices=train_indices,
        proportions=[0.2, 0.4, 1.0],
        repeats=2,
        seed=2026,
    )
    right = sample_size_impact.build_nested_train_subsets(
        train_indices=train_indices,
        proportions=[0.2, 0.4, 1.0],
        repeats=2,
        seed=2026,
    )
    assert np.array_equal(left[(1, 0.2)], right[(1, 0.2)])
    assert np.array_equal(left[(2, 0.4)], right[(2, 0.4)])
