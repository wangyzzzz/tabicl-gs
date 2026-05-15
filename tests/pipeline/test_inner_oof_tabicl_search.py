from pathlib import Path

import numpy as np
import pandas as pd

from tabicl_gs.pipeline.inner_oof_tabicl_search import (
    build_block_search_space,
    run_inner_oof_tabicl_block_search,
)


def test_build_block_search_space_preserves_order():
    blocks = build_block_search_space([300, 500, 750, 1000, 1500, 2000])
    assert blocks == [300, 500, 750, 1000, 1500, 2000]


def test_run_inner_oof_tabicl_block_search_collects_best_metrics(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "tabicl_gs.pipeline.inner_oof_tabicl_search._load_fold1_data",
        lambda config: (
            np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32),
            np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        "tabicl_gs.pipeline.inner_oof_tabicl_search.make_outer_cv_splits",
        lambda X, n_splits, seed: [
            (np.array([0, 1]), np.array([2, 3])),
            (np.array([2, 3]), np.array([0, 1])),
        ],
    )

    monkeypatch.setattr(
        "tabicl_gs.pipeline.inner_oof_tabicl_search.resolve_two_stage_model_specs",
        lambda config: [
            type(
                "Spec",
                (),
                {
                    "name": "TabICLv2-2stage",
                    "stage1_backend": "tabicl",
                    "stage2_backend": "tabicl",
                    "stage1_config": {},
                    "stage2_config": {},
                },
            )()
        ],
    )
    monkeypatch.setattr(
        "tabicl_gs.pipeline.inner_oof_tabicl_search._resolve_stage2_feature_mode",
        lambda model_spec: "reduced",
    )
    monkeypatch.setattr(
        "tabicl_gs.pipeline.inner_oof_tabicl_search._resolve_second_stage_adjustment",
        lambda config, model_name: None,
    )

    call_state = {"counter": 0}

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
        return (
            np.asarray(X_train, dtype=np.float32),
            np.asarray(X_eval, dtype=np.float32),
            [{"reduced_embedding_dim": 1}],
            1,
        )

    def fake_fit_stage2_model(stage2_backend, X_train_stage2, y_train, X_valid_stage2, stage2_config, seed):
        call_state["counter"] += 1
        inner_id = call_state["counter"]
        group_size = int(X_train_stage2.shape[0] + X_valid_stage2.shape[0] + 296) if False else None
        if seed < 12000:
            raise AssertionError("unexpected seed")
        if X_train_stage2.shape[0] == 2 and X_valid_stage2.shape[0] == 2:
            pass
        if inner_id in {1, 2}:
            pred = np.array([2.1, 2.9], dtype=np.float32) if inner_id == 1 else np.array([0.1, 0.9], dtype=np.float32)
        else:
            pred = np.array([1.7, 2.3], dtype=np.float32) if inner_id == 3 else np.array([0.4, 0.6], dtype=np.float32)
        return object(), pred, {}

    monkeypatch.setattr("tabicl_gs.pipeline.inner_oof_tabicl_search._build_stage_features", fake_build_stage_features)
    monkeypatch.setattr("tabicl_gs.pipeline.inner_oof_tabicl_search._prepare_stage2_config", lambda *args, **kwargs: {})
    monkeypatch.setattr("tabicl_gs.pipeline.inner_oof_tabicl_search.fit_stage2_model", fake_fit_stage2_model)

    summary = run_inner_oof_tabicl_block_search(
        base_config={
            "seed": 2026,
            "trait_col": "Trait",
            "grouping_strategy": "window",
            "outer_cv_folds": 5,
            "stage1": {"tabicl": {}},
            "baselines": {},
        },
        output_root=tmp_path,
        group_sizes=[300, 500],
        inner_folds=2,
    )

    assert list(summary["group_size"]) == [300, 500]
    assert (tmp_path / "fold1_inner_oof_block_search.csv").exists()
    assert summary.iloc[0]["inner_oof_pearson"] > summary.iloc[1]["inner_oof_pearson"]


def test_load_fold1_data_aligns_plink_and_phenotype_by_intersection(monkeypatch):
    import pandas as pd

    monkeypatch.setattr(
        "tabicl_gs.pipeline.inner_oof_tabicl_search.read_phenotype_table",
        lambda path, sample_id_col="sample_id": pd.DataFrame(
            {
                "sample_id": ["s3", "s1", "s4"],
                "Trait": [3.0, 1.0, 4.0],
            }
        ),
    )
    monkeypatch.setattr(
        "tabicl_gs.pipeline.inner_oof_tabicl_search.plink_num_snps",
        lambda prefix: 4,
    )
    monkeypatch.setattr(
        "tabicl_gs.pipeline.inner_oof_tabicl_search.subsample_snp_indices",
        lambda n_snps, max_snps, seed: [0, 1, 2, 3],
    )

    class DummyPlink:
        matrix = np.array(
            [
                [10.0, 11.0, 12.0, 13.0],
                [20.0, 21.0, 22.0, 23.0],
                [30.0, 31.0, 32.0, 33.0],
                [40.0, 41.0, 42.0, 43.0],
            ],
            dtype=np.float32,
        )
        sample_ids = ["s1", "s2", "s3", "s4"]

    monkeypatch.setattr(
        "tabicl_gs.pipeline.inner_oof_tabicl_search.load_plink_matrix",
        lambda prefix, snp_indices=None: DummyPlink(),
    )
    monkeypatch.setattr(
        "tabicl_gs.pipeline.inner_oof_tabicl_search.make_outer_cv_splits",
        lambda X, n_splits, seed: [(np.array([0, 1]), np.array([2]))],
    )

    X, y = __import__("tabicl_gs.pipeline.inner_oof_tabicl_search", fromlist=["_load_fold1_data"])._load_fold1_data(
        {
            "phenotype_csv": "dummy.csv",
            "plink_prefix": "dummy",
            "phenotype_sample_id_col": "sample_id",
            "trait_col": "Trait",
            "max_snps": 10000,
            "seed": 2026,
            "outer_cv_folds": 5,
        }
    )

    assert X.shape == (2, 4)
    assert X.tolist() == [[10.0, 11.0, 12.0, 13.0], [30.0, 31.0, 32.0, 33.0]]
    assert y.tolist() == [1.0, 3.0]
