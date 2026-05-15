from tabicl_gs.pipeline.variance_target_sweep import build_variance_target_config


def test_build_variance_target_config_sets_target_and_disables_baselines():
    base = {
        "trait_col": "Heading_date",
        "grouping_strategy": "window",
        "group_size": 100,
        "include_block_scalar": False,
        "main_models": [{"name": "TabICLv2-2stage", "stage1_backend": "tabicl", "stage2_backend": "tabicl"}],
        "stage1": {"tabicl": {"embedding_reduce_dim": 16}},
        "tuning": {"enabled": True},
        "baselines": {"gblup": True, "bayesA": True, "bayesB": True, "bayesLasso": True},
    }

    cfg = build_variance_target_config(
        base_config=base,
        trait_col="Num_panicles",
        group_size=200,
        include_block_scalar=False,
        variance_target=0.95,
        output_dir="outputs/test",
    )

    assert cfg["trait_col"] == "Num_panicles"
    assert cfg["group_size"] == 200
    assert cfg["stage1"]["tabicl"]["embedding_reduce_dim"] is None
    assert cfg["stage1"]["tabicl"]["embedding_explained_variance_target"] == 0.95
    assert cfg["stage1"]["tabicl"]["track_full_explained_variance"] is True
    assert cfg["baselines"]["gblup"] is False
