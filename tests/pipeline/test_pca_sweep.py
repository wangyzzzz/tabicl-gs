from tabicl_gs.pipeline.pca_sweep import build_pca_sweep_config


def test_build_pca_sweep_config_overrides_trait_group_and_pca():
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

    cfg = build_pca_sweep_config(
        base_config=base,
        trait_col="Num_panicles",
        group_size=200,
        include_block_scalar=False,
        pca_dim=32,
        output_dir="outputs/test",
    )

    assert cfg["trait_col"] == "Num_panicles"
    assert cfg["group_size"] == 200
    assert cfg["stage1"]["tabicl"]["embedding_reduce_dim"] == 32
    assert cfg["tuning"]["enabled"] is False
    assert cfg["baselines"]["gblup"] is False
    assert cfg["save_block_summaries"] is True
