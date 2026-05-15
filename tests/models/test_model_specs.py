from tabicl_gs.models.model_specs import resolve_two_stage_model_specs


def test_resolve_model_specs_supports_legacy_single_tabicl_config():
    config = {
        "stage1": {"n_estimators": 1, "embedding_reduce_dim": 8},
        "stage2": {"n_estimators": 1},
    }

    specs = resolve_two_stage_model_specs(config)

    assert len(specs) == 1
    assert specs[0].name == "TabICLv2-2stage"
    assert specs[0].stage1_backend == "tabicl"
    assert specs[0].stage2_backend == "tabicl"
    assert specs[0].stage1_config["embedding_reduce_dim"] == 8


def test_resolve_model_specs_supports_multiple_backends():
    config = {
        "main_models": [
            {"name": "TabICLv2-2stage", "stage1_backend": "tabicl", "stage2_backend": "tabicl"},
            {"name": "TabPFN-2stage", "stage1_backend": "tabpfn", "stage2_backend": "tabpfn"},
            {"name": "XGBoost-2stage", "stage1_backend": "xgboost", "stage2_backend": "xgboost"},
            {"name": "TabICLv2-Mixture", "stage1_backend": "tabicl", "stage2_backend": "sample_mixture"},
            {"name": "TabICLv2-CalibratedCorrection", "stage1_backend": "tabicl", "stage2_backend": "calibrated_correction"},
        ],
        "stage1": {
            "tabicl": {"embedding_reduce_dim": 16},
            "tabpfn": {"embedding_reduce_dim": 4},
            "xgboost": {"embedding_reduce_dim": 8},
        },
        "stage2": {
            "tabicl": {"device": "cuda"},
            "tabpfn": {"device": "cuda"},
            "xgboost": {"device": "cpu"},
            "sample_mixture": {"device": "cpu", "expert_backend": "tabicl", "use_prior_prediction": True},
            "calibrated_correction": {"device": "cpu", "expert_backend": "tabicl", "use_prior_prediction": True},
        },
    }

    specs = resolve_two_stage_model_specs(config)

    assert [spec.name for spec in specs] == [
        "TabICLv2-2stage",
        "TabPFN-2stage",
        "XGBoost-2stage",
        "TabICLv2-Mixture",
        "TabICLv2-CalibratedCorrection",
    ]
    assert specs[1].stage1_config["embedding_reduce_dim"] == 4
    assert specs[2].stage2_config["device"] == "cpu"
    assert specs[3].stage2_config["expert_backend"] == "tabicl"
    assert specs[4].stage2_config["use_prior_prediction"] is True


def test_group_shared_gate_can_configure_xgboost_expert():
    config = {
        "main_models": [
            {
                "name": "TabICLv2-GroupSharedGate-XGBoost-Prior",
                "stage1_backend": "tabicl",
                "stage2_backend": "group_shared_gate",
            }
        ],
        "stage1": {"tabicl": {"embedding_reduce_dim": None}},
        "stage2": {
            "group_shared_gate": {
                "use_prior_prediction": True,
                "use_dual_priors": True,
                "expert_backend": "xgboost",
                "expert_config": {
                    "n_estimators": 8,
                    "max_depth": 2,
                    "device": "cpu",
                },
            }
        },
    }

    specs = resolve_two_stage_model_specs(config)

    assert len(specs) == 1
    assert specs[0].stage1_backend == "tabicl"
    assert specs[0].stage2_backend == "group_shared_gate"
    assert specs[0].stage2_config["expert_backend"] == "xgboost"
    assert specs[0].stage2_config["expert_config"]["n_estimators"] == 8
