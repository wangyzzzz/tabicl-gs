import numpy as np

from tabicl_gs.models.factory import fit_stage2_model


def test_fit_stage2_model_supports_block_attention_backend():
    rng = np.random.default_rng(11)
    X_train = rng.normal(size=(16, 9)).astype(np.float32)
    y_train = (X_train[:, 0] * 0.4 - X_train[:, 4] * 0.2 + X_train[:, 8] * 0.1).astype(np.float32)
    X_test = rng.normal(size=(5, 9)).astype(np.float32)

    model, pred, device = fit_stage2_model(
        "block_attention",
        X_train,
        y_train,
        X_test,
        config={
            "block_input_dims": [3, 4, 2],
            "model_dim": 16,
            "num_heads": 4,
            "num_layers": 1,
            "ff_multiplier": 2,
            "dropout": 0.0,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "max_epochs": 5,
            "batch_size": None,
            "device": "cpu",
        },
        seed=17,
    )

    assert pred.shape == (5,)
    assert np.isfinite(pred).all()
    assert device == "cpu"
    assert model.config.block_input_dims == [3, 4, 2]


def test_fit_stage2_model_supports_sample_mixture_backend():
    rng = np.random.default_rng(21)
    X_core = rng.normal(size=(20, 5)).astype(np.float32)
    prior = (0.5 * X_core[:, 0] - 0.2 * X_core[:, 1]).astype(np.float32)
    y = np.where(X_core[:, 0] > 0, X_core[:, 0] + 0.1 * X_core[:, 2], prior).astype(np.float32)
    X_train = np.concatenate([X_core, prior.reshape(-1, 1)], axis=1).astype(np.float32)
    X_test_core = rng.normal(size=(4, 5)).astype(np.float32)
    X_test_prior = (0.5 * X_test_core[:, 0] - 0.2 * X_test_core[:, 1]).astype(np.float32)
    X_test = np.concatenate([X_test_core, X_test_prior.reshape(-1, 1)], axis=1).astype(np.float32)

    model, pred, device = fit_stage2_model(
        "sample_mixture",
        X_train,
        y,
        X_test,
        config={
            "expert_backend": "xgboost",
            "expert_config": {
                "n_estimators": 8,
                "max_depth": 2,
                "learning_rate": 0.1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "tree_method": "hist",
                "device": "cpu",
                "n_jobs": 1,
            },
            "use_prior_prediction": True,
            "hidden_dim": 8,
            "lr": 1e-2,
            "weight_decay": 0.0,
            "max_epochs": 50,
            "device": "cpu",
        },
        seed=17,
    )

    assert pred.shape == (4,)
    assert np.isfinite(pred).all()
    assert device == "cpu"


def test_fit_stage2_model_supports_static_block_weight_backend():
    rng = np.random.default_rng(31)
    X_train = rng.normal(size=(18, 9)).astype(np.float32)
    y_train = (0.3 * X_train[:, 0] - 0.2 * X_train[:, 4] + 0.4 * X_train[:, 8]).astype(np.float32)
    X_test = rng.normal(size=(4, 9)).astype(np.float32)

    model, pred, device = fit_stage2_model(
        "static_block_weight",
        X_train,
        y_train,
        X_test,
        config={
            "block_input_dims": [3, 3, 3],
            "prior_scores": [0.1, 0.2, 0.8],
            "model_dim": 16,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "max_epochs": 5,
            "device": "cpu",
        },
        seed=17,
    )

    assert pred.shape == (4,)
    assert np.isfinite(pred).all()
    assert device == "cpu"
    assert model.config.block_input_dims == [3, 3, 3]


def test_fit_stage2_model_supports_group_weight_pooling_backend():
    rng = np.random.default_rng(41)
    X_train = rng.normal(size=(18, 9)).astype(np.float32)
    y_train = (0.3 * X_train[:, 0] - 0.2 * X_train[:, 4] + 0.4 * X_train[:, 8]).astype(np.float32)
    X_test = rng.normal(size=(4, 9)).astype(np.float32)

    model, pred, device = fit_stage2_model(
        "group_weight_pooling",
        X_train,
        y_train,
        X_test,
        config={
            "block_input_dims": [3, 3, 3],
            "prior_scores": [0.1, 0.2, 0.8],
            "num_groups": 3,
            "model_dim": 16,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "max_epochs": 5,
            "device": "cpu",
        },
        seed=17,
    )

    assert pred.shape == (4,)
    assert np.isfinite(pred).all()
    assert device == "cpu"
    assert model.config.block_input_dims == [3, 3, 3]


def test_fit_stage2_model_supports_group_shared_gate_backend():
    rng = np.random.default_rng(51)
    X_core = rng.normal(size=(20, 6)).astype(np.float32)
    y_tabicl = (0.7 * X_core[:, 0] - 0.3 * X_core[:, 1]).astype(np.float32)
    y_bayesb = (-0.4 * X_core[:, 0] + 0.2 * X_core[:, 2]).astype(np.float32)
    y_gblup = (0.5 * X_core[:, 0] + 0.1 * X_core[:, 3]).astype(np.float32)
    X_train = np.concatenate([X_core, y_tabicl.reshape(-1, 1), y_bayesb.reshape(-1, 1), y_gblup.reshape(-1, 1)], axis=1)
    y_train = (0.4 * y_tabicl + 0.4 * y_bayesb + 0.2 * y_gblup).astype(np.float32)
    X_test_core = rng.normal(size=(4, 6)).astype(np.float32)
    X_test_tabicl = (0.7 * X_test_core[:, 0] - 0.3 * X_test_core[:, 1]).astype(np.float32)
    X_test_bayesb = (-0.4 * X_test_core[:, 0] + 0.2 * X_test_core[:, 2]).astype(np.float32)
    X_test_gblup = (0.5 * X_test_core[:, 0] + 0.1 * X_test_core[:, 3]).astype(np.float32)
    X_test = np.concatenate(
        [X_test_core, X_test_tabicl.reshape(-1, 1), X_test_bayesb.reshape(-1, 1), X_test_gblup.reshape(-1, 1)],
        axis=1,
    )

    model, pred, device = fit_stage2_model(
        "group_shared_gate",
        X_train,
        y_train,
        X_test,
        config={
            "block_input_dims": [3, 3],
            "prior_scores": [0.2, 0.8],
            "num_groups": 3,
            "group_mode": "embedding_group",
            "hidden_dim": 8,
            "max_epochs": 30,
            "device": "cpu",
            "use_prior_prediction": True,
            "use_dual_priors": True,
        },
        seed=17,
    )

    assert pred.shape == (4,)
    assert np.isfinite(pred).all()
    assert device in {"cpu", "cuda"}
    assert model.gate_model.config.group_mode == "embedding_group"
    assert model.gate_model.config.assignment_mode == "nearest_centroid"
