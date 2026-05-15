import numpy as np

from tabicl_gs.models.factory import fit_stage2_model


def test_fit_stage2_model_supports_calibrated_correction_with_oof_gate():
    rng = np.random.default_rng(123)
    X_core = rng.normal(size=(24, 6)).astype(np.float32)
    y_bayesb = (-0.7 * X_core[:, 0] + 0.3 * X_core[:, 2]).astype(np.float32)
    y = np.where(X_core[:, 0] > 0, 0.8 * X_core[:, 0] + 0.2 * X_core[:, 1], y_bayesb).astype(np.float32)
    X_train = np.concatenate([X_core, y_bayesb.reshape(-1, 1)], axis=1).astype(np.float32)

    X_test_core = rng.normal(size=(5, 6)).astype(np.float32)
    y_bayesb_test = (-0.7 * X_test_core[:, 0] + 0.3 * X_test_core[:, 2]).astype(np.float32)
    X_test = np.concatenate([X_test_core, y_bayesb_test.reshape(-1, 1)], axis=1).astype(np.float32)

    model, pred, device = fit_stage2_model(
        "calibrated_correction",
        X_train,
        y,
        X_test,
        config={
            "expert_backend": "xgboost",
            "use_prior_prediction": True,
            "use_oof_gate_training": True,
            "oof_splits": 3,
            "hidden_dim": 8,
            "lr": 1e-2,
            "weight_decay": 0.0,
            "max_epochs": 80,
            "device": "cpu",
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
        },
        seed=17,
    )

    assert pred.shape == (5,)
    assert np.isfinite(pred).all()
    assert device == "cpu"
    assert hasattr(model, "predict")
