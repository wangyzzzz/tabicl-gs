from pathlib import Path

import numpy as np

from tabicl_gs.models.baselines import BaselineRunResult
import tabicl_gs.models.factory as factory
import tabicl_gs.models.baselines as baselines


def test_fit_expert_regressor_supports_rkhs_backend(monkeypatch, tmp_path: Path):
    calls: list[dict[str, object]] = []

    def fake_run_r_baseline(
        model_name,
        X_train,
        y_train,
        X_test,
        output_dir,
        rscript_path,
        seed,
        sommer_method=None,
        keep_artifacts=True,
        return_beta=False,
        bandwidth_scale=None,
    ):
        calls.append(
            {
                "model_name": model_name,
                "train_shape": tuple(X_train.shape),
                "test_shape": tuple(X_test.shape),
                "output_dir": str(output_dir),
                "seed": seed,
                "bandwidth_scale": bandwidth_scale,
            }
        )
        pred = np.full(X_test.shape[0], float(np.mean(y_train)), dtype=np.float32)
        return BaselineRunResult(
            predictions=pred,
            metadata={"model": model_name, "device": "R"},
            command=["Rscript"],
            beta=None,
        )

    monkeypatch.setattr(baselines, "run_r_baseline", fake_run_r_baseline)

    X_train = np.arange(24, dtype=np.float32).reshape(6, 4)
    y_train = np.linspace(-1.0, 1.0, 6, dtype=np.float32)
    X_eval = np.arange(8, dtype=np.float32).reshape(2, 4)

    model, pred, device = factory._fit_expert_regressor(
        "rkhs",
        {
            "rscript_path": "Rscript",
            "output_dir": str(tmp_path / "rkhs_expert"),
            "keep_artifacts": False,
            "bandwidth_scale": 1.5,
        },
        X_train,
        y_train,
        X_eval,
        seed=11,
    )

    assert pred.shape == (2,)
    assert np.isfinite(pred).all()
    assert device == "R"
    assert calls[0]["model_name"] == "RKHS"
    assert calls[0]["train_shape"] == (6, 4)
    assert calls[0]["test_shape"] == (2, 4)
    assert calls[0]["bandwidth_scale"] == 1.5

    train_pred = model.predict(X_train[:3])

    assert train_pred.shape == (3,)
    assert np.isfinite(train_pred).all()
    assert len(calls) == 2
    assert calls[1]["model_name"] == "RKHS"
    assert calls[1]["test_shape"] == (3, 4)


def test_fit_stage2_model_supports_group_shared_gate_with_rkhs_expert(monkeypatch, tmp_path: Path):
    def fake_run_r_baseline(
        model_name,
        X_train,
        y_train,
        X_test,
        output_dir,
        rscript_path,
        seed,
        sommer_method=None,
        keep_artifacts=True,
        return_beta=False,
        bandwidth_scale=None,
    ):
        pred = (
            0.4 * X_test[:, 0] - 0.1 * X_test[:, 1] + float(np.mean(y_train))
        ).astype(np.float32)
        return BaselineRunResult(
            predictions=pred,
            metadata={"model": model_name, "device": "R"},
            command=["Rscript"],
            beta=None,
        )

    monkeypatch.setattr(baselines, "run_r_baseline", fake_run_r_baseline)

    rng = np.random.default_rng(7)
    X_core = rng.normal(size=(18, 6)).astype(np.float32)
    y_bayesb = (0.5 * X_core[:, 0] - 0.2 * X_core[:, 2]).astype(np.float32)
    y_gblup = (0.4 * X_core[:, 0] + 0.1 * X_core[:, 3]).astype(np.float32)
    y = (0.6 * y_bayesb + 0.4 * y_gblup + 0.2 * X_core[:, 1]).astype(np.float32)
    X_train = np.concatenate([X_core, y_bayesb.reshape(-1, 1), y_gblup.reshape(-1, 1)], axis=1)

    X_test_core = rng.normal(size=(4, 6)).astype(np.float32)
    y_bayesb_test = (0.5 * X_test_core[:, 0] - 0.2 * X_test_core[:, 2]).astype(np.float32)
    y_gblup_test = (0.4 * X_test_core[:, 0] + 0.1 * X_test_core[:, 3]).astype(np.float32)
    X_test = np.concatenate([X_test_core, y_bayesb_test.reshape(-1, 1), y_gblup_test.reshape(-1, 1)], axis=1)

    model, pred, device = factory.fit_stage2_model(
        "group_shared_gate",
        X_train,
        y,
        X_test,
        config={
            "use_prior_prediction": True,
            "use_dual_priors": True,
            "expert_backend": "rkhs",
            "expert_config": {
                "rscript_path": "Rscript",
                "output_dir": str(tmp_path / "gate_rkhs"),
                "keep_artifacts": False,
            },
            "block_input_dims": [3, 3],
            "prior_scores": [0.2, 0.8],
            "num_groups": 2,
            "group_mode": "embedding_group",
            "assignment_mode": "nearest_centroid",
            "device": "cpu",
        },
        seed=19,
    )

    assert pred.shape == (4,)
    assert np.isfinite(pred).all()
    assert device == "R"
    assert hasattr(model, "predict")
    summary = model.get_group_summary()
    assert "alpha_group" in summary
    assert "w_group" in summary
