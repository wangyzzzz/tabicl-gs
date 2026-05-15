import numpy as np

import tabicl_gs.pipeline.experiment as experiment


def test_compute_oof_baseline_prior_refits_on_inner_train(monkeypatch, tmp_path):
    calls = []

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
    ):
        calls.append(
            {
                "model_name": model_name,
                "n_train": X_train.shape[0],
                "n_test": X_test.shape[0],
                "output_dir": str(output_dir),
            }
        )

        class Result:
            predictions = np.full(X_test.shape[0], float(X_train.shape[0]), dtype=np.float32)
            metadata = {"device": "R"}
            command = ["fake"]
            beta = None

        return Result()

    monkeypatch.setattr(experiment, "run_r_baseline", fake_run_r_baseline)

    X = np.arange(30, dtype=np.float32).reshape(10, 3)
    y = np.arange(10, dtype=np.float32)
    oof, summary = experiment._compute_oof_baseline_prior_predictions(
        baseline_model="GBLUP",
        fold_dir=tmp_path,
        X_train_base=X,
        y_train=y,
        config={
            "seed": 2026,
            "baselines": {
                "rscript_path": "Rscript",
                "sommer_method": "mmer",
            },
        },
        fold_id=1,
        n_splits=5,
    )

    assert oof.shape == (10,)
    assert len(calls) == 5
    assert {call["n_train"] for call in calls} == {8}
    assert {call["n_test"] for call in calls} == {2}
    assert summary["oof_splits"] == 5
