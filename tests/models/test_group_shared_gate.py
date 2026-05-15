import numpy as np

from tabicl_gs.models.group_shared_gate import GroupSharedGateRegressor


def test_group_shared_gate_regressor_fit_predict_shape():
    rng = np.random.default_rng(0)
    X_core = rng.normal(size=(40, 6)).astype(np.float32)
    y_tabicl = (0.8 * X_core[:, 0] - 0.2 * X_core[:, 1]).astype(np.float32)
    y_bayesb = (-0.5 * X_core[:, 0] + 0.3 * X_core[:, 2]).astype(np.float32)
    y_gblup = (0.6 * X_core[:, 0] + 0.2 * X_core[:, 3]).astype(np.float32)
    X = np.concatenate([X_core, y_tabicl.reshape(-1, 1), y_bayesb.reshape(-1, 1), y_gblup.reshape(-1, 1)], axis=1)
    y = (0.5 * y_tabicl + 0.3 * y_bayesb + 0.2 * y_gblup).astype(np.float32)

    model = GroupSharedGateRegressor(
        block_input_dims=[3, 3],
        prior_scores=[-1.0, 1.0],
        num_groups=3,
        group_mode="prior_guided_group",
        hidden_dim=8,
        max_epochs=20,
    )
    model.fit(X_core, y, y_tabicl, y_bayesb, y_gblup)
    pred = model.predict(X_core[:5], y_tabicl[:5], y_bayesb[:5], y_gblup[:5])

    assert pred.shape == (5,)
    assert np.isfinite(pred).all()
    summary = model.get_group_summary()
    assert "group_counts" in summary
    assert len(summary["group_counts"]) == 3
    assert "alpha_group" in summary
    assert "w_group" in summary


def test_group_shared_gate_supports_global_oof_group_calibration_with_centroid_assignment():
    X_train_core = np.array(
        [
            [3.0, 1.0, 2.0, 0.5],
            [2.8, 1.2, 2.1, 0.3],
            [-3.0, -1.0, -2.0, -0.5],
            [-2.7, -1.3, -2.2, -0.2],
        ],
        dtype=np.float32,
    )
    y_tabicl = np.array([1.2, 1.1, -1.0, -0.9], dtype=np.float32)
    y_bayesb = np.array([0.8, 0.75, -0.7, -0.65], dtype=np.float32)
    y_gblup = np.array([0.6, 0.55, -0.5, -0.45], dtype=np.float32)
    y = np.array([1.0, 0.95, -0.8, -0.75], dtype=np.float32)

    model = GroupSharedGateRegressor(
        block_input_dims=[2, 2],
        prior_scores=[-1.0, 1.0],
        num_groups=2,
        group_mode="embedding_group",
        assignment_mode="nearest_centroid",
        hidden_dim=8,
        max_epochs=20,
    )

    X_group_train = model.build_group_features(X_train_core)
    model.fit_from_group_features(X_group_train, y, y_tabicl, y_bayesb, y_gblup)

    X_test_core = np.array(
        [
            [2.9, 0.9, 2.2, 0.4],
            [-2.9, -0.8, -2.1, -0.3],
        ],
        dtype=np.float32,
    )
    X_group_test = model.build_group_features(X_test_core)
    group_assignments = model.predict_group_assignments_from_group_features(X_group_test)
    pred = model.predict_from_group_features(
        X_group_test,
        y_tabicl=np.array([1.15, -0.95], dtype=np.float32),
        y_bayesb=np.array([0.78, -0.68], dtype=np.float32),
        y_gblup=np.array([0.58, -0.48], dtype=np.float32),
    )

    assert sorted(np.unique(group_assignments).tolist()) == [0, 1]
    assert int(group_assignments[0]) != int(group_assignments[1])
    assert pred.shape == (2,)
    assert np.isfinite(pred).all()

    summary = model.get_group_summary()
    assert summary["group_counts"] == [2, 2]
    assert "group_centroids" in summary
    assert len(summary["group_centroids"]) == 2
    assert len(summary["alpha_group"]) == 2
    assert len(summary["w_group"]) == 2


def test_group_shared_gate_can_select_bayes_family_by_group():
    X_group = np.array(
        [
            [-2.0],
            [-1.8],
            [1.8],
            [2.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0.0, 0.1, 5.0, 5.1], dtype=np.float32)
    y_tabicl = y.copy()
    y_gblup = np.zeros_like(y)
    y_bayesb = np.array([0.0, 0.1, 1.0, 1.1], dtype=np.float32)
    y_bayeslasso = np.array([1.0, 1.1, 5.0, 5.1], dtype=np.float32)

    model = GroupSharedGateRegressor(
        block_input_dims=[1],
        prior_scores=[1.0],
        num_groups=2,
        group_mode="embedding_group",
        random_state=0,
    )
    model.fit_from_group_features(
        X_group,
        y,
        y_tabicl,
        y_bayesb,
        y_gblup,
        y_bayes_candidates={"BayesLasso": y_bayeslasso},
    )

    summary = model.get_group_summary()
    assert "bayes_family_group" in summary
    assert "BayesB" in summary["bayes_family_group"]
    assert "BayesLasso" in summary["bayes_family_group"]


def test_group_shared_gate_supports_triple_prior_weights():
    X_group = np.array(
        [
            [-2.0],
            [-1.5],
            [1.5],
            [2.0],
        ],
        dtype=np.float32,
    )
    y_tabicl = np.array([0.5, 0.7, 2.3, 2.5], dtype=np.float32)
    y_bayesb = np.array([0.1, 0.2, 1.9, 2.0], dtype=np.float32)
    y_gblup = np.array([0.2, 0.3, 1.5, 1.6], dtype=np.float32)
    y_rkhs = np.array([0.4, 0.6, 2.1, 2.4], dtype=np.float32)
    y = np.array([0.45, 0.65, 2.2, 2.45], dtype=np.float32)

    model = GroupSharedGateRegressor(
        block_input_dims=[1],
        prior_scores=[1.0],
        num_groups=2,
        group_mode="embedding_group",
        random_state=0,
    )
    model.fit_from_group_features(
        X_group,
        y,
        y_tabicl,
        y_bayesb,
        y_gblup,
        y_bayes_candidates={"RKHS": y_rkhs},
    )

    pred = model.predict_from_group_features(
        X_group,
        y_tabicl=y_tabicl,
        y_bayesb=y_bayesb,
        y_gblup=y_gblup,
        y_bayes_candidates={"RKHS": y_rkhs},
    )
    summary = model.get_group_summary()

    assert pred.shape == (4,)
    assert np.isfinite(pred).all()
    assert summary["prior_names"] == ["BayesB", "GBLUP", "RKHS"]
    assert len(summary["prior_weight_group"]) == 2
    for row in summary["prior_weight_group"]:
        assert len(row) == 3
        assert np.isclose(sum(row), 1.0, atol=1e-5)
    assert "w_group" in summary


def test_group_shared_gate_from_summary_keeps_legacy_dual_prior_compatibility():
    summary = {
        "group_mode": "embedding_group",
        "assignment_mode": "nearest_centroid",
        "group_counts": [2, 2],
        "group_probs_mean": [0.5, 0.5],
        "alpha_group": [0.8, 0.3],
        "w_group": [0.4, 0.6],
        "bayes_family_group": ["BayesB", "BayesB"],
        "group_centroids": [[-1.0], [1.0]],
        "scaler_mean": [0.0],
        "scaler_scale": [1.0],
    }
    model = GroupSharedGateRegressor.from_summary(
        summary,
        block_input_dims=[1],
        prior_scores=[1.0],
        random_state=7,
    )
    pred = model.predict_from_group_features(
        np.array([[-1.2], [1.2]], dtype=np.float32),
        y_tabicl=np.array([1.0, 2.0], dtype=np.float32),
        y_bayesb=np.array([0.8, 1.8], dtype=np.float32),
        y_gblup=np.array([0.5, 1.5], dtype=np.float32),
    )

    assert pred.shape == (2,)
    assert np.isfinite(pred).all()
