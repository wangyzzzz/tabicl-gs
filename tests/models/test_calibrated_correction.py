import numpy as np

from tabicl_gs.models.calibrated_correction import CalibratedCorrectionRegressor


def test_calibrated_correction_regressor_fit_predict_shape():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(24, 5)).astype(np.float32)
    y_tabicl = (0.9 * X[:, 0] + 0.2 * X[:, 1]).astype(np.float32)
    y_bayesb = (-0.6 * X[:, 0] + 0.3 * X[:, 2]).astype(np.float32)
    y = (y_bayesb + (X[:, 0] > 0).astype(np.float32) * (y_tabicl - y_bayesb)).astype(np.float32)

    model = CalibratedCorrectionRegressor(
        hidden_dim=8,
        lr=1e-2,
        weight_decay=0.0,
        max_epochs=80,
        device="cpu",
        random_state=1,
    )
    model.fit(X, y, y_tabicl, y_bayesb)
    pred = model.predict(X[:6], y_tabicl[:6], y_bayesb[:6])

    assert pred.shape == (6,)
    assert np.isfinite(pred).all()


def test_calibrated_correction_regressor_learns_correction_strength():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 4)).astype(np.float32)
    y_tabicl = (1.1 * X[:, 0] - 0.2 * X[:, 1]).astype(np.float32)
    y_bayesb = (-0.8 * X[:, 0] + 0.4 * X[:, 2]).astype(np.float32)
    gate = np.where(X[:, 0] > 0, 1.0, 0.0).astype(np.float32)
    y = (y_bayesb + gate * (y_tabicl - y_bayesb)).astype(np.float32)

    model = CalibratedCorrectionRegressor(
        hidden_dim=8,
        lr=5e-3,
        weight_decay=0.0,
        max_epochs=120,
        device="cpu",
        random_state=3,
    )
    model.fit(X, y, y_tabicl, y_bayesb)
    pred = model.predict(X, y_tabicl, y_bayesb)

    corr = float(np.corrcoef(y, pred)[0, 1])
    assert corr > 0.95


def test_calibrated_correction_regressor_supports_dual_priors():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(60, 5)).astype(np.float32)
    y_tabicl = (1.0 * X[:, 0] - 0.2 * X[:, 1]).astype(np.float32)
    y_bayesb = (-0.8 * X[:, 0] + 0.3 * X[:, 2]).astype(np.float32)
    y_gblup = (0.7 * X[:, 0] + 0.2 * X[:, 3]).astype(np.float32)
    alpha = np.where(X[:, 1] > 0, 1.0, 0.0).astype(np.float32)
    y_prior = alpha * y_bayesb + (1.0 - alpha) * y_gblup
    gate = np.where(X[:, 0] > 0, 1.0, 0.0).astype(np.float32)
    y = (y_prior + gate * (y_tabicl - y_prior)).astype(np.float32)

    model = CalibratedCorrectionRegressor(
        hidden_dim=8,
        lr=5e-3,
        weight_decay=0.0,
        max_epochs=150,
        device="cpu",
        random_state=9,
        use_dual_priors=True,
    )
    model.fit(X, y, y_tabicl, y_bayesb, y_gblup)
    pred = model.predict(X, y_tabicl, y_bayesb, y_gblup)

    corr = float(np.corrcoef(y, pred)[0, 1])
    assert corr > 0.9
