import numpy as np

from tabicl_gs.models.sample_mixture import SampleWiseMixtureRegressor


def test_sample_mixture_regressor_fit_predict_shape():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(20, 6)).astype(np.float32)
    expert_pred = (X[:, 0] + 0.2 * X[:, 1]).astype(np.float32)
    prior_pred = (-0.5 * X[:, 0] + 0.4 * X[:, 2]).astype(np.float32)
    y = np.where(X[:, 0] > 0, expert_pred, prior_pred).astype(np.float32)

    model = SampleWiseMixtureRegressor(
        hidden_dim=8,
        dropout=0.0,
        lr=1e-2,
        weight_decay=0.0,
        max_epochs=50,
        batch_size=None,
        device="cpu",
        random_state=7,
    )
    model.fit(X, y, expert_pred, prior_pred)
    pred = model.predict(X[:5], expert_pred[:5], prior_pred[:5])

    assert pred.shape == (5,)
    assert np.isfinite(pred).all()


def test_sample_mixture_regressor_learns_samplewise_weights():
    rng = np.random.default_rng(321)
    X = rng.normal(size=(40, 4)).astype(np.float32)
    expert_pred = (1.2 * X[:, 0] - 0.1 * X[:, 1]).astype(np.float32)
    prior_pred = (-0.9 * X[:, 0] + 0.3 * X[:, 2]).astype(np.float32)
    y = np.where(X[:, 0] > 0, expert_pred, prior_pred).astype(np.float32)

    model = SampleWiseMixtureRegressor(
        hidden_dim=8,
        dropout=0.0,
        lr=5e-3,
        weight_decay=0.0,
        max_epochs=120,
        batch_size=None,
        device="cpu",
        random_state=11,
    )
    model.fit(X, y, expert_pred, prior_pred)
    pred = model.predict(X, expert_pred, prior_pred)

    corr = float(np.corrcoef(y, pred)[0, 1])
    assert corr > 0.95
