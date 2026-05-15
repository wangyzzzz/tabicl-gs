import numpy as np

from tabicl_gs.models.vi_sparse_prior import VISparsePriorRegressor


def test_vi_sparse_prior_fit_predict_shape_and_effects():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(48, 12)).astype(np.float32)
    beta = np.zeros(12, dtype=np.float32)
    beta[[1, 5, 9]] = np.array([1.2, -0.8, 0.5], dtype=np.float32)
    y = (X @ beta + rng.normal(scale=0.05, size=48)).astype(np.float32)

    model = VISparsePriorRegressor(
        max_epochs=180,
        lr=0.03,
        prior_precision=1.0,
        random_state=7,
        device="cpu",
    )
    model.fit(X, y)
    pred = model.predict(X[:6])

    assert pred.shape == (6,)
    assert np.isfinite(pred).all()
    assert model.coef_.shape == (12,)
    assert model.coef_var_.shape == (12,)
    assert np.isfinite(model.coef_).all()
    assert np.isfinite(model.coef_var_).all()

    corr = float(np.corrcoef(y, model.predict(X))[0, 1])
    assert corr > 0.9


def test_vi_sparse_prior_returns_block_scores():
    rng = np.random.default_rng(321)
    X = rng.normal(size=(40, 8)).astype(np.float32)
    y = (1.5 * X[:, 0] - 0.5 * X[:, 6]).astype(np.float32)
    block_summaries = [
        {"snp_indices": [0, 1, 2]},
        {"snp_indices": [3, 4, 5]},
        {"snp_indices": [6, 7]},
    ]

    model = VISparsePriorRegressor(max_epochs=120, lr=0.03, random_state=11, device="cpu")
    model.fit(X, y)
    scores = model.block_prior_scores(block_summaries)

    assert scores.shape == (3,)
    assert np.isfinite(scores).all()
    assert np.argmax(scores) in {0, 2}
    assert scores[[0, 2]].max() > scores[1]


def test_vi_sparse_prior_scales_prior_for_high_dimensional_snps():
    model = VISparsePriorRegressor(prior_scale=1.0, scale_prior_by_n_features=True)

    assert np.isclose(model._effective_prior_scale(10000), 0.01)
    assert np.isclose(model._effective_prior_scale(100), 0.1)


def test_vi_sparse_prior_calibration_collapses_weak_or_negative_train_signal():
    y = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
    raw_train = -100.0 * y
    raw_test = np.array([-10000.0, 0.0, 10000.0], dtype=np.float32)
    model = VISparsePriorRegressor(
        calibration_enabled=True,
        min_train_corr_for_prior=0.05,
        max_calibration_z=3.0,
    )
    model.y_mean_ = float(np.mean(y))
    model.y_scale_ = float(np.std(y))

    model._fit_prediction_calibration(raw_train, y)
    pred = model._calibrate_raw_predictions(raw_test)

    assert model.calibration_train_corr_ < 0.0
    assert model.calibration_slope_ == 0.0
    assert np.allclose(pred, np.full_like(raw_test, np.mean(y), dtype=np.float32))


def test_vi_sparse_prior_calibration_clips_extrapolated_predictions():
    y = np.linspace(-2.0, 2.0, 40, dtype=np.float32)
    raw_train = 50.0 * y
    raw_test = np.array([-1e6, 0.0, 1e6], dtype=np.float32)
    model = VISparsePriorRegressor(
        calibration_enabled=True,
        min_train_corr_for_prior=0.05,
        max_calibration_z=2.0,
    )
    model.y_mean_ = float(np.mean(y))
    model.y_scale_ = float(np.std(y))

    model._fit_prediction_calibration(raw_train, y)
    pred = model._calibrate_raw_predictions(raw_test)

    assert model.calibration_train_corr_ > 0.99
    assert np.isfinite(pred).all()
    assert pred.max() <= np.mean(y) + 2.1 * np.std(y)
    assert pred.min() >= np.mean(y) - 2.1 * np.std(y)
