import numpy as np

from tabicl_gs.models.block_weight_pooling import GroupWeightedPoolingRegressor, StaticBlockWeightedRegressor


def test_static_block_weight_regressor_fit_predict_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(24, 12)).astype(np.float32)
    y = rng.normal(size=24).astype(np.float32)
    prior = np.linspace(-1.0, 1.0, 3).astype(np.float32)

    model = StaticBlockWeightedRegressor(
        block_input_dims=[4, 4, 4],
        prior_scores=prior,
        model_dim=8,
        device="cpu",
        max_epochs=3,
    )
    model.fit(X, y)
    pred = model.predict(X[:5])

    assert pred.shape == (5,)
    weights = model.get_block_weights()
    assert weights.shape == (3,)
    assert np.isclose(weights.sum(), 1.0, atol=1e-5)


def test_group_weighted_pooling_regressor_fit_predict_shape():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(24, 12)).astype(np.float32)
    y = rng.normal(size=24).astype(np.float32)
    prior = np.array([-1.0, 0.2, 1.5], dtype=np.float32)

    model = GroupWeightedPoolingRegressor(
        block_input_dims=[4, 4, 4],
        prior_scores=prior,
        num_groups=3,
        model_dim=8,
        device="cpu",
        max_epochs=3,
    )
    model.fit(X, y)
    pred = model.predict(X[:6])

    assert pred.shape == (6,)
    weights = model.get_group_weights()
    assert weights.shape == (3,)
    assert np.isclose(weights.sum(), 1.0, atol=1e-5)

