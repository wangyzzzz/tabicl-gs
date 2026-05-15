import numpy as np

from tabicl_gs.models.block_attention import (
    BlockAttentionRegressor,
    flat_features_to_block_tensor,
)


def test_flat_features_to_block_tensor_zero_pads_variable_dims():
    X = np.array(
        [
            [1.0, 2.0, 10.0, 11.0, 12.0],
            [3.0, 4.0, 20.0, 21.0, 22.0],
        ],
        dtype=np.float32,
    )
    tensor = flat_features_to_block_tensor(X, block_input_dims=[2, 3])
    assert tensor.shape == (2, 2, 3)
    assert np.allclose(tensor[:, 0, :2], X[:, :2])
    assert np.allclose(tensor[:, 0, 2], 0.0)
    assert np.allclose(tensor[:, 1, :], X[:, 2:])


def test_block_attention_regressor_fit_predict_shape():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(12, 7)).astype(np.float32)
    y = (0.5 * X[:, 0] - 0.25 * X[:, 4] + 0.1 * X[:, 6]).astype(np.float32)

    model = BlockAttentionRegressor(
        block_input_dims=[2, 3, 2],
        model_dim=16,
        num_heads=4,
        num_layers=1,
        ff_multiplier=2,
        dropout=0.0,
        lr=1e-3,
        weight_decay=0.0,
        max_epochs=5,
        batch_size=None,
        device="cpu",
        random_state=13,
    )
    model.fit(X, y)
    pred = model.predict(X[:4])

    assert pred.shape == (4,)
    assert np.isfinite(pred).all()


def test_block_attention_regressor_supports_prior_token_mode():
    rng = np.random.default_rng(19)
    # last block is a 1-d prior token
    X = rng.normal(size=(10, 6)).astype(np.float32)
    y = (0.3 * X[:, 0] + 0.2 * X[:, 5]).astype(np.float32)

    model = BlockAttentionRegressor(
        block_input_dims=[2, 3, 1],
        model_dim=16,
        num_heads=4,
        num_layers=1,
        ff_multiplier=2,
        dropout=0.0,
        lr=1e-3,
        weight_decay=0.0,
        max_epochs=5,
        batch_size=None,
        device="cpu",
        random_state=23,
        use_prior_token=True,
    )
    model.fit(X, y)
    pred = model.predict(X[:3])

    assert pred.shape == (3,)
    assert np.isfinite(pred).all()


def test_block_attention_regressor_supports_block_prior_mode():
    rng = np.random.default_rng(31)
    # last 3 dims are reserved for block prior features
    X = rng.normal(size=(12, 10)).astype(np.float32)
    y = (0.2 * X[:, 0] + 0.4 * X[:, 4] - 0.1 * X[:, 8]).astype(np.float32)

    model = BlockAttentionRegressor(
        block_input_dims=[2, 3, 2],
        model_dim=16,
        num_heads=4,
        num_layers=1,
        ff_multiplier=2,
        dropout=0.0,
        lr=1e-3,
        weight_decay=0.0,
        max_epochs=5,
        batch_size=None,
        device="cpu",
        random_state=37,
        use_block_prior=True,
    )
    model.fit(X, y)
    pred = model.predict(X[:4])

    assert pred.shape == (4,)
    assert np.isfinite(pred).all()
