import numpy as np

from tabicl_gs.pipeline.block_prior import (
    aggregate_beta_to_block_prior,
    build_block_level_prior_matrix,
    compute_sample_block_prior_predictions,
    interleave_block_prior_features,
)


def test_aggregate_beta_to_block_prior_supports_l2_summary():
    beta = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    block_summaries = [
        {"snp_indices": [0, 1], "snp_ids": ["a", "b"]},
        {"snp_indices": [2, 3], "snp_ids": ["c", "d"]},
    ]
    prior = aggregate_beta_to_block_prior(beta, block_summaries, method="l2")
    assert prior.shape == (2,)
    assert np.isfinite(prior).all()
    assert prior[1] > prior[0]


def test_interleave_block_prior_features_appends_one_scalar_per_block():
    X = np.array(
        [
            [1.0, 2.0, 10.0, 11.0, 12.0],
            [3.0, 4.0, 20.0, 21.0, 22.0],
        ],
        dtype=np.float32,
    )
    block_summaries = [
        {"reduced_embedding_dim": 2, "include_block_scalar": False},
        {"reduced_embedding_dim": 3, "include_block_scalar": False},
    ]
    prior = np.array([0.1, 0.2], dtype=np.float32)
    out = interleave_block_prior_features(X, block_summaries, prior, feature_mode="reduced")

    assert out.shape == (2, 7)
    assert np.allclose(out[:, :2], X[:, :2])
    assert np.allclose(out[:, 2], 0.1)
    assert np.allclose(out[:, 3:6], X[:, 2:5])
    assert np.allclose(out[:, 6], 0.2)


def test_compute_sample_block_prior_predictions_matches_blockwise_beta_dot_products():
    X = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [0.5, 1.0, -1.0, 2.0],
        ],
        dtype=np.float32,
    )
    beta = np.array([0.1, 0.2, -0.5, 1.0], dtype=np.float32)
    block_summaries = [
        {"snp_indices": [0, 1], "snp_ids": ["a", "b"]},
        {"snp_indices": [2, 3], "snp_ids": ["c", "d"]},
    ]

    out = compute_sample_block_prior_predictions(X, beta, block_summaries, normalize=False)

    expected = np.array(
        [
            [1.0 * 0.1 + 2.0 * 0.2, 3.0 * -0.5 + 4.0 * 1.0],
            [0.5 * 0.1 + 1.0 * 0.2, -1.0 * -0.5 + 2.0 * 1.0],
        ],
        dtype=np.float32,
    )
    assert out.shape == (2, 2)
    assert np.allclose(out, expected)


def test_interleave_block_prior_features_supports_sample_specific_matrix():
    X = np.array(
        [
            [1.0, 2.0, 10.0, 11.0, 12.0],
            [3.0, 4.0, 20.0, 21.0, 22.0],
        ],
        dtype=np.float32,
    )
    block_summaries = [
        {"reduced_embedding_dim": 2, "include_block_scalar": False},
        {"reduced_embedding_dim": 3, "include_block_scalar": False},
    ]
    prior = np.array(
        [
            [0.1, 0.2],
            [0.3, 0.4],
        ],
        dtype=np.float32,
    )

    out = interleave_block_prior_features(X, block_summaries, prior, feature_mode="reduced")

    assert out.shape == (2, 7)
    assert np.allclose(out[:, :2], X[:, :2])
    assert np.allclose(out[:, 2], np.array([0.1, 0.3], dtype=np.float32))
    assert np.allclose(out[:, 3:6], X[:, 2:5])
    assert np.allclose(out[:, 6], np.array([0.2, 0.4], dtype=np.float32))


def test_build_block_level_prior_matrix_stacks_train_and_test_predictions():
    train_preds = [
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([3.0, 4.0], dtype=np.float32),
    ]
    test_preds = [
        np.array([10.0], dtype=np.float32),
        np.array([20.0], dtype=np.float32),
    ]

    train_out, test_out = build_block_level_prior_matrix(train_preds, test_preds, normalize=False)

    assert train_out.shape == (2, 2)
    assert test_out.shape == (1, 2)
    assert np.allclose(train_out, np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32))
    assert np.allclose(test_out, np.array([[10.0, 20.0]], dtype=np.float32))
