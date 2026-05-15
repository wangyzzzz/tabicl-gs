import numpy as np

from tabicl_gs.pipeline.block_quality import apply_block_weights, compute_block_weights


def test_compute_block_weights_prefers_high_r2_and_low_variance():
    block_summaries = [
        {"scalar_train_pearson": 0.8},
        {"scalar_train_pearson": 0.2},
    ]
    scores, weights = compute_block_weights(block_summaries, {})
    assert scores[0] > scores[1]
    assert weights[0] > weights[1]


def test_apply_block_weights_scales_each_block_feature_matrix():
    features = [np.ones((2, 3), dtype=np.float32), np.ones((2, 3), dtype=np.float32) * 2]
    weighted = apply_block_weights(features, np.array([0.5, 1.5], dtype=np.float32))
    assert np.allclose(weighted[0], 0.5)
    assert np.allclose(weighted[1], 3.0)
