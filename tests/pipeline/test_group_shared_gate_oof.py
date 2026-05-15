import numpy as np

from tabicl_gs.pipeline.dual_prior_fold_search import _build_group_features_from_stage2


def test_build_group_features_from_stage2_prior_guided_group_reduces_to_group_summaries():
    X_stage2 = np.array(
        [
            [1.0, 3.0, 2.0, 4.0],
            [2.0, 6.0, 3.0, 9.0],
        ],
        dtype=np.float32,
    )
    block_summaries = [
        {"reduced_embedding_dim": 2},
        {"reduced_embedding_dim": 2},
    ]
    prior_scores = np.array([-1.0, 1.0], dtype=np.float32)

    group_features, block_group_ids = _build_group_features_from_stage2(
        X_stage2,
        block_summaries,
        group_mode="prior_guided_group",
        num_groups=2,
        prior_scores=prior_scores,
        include_block_scalar=False,
    )

    assert block_group_ids.tolist() == [0, 1]
    assert group_features.shape == (2, 2)
    assert np.allclose(group_features, np.array([[2.236068, 3.162278], [4.472136, 6.708204]], dtype=np.float32), atol=1e-5)


def test_build_group_features_from_stage2_embedding_group_keeps_blockwise_summary_width():
    X_stage2 = np.array(
        [
            [1.0, 3.0, 2.0, 4.0],
            [2.0, 6.0, 3.0, 9.0],
        ],
        dtype=np.float32,
    )
    block_summaries = [
        {"reduced_embedding_dim": 2},
        {"reduced_embedding_dim": 2},
    ]

    group_features, block_group_ids = _build_group_features_from_stage2(
        X_stage2,
        block_summaries,
        group_mode="embedding_group",
        num_groups=3,
        prior_scores=None,
        include_block_scalar=False,
    )

    assert block_group_ids is None
    assert group_features.shape == (2, 2)
    assert np.allclose(group_features, np.array([[2.236068, 3.162278], [4.472136, 6.708204]], dtype=np.float32), atol=1e-5)
