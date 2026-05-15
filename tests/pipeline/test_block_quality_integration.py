from tabicl_gs.pipeline.block_quality import compute_block_weights


def test_second_stage_weighting_prefers_high_quality_blocks():
    summaries = [
        {"scalar_train_pearson": 0.8},
        {"scalar_train_pearson": 0.2},
    ]
    scores, weights = compute_block_weights(
        summaries,
        {"metric_weights": {"scalar_train_pearson": 1.0}, "weight_floor": 0.5, "weight_ceiling": 1.5},
    )
    assert scores[0] > scores[1]
    assert weights[0] > weights[1]
