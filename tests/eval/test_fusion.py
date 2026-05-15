import numpy as np

from tabicl_gs.eval.fusion import fuse_predictions, search_fusion_weight


def test_fuse_predictions_weighted_average():
    pred_tabicl = np.array([1.0, 3.0], dtype=np.float32)
    pred_baseline = np.array([3.0, 1.0], dtype=np.float32)
    fused = fuse_predictions(pred_tabicl, pred_baseline, weight_tabicl=0.25)
    assert np.allclose(fused, np.array([2.5, 1.5], dtype=np.float32))


def test_search_fusion_weight_prefers_better_source():
    y_train = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    pred_tabicl = np.array([0.0, 1.1, 1.9, 3.0], dtype=np.float32)
    pred_baseline = np.array([0.3, 0.7, 2.4, 2.7], dtype=np.float32)

    result = search_fusion_weight(
        y_train=y_train,
        pred_tabicl_train=pred_tabicl,
        pred_baseline_train=pred_baseline,
        metric_name="pearson",
        grid_size=51,
    )

    assert result.weight_tabicl > 0.5
    assert result.train_metrics["pearson"] >= 0.99
