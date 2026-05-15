import pandas as pd

from tabicl_gs.eval.summary import summarize_metrics_frame


def test_summarize_metrics_frame_groups_by_strategy_and_model():
    frame = pd.DataFrame(
        [
            {"strategy": "random", "model": "A", "pearson": 0.5, "rmse": 2.0, "mae": 1.0, "r2": 0.1},
            {"strategy": "random", "model": "A", "pearson": 0.7, "rmse": 4.0, "mae": 3.0, "r2": 0.3},
            {"strategy": "window", "model": "B", "pearson": 0.9, "rmse": 1.0, "mae": 0.5, "r2": 0.6},
        ]
    )

    summary = summarize_metrics_frame(frame)

    assert list(summary.columns) == ["strategy", "model", "pearson", "rmse", "mae", "r2"]
    row = summary[(summary["strategy"] == "random") & (summary["model"] == "A")].iloc[0]
    assert row["pearson"] == 0.6
    assert row["rmse"] == 3.0
