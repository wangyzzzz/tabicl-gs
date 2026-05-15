from __future__ import annotations

import pandas as pd


def summarize_metrics_frame(frame: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [column for column in ["pearson", "rmse", "mae", "r2"] if column in frame.columns]
    return frame.groupby(["strategy", "model"], as_index=False)[metric_columns].mean()
