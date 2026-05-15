from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_METRIC_WEIGHTS = {
    "scalar_train_pearson": 1.0,
}

LOWER_IS_BETTER = set()


def _rank_scores(values: list[float], higher_is_better: bool) -> np.ndarray:
    series = pd.Series(values, dtype=float)
    mask = np.isfinite(series.to_numpy())
    if not mask.any():
        return np.full(len(values), 0.5, dtype=np.float32)
    valid = series[mask]
    ranks = valid.rank(method="average", pct=True).to_numpy(dtype=np.float32)
    if not higher_is_better:
        ranks = 1.0 - ranks
    output = np.full(len(values), 0.5, dtype=np.float32)
    output[mask] = ranks
    return output


def compute_block_quality_scores(
    block_summaries: list[dict[str, Any]],
    metric_weights: dict[str, float] | None = None,
) -> np.ndarray:
    metric_weights = metric_weights or DEFAULT_METRIC_WEIGHTS
    accum = np.zeros(len(block_summaries), dtype=np.float32)
    total_weight = 0.0
    for metric_name, metric_weight in metric_weights.items():
        values = [summary.get(metric_name, np.nan) for summary in block_summaries]
        if metric_name == "scalar_train_pearson":
            values = [abs(value) if pd.notna(value) else value for value in values]
        higher_is_better = metric_name not in LOWER_IS_BETTER
        scores = _rank_scores(values, higher_is_better=higher_is_better)
        accum += metric_weight * scores
        total_weight += metric_weight
    if total_weight <= 0:
        return np.full(len(block_summaries), 0.5, dtype=np.float32)
    return accum / total_weight


def compute_block_weights(
    block_summaries: list[dict[str, Any]],
    weighting_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    scores = compute_block_quality_scores(
        block_summaries,
        metric_weights=weighting_config.get("metric_weights"),
    )
    weight_floor = float(weighting_config.get("weight_floor", 0.5))
    weight_ceiling = float(weighting_config.get("weight_ceiling", 1.5))
    weights = weight_floor + scores * (weight_ceiling - weight_floor)
    return scores.astype(np.float32), weights.astype(np.float32)


def apply_block_weights(
    block_features: list[np.ndarray],
    weights: np.ndarray,
) -> list[np.ndarray]:
    return [feature * float(weight) for feature, weight in zip(block_features, weights)]
