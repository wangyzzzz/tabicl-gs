from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tabicl_gs.eval.metrics import regression_metrics


@dataclass(frozen=True)
class FusionSearchResult:
    weight_tabicl: float
    best_metric_value: float
    train_metrics: dict[str, float]
    fused_predictions: np.ndarray


def fuse_predictions(
    pred_tabicl: np.ndarray,
    pred_baseline: np.ndarray,
    weight_tabicl: float,
) -> np.ndarray:
    pred_tabicl = np.asarray(pred_tabicl, dtype=np.float32)
    pred_baseline = np.asarray(pred_baseline, dtype=np.float32)
    if pred_tabicl.shape != pred_baseline.shape:
        raise ValueError(f"Prediction shapes must match, got {pred_tabicl.shape} vs {pred_baseline.shape}")
    weight = float(weight_tabicl)
    return (weight * pred_tabicl + (1.0 - weight) * pred_baseline).astype(np.float32)


def search_fusion_weight(
    y_train: np.ndarray,
    pred_tabicl_train: np.ndarray,
    pred_baseline_train: np.ndarray,
    metric_name: str = "pearson",
    grid_size: int = 101,
) -> FusionSearchResult:
    if grid_size < 2:
        raise ValueError(f"grid_size must be >= 2, got {grid_size}")
    y_train = np.asarray(y_train, dtype=np.float32)
    pred_tabicl_train = np.asarray(pred_tabicl_train, dtype=np.float32)
    pred_baseline_train = np.asarray(pred_baseline_train, dtype=np.float32)

    if y_train.shape != pred_tabicl_train.shape or y_train.shape != pred_baseline_train.shape:
        raise ValueError(
            f"Training target and predictions must have the same shape, got "
            f"{y_train.shape}, {pred_tabicl_train.shape}, {pred_baseline_train.shape}"
        )

    best_weight = 0.5
    best_value = -np.inf
    best_metrics: dict[str, float] | None = None
    best_pred: np.ndarray | None = None

    for weight in np.linspace(0.0, 1.0, grid_size):
        fused = fuse_predictions(pred_tabicl_train, pred_baseline_train, float(weight))
        metrics = regression_metrics(y_train, fused)
        metric_value = float(metrics[metric_name])
        if metric_value > best_value:
            best_weight = float(weight)
            best_value = metric_value
            best_metrics = metrics
            best_pred = fused

    return FusionSearchResult(
        weight_tabicl=best_weight,
        best_metric_value=best_value,
        train_metrics=best_metrics or {},
        fused_predictions=np.asarray(best_pred, dtype=np.float32),
    )
