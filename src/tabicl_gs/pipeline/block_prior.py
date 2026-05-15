from __future__ import annotations

from typing import Sequence

import numpy as np


def aggregate_beta_to_block_prior(
    beta: np.ndarray,
    block_summaries: list[dict],
    method: str = "l2",
) -> np.ndarray:
    beta = np.asarray(beta, dtype=np.float32).reshape(-1)
    priors = []
    for summary in block_summaries:
        snp_ids = summary.get("snp_ids") or []
        if not snp_ids:
            priors.append(0.0)
            continue
        indices = summary.get("snp_indices")
        if indices is None:
            raise ValueError("block_summaries must include snp_indices for block prior aggregation.")
        values = beta[np.asarray(indices, dtype=int)]
        if method == "l1":
            priors.append(float(np.sum(np.abs(values))))
        elif method == "l2":
            priors.append(float(np.sqrt(np.sum(values**2))))
        elif method == "mean_abs":
            priors.append(float(np.mean(np.abs(values))))
        else:
            raise ValueError(f"Unsupported block prior method: {method}")
    prior = np.asarray(priors, dtype=np.float32)
    if prior.size == 0:
        return prior
    scale = float(prior.std())
    if scale <= 1e-8:
        return np.zeros_like(prior, dtype=np.float32)
    return ((prior - float(prior.mean())) / scale).astype(np.float32)


def compute_sample_block_prior_predictions(
    X: np.ndarray,
    beta: np.ndarray,
    block_summaries: list[dict],
    normalize: bool = True,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32).reshape(-1)
    outputs = []
    for summary in block_summaries:
        indices = summary.get("snp_indices")
        if indices is None:
            raise ValueError("block_summaries must include snp_indices for sample-specific block prior predictions.")
        index_array = np.asarray(indices, dtype=int)
        if index_array.size == 0:
            outputs.append(np.zeros((X.shape[0], 1), dtype=np.float32))
            continue
        block_pred = X[:, index_array] @ beta[index_array]
        outputs.append(np.asarray(block_pred, dtype=np.float32).reshape(-1, 1))

    if not outputs:
        return np.zeros((X.shape[0], 0), dtype=np.float32)

    out = np.concatenate(outputs, axis=1).astype(np.float32)
    if not normalize or out.size == 0:
        return out

    mean = out.mean(axis=0, keepdims=True)
    std = out.std(axis=0, keepdims=True)
    std = np.where(std <= 1e-8, 1.0, std)
    return ((out - mean) / std).astype(np.float32)


def build_block_level_prior_matrix(
    train_predictions: Sequence[np.ndarray],
    test_predictions: Sequence[np.ndarray],
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if len(train_predictions) != len(test_predictions):
        raise ValueError(
            f"train_predictions length ({len(train_predictions)}) must match test_predictions length ({len(test_predictions)})."
        )
    if not train_predictions:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)

    train_cols = [np.asarray(pred, dtype=np.float32).reshape(-1, 1) for pred in train_predictions]
    test_cols = [np.asarray(pred, dtype=np.float32).reshape(-1, 1) for pred in test_predictions]
    train_out = np.concatenate(train_cols, axis=1).astype(np.float32)
    test_out = np.concatenate(test_cols, axis=1).astype(np.float32)

    if not normalize or train_out.size == 0:
        return train_out, test_out

    mean = train_out.mean(axis=0, keepdims=True)
    std = train_out.std(axis=0, keepdims=True)
    std = np.where(std <= 1e-8, 1.0, std)
    return ((train_out - mean) / std).astype(np.float32), ((test_out - mean) / std).astype(np.float32)


def append_block_prior_token(
    X_train_stage2: np.ndarray,
    X_test_stage2: np.ndarray | None,
    block_prior: Sequence[float],
) -> tuple[np.ndarray, np.ndarray | None]:
    prior = np.asarray(block_prior, dtype=np.float32).reshape(1, -1)
    prior_train = np.repeat(prior, X_train_stage2.shape[0], axis=0)
    train_out = np.concatenate([X_train_stage2.astype(np.float32), prior_train], axis=1).astype(np.float32)
    if X_test_stage2 is None:
        return train_out, None
    prior_test = np.repeat(prior, X_test_stage2.shape[0], axis=0)
    test_out = np.concatenate([X_test_stage2.astype(np.float32), prior_test], axis=1).astype(np.float32)
    return train_out, test_out


def interleave_block_prior_features(
    X_stage2: np.ndarray,
    block_summaries: list[dict],
    block_prior: Sequence[float] | np.ndarray,
    feature_mode: str = "reduced",
) -> np.ndarray:
    X_stage2 = np.asarray(X_stage2, dtype=np.float32)
    prior = np.asarray(block_prior, dtype=np.float32)
    if prior.ndim == 1:
        if len(block_summaries) != len(prior):
            raise ValueError(
                f"block_summaries length ({len(block_summaries)}) must match block_prior length ({len(prior)})."
            )
        prior_matrix = np.repeat(prior.reshape(1, -1), X_stage2.shape[0], axis=0).astype(np.float32)
    elif prior.ndim == 2:
        if prior.shape[0] != X_stage2.shape[0]:
            raise ValueError(
                f"Sample dimension mismatch between X_stage2 ({X_stage2.shape[0]}) and block_prior ({prior.shape[0]})."
            )
        if prior.shape[1] != len(block_summaries):
            raise ValueError(
                f"block_summaries length ({len(block_summaries)}) must match block_prior width ({prior.shape[1]})."
            )
        prior_matrix = prior.astype(np.float32)
    else:
        raise ValueError(f"block_prior must be 1D or 2D, got shape={prior.shape}")

    pieces = []
    start = 0
    for idx, summary in enumerate(block_summaries):
        dim_key = "raw_embedding_dim" if feature_mode == "raw" else "reduced_embedding_dim"
        block_dim = int(summary[dim_key])
        if summary.get("include_block_scalar", False):
            block_dim += 1
        end = start + block_dim
        pieces.append(X_stage2[:, start:end])
        pieces.append(prior_matrix[:, idx : idx + 1].astype(np.float32))
        start = end
    if start != X_stage2.shape[1]:
        raise ValueError(f"Feature width mismatch after interleave: consumed {start}, total {X_stage2.shape[1]}")
    return np.concatenate(pieces, axis=1).astype(np.float32)
