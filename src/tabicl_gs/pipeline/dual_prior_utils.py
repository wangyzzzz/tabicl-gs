from __future__ import annotations

from typing import Any

import numpy as np


def reconstruct_train_predictions_from_inner_cache(
    inner_cache: list[dict[str, Any]],
    train_size: int,
    key: str,
) -> np.ndarray:
    out = np.zeros(int(train_size), dtype=np.float32)
    covered = np.zeros(int(train_size), dtype=bool)
    for inner_meta in inner_cache:
        valid_idx = np.asarray(inner_meta["inner_valid_idx"], dtype=np.int64)
        values = np.asarray(inner_meta[key], dtype=np.float32).reshape(-1)
        out[valid_idx] = values
        covered[valid_idx] = True
    if not np.all(covered):
        raise ValueError(f"Inner cache does not fully cover outer-train for key={key}.")
    return out.astype(np.float32)


def resolve_gate_prior_train_predictions(
    cached: dict[str, Any],
    prior_train_source: str = "full_train",
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray] | None]:
    def _collect_prior_candidates(payload: dict[str, Any], suffix: str) -> dict[str, np.ndarray] | None:
        candidates: dict[str, np.ndarray] = {}
        for key, prior_name in (
            (f"bayeslasso_{suffix}", "BayesLasso"),
            (f"rkhs_{suffix}", "RKHS"),
        ):
            if key in payload:
                candidates[prior_name] = np.asarray(payload[key], dtype=np.float32)
        return candidates or None

    source = str(prior_train_source).lower()
    if source == "full_train":
        bayesb_train = np.asarray(cached["bayesb_train"], dtype=np.float32)
        gblup_train = np.asarray(cached["gblup_train"], dtype=np.float32)
        bayes_candidates = _collect_prior_candidates(cached, "train")
        return bayesb_train, gblup_train, bayes_candidates
    if source == "inner_oof":
        train_size = int(np.asarray(cached["y_outer_train"], dtype=np.float32).shape[0])
        inner_cache = list(cached.get("inner_cache", []))
        if not inner_cache:
            raise ValueError("inner_oof prior_train_source requires non-empty inner_cache.")
        bayesb_train = reconstruct_train_predictions_from_inner_cache(inner_cache, train_size, "bayesb_valid")
        gblup_train = reconstruct_train_predictions_from_inner_cache(inner_cache, train_size, "gblup_valid")
        bayes_candidates = {}
        if any("bayeslasso_valid" in inner_meta for inner_meta in inner_cache):
            bayes_candidates["BayesLasso"] = reconstruct_train_predictions_from_inner_cache(
                inner_cache, train_size, "bayeslasso_valid"
            )
        if any("rkhs_valid" in inner_meta for inner_meta in inner_cache):
            bayes_candidates["RKHS"] = reconstruct_train_predictions_from_inner_cache(
                inner_cache, train_size, "rkhs_valid"
            )
        if not bayes_candidates:
            bayes_candidates = None
        return bayesb_train, gblup_train, bayes_candidates
    raise ValueError(f"Unsupported prior_train_source: {prior_train_source}")
