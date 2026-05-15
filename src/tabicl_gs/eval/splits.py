from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold


def make_outer_cv_splits(X: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(train_idx, test_idx) for train_idx, test_idx in splitter.split(X)]
