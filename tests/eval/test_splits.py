import numpy as np

from tabicl_gs.eval.splits import make_outer_cv_splits


def test_outer_cv_produces_full_partition():
    X = np.zeros((20, 4))
    splits = make_outer_cv_splits(X, n_splits=5, seed=2026)
    test_indices = sorted(idx for _, test_idx in splits for idx in test_idx.tolist())
    assert len(splits) == 5
    assert test_indices == list(range(20))
