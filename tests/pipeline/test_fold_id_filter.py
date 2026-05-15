from tabicl_gs.pipeline.experiment import _filter_splits_by_fold_ids


def test_filter_splits_by_fold_ids_keeps_requested_order():
    splits = [("fold1_train", "fold1_test"), ("fold2_train", "fold2_test"), ("fold3_train", "fold3_test")]

    filtered = _filter_splits_by_fold_ids(splits, [3, 1])

    assert filtered == [
        ("fold1_train", "fold1_test"),
        ("fold3_train", "fold3_test"),
    ]


def test_filter_splits_by_fold_ids_raises_on_missing_fold():
    splits = [("fold1_train", "fold1_test"), ("fold2_train", "fold2_test")]

    try:
        _filter_splits_by_fold_ids(splits, [4])
    except ValueError as exc:
        assert "Requested fold ids" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing fold id.")
