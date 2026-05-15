from tabicl_gs.data.grouping import (
    build_blocks,
    expected_num_blocks,
    make_random_groups,
    make_window_groups,
    subsample_snp_indices,
)


def test_random_groups_cover_all_snps_without_overlap():
    groups = make_random_groups(list(range(10000)), group_size=100, seed=7)
    flat = [marker for group in groups for marker in group]
    assert len(groups) == 100
    assert len(flat) == 10000
    assert len(set(flat)) == 10000


def test_random_groups_reproducible_with_seed():
    left = make_random_groups(list(range(1000)), group_size=100, seed=3)
    right = make_random_groups(list(range(1000)), group_size=100, seed=3)
    assert left == right


def test_window_groups_preserve_order():
    groups = make_window_groups(list(range(1000)), group_size=100)
    assert groups[0] == list(range(100))
    assert groups[1] == list(range(100, 200))
    assert groups[-1] == list(range(900, 1000))


def test_subsample_snp_indices_is_sorted_and_reproducible():
    left = subsample_snp_indices(n_snps=30000, max_snps=10000, seed=2026)
    right = subsample_snp_indices(n_snps=30000, max_snps=10000, seed=2026)
    assert left == right
    assert len(left) == 10000
    assert left == sorted(left)


def test_build_blocks_dispatches_window_strategy():
    groups = build_blocks(
        snp_indices=list(range(1000)),
        strategy="window",
        group_size=100,
        seed=2026,
    )
    assert groups[0] == list(range(100))


def test_window_groups_can_pad_incomplete_last_block():
    groups = make_window_groups(list(range(10)), group_size=6, pad_incomplete=True, pad_value=-1)
    assert groups == [list(range(6)), [6, 7, 8, 9, -1, -1]]


def test_expected_num_blocks_uses_ceiling():
    assert expected_num_blocks(10000, 300) == 34
