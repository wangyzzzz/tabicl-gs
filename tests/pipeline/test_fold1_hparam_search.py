from tabicl_gs.pipeline.fold1_hparam_search import build_search_combinations


def test_build_search_combinations_returns_cartesian_product():
    combos = build_search_combinations([300, 500], [50, 90, 99])

    assert len(combos) == 6
    assert combos[0] == {"group_size": 300, "variance_target": 0.50, "variance_target_pct": 50}
    assert combos[-1] == {"group_size": 500, "variance_target": 0.99, "variance_target_pct": 99}
