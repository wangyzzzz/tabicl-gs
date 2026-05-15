from __future__ import annotations

import argparse

from tabicl_gs.config import deep_update, load_experiment_config
from tabicl_gs.pipeline.staged_tabicl_xgboost_search import fit_group_gate_from_best_xgboost_oof


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit dual-prior group gate from cached best-XGBoost OOF predictions without re-running XGBoost."
    )
    parser.add_argument("--config", default="configs/tabicl_block/window_tabicl_group_shared_gate_prior.yaml")
    parser.add_argument("--feature-cache-root", required=True)
    parser.add_argument("--xgboost-search-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--trait-col", default=None)
    parser.add_argument("--fold-id", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_experiment_config(args.config)
    runtime_override = {}
    if args.trait_col:
        runtime_override["trait_col"] = args.trait_col
    if runtime_override:
        base_config = deep_update(base_config, runtime_override)
    summary = fit_group_gate_from_best_xgboost_oof(
        base_config=base_config,
        fold_id=args.fold_id,
        feature_cache_root=args.feature_cache_root,
        xgboost_search_root=args.xgboost_search_root,
        output_root=args.output_root,
    )
    print(summary)


if __name__ == "__main__":
    main()

