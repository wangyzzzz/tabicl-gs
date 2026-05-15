from __future__ import annotations

import argparse

from tabicl_gs.config import deep_update, load_experiment_config
from tabicl_gs.pipeline.staged_tabicl_xgboost_search import build_pure_tabicl_oof_feature_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build inner-fold feature cache for a fixed best block on pure TabICL stage1.")
    parser.add_argument("--config", default="configs/tabicl_block/window_tabicl_group_shared_gate_prior.yaml")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--trait-col", default=None)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--fold-id", type=int, default=1)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_experiment_config(args.config)
    runtime_override = {}
    if args.trait_col:
        runtime_override["trait_col"] = args.trait_col
    if runtime_override:
        base_config = deep_update(base_config, runtime_override)
    cache = build_pure_tabicl_oof_feature_cache(
        base_config=base_config,
        fold_id=args.fold_id,
        group_size=args.group_size,
        cache_root=args.output_root,
        inner_folds=args.inner_folds,
        force_rebuild=bool(args.force_rebuild),
    )
    print(cache["metadata"])


if __name__ == "__main__":
    main()

