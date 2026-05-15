from __future__ import annotations

import argparse

from tabicl_gs.pipeline.staged_tabicl_xgboost_search import run_fold1_xgboost_search_on_cached_block


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fold1 OOF XGBoost search on cached pure-TabICL block features.")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_fold1_xgboost_search_on_cached_block(
        cache_root=args.cache_root,
        output_root=args.output_root,
        n_trials=args.n_trials,
        seed=args.seed,
    )
    print(summary)


if __name__ == "__main__":
    main()

