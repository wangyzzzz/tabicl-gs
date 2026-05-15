from __future__ import annotations

import argparse

from tabicl_gs.config import deep_update, load_experiment_config
from tabicl_gs.pipeline.dual_prior_fold_search import run_fold1_dual_prior_block_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fold1 inner-CV block search for block-level stage2 models.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--trait-col", default=None)
    parser.add_argument("--min-block", type=int, required=True)
    parser.add_argument("--max-block", type=int, required=True)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--inner-folds", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_experiment_config(args.config)
    runtime_override = {}
    if args.trait_col:
        runtime_override["trait_col"] = args.trait_col
    if args.output_root:
        runtime_override["output_dir"] = args.output_root
    if runtime_override:
        base_config = deep_update(base_config, runtime_override)
    summary = run_fold1_dual_prior_block_search(
        base_config=base_config,
        output_root=args.output_root,
        min_block=args.min_block,
        max_block=args.max_block,
        n_trials=args.n_trials,
        seed=args.seed,
        inner_folds=args.inner_folds,
    )
    print(summary)


if __name__ == "__main__":
    main()

