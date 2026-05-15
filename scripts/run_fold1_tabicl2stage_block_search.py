from __future__ import annotations

import argparse

from tabicl_gs.config import deep_update, load_experiment_config
from tabicl_gs.pipeline.inner_oof_tabicl_search import run_fold1_tabicl2stage_block_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fold1 Optuna block search with no-prior TabICL->TabICL.")
    parser.add_argument("--config", default="configs/tabicl_block/window_tabicl_dynamic99_traitscan.yaml")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--trait-col", default=None)
    parser.add_argument("--plink-prefix", default=None)
    parser.add_argument("--phenotype-csv", default=None)
    parser.add_argument("--phenotype-sample-id-col", default=None)
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
    if args.plink_prefix:
        runtime_override["plink_prefix"] = args.plink_prefix
    if args.phenotype_csv:
        runtime_override["phenotype_csv"] = args.phenotype_csv
    if args.phenotype_sample_id_col:
        runtime_override["phenotype_sample_id_col"] = args.phenotype_sample_id_col
    if runtime_override:
        base_config = deep_update(base_config, runtime_override)
    summary = run_fold1_tabicl2stage_block_search(
        base_config=base_config,
        output_root=args.output_root,
        min_block=int(args.min_block),
        max_block=int(args.max_block),
        n_trials=int(args.n_trials),
        seed=int(args.seed),
        inner_folds=int(args.inner_folds),
    )
    print(summary)


if __name__ == "__main__":
    main()
