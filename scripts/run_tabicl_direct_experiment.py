from __future__ import annotations

import argparse

from tabicl_gs.config import deep_update, load_experiment_config
from tabicl_gs.pipeline.direct_experiment import run_direct_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run direct 10000-SNP TabICLv2 experiment.")
    parser.add_argument("--config", default="configs/tabicl_direct/base.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--trait-col", default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    runtime_override = {}
    if args.output_dir:
        runtime_override["output_dir"] = args.output_dir
    if args.trait_col:
        runtime_override["trait_col"] = args.trait_col
    if runtime_override:
        config = deep_update(config, runtime_override)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    metrics = run_direct_experiment(config, max_folds=args.max_folds)
    print(metrics)


if __name__ == "__main__":
    main()
