from __future__ import annotations

import argparse

from tabicl_gs.config import load_experiment_config
from tabicl_gs.pipeline.variance_target_sweep import run_variance_target_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run explained-variance-target sweep for TabICLv2 2-stage window.")
    parser.add_argument("--config", default="configs/tabicl_block/window_tabicl_optuna20.yaml")
    parser.add_argument("--trait-col", required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--include-block-scalar", choices=["true", "false"], required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--variance-targets", nargs="+", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_experiment_config(args.config)
    summary = run_variance_target_sweep(
        base_config=base_config,
        trait_col=args.trait_col,
        group_size=args.group_size,
        include_block_scalar=(args.include_block_scalar == "true"),
        variance_targets=args.variance_targets,
        output_root=args.output_root,
    )
    print(summary)


if __name__ == "__main__":
    main()
