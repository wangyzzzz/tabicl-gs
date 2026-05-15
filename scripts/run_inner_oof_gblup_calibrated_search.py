from __future__ import annotations

import argparse

from tabicl_gs.config import load_experiment_config
from tabicl_gs.pipeline.fold1_hparam_search import run_inner_oof_hparam_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inner-OOF objective search for GBLUP-CalibratedCorrection.")
    parser.add_argument("--config", default="configs/tabicl_block/window_tabicl_calibrated_correction_gblup_y.yaml")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--fold-id", type=int, required=True)
    parser.add_argument("--group-sizes", nargs="+", type=int, required=True)
    parser.add_argument("--variance-target-pcts", nargs="+", type=int, required=True)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_experiment_config(args.config)
    results = run_inner_oof_hparam_search(
        base_config=base_config,
        output_root=args.output_root,
        fold_id=args.fold_id,
        group_sizes=args.group_sizes,
        variance_target_pcts=args.variance_target_pcts,
        inner_folds=args.inner_folds,
        seed=args.seed,
    )
    print(results)


if __name__ == "__main__":
    main()
