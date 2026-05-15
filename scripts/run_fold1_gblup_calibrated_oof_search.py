from __future__ import annotations

import argparse

from tabicl_gs.config import load_experiment_config
from tabicl_gs.pipeline.fold1_hparam_search import rerun_best_oof_and_non_oof, run_fold1_optuna_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fold1 OOF hyperparameter search for GBLUP-CalibratedCorrection.")
    parser.add_argument("--config", default="configs/tabicl_block/window_tabicl_calibrated_correction_gblup_y.yaml")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--group-sizes", nargs="+", type=int, required=True)
    parser.add_argument("--variance-target-pcts", nargs="+", type=int, required=True)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_experiment_config(args.config)
    trials = run_fold1_optuna_search(
        base_config=base_config,
        output_root=args.output_root,
        group_sizes=args.group_sizes,
        variance_target_pcts=args.variance_target_pcts,
        n_trials=args.n_trials,
        seed=args.seed,
    )
    best = trials.sort_values("pearson", ascending=False).iloc[0].to_dict()
    comparison = rerun_best_oof_and_non_oof(
        base_config=base_config,
        output_root=args.output_root,
        best_params=best,
    )
    print(trials)
    print(comparison)


if __name__ == "__main__":
    main()
