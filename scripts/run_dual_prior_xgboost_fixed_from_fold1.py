from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tabicl_gs.config import deep_update, load_experiment_config
from tabicl_gs.pipeline.dual_prior_fold_search import run_dual_prior_fixed_block_on_fold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-block dual-prior XGBoost evaluation on selected folds.")
    parser.add_argument("--config", default="configs/tabicl_block/window_tabicl_group_shared_gate_xgboost_prior.yaml")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--trait-col", default=None)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--fold-ids", nargs="+", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_experiment_config(args.config)
    runtime_override = {}
    if args.trait_col:
        runtime_override["trait_col"] = args.trait_col
    if runtime_override:
        base_config = deep_update(base_config, runtime_override)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold_id in args.fold_ids:
        row = run_dual_prior_fixed_block_on_fold(
            base_config=base_config,
            fold_id=int(fold_id),
            group_size=int(args.group_size),
            output_dir=output_root / f"fold_{int(fold_id)}",
        )
        rows.append(row)
        pd.DataFrame(rows).to_csv(output_root / "fold_summary.csv", index=False)
    print(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
