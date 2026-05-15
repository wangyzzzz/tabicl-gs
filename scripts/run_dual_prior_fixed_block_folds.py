from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tabicl_gs.config import deep_update, load_experiment_config
from tabicl_gs.pipeline.dual_prior_fold_search import (
    run_dual_prior_fixed_block_on_fold,
    run_dual_prior_fixed_block_with_frozen_gate_on_fold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dual-prior fixed-block evaluation on selected folds.")
    parser.add_argument("--config", default="configs/tabicl_block/window_tabicl_group_shared_gate_prior.yaml")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--trait-col", default=None)
    parser.add_argument("--plink-prefix", default=None)
    parser.add_argument("--phenotype-csv", default=None)
    parser.add_argument("--phenotype-sample-id-col", default=None)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--num-groups", type=int, default=None)
    parser.add_argument("--fold-ids", nargs="+", type=int, required=True)
    parser.add_argument("--frozen-gate-summary-path", default=None)
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
    num_groups = getattr(args, "num_groups", None)
    if num_groups is not None:
        runtime_override.setdefault("stage2", {}).setdefault("group_shared_gate", {})["num_groups"] = int(num_groups)
    if runtime_override:
        base_config = deep_update(base_config, runtime_override)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "fold_summary.csv"
    rows = []
    if summary_path.exists():
        rows = pd.read_csv(summary_path).to_dict("records")
    existing_folds = {int(row["fold"]) for row in rows if "fold" in row}
    for fold_id in args.fold_ids:
        if int(fold_id) in existing_folds:
            continue
        if args.frozen_gate_summary_path and int(fold_id) != 1:
            row = run_dual_prior_fixed_block_with_frozen_gate_on_fold(
                base_config=base_config,
                fold_id=int(fold_id),
                group_size=int(args.group_size),
                gate_summary_path=args.frozen_gate_summary_path,
                output_dir=output_root / f"fold_{int(fold_id)}",
            )
        else:
            row = run_dual_prior_fixed_block_on_fold(
                base_config=base_config,
                fold_id=int(fold_id),
                group_size=int(args.group_size),
                output_dir=output_root / f"fold_{int(fold_id)}",
            )
        rows.append(row)
        pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
