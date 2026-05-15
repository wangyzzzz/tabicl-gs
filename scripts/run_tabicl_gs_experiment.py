from __future__ import annotations

import argparse
from pathlib import Path

from tabicl_gs.config import deep_update, load_experiment_config
from tabicl_gs.pipeline.experiment import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TabICLv2 two-stage GS prototype.")
    parser.add_argument("--config", default="configs/tabicl_block/base.yaml")
    parser.add_argument("--strategy", choices=["random", "window"], required=True)
    parser.add_argument("--plink-prefix", default=None)
    parser.add_argument("--phenotype-csv", default=None)
    parser.add_argument("--phenotype-sample-id-col", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--trait-col", default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--fold-ids", nargs="+", type=int, default=None)
    parser.add_argument("--stage1-norm", default=None)
    parser.add_argument("--stage1-norms", nargs="+", default=None)
    parser.add_argument("--stage1-n-estimators", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    override_path = Path(f"configs/tabicl_block/{args.strategy}.yaml")
    config = load_experiment_config(args.config, override_path)
    runtime_override = {}
    if args.plink_prefix:
        runtime_override["plink_prefix"] = args.plink_prefix
    if args.phenotype_csv:
        runtime_override["phenotype_csv"] = args.phenotype_csv
    if args.phenotype_sample_id_col:
        runtime_override["phenotype_sample_id_col"] = args.phenotype_sample_id_col
    if args.output_dir:
        runtime_override["output_dir"] = args.output_dir
    if args.trait_col:
        runtime_override["trait_col"] = args.trait_col
    if args.stage1_norms:
        runtime_override.setdefault("stage1", {}).setdefault("tabicl", {})["norm_methods"] = args.stage1_norms
    elif args.stage1_norm:
        runtime_override.setdefault("stage1", {}).setdefault("tabicl", {})["norm_methods"] = [args.stage1_norm]
    if args.stage1_n_estimators is not None:
        runtime_override.setdefault("stage1", {}).setdefault("tabicl", {})["n_estimators"] = args.stage1_n_estimators
    if args.group_size is not None:
        runtime_override["group_size"] = args.group_size
    if runtime_override:
        config = deep_update(config, runtime_override)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    else:
        config["output_dir"] = f"outputs/tabicl_block/{args.strategy}"
    metrics = run_experiment(config, max_folds=args.max_folds, fold_ids=args.fold_ids)
    print(metrics)


if __name__ == "__main__":
    main()
