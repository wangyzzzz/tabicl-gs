from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tabicl_gs.config import deep_update, load_experiment_config
from tabicl_gs.pipeline.experiment import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-block no-prior evaluation with run_experiment on selected folds.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--trait-col", default=None)
    parser.add_argument("--plink-prefix", default=None)
    parser.add_argument("--phenotype-csv", default=None)
    parser.add_argument("--phenotype-sample-id-col", default=None)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--fold-ids", nargs="+", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_experiment_config(args.config)
    runtime_override = {
        "group_size": int(args.group_size),
        "output_dir": args.output_root,
    }
    if args.trait_col:
        runtime_override["trait_col"] = args.trait_col
    if args.plink_prefix:
        runtime_override["plink_prefix"] = args.plink_prefix
    if args.phenotype_csv:
        runtime_override["phenotype_csv"] = args.phenotype_csv
    if args.phenotype_sample_id_col:
        runtime_override["phenotype_sample_id_col"] = args.phenotype_sample_id_col
    base_config = deep_update(base_config, runtime_override)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    metrics = run_experiment(base_config, fold_ids=[int(f) for f in args.fold_ids])
    model_name = str(base_config["main_models"][0]["name"])
    rows = metrics[metrics["model"] == model_name].copy()
    rows.to_csv(output_root / "fold_summary.csv", index=False)
    print(rows)


if __name__ == "__main__":
    main()
