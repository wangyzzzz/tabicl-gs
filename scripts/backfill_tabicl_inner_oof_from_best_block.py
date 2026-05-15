from __future__ import annotations

import argparse
from pathlib import Path

from tabicl_gs.config import deep_update, load_experiment_config
from tabicl_gs.pipeline.inner_oof_tabicl_search import _evaluate_single_group_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill fold1 TabICL inner OOF for a fixed best block without rerunning outer-test."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True, help="Existing no-prior output root, e.g. .../tabicl_tabicl_no_prior")
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--trait-col", default=None)
    parser.add_argument("--plink-prefix", default=None)
    parser.add_argument("--phenotype-csv", default=None)
    parser.add_argument("--phenotype-sample-id-col", default=None)
    parser.add_argument("--inner-folds", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_experiment_config(args.config)
    runtime_override = {
        "group_size": int(args.group_size),
        "output_dir": str(args.output_root),
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
    fold1_dir = output_root / "fold_1"
    fold1_dir.mkdir(parents=True, exist_ok=True)
    row = _evaluate_single_group_size(
        base_config=base_config,
        group_size=int(args.group_size),
        inner_folds=int(args.inner_folds),
        output_dir=fold1_dir,
        save_oof_bundle=True,
        write_best_block=False,
    )
    print(row)


if __name__ == "__main__":
    main()
