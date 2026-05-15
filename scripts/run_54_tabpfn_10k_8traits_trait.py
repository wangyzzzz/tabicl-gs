from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tabicl_gs.config import deep_update, load_experiment_config
from tabicl_gs.pipeline.experiment import run_experiment
from tabicl_gs.pipeline.inner_oof_tabicl_search import run_fold1_tabicl2stage_block_search
from scripts.run_54_marker_count_decoupled_trait import (
    _dataset_slug,
    _ensure_dataset_resources,
    _resolve_trait_col_name,
    _trait_slug,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 10K no-prior TabPFN for the SNP-count 8-trait panel.")
    parser.add_argument("--dataset", required=True, choices=["rice529", "cotton1245", "soybean951", "wheat406"])
    parser.add_argument("--trait-col", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config-no-prior", default="configs/tabicl_block/window_tabpfn_dynamic_traitscan.yaml")
    parser.add_argument("--marker-count", type=int, default=10000)
    parser.add_argument("--fold-ids", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--block-min", type=int, default=None)
    parser.add_argument("--block-max", type=int, default=None)
    parser.add_argument("--block-trials", type=int, default=10)
    parser.add_argument("--block-inner-folds", type=int, default=3)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _auto_block_bounds(n_snps: int) -> tuple[int, int]:
    min_block = min(int(n_snps), max(50, int(round(float(n_snps) / 50.0))))
    max_block = min(int(n_snps), max(min_block, int(round(float(n_snps) / 7.0))))
    return int(min_block), int(max_block)


def _preferred_block_bounds(marker_count: int, n_snps: int) -> tuple[int, int]:
    if int(marker_count) == 10000:
        min_block, max_block = 200, 1500
        min_block = min(int(n_snps), int(min_block))
        max_block = min(int(n_snps), int(max_block))
        max_block = max(min_block, max_block)
        return int(min_block), int(max_block)
    return _auto_block_bounds(int(n_snps))


def _complete(no_prior_root: Path) -> bool:
    if not (no_prior_root / "fold_metrics.csv").exists():
        return False
    required = [
        no_prior_root / "fold_1" / "tabicl_inner_oof_predictions.npy",
        no_prior_root / "fold_1" / "tabicl_inner_oof_targets.npy",
        no_prior_root / "fold_1" / "tabicl_inner_oof_summary.json",
    ]
    return all(path.exists() for path in required)


def main() -> None:
    args = parse_args()
    dataset = str(args.dataset)
    dataset_slug = _dataset_slug(dataset)
    trait_input = str(args.trait_col)
    trait_slug = _trait_slug(trait_input)
    output_root = Path(args.output_root)

    resources = _ensure_dataset_resources(dataset, int(args.marker_count), int(args.seed))
    resolved_trait_col = _resolve_trait_col_name(list(resources["trait_cols"]), trait_input)
    trait_root = output_root / dataset_slug / trait_slug
    block_search_root = trait_root / "fold1_tabpfn_block_search"
    no_prior_root = trait_root / "no_prior_tabpfn"
    manifest_path = trait_root / "run_manifest.json"

    block_min, block_max = _preferred_block_bounds(int(args.marker_count), int(resources["subset_snp_count"]))
    if args.block_min is not None:
        block_min = int(args.block_min)
    if args.block_max is not None:
        block_max = int(args.block_max)
    if block_min > block_max:
        raise ValueError(f"block_min ({block_min}) cannot exceed block_max ({block_max}).")

    best_block_path = block_search_root / "best_block.json"
    if not best_block_path.exists():
        base_config = load_experiment_config(str(args.config_no_prior))
        base_config = deep_update(
            base_config,
            {
                "trait_col": resolved_trait_col,
                "plink_prefix": str(resources["subset_plink_prefix"]),
                "phenotype_csv": str(resources["prepared_phenotype_csv"]),
                "phenotype_sample_id_col": str(resources["sample_id_col"]),
                "max_snps": int(resources["subset_snp_count"]),
            },
        )
        run_fold1_tabicl2stage_block_search(
            base_config=base_config,
            output_root=block_search_root,
            min_block=int(block_min),
            max_block=int(block_max),
            n_trials=int(args.block_trials),
            seed=int(args.seed),
            inner_folds=int(args.block_inner_folds),
        )

    best_block = int(json.loads(best_block_path.read_text(encoding="utf-8"))["group_size"])
    _write_json(
        manifest_path,
        {
            "dataset": dataset,
            "dataset_slug": dataset_slug,
            "trait_col": resolved_trait_col,
            "trait_input": trait_input,
            "trait_slug": trait_slug,
            "requested_marker_count": int(args.marker_count),
            "effective_marker_count": int(resources["subset_snp_count"]),
            "raw_snp_count": int(resources["raw_snp_count"]),
            "seed": int(args.seed),
            "fold_ids": [int(f) for f in args.fold_ids],
            "plink_prefix": str(resources["subset_plink_prefix"]),
            "phenotype_csv": str(resources["prepared_phenotype_csv"]),
            "sample_id_col": str(resources["sample_id_col"]),
            "block_search_root": str(block_search_root),
            "block_search_min": int(block_min),
            "block_search_max": int(block_max),
            "block_search_trials": int(args.block_trials),
            "block_search_inner_folds": int(args.block_inner_folds),
            "best_block_group_size": int(best_block),
            "no_prior_root": str(no_prior_root),
            "model_family": "TabPFN",
        },
    )

    if _complete(no_prior_root):
        return

    config = load_experiment_config(str(args.config_no_prior))
    runtime_override: dict[str, object] = {
        "output_dir": str(no_prior_root),
        "trait_col": resolved_trait_col,
        "plink_prefix": str(resources["subset_plink_prefix"]),
        "phenotype_csv": str(resources["prepared_phenotype_csv"]),
        "phenotype_sample_id_col": str(resources["sample_id_col"]),
        "max_snps": int(resources["subset_snp_count"]),
        "group_size": int(best_block),
        "tabicl_inner_oof_fold": 1,
        "tabicl_inner_oof_enabled": True,
    }
    config = deep_update(config, runtime_override)
    no_prior_root.mkdir(parents=True, exist_ok=True)
    run_experiment(config, fold_ids=[int(f) for f in args.fold_ids])


if __name__ == "__main__":
    main()
