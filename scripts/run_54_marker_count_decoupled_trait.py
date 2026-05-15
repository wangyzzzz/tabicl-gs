from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from tabicl_gs.config import deep_update, load_experiment_config
from tabicl_gs.data.plink import (
    convert_matrix_csv_to_plink,
    plink_num_snps,
    read_phenotype_table,
    write_subsampled_plink,
)
from tabicl_gs.data.grouping import subsample_snp_indices
from tabicl_gs.pipeline.experiment import run_experiment
from tabicl_gs.pipeline.inner_oof_tabicl_search import run_fold1_tabicl2stage_block_search


DATASET_SPECS: dict[str, dict[str, object]] = {
    "rice529": {
        "dataset_title": "rice529",
        "phenotype_csv": "genome/rice529/rice529_phe.csv",
        "phenotype_sep": ",",
        "sample_id_col": "sample_id",
        "raw_plink_prefix": "genome/rice529/plink/rice529",
        "genotype_csv": "genome/rice529/rice529_gen.csv",
    },
    "cotton1245": {
        "dataset_title": "Cotton1245",
        "phenotype_csv": "genome/Cotton1245/Cotton_all.txt",
        "phenotype_sep": "\t",
        "sample_id_col": "Taxa",
        "raw_plink_prefix": "genome/Cotton1245/Cotton_all",
    },
    "soybean951": {
        "dataset_title": "Soybean951",
        "phenotype_csv": "genome/Soybean951/Soybean_all.txt",
        "phenotype_sep": "\t",
        "sample_id_col": "Taxa",
        "raw_plink_prefix": "genome/Soybean951/Soybean_1500K",
    },
    "wheat406": {
        "dataset_title": "wheat406",
        "phenotype_csv": "genome/wheat406/wheat406_phe.csv",
        "phenotype_sep": ",",
        "sample_id_col": "sample_id",
        "raw_plink_prefix": "genome/wheat406/plink/wheat406_raw",
        "genotype_csv": "genome/wheat406/wheat406_gen.csv",
        "genotype_sep": ",",
        "genotype_has_header": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run decoupled marker-count experiment for one trait.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_SPECS))
    parser.add_argument("--trait-col", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config-no-prior", default="configs/tabicl_block/window_tabicl_dynamic99_traitscan.yaml")
    parser.add_argument("--config-baseline", default="configs/tabicl_block/window_baseline_only_3models_liudang.yaml")
    parser.add_argument("--marker-counts", nargs="+", type=int, default=[2000, 50000])
    parser.add_argument("--fold-ids", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--block-min", type=int, default=None)
    parser.add_argument("--block-max", type=int, default=None)
    parser.add_argument("--block-trials", type=int, default=10)
    parser.add_argument("--block-inner-folds", type=int, default=3)
    return parser.parse_args()


def _dataset_slug(name: str) -> str:
    return str(name).lower()


def _trait_slug(trait_col: str) -> str:
    return trait_col.lower().replace("/", "__").replace(" ", "__")


def _read_header(path: str | Path, sep: str) -> list[str]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=sep)
        return next(reader)


def _write_phenotype_csv(src_path: str | Path, dst_path: str | Path, sep: str, sample_id_col: str) -> Path:
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with src_path.open("r", encoding="utf-8", newline="") as src, dst_path.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.reader(src, delimiter=sep)
        writer = csv.writer(dst)
        rows = list(reader)
        if not rows:
            raise ValueError(f"Empty phenotype file: {src_path}")
        header = list(rows[0])
        if sample_id_col not in header:
            header = [sample_id_col] + header
            writer.writerow(header)
            for index, row in enumerate(rows[1:], start=1):
                writer.writerow([f"line_{index:04d}"] + list(row))
        else:
            writer.writerow(header)
            for row in rows[1:]:
                writer.writerow(row)
    return dst_path


def _write_genotype_matrix_csv(
    src_path: str | Path,
    dst_path: str | Path,
    sep: str,
    has_header: bool,
) -> Path:
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with src_path.open("r", encoding="utf-8", newline="") as src, dst_path.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.reader(src, delimiter=sep)
        writer = csv.writer(dst)
        for row_id, row in enumerate(reader):
            if has_header and row_id == 0:
                continue
            if not row:
                continue
            writer.writerow(row[1:] if has_header else row)
    return dst_path


def _ensure_dataset_resources(dataset: str, max_snps: int, seed: int) -> dict[str, object]:
    spec = DATASET_SPECS[dataset]
    raw_plink_prefix = Path(str(spec["raw_plink_prefix"]))
    raw_plink_prefix.parent.mkdir(parents=True, exist_ok=True)
    raw_bed = raw_plink_prefix.with_suffix(".bed")

    if dataset == "rice529":
        phenotype_csv = Path(str(spec["phenotype_csv"]))
        sample_id_col = str(spec["sample_id_col"])
        if not raw_bed.exists():
            convert_matrix_csv_to_plink(
                genotype_csv=str(spec["genotype_csv"]),
                phenotype_csv=phenotype_csv,
                plink_prefix=raw_plink_prefix,
                sample_id_col=sample_id_col,
            )
        prepared_phenotype_csv = phenotype_csv
        trait_cols = [str(col) for col in read_phenotype_table(phenotype_csv, sample_id_col=sample_id_col).columns if str(col) != sample_id_col]
    else:
        phenotype_csv = str(spec["phenotype_csv"])
        phenotype_sep = str(spec["phenotype_sep"])
        sample_id_col = str(spec["sample_id_col"])
        prepared_phenotype_csv = raw_plink_prefix.parent / "prepared_phenotype.csv"
        _write_phenotype_csv(phenotype_csv, prepared_phenotype_csv, sep=phenotype_sep, sample_id_col=sample_id_col)
        if not raw_bed.exists():
            genotype_csv = spec.get("genotype_csv")
            if genotype_csv is None:
                raise FileNotFoundError(f"Missing raw PLINK for {dataset}: {raw_bed}")
            prepared_genotype_csv = raw_plink_prefix.parent / "prepared_genotype.csv"
            _write_genotype_matrix_csv(
                genotype_csv,
                prepared_genotype_csv,
                sep=str(spec.get("genotype_sep", ",")),
                has_header=bool(spec.get("genotype_has_header", False)),
            )
            convert_matrix_csv_to_plink(
                genotype_csv=prepared_genotype_csv,
                phenotype_csv=prepared_phenotype_csv,
                plink_prefix=raw_plink_prefix,
                sample_id_col=sample_id_col,
            )
        trait_cols = [col for col in _read_header(phenotype_csv, phenotype_sep) if col != sample_id_col]

    total_snps = plink_num_snps(raw_plink_prefix)
    selected_indices = subsample_snp_indices(int(total_snps), int(max_snps), int(seed))
    subset_prefix = raw_plink_prefix.parent / f"{_dataset_slug(dataset)}_max{int(max_snps)}_seed{int(seed)}"
    if not subset_prefix.with_suffix(".bed").exists():
        write_subsampled_plink(raw_plink_prefix, subset_prefix, selected_indices)

    return {
        "dataset": dataset,
        "dataset_title": str(spec["dataset_title"]),
        "raw_plink_prefix": str(raw_plink_prefix),
        "prepared_phenotype_csv": str(prepared_phenotype_csv),
        "subset_plink_prefix": str(subset_prefix),
        "sample_id_col": str(spec["sample_id_col"]),
        "trait_cols": trait_cols,
        "raw_snp_count": int(total_snps),
        "subset_snp_count": min(int(total_snps), int(max_snps)),
        "requested_max_snps": int(max_snps),
        "seed": int(seed),
    }


def _resolve_trait_col_name(trait_cols: list[str], trait_col_or_slug: str) -> str:
    if trait_col_or_slug in trait_cols:
        return trait_col_or_slug
    slug_map = {_trait_slug(col): col for col in trait_cols}
    if trait_col_or_slug in slug_map:
        return str(slug_map[trait_col_or_slug])
    raise KeyError(
        f"Trait '{trait_col_or_slug}' not found. Available columns include: {trait_cols[:20]}"
    )


def _auto_block_bounds(n_snps: int) -> tuple[int, int]:
    min_block = min(int(n_snps), max(50, int(round(float(n_snps) / 50.0))))
    max_block = min(int(n_snps), max(min_block, int(round(float(n_snps) / 7.0))))
    return int(min_block), int(max_block)


def _preferred_block_bounds(marker_count: int, n_snps: int) -> tuple[int, int]:
    fixed_bounds = {
        2000: (50, 400),
        50000: (1000, 5000),
    }
    if int(marker_count) in fixed_bounds:
        min_block, max_block = fixed_bounds[int(marker_count)]
        min_block = min(int(n_snps), int(min_block))
        max_block = min(int(n_snps), int(max_block))
        max_block = max(min_block, max_block)
        return int(min_block), int(max_block)
    return _auto_block_bounds(int(n_snps))


def _run_family(
    *,
    base_config_path: str,
    output_dir: Path,
    trait_col: str,
    plink_prefix: str,
    phenotype_csv: str,
    sample_id_col: str,
    group_size: int | None,
    fold_ids: list[int],
    max_snps: int,
    baseline_inner_oof: dict[str, object] | None = None,
    tabicl_inner_oof_fold: int | None = None,
) -> None:
    config = load_experiment_config(base_config_path)
    runtime_override: dict[str, object] = {
        "output_dir": str(output_dir),
        "trait_col": trait_col,
        "plink_prefix": plink_prefix,
        "phenotype_csv": phenotype_csv,
        "phenotype_sample_id_col": sample_id_col,
        "max_snps": int(max_snps),
    }
    if group_size is not None:
        runtime_override["group_size"] = int(group_size)
    if baseline_inner_oof is not None:
        runtime_override["baseline_inner_oof"] = baseline_inner_oof
    if tabicl_inner_oof_fold is not None:
        runtime_override["tabicl_inner_oof_fold"] = int(tabicl_inner_oof_fold)
        runtime_override["tabicl_inner_oof_enabled"] = True
    config = deep_update(config, runtime_override)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_experiment(config, fold_ids=[int(f) for f in fold_ids])


def _compare_complete(path: Path) -> bool:
    return (path / "compare_main.csv").exists() and (path / "compare_main.json").exists()


def _run_compare(
    *,
    dataset_slug: str,
    trait_slug: str,
    tabicl_root: Path,
    baseline_root: Path,
    compare_root: Path,
) -> None:
    if _compare_complete(compare_root):
        return
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "src")
    compare_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "scripts/compare_decoupled_weight_schemes.py",
            "--trait-no-prior-root",
            str(tabicl_root),
            "--trait-baseline-root",
            str(baseline_root),
            "--dataset",
            dataset_slug,
            "--trait-slug",
            trait_slug,
            "--output-csv",
            str(compare_root / "compare_main.csv"),
            "--output-json",
            str(compare_root / "compare_main.json"),
        ],
        check=True,
        env=env,
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json_if_exists(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    dataset = str(args.dataset)
    dataset_slug = _dataset_slug(dataset)
    trait_input = str(args.trait_col)
    trait_slug = _trait_slug(trait_input)
    output_root = Path(args.output_root)
    marker_roots: list[str] = []

    for marker_count in [int(value) for value in args.marker_counts]:
        resources = _ensure_dataset_resources(dataset, int(marker_count), int(args.seed))
        resolved_trait_col = _resolve_trait_col_name(list(resources["trait_cols"]), trait_input)
        marker_dir = output_root / dataset_slug / trait_slug / f"maxsnps_{int(marker_count):05d}"
        block_search_root = marker_dir / "fold1_tabicl_block_search"
        no_prior_root = marker_dir / "no_prior"
        baseline_root = marker_dir / "baseline_3models"
        compare_root = marker_dir / "compare"
        manifest_path = marker_dir / "run_manifest.json"
        marker_roots.append(str(marker_dir))

        if manifest_path.exists() and _compare_complete(compare_root):
            continue

        block_min, block_max = _preferred_block_bounds(int(marker_count), int(resources["subset_snp_count"]))
        if args.block_min is not None:
            block_min = int(args.block_min)
        if args.block_max is not None:
            block_max = int(args.block_max)
        if block_min > block_max:
            raise ValueError(f"block_min ({block_min}) cannot exceed block_max ({block_max}).")

        best_block_path = block_search_root / "best_block.json"
        existing_manifest = _read_json_if_exists(manifest_path)
        should_rerun_block_search = not best_block_path.exists()
        if existing_manifest is not None:
            old_min = existing_manifest.get("block_search_min")
            old_max = existing_manifest.get("block_search_max")
            old_trials = existing_manifest.get("block_search_trials")
            old_inner_folds = existing_manifest.get("block_search_inner_folds")
            if (
                int(old_min) != int(block_min)
                or int(old_max) != int(block_max)
                or int(old_trials) != int(args.block_trials)
                or int(old_inner_folds) != int(args.block_inner_folds)
            ):
                should_rerun_block_search = True

        if should_rerun_block_search:
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
                "requested_marker_count": int(marker_count),
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
                "baseline_root": str(baseline_root),
                "compare_root": str(compare_root),
            },
        )

        _run_family(
            base_config_path=str(args.config_no_prior),
            output_dir=no_prior_root,
            trait_col=resolved_trait_col,
            plink_prefix=str(resources["subset_plink_prefix"]),
            phenotype_csv=str(resources["prepared_phenotype_csv"]),
            sample_id_col=str(resources["sample_id_col"]),
            group_size=int(best_block),
            fold_ids=[int(f) for f in args.fold_ids],
            max_snps=int(resources["subset_snp_count"]),
            tabicl_inner_oof_fold=1,
        )

        _run_family(
            base_config_path=str(args.config_baseline),
            output_dir=baseline_root,
            trait_col=resolved_trait_col,
            plink_prefix=str(resources["subset_plink_prefix"]),
            phenotype_csv=str(resources["prepared_phenotype_csv"]),
            sample_id_col=str(resources["sample_id_col"]),
            group_size=None,
            fold_ids=[int(f) for f in args.fold_ids],
            max_snps=int(resources["subset_snp_count"]),
            baseline_inner_oof={
                "enabled": True,
                "fold": 1,
                "n_splits": 3,
                "models": ["GBLUP", "BayesB", "RKHS"],
            },
        )

        _run_compare(
            dataset_slug=dataset_slug,
            trait_slug=trait_slug,
            tabicl_root=no_prior_root,
            baseline_root=baseline_root,
            compare_root=compare_root,
        )

    _write_json(output_root / dataset_slug / trait_slug / "marker_roots.json", {"marker_roots": marker_roots})


if __name__ == "__main__":
    main()
