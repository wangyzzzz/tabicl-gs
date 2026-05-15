from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from tabicl_gs.config import deep_update, load_experiment_config
from tabicl_gs.data.grouping import subsample_snp_indices
from tabicl_gs.data.plink import align_phenotype_to_sample_ids, load_plink_matrix, plink_num_snps, read_phenotype_table
from tabicl_gs.eval.splits import make_outer_cv_splits
from tabicl_gs.pipeline.experiment import run_experiment
from tabicl_gs.pipeline.sample_size_impact import build_nested_train_subsets

DATASET_TITLES = {
    "rice529": "rice529",
    "cotton1245": "Cotton1245",
    "soybean951": "Soybean951",
    "wheat406": "wheat406",
}

DATASET_SUMMARY_PATHS = {
    "rice529": Path("genome/rice529/plink/rice529_cache_summary.json"),
    "cotton1245": Path("genome/Cotton1245/cotton1245_cache_summary.json"),
    "soybean951": Path("genome/Soybean951/soybean951_cache_summary.json"),
    "wheat406": Path("genome/wheat406/plink/wheat406_cache_summary.json"),
}

DATASET_PREP_SCRIPTS = {
    "rice529": [
        "scripts/prepare_rice529_plink_cache.py",
        [
            "--genotype-csv",
            "genome/rice529/rice529_gen.csv",
            "--phenotype-csv",
            "genome/rice529/rice529_phe.csv",
            "--plink-prefix",
            "genome/rice529/plink/rice529",
            "--sample-id-col",
            "sample_id",
        ],
    ],
    "cotton1245": [
        "scripts/prepare_multi_dataset_plink_cache.py",
        ["--dataset", "Cotton1245"],
    ],
    "soybean951": [
        "scripts/prepare_multi_dataset_plink_cache.py",
        ["--dataset", "Soybean951"],
    ],
    "wheat406": [
        "scripts/prepare_multi_dataset_plink_cache.py",
        ["--dataset", "wheat406"],
    ],
}

DUAL_BEST_BLOCK_ROOTS = {
    "rice529": Path("outputs/rice529_10traits_tabicl_tabicl_tabiclxgb_dualprior"),
    "cotton1245": Path("outputs/multidataset_alltraits_dualprior/cotton1245"),
    "soybean951": Path("outputs/multidataset_alltraits_dualprior/soybean951"),
    "wheat406": Path("outputs/multidataset_alltraits_dualprior/wheat406"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run decoupled sample-size experiment for one trait.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_TITLES))
    parser.add_argument("--trait-col", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config-no-prior", default="configs/tabicl_block/window_tabicl_dynamic99_traitscan.yaml")
    parser.add_argument("--config-baseline", default="configs/tabicl_block/window_baseline_only_3models_liudang.yaml")
    parser.add_argument("--proportions", nargs="+", type=float, default=[0.2, 0.6])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fold-ids", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--max-snps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--best-block-root", default=None)
    return parser.parse_args()


def _run_command(cmd: list[str]) -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "src")
    subprocess.run(cmd, check=True, env=env)


def _prepare_dataset_if_needed(dataset: str) -> dict:
    summary_path = DATASET_SUMMARY_PATHS[dataset]
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    script_path, extra_args = DATASET_PREP_SCRIPTS[dataset]
    cmd = [sys.executable, script_path, *extra_args, "--max-snps", "10000", "--seed", "2026"]
    _run_command(cmd)
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _trait_slug(trait_col: str) -> str:
    return trait_col.lower().replace("/", "__").replace(" ", "__")


def _resolve_trait_col_name(phenotype, trait_col_or_slug: str) -> str:
    columns = [str(col) for col in phenotype.columns]
    if trait_col_or_slug in columns:
        return trait_col_or_slug
    slug_map = {_trait_slug(col): col for col in columns}
    if trait_col_or_slug in slug_map:
        return str(slug_map[trait_col_or_slug])
    raise KeyError(
        f"Trait '{trait_col_or_slug}' not found in phenotype columns. "
        f"Available columns include: {columns[:20]}"
    )


def _best_block_path(dataset: str, trait_slug: str, override_root: str | None) -> Path:
    if override_root:
        return Path(override_root) / trait_slug / "fold1_tabicl_block_search" / "best_block.json"
    return DUAL_BEST_BLOCK_ROOTS[dataset] / trait_slug / "fold1_tabicl_block_search" / "best_block.json"


def _load_best_block(dataset: str, trait_col: str, override_root: str | None) -> int:
    path = _best_block_path(dataset, _trait_slug(trait_col), override_root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return int(payload["group_size"])


def _load_trait_arrays(dataset: str, trait_col: str, max_snps: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    summary = _prepare_dataset_if_needed(dataset)
    phenotype_csv = summary["prepared_phenotype_csv"]
    sample_id_col = summary["sample_id_col"]
    plink_prefix = summary["subset_plink_prefix"]

    phenotype = read_phenotype_table(phenotype_csv, sample_id_col=sample_id_col)
    resolved_trait_col = _resolve_trait_col_name(phenotype, trait_col)
    total_snps = plink_num_snps(plink_prefix)
    selected_snp_indices = subsample_snp_indices(int(total_snps), int(max_snps), int(seed))
    plink_data = load_plink_matrix(plink_prefix, snp_indices=selected_snp_indices)
    phenotype, keep_indices = align_phenotype_to_sample_ids(
        phenotype,
        plink_data.sample_ids,
        sample_id_col=sample_id_col,
    )
    genotype = plink_data.matrix[np.asarray(keep_indices, dtype=np.int64)].astype(np.float32)
    target = phenotype[resolved_trait_col].to_numpy(dtype=np.float32)
    valid_mask = np.isfinite(target)
    genotype = genotype[valid_mask]
    target = target[valid_mask]
    summary = dict(summary)
    summary["resolved_trait_col"] = resolved_trait_col
    return genotype, target, summary


def _build_sample_override(
    *,
    genotype: np.ndarray,
    outer_cv_folds: int,
    seed: int,
    proportion: float,
    repeat: int,
) -> dict[str, object]:
    splits = make_outer_cv_splits(genotype, int(outer_cv_folds), int(seed))
    fold_subsets: dict[str, list[int]] = {}
    for fold_id, (train_idx, _test_idx) in enumerate(splits, start=1):
        fold_train_indices = np.arange(len(train_idx), dtype=np.int64)
        subset_map = build_nested_train_subsets(
            train_indices=fold_train_indices,
            proportions=[float(proportion)],
            repeats=int(repeat),
            seed=int(seed) + int(fold_id) * 1000,
        )
        subset_indices = subset_map[(int(repeat), float(proportion))]
        fold_subsets[str(int(fold_id))] = np.asarray(subset_indices, dtype=np.int64).tolist()
    return {
        "fold_subsets": fold_subsets,
        "proportion": float(proportion),
        "repeat": int(repeat),
        "selection_tag": f"p_{float(proportion):.2f}_repeat_{int(repeat)}",
        "note": "decoupled_sample_size",
    }


def _run_family(
    *,
    base_config_path: str,
    output_dir: Path,
    dataset: str,
    trait_col: str,
    plink_prefix: str,
    phenotype_csv: str,
    sample_id_col: str,
    group_size: int | None,
    fold_ids: list[int],
    sample_override: dict[str, object] | None,
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
    }
    if group_size is not None:
        runtime_override["group_size"] = int(group_size)
    if baseline_inner_oof is not None:
        runtime_override["baseline_inner_oof"] = baseline_inner_oof
    if tabicl_inner_oof_fold is not None:
        runtime_override["tabicl_inner_oof_fold"] = int(tabicl_inner_oof_fold)
        runtime_override["tabicl_inner_oof_enabled"] = True
    config = deep_update(config, runtime_override)
    if sample_override is not None:
        config = deep_update(config, {"_sample_size_override": sample_override})
    output_dir.mkdir(parents=True, exist_ok=True)
    run_experiment(config, fold_ids=[int(f) for f in fold_ids])


def _compare_complete(path: Path) -> bool:
    required = [
        path / "compare_main.csv",
        path / "compare_main.json",
    ]
    return all(item.exists() for item in required)


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


def main() -> None:
    args = parse_args()
    dataset = str(args.dataset)
    trait_col = str(args.trait_col)
    dataset_slug = dataset
    trait_slug = _trait_slug(trait_col)
    output_root = Path(args.output_root)
    repeat_roots = []

    best_block = _load_best_block(dataset, trait_col, args.best_block_root)
    summary = _prepare_dataset_if_needed(dataset)
    plink_prefix = str(summary["subset_plink_prefix"])
    phenotype_csv = str(summary["prepared_phenotype_csv"])
    sample_id_col = str(summary["sample_id_col"])
    genotype, _target, _dataset_summary = _load_trait_arrays(dataset, trait_col, int(args.max_snps), int(args.seed))
    resolved_trait_col = str(_dataset_summary["resolved_trait_col"])

    for proportion in [float(p) for p in args.proportions]:
        for repeat in range(1, int(args.repeats) + 1):
            repeat_root = output_root / dataset_slug / trait_slug / f"p_{float(proportion):.2f}" / f"repeat_{repeat}"
            no_prior_root = repeat_root / "no_prior"
            baseline_root = repeat_root / "baseline_3models"
            compare_root = repeat_root / "compare"
            manifest_path = repeat_root / "run_manifest.json"
            if manifest_path.exists() and _compare_complete(compare_root):
                repeat_roots.append(str(repeat_root))
                continue

            sample_override = _build_sample_override(
                genotype=genotype,
                outer_cv_folds=5,
                seed=int(args.seed),
                proportion=proportion,
                repeat=repeat,
            )
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "dataset": dataset,
                        "trait_col": resolved_trait_col,
                        "trait_input": trait_col,
                        "trait_slug": trait_slug,
                        "group_size": int(best_block),
                        "proportion": float(proportion),
                        "repeat": int(repeat),
                        "fold_ids": [int(f) for f in args.fold_ids],
                        "max_snps": int(args.max_snps),
                        "seed": int(args.seed),
                        "no_prior_root": str(no_prior_root),
                        "baseline_root": str(baseline_root),
                        "compare_root": str(compare_root),
                        "sample_override": sample_override,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            _run_family(
                base_config_path=str(args.config_no_prior),
                output_dir=no_prior_root,
                dataset=dataset,
                trait_col=resolved_trait_col,
                plink_prefix=plink_prefix,
                phenotype_csv=phenotype_csv,
                sample_id_col=sample_id_col,
                group_size=int(best_block),
                fold_ids=[int(f) for f in args.fold_ids],
                sample_override=sample_override,
                tabicl_inner_oof_fold=1,
            )

            _run_family(
                base_config_path=str(args.config_baseline),
                output_dir=baseline_root,
                dataset=dataset,
                trait_col=resolved_trait_col,
                plink_prefix=plink_prefix,
                phenotype_csv=phenotype_csv,
                sample_id_col=sample_id_col,
                group_size=None,
                fold_ids=[int(f) for f in args.fold_ids],
                sample_override=sample_override,
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
            repeat_roots.append(str(repeat_root))

    (output_root / dataset_slug / trait_slug / "repeat_roots.json").write_text(
        json.dumps(repeat_roots, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
