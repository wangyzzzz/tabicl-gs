from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tabicl_gs.eval.metrics import regression_metrics


DATASETS = {
    "rice529": {
        "sample_id_col": "sample_id",
        "raw_plink_prefix": "genome/rice529/plink/rice529",
        "prepared_phenotype_csv": "genome/rice529/rice529_phe.csv",
    },
    "Cotton1245": {
        "sample_id_col": "Taxa",
        "raw_plink_prefix": "genome/Cotton1245/Cotton_all",
    },
    "Soybean951": {
        "sample_id_col": "Taxa",
        "raw_plink_prefix": "genome/Soybean951/Soybean_1500K",
    },
    "pig3534": {
        "sample_id_col": "ID",
        "raw_plink_prefix": "genome/pig3534/plink/pig3534_raw",
    },
    "wheat406": {
        "sample_id_col": "sample_id",
        "raw_plink_prefix": "genome/wheat406/plink/wheat406_raw",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GBLUP prior-only from existing single-prior outputs.")
    parser.add_argument("--single-root", required=True, help="Trait root containing tabicl_gblup_single_prior/")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--trait-col", required=True)
    return parser.parse_args()


def _load_fold_data_for_prior_eval(config: dict, fold_id: int):
    from tabicl_gs.pipeline.dual_prior_fold_search import _load_fold_data

    return _load_fold_data(config, fold_id)


def _build_base_config(dataset: str, trait_col: str) -> dict:
    spec = DATASETS[dataset]
    raw_plink_prefix = Path(spec["raw_plink_prefix"])
    subset_prefix = raw_plink_prefix.parent / f"{dataset.lower()}_max10000_seed2026"
    prepared_phenotype_csv = Path(spec.get("prepared_phenotype_csv", raw_plink_prefix.parent / "prepared_phenotype.csv"))
    return {
        "seed": 2026,
        "max_snps": 10000,
        "outer_cv_folds": 5,
        "plink_prefix": str(subset_prefix),
        "phenotype_csv": str(prepared_phenotype_csv),
        "phenotype_sample_id_col": spec["sample_id_col"],
        "trait_col": trait_col,
    }


def _load_gblup_test_prediction(fold_dir: Path) -> np.ndarray:
    prediction_csv = fold_dir / "_residual_target" / "GBLUP" / "test_fit" / "predictions.csv"
    return np.loadtxt(prediction_csv, delimiter=",", dtype=np.float32).reshape(-1)


def evaluate_single_prior_only(
    single_root: Path,
    dataset: str,
    trait_col: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    single_dir = single_root / "tabicl_gblup_single_prior"
    base_config = _build_base_config(dataset, trait_col)
    rows = []
    for fold_id in range(1, 6):
        fold_dir = single_dir / f"fold_{fold_id}"
        _, _, _, y_true = _load_fold_data_for_prior_eval(base_config, fold_id=fold_id)
        pred = _load_gblup_test_prediction(fold_dir)
        metric = regression_metrics(y_true, pred)
        rows.append(
            {
                "fold": int(fold_id),
                "w_gblup": 1.0,
                "pearson": float(metric["pearson"]),
                "r2": float(metric["r2"]),
                "rmse": float(metric["rmse"]),
                "mae": float(metric["mae"]),
            }
        )
    metadata: dict[str, object] = {
        "weights": [1.0],
        "weight_source": "outer_test_gblup_prediction",
        "weight_source_dir": str(single_dir),
    }
    return pd.DataFrame(rows), metadata


def main() -> None:
    args = parse_args()
    single_root = Path(args.single_root)
    fold_df, metadata = evaluate_single_prior_only(
        single_root=single_root,
        dataset=args.dataset,
        trait_col=args.trait_col,
    )
    out_dir = single_root / "prior_only_gblup"
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    summary = {
        "weights": list(metadata["weights"]),
        "weight_source": str(metadata["weight_source"]),
        "weight_source_dir": str(metadata["weight_source_dir"]),
        "pearson_mean": float(fold_df["pearson"].mean()),
        "r2_mean": float(fold_df["r2"].mean()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(fold_df.to_csv(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
