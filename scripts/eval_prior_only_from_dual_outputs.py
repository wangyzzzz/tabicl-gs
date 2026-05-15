from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tabicl_gs.eval.metrics import regression_metrics
from tabicl_gs.eval.splits import make_outer_cv_splits


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
    parser = argparse.ArgumentParser(description="Evaluate BayesB+GBLUP prior-only from existing dual-prior caches.")
    parser.add_argument("--dual-root", required=True, help="Trait root containing fold1 search and tabicl_tabicl_dual_prior/")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--trait-col", required=True)
    return parser.parse_args()


def _load_npy(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32).reshape(-1)


def _load_prior_prediction(prior_cache: Path, stem: str, fallback_csv: Path) -> np.ndarray:
    npy_path = prior_cache / f"{stem}.npy"
    if npy_path.exists():
        return _load_npy(npy_path)
    return np.loadtxt(fallback_csv, delimiter=",", dtype=np.float32).reshape(-1)


def _load_fold_data_for_prior_eval(config: dict, fold_id: int):
    from tabicl_gs.pipeline.dual_prior_fold_search import _load_fold_data

    return _load_fold_data(config, fold_id)


def _clip_alpha_targets(y_true: np.ndarray, y_bayesb: np.ndarray, y_gblup: np.ndarray) -> np.ndarray:
    gap = y_bayesb - y_gblup
    out = np.zeros_like(y_true, dtype=np.float32)
    stable = np.abs(gap) > 1e-6
    out[stable] = (y_true[stable] - y_gblup[stable]) / gap[stable]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


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


def _estimate_alpha_from_fold1_oof(dual_dir: Path, base_config: dict) -> float:
    fold1_prior = dual_dir / "fold_1" / "prior_cache"
    inner_dirs = sorted([p for p in fold1_prior.iterdir() if p.is_dir() and p.name.startswith("inner_")])
    if not inner_dirs:
        raise FileNotFoundError(f"No inner OOF prior caches found under {fold1_prior}")

    X_train, y_train, _, _ = _load_fold_data_for_prior_eval(base_config, fold_id=1)
    inner_splits = make_outer_cv_splits(X_train, 3, int(base_config["seed"]) + 1)
    bayesb_oof = np.zeros_like(y_train, dtype=np.float32)
    gblup_oof = np.zeros_like(y_train, dtype=np.float32)
    covered = np.zeros_like(y_train, dtype=bool)

    for inner_id, inner_dir in enumerate(inner_dirs, start=1):
        _, valid_idx = inner_splits[inner_id - 1]
        valid_idx = np.asarray(valid_idx, dtype=np.int64)
        bayesb_valid = _load_npy(inner_dir / "bayesb_valid.npy")
        gblup_valid = _load_npy(inner_dir / "gblup_valid.npy")
        bayesb_oof[valid_idx] = bayesb_valid
        gblup_oof[valid_idx] = gblup_valid
        covered[valid_idx] = True

    if not np.all(covered):
        raise ValueError(f"Inner OOF valid indices do not fully cover outer-train under {fold1_prior}")

    alpha_targets = _clip_alpha_targets(y_train, bayesb_oof, gblup_oof)
    return float(np.mean(alpha_targets))


def _load_fold_predictions(
    dual_fold_dir: Path,
    base_config: dict,
    fold_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, _, _, y_true = _load_fold_data_for_prior_eval(base_config, fold_id=fold_id)
    prior_cache = dual_fold_dir / "prior_cache"
    bayesb_test = _load_prior_prediction(
        prior_cache,
        "bayesb_test",
        prior_cache / "bayesb_outer" / "eval_fit" / "predictions.csv",
    )
    gblup_test = _load_prior_prediction(
        prior_cache,
        "gblup_test",
        prior_cache / "gblup_outer" / "_residual_target" / "GBLUP" / "test_fit" / "predictions.csv",
    )
    return y_true, bayesb_test, gblup_test


def evaluate_prior_only(
    dual_root: Path,
    dataset: str,
    trait_col: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    dual_dir = dual_root / "tabicl_tabicl_dual_prior"
    base_config = _build_base_config(dataset, trait_col)
    alpha = _estimate_alpha_from_fold1_oof(dual_dir, base_config)
    rows = []
    for fold_id in range(1, 6):
        fold_dir = dual_dir / f"fold_{fold_id}"
        y_true, bayesb_test, gblup_test = _load_fold_predictions(
            fold_dir,
            base_config,
            fold_id,
        )
        pred = alpha * bayesb_test + (1.0 - alpha) * gblup_test
        metric = regression_metrics(y_true, pred)
        rows.append(
            {
                "fold": int(fold_id),
                "alpha": float(alpha),
                "pearson": float(metric["pearson"]),
                "r2": float(metric["r2"]),
                "rmse": float(metric["rmse"]),
                "mae": float(metric["mae"]),
            }
        )
    metadata: dict[str, object] = {
        "alpha": float(alpha),
        "alpha_source": "fold1_inner_val_oof_prior",
        "alpha_source_dir": str(dual_dir / "fold_1" / "prior_cache"),
    }
    return pd.DataFrame(rows), metadata


def main() -> None:
    args = parse_args()
    dual_root = Path(args.dual_root)
    fold_df, metadata = evaluate_prior_only(
        dual_root,
        dataset=args.dataset,
        trait_col=args.trait_col,
    )
    out_dir = dual_root / "prior_only_bayesb_gblup"
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    summary = {
        "alpha": float(metadata["alpha"]),
        "alpha_source": str(metadata["alpha_source"]),
        "alpha_source_dir": str(metadata["alpha_source_dir"]),
        "pearson_mean": float(fold_df["pearson"].mean()),
        "r2_mean": float(fold_df["r2"].mean()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(fold_df.to_csv(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
