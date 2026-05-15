from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tabicl_gs.data.grouping import subsample_snp_indices
from tabicl_gs.data.plink import impute_by_train_mean, load_plink_matrix, plink_num_snps, read_phenotype_table
from tabicl_gs.eval.metrics import regression_metrics
from tabicl_gs.eval.splits import make_outer_cv_splits
from tabicl_gs.models.baselines import run_r_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Yield GBLUP 5-fold accuracy and REML H2 for full/median-split subsets.")
    parser.add_argument("--plink-prefix", default="genome/rice529/plink/rice529")
    parser.add_argument("--phenotype-csv", default="genome/rice529/rice529_phe.csv")
    parser.add_argument("--phenotype-sample-id-col", default="sample_id")
    parser.add_argument("--trait-col", default="Yield")
    parser.add_argument("--max-snps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--rscript-path", default="/data/yes/envs/r_env/bin/Rscript")
    parser.add_argument("--sommer-method", default="mmer", choices=["mmer", "mmes"])
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    total_snps = plink_num_snps(args.plink_prefix)
    snp_indices = subsample_snp_indices(total_snps, args.max_snps, args.seed)
    plink = load_plink_matrix(args.plink_prefix, snp_indices=snp_indices)
    phenotype = read_phenotype_table(args.phenotype_csv, sample_id_col=args.phenotype_sample_id_col)
    phenotype = phenotype.set_index(args.phenotype_sample_id_col).loc[plink.sample_ids].reset_index()
    y = phenotype[args.trait_col].to_numpy(dtype=np.float32)
    X = plink.matrix.astype(np.float32)
    valid = np.isfinite(y)
    return X[valid], y[valid]


def save_matrix(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(values, dtype=np.float32), delimiter=",", fmt="%.6f")


def run_h2(
    *,
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
    rscript_path: str,
    sommer_method: str,
    seed: int,
) -> dict[str, Any]:
    h2_dir = output_dir / "h2_reml"
    h2_dir.mkdir(parents=True, exist_ok=True)
    x_path = h2_dir / "x.csv"
    y_path = h2_dir / "y.csv"
    out_path = h2_dir / "h2.json"
    save_matrix(x_path, X)
    save_matrix(y_path, y.reshape(-1, 1))
    command = [
        rscript_path,
        str(project_root() / "r" / "run_gblup_h2.R"),
        "--x",
        str(x_path),
        "--y",
        str(y_path),
        "--out",
        str(out_path),
        "--seed",
        str(seed),
        "--sommer-method",
        sommer_method,
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    if completed.stdout:
        (h2_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    if completed.stderr:
        (h2_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    return json.loads(out_path.read_text(encoding="utf-8"))


def run_group(
    *,
    X: np.ndarray,
    y: np.ndarray,
    group_name: str,
    output_root: Path,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    group_dir = output_root / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    h2 = run_h2(
        X=X,
        y=y,
        output_dir=group_dir,
        rscript_path=args.rscript_path,
        sommer_method=args.sommer_method,
        seed=args.seed,
    )
    rows: list[dict[str, Any]] = []
    splits = make_outer_cv_splits(X, n_splits=args.folds, seed=args.seed)
    for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train, X_test = impute_by_train_mean(X[train_idx], X[test_idx])
        y_train = y[train_idx]
        y_test = y[test_idx]
        result = run_r_baseline(
            model_name="GBLUP",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            output_dir=group_dir / f"fold_{fold_id}" / "GBLUP",
            rscript_path=args.rscript_path,
            seed=args.seed + fold_id,
            sommer_method=args.sommer_method,
            keep_artifacts=True,
            return_beta=False,
        )
        metrics = regression_metrics(y_test, result.predictions)
        rows.append(
            {
                "group": group_name,
                "fold": fold_id,
                "n_samples": int(X.shape[0]),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "pearson": float(metrics["pearson"]),
                "r2": float(metrics["r2"]),
                "rmse": float(metrics["rmse"]),
                "mae": float(metrics["mae"]),
                "h2": float(h2["h2"]),
                "genetic_variance": float(h2["genetic_variance"]),
                "residual_variance": float(h2["residual_variance"]),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(group_dir / "fold_metrics.csv", index=False)
    return frame, h2


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    X, y = load_dataset(args)
    median = float(np.median(y))
    groups = {
        "full": np.ones_like(y, dtype=bool),
        "yield_low_median": y <= median,
        "yield_high_median": y > median,
    }
    all_rows = []
    h2_rows = []
    for group_name, mask in groups.items():
        group_X = X[mask]
        group_y = y[mask]
        frame, h2 = run_group(X=group_X, y=group_y, group_name=group_name, output_root=output_root, args=args)
        all_rows.append(frame)
        h2_rows.append(
            {
                "group": group_name,
                "n_samples": int(group_X.shape[0]),
                "yield_median_full": median,
                "yield_min": float(np.min(group_y)),
                "yield_max": float(np.max(group_y)),
                "yield_mean": float(np.mean(group_y)),
                "h2": float(h2["h2"]),
                "genetic_variance": float(h2["genetic_variance"]),
                "residual_variance": float(h2["residual_variance"]),
            }
        )
    fold_metrics = pd.concat(all_rows, ignore_index=True)
    fold_metrics.to_csv(output_root / "fold_metrics.csv", index=False)
    summary = (
        fold_metrics.groupby("group", as_index=False)
        .agg(
            n_samples=("n_samples", "first"),
            mean_pearson=("pearson", "mean"),
            std_pearson=("pearson", "std"),
            mean_r2=("r2", "mean"),
            std_r2=("r2", "std"),
            mean_rmse=("rmse", "mean"),
            mean_mae=("mae", "mean"),
            h2=("h2", "first"),
            genetic_variance=("genetic_variance", "first"),
            residual_variance=("residual_variance", "first"),
        )
    )
    summary.to_csv(output_root / "summary.csv", index=False)
    pd.DataFrame(h2_rows).to_csv(output_root / "h2_summary.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()
