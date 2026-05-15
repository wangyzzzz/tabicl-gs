from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tabicl_gs.data.grouping import subsample_snp_indices
from tabicl_gs.data.plink import align_phenotype_to_sample_ids, impute_by_train_mean, load_plink_matrix, plink_num_snps, read_phenotype_table
from tabicl_gs.eval.metrics import regression_metrics
from tabicl_gs.eval.splits import make_outer_cv_splits
from tabicl_gs.models.baselines import run_r_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run direct RKHS + BayesB + GBLUP dual-prior evaluation.")
    parser.add_argument("--plink-prefix", required=True)
    parser.add_argument("--phenotype-csv", required=True)
    parser.add_argument("--phenotype-sample-id-col", required=True)
    parser.add_argument("--trait-col", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rscript-path", required=True)
    parser.add_argument("--max-snps", type=int, default=10000)
    parser.add_argument("--outer-cv-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fold-ids", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--frozen-gate-path", default=None)
    parser.add_argument("--bandwidth-scale", type=float, default=1.0)
    parser.add_argument("--sommer-method", default="mmer")
    return parser.parse_args()


def _clip_targets(y_true: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_left = np.asarray(y_left, dtype=np.float32).reshape(-1)
    y_right = np.asarray(y_right, dtype=np.float32).reshape(-1)
    gap = y_left - y_right
    out = np.zeros_like(y_true, dtype=np.float32)
    stable = np.abs(gap) > 1e-6
    out[stable] = (y_true[stable] - y_right[stable]) / gap[stable]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def fit_single_group_gate(
    *,
    y_true: np.ndarray,
    y_model: np.ndarray,
    y_bayesb: np.ndarray,
    y_gblup: np.ndarray,
) -> dict[str, float]:
    alpha_targets = _clip_targets(y_true, y_bayesb, y_gblup)
    alpha = float(np.clip(np.mean(alpha_targets), 0.0, 1.0))
    y_prior = alpha * np.asarray(y_bayesb, dtype=np.float32) + (1.0 - alpha) * np.asarray(y_gblup, dtype=np.float32)
    w_targets = _clip_targets(y_true, y_model, y_prior)
    w = float(np.clip(np.mean(w_targets), 0.0, 1.0))
    return {"alpha": alpha, "w": w}


def apply_dual_prior_gate(
    *,
    y_model: np.ndarray,
    y_bayesb: np.ndarray,
    y_gblup: np.ndarray,
    alpha: float,
    w: float,
) -> np.ndarray:
    y_model = np.asarray(y_model, dtype=np.float32).reshape(-1)
    y_bayesb = np.asarray(y_bayesb, dtype=np.float32).reshape(-1)
    y_gblup = np.asarray(y_gblup, dtype=np.float32).reshape(-1)
    y_prior = float(alpha) * y_bayesb + (1.0 - float(alpha)) * y_gblup
    return (y_prior + float(w) * (y_model - y_prior)).astype(np.float32)


def _load_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    phenotype = read_phenotype_table(args.phenotype_csv, sample_id_col=args.phenotype_sample_id_col)
    total_snps = plink_num_snps(args.plink_prefix)
    selected_snp_indices = subsample_snp_indices(total_snps, args.max_snps, args.seed)
    plink = load_plink_matrix(args.plink_prefix, snp_indices=selected_snp_indices)
    phenotype, keep_indices = align_phenotype_to_sample_ids(
        phenotype,
        plink.sample_ids,
        sample_id_col=args.phenotype_sample_id_col,
    )
    genotype = plink.matrix[np.asarray(keep_indices, dtype=np.int64)].astype(np.float32)
    target = phenotype[args.trait_col].to_numpy(dtype=np.float32)
    valid_mask = np.isfinite(target)
    return genotype[valid_mask], target[valid_mask]


def _run_r_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
) -> np.ndarray:
    result = run_r_baseline(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        output_dir=output_dir,
        rscript_path=args.rscript_path,
        seed=args.seed,
        sommer_method=args.sommer_method,
        keep_artifacts=True,
        bandwidth_scale=args.bandwidth_scale if model_name == "RKHS" else None,
    )
    return np.asarray(result.predictions, dtype=np.float32)


def _estimate_fold1_gate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, float]:
    inner_splits = make_outer_cv_splits(X_train, args.inner_folds, args.seed + 1)
    rkhs_oof = np.zeros(X_train.shape[0], dtype=np.float32)
    bayesb_oof = np.zeros(X_train.shape[0], dtype=np.float32)
    gblup_oof = np.zeros(X_train.shape[0], dtype=np.float32)
    for inner_id, (inner_train_idx, inner_valid_idx) in enumerate(inner_splits, start=1):
        X_inner_train = X_train[inner_train_idx]
        y_inner_train = y_train[inner_train_idx]
        X_inner_valid = X_train[inner_valid_idx]
        X_inner_train, X_inner_valid = impute_by_train_mean(X_inner_train, X_inner_valid)
        inner_dir = output_dir / f"inner_{inner_id}"
        rkhs_oof[inner_valid_idx] = _run_r_model("RKHS", X_inner_train, y_inner_train, X_inner_valid, inner_dir / "rkhs", args)
        bayesb_oof[inner_valid_idx] = _run_r_model("BayesB", X_inner_train, y_inner_train, X_inner_valid, inner_dir / "bayesb", args)
        gblup_oof[inner_valid_idx] = _run_r_model("GBLUP", X_inner_train, y_inner_train, X_inner_valid, inner_dir / "gblup", args)
    gate = fit_single_group_gate(
        y_true=y_train,
        y_model=rkhs_oof,
        y_bayesb=bayesb_oof,
        y_gblup=gblup_oof,
    )
    gate["num_groups"] = 1
    return gate


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y = _load_dataset(args)
    splits = make_outer_cv_splits(X, args.outer_cv_folds, args.seed)
    rows: list[dict[str, float | int | str]] = []
    frozen_gate: dict[str, float] | None = None
    if args.frozen_gate_path:
        frozen_gate = json.loads(Path(args.frozen_gate_path).read_text(encoding="utf-8"))

    for fold_id in args.fold_ids:
        train_idx, test_idx = splits[int(fold_id) - 1]
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]
        X_train, X_test = impute_by_train_mean(X_train, X_test)
        fold_dir = output_dir / f"fold_{int(fold_id)}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        if int(fold_id) == 1 and frozen_gate is None:
            gate = _estimate_fold1_gate(X_train, y_train, args, fold_dir / "gate_oof")
            (fold_dir / "single_group_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
            frozen_gate = {"alpha": float(gate["alpha"]), "w": float(gate["w"]), "num_groups": 1}
        elif frozen_gate is None:
            raise ValueError("fold>1 requires frozen gate from fold1.")

        rkhs_test = _run_r_model("RKHS", X_train, y_train, X_test, fold_dir / "rkhs_test", args)
        bayesb_test = _run_r_model("BayesB", X_train, y_train, X_test, fold_dir / "bayesb_test", args)
        gblup_test = _run_r_model("GBLUP", X_train, y_train, X_test, fold_dir / "gblup_test", args)
        pred = apply_dual_prior_gate(
            y_model=rkhs_test,
            y_bayesb=bayesb_test,
            y_gblup=gblup_test,
            alpha=float(frozen_gate["alpha"]),
            w=float(frozen_gate["w"]),
        )
        metric = regression_metrics(y_test, pred)
        row = {
            "fold": int(fold_id),
            "model": "RKHS-direct-dual-prior",
            "pearson": float(metric["pearson"]),
            "r2": float(metric["r2"]),
            "rmse": float(metric["rmse"]),
            "mae": float(metric["mae"]),
            "alpha": float(frozen_gate["alpha"]),
            "w": float(frozen_gate["w"]),
            "num_groups": 1,
            "n_snps": int(X.shape[1]),
            "output_dir": str(fold_dir),
        }
        rows.append(row)
        pd.DataFrame([row]).to_csv(fold_dir / "fold_metrics.csv", index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "fold_summary.csv", index=False)
    print(summary.to_csv(index=False))


if __name__ == "__main__":
    main()
