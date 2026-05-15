from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabicl_gs.config import deep_update, load_experiment_config
from tabicl_gs.eval.metrics import regression_metrics
from tabicl_gs.pipeline.dual_prior_fold_search import _load_fold_data
from tabicl_gs.pipeline.experiment import (
    _compute_oof_baseline_prior_predictions,
    _save_baseline_inner_oof_bundle,
    impute_by_train_mean,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill fold1 baseline inner OOF artifacts without rerunning full outer-test."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True, help="Existing baseline-only trait root")
    parser.add_argument("--baseline-model", required=True, choices=["GBLUP", "BayesA", "BayesB", "BayesLasso", "RKHS"])
    parser.add_argument("--trait-col", default=None)
    parser.add_argument("--plink-prefix", default=None)
    parser.add_argument("--phenotype-csv", default=None)
    parser.add_argument("--phenotype-sample-id-col", default=None)
    parser.add_argument("--fold-id", type=int, default=1)
    parser.add_argument("--inner-folds", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_experiment_config(args.config)
    runtime_override = {
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

    fold_id = int(args.fold_id)
    output_root = Path(args.output_root)
    fold_dir = output_root / f"fold_{fold_id}"
    baseline_dir = fold_dir / args.baseline_model
    baseline_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, _X_test, _y_test = _load_fold_data(base_config, fold_id=fold_id)
    X_train_base, _ = impute_by_train_mean(X_train, X_train)
    oof_pred, oof_summary = _compute_oof_baseline_prior_predictions(
        baseline_model=args.baseline_model,
        fold_dir=baseline_dir / "_inner_oof",
        X_train_base=X_train_base,
        y_train=y_train,
        config=base_config,
        fold_id=fold_id,
        n_splits=int(args.inner_folds),
    )
    metrics = regression_metrics(y_train, oof_pred)
    _save_baseline_inner_oof_bundle(
        baseline_dir,
        baseline_name=args.baseline_model,
        y_true=y_train,
        y_pred=oof_pred,
        metadata={
            **oof_summary,
            "source": "baseline_inner_oof",
            "fold": fold_id,
            "inner_oof_pearson": float(metrics["pearson"]),
            "inner_oof_r2": float(metrics["r2"]),
        },
    )
    print(
        json.dumps(
            {
                "baseline_model": args.baseline_model,
                "fold": fold_id,
                "output_dir": str(baseline_dir),
                "inner_oof_pearson": float(metrics["pearson"]),
                "inner_oof_r2": float(metrics["r2"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
