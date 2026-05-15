from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize 5.4-duli-liudang fusion results.")
    parser.add_argument("--fusion-root", required=True)
    parser.add_argument("--no-prior-root", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_metric_from_fold_metrics(path: Path, model_name: str | None = None) -> dict[str, float]:
    frame = pd.read_csv(path)
    if model_name is not None:
        frame = frame[frame["model"].astype(str) == model_name].copy()
    if frame.empty:
        return {"pearson": float("nan"), "r2": float("nan")}
    return {
        "pearson": float(frame["pearson"].mean()),
        "r2": float(frame["r2"].mean()),
    }


def _load_trait_metrics(trait_root: Path) -> dict[str, object]:
    raise RuntimeError("_load_trait_metrics now requires explicit roots")


def _load_trait_metrics_with_roots(trait_root: Path, no_prior_root: Path, baseline_root: Path) -> dict[str, object]:
    no_prior = _mean_metric_from_fold_metrics(no_prior_root / trait_root.parent.name / trait_root.name / "fold_metrics.csv")
    baseline_root = baseline_root / trait_root.parent.name / trait_root.name
    baselines = {
        model: _mean_metric_from_fold_metrics(baseline_root / "fold_metrics.csv", model)
        for model in ("BayesB", "GBLUP", "RKHS")
    }
    fusion = {
        "only_single": _mean_metric_from_fold_metrics(trait_root / "prior_only_gblup" / "fold_metrics.csv"),
        "single_two_step_clip": _mean_metric_from_fold_metrics(trait_root / "tabicl_gblup_single_prior" / "fold_metrics.csv"),
        "single_two_step_ls": _mean_metric_from_fold_metrics(trait_root / "tabicl_gblup_single_prior" / "fold_metrics.csv"),
        "single_all_ls": _mean_metric_from_fold_metrics(trait_root / "tabicl_gblup_single_prior" / "fold_metrics.csv"),
        "only_dual": _mean_metric_from_fold_metrics(trait_root / "prior_only_bayesb_gblup" / "fold_metrics.csv"),
        "dual_two_step_clip": _mean_metric_from_fold_metrics(trait_root / "tabicl_tabicl_dual_prior" / "fold_metrics.csv"),
        "dual_two_step_ls": _mean_metric_from_fold_metrics(trait_root / "tabicl_tabicl_dual_prior" / "fold_metrics.csv"),
        "dual_all_ls": _mean_metric_from_fold_metrics(trait_root / "tabicl_tabicl_dual_prior" / "fold_metrics.csv"),
        "only_triple": _mean_metric_from_fold_metrics(trait_root / "prior_only_triple" / "fold_metrics.csv"),
        "triple_two_step_clip": _mean_metric_from_fold_metrics(trait_root / "tabicl_tabicl_triple_prior" / "fold_metrics.csv"),
        "triple_two_step_ls": _mean_metric_from_fold_metrics(trait_root / "tabicl_tabicl_triple_prior" / "fold_metrics.csv"),
        "triple_all_ls": _mean_metric_from_fold_metrics(trait_root / "tabicl_tabicl_triple_prior" / "fold_metrics.csv"),
    }
    best_baseline_name = max(baselines, key=lambda k: baselines[k]["pearson"])
    best_baseline = baselines[best_baseline_name]["pearson"]
    out = {
        "dataset": trait_root.parent.name,
        "trait_slug": trait_root.name,
        "no_prior": no_prior["pearson"],
        "BayesB": baselines["BayesB"]["pearson"],
        "GBLUP": baselines["GBLUP"]["pearson"],
        "RKHS": baselines["RKHS"]["pearson"],
        "best_baseline": best_baseline,
        "best_baseline_name": best_baseline_name,
        **{k: v["pearson"] for k, v in fusion.items()},
    }
    return out


def main() -> None:
    args = parse_args()
    fusion_root = Path(args.fusion_root)
    rows = []
    for dataset_dir in sorted([p for p in fusion_root.iterdir() if p.is_dir()]):
        for trait_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
            req = [
                trait_dir / "tabicl_bayesb_single_prior" / "fold_metrics.csv",
                trait_dir / "tabicl_gblup_single_prior" / "fold_metrics.csv",
                trait_dir / "tabicl_rkhs_single_prior" / "fold_metrics.csv",
                trait_dir / "tabicl_tabicl_dual_prior" / "fold_metrics.csv",
                trait_dir / "tabicl_tabicl_triple_prior" / "fold_metrics.csv",
                trait_dir / "prior_only_bayesb" / "fold_metrics.csv",
                trait_dir / "prior_only_gblup" / "fold_metrics.csv",
                trait_dir / "prior_only_rkhs" / "fold_metrics.csv",
                trait_dir / "prior_only_bayesb_gblup" / "fold_metrics.csv",
                trait_dir / "prior_only_triple" / "fold_metrics.csv",
            ]
            if not all(p.exists() for p in req):
                continue
            rows.append(_load_trait_metrics_with_roots(trait_dir, Path(args.no_prior_root), Path(args.baseline_root)))

    frame = pd.DataFrame(rows).sort_values(["dataset", "trait_slug"]).reset_index(drop=True)
    frame.to_csv(args.output_csv, index=False)

    summary = {
        "traits_completed": int(len(frame)),
        "means": {
            col: float(frame[col].mean())
            for col in [
                "no_prior",
                "BayesB",
                "GBLUP",
                "RKHS",
                "only_single",
                "single_two_step_clip",
                "single_two_step_ls",
                "single_all_ls",
                "only_dual",
                "dual_two_step_clip",
                "dual_two_step_ls",
                "dual_all_ls",
                "only_triple",
                "triple_two_step_clip",
                "triple_two_step_ls",
                "triple_all_ls",
            ]
            if col in frame.columns
        },
        "relative_pct": {
            col: {
                "vs_no_prior_pct": float((frame[col].mean() - frame["no_prior"].mean()) / abs(frame["no_prior"].mean()) * 100.0),
                "vs_best_baseline_pct": float((frame[col].mean() - frame["best_baseline"].mean()) / abs(frame["best_baseline"].mean()) * 100.0),
            }
            for col in [
                "only_single",
                "single_two_step_clip",
                "single_two_step_ls",
                "single_all_ls",
                "only_dual",
                "dual_two_step_clip",
                "dual_two_step_ls",
                "dual_all_ls",
                "only_triple",
                "triple_two_step_clip",
                "triple_two_step_ls",
                "triple_all_ls",
            ]
        },
    }
    Path(args.output_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(frame.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
