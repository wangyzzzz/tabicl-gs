from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize triple-prior TabICL, only-triple-prior, no-prior TabICL, and baselines."
    )
    parser.add_argument("--triple-root", required=True)
    parser.add_argument("--rice-baseline-root", required=True)
    parser.add_argument("--multi-baseline-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


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


def _load_no_prior(trait_root: Path) -> dict[str, float]:
    return _mean_metric_from_fold_metrics(trait_root / "tabicl_tabicl_no_prior" / "fold_metrics.csv")


def _load_triple_prior(trait_root: Path) -> dict[str, float]:
    fold_paths = sorted((trait_root / "tabicl_tabicl_triple_prior").glob("fold_*/fold_metrics.csv"))
    rows: list[pd.DataFrame] = []
    for fold_path in fold_paths:
        rows.append(pd.read_csv(fold_path))
    if not rows:
        return {"pearson": float("nan"), "r2": float("nan")}
    frame = pd.concat(rows, ignore_index=True)
    return {
        "pearson": float(frame["pearson"].mean()),
        "r2": float(frame["r2"].mean()),
    }


def _load_only_triple_prior(trait_root: Path) -> dict[str, float]:
    metrics_path = trait_root / "prior_only_triple" / "fold_metrics.csv"
    return _mean_metric_from_fold_metrics(metrics_path)


def _baseline_trait_dir(dataset_slug: str, trait_slug: str, rice_root: Path, multi_root: Path) -> Path:
    if dataset_slug == "rice529":
        return rice_root / trait_slug
    return multi_root / dataset_slug / trait_slug


def _load_baselines(dataset_slug: str, trait_slug: str, rice_root: Path, multi_root: Path) -> dict[str, float]:
    metrics_path = _baseline_trait_dir(dataset_slug, trait_slug, rice_root, multi_root) / "fold_metrics.csv"
    frame = pd.read_csv(metrics_path)
    out: dict[str, float] = {}
    for model_name in ("BayesA", "BayesB", "BayesLasso", "RKHS", "GBLUP"):
        sub = frame[frame["model"].astype(str) == model_name].copy()
        out[f"{model_name}_pearson"] = float(sub["pearson"].mean()) if not sub.empty else float("nan")
        out[f"{model_name}_r2"] = float(sub["r2"].mean()) if not sub.empty else float("nan")
    return out


def build_compare(triple_root: Path, rice_root: Path, multi_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_dir in sorted([p for p in triple_root.iterdir() if p.is_dir() and p.name != "logs"]):
        dataset_slug = dataset_dir.name
        for trait_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
            trait_slug = trait_dir.name
            row: dict[str, object] = {
                "dataset": dataset_slug,
                "trait_slug": trait_slug,
            }
            no_prior = _load_no_prior(trait_dir)
            triple = _load_triple_prior(trait_dir)
            only_triple = _load_only_triple_prior(trait_dir)
            baselines = _load_baselines(dataset_slug, trait_slug, rice_root, multi_root)
            row["no_prior_tabicl_pearson"] = no_prior["pearson"]
            row["no_prior_tabicl_r2"] = no_prior["r2"]
            row["triple_prior_tabicl_pearson"] = triple["pearson"]
            row["triple_prior_tabicl_r2"] = triple["r2"]
            row["only_triple_prior_pearson"] = only_triple["pearson"]
            row["only_triple_prior_r2"] = only_triple["r2"]
            row.update(baselines)
            baseline_pearsons = {
                model_name: float(row[f"{model_name}_pearson"])
                for model_name in ("BayesA", "BayesB", "BayesLasso", "RKHS", "GBLUP")
            }
            best_baseline_model = max(baseline_pearsons, key=baseline_pearsons.get)
            best_baseline_pearson = baseline_pearsons[best_baseline_model]
            row["best_baseline_model"] = best_baseline_model
            row["best_baseline_pearson"] = best_baseline_pearson
            row["delta_triple_vs_best_baseline"] = float(row["triple_prior_tabicl_pearson"]) - best_baseline_pearson
            row["delta_only_triple_vs_best_baseline"] = float(row["only_triple_prior_pearson"]) - best_baseline_pearson
            row["delta_no_prior_vs_best_baseline"] = float(row["no_prior_tabicl_pearson"]) - best_baseline_pearson
            row["delta_triple_vs_only_triple"] = (
                float(row["triple_prior_tabicl_pearson"]) - float(row["only_triple_prior_pearson"])
            )
            row["delta_triple_vs_no_prior"] = (
                float(row["triple_prior_tabicl_pearson"]) - float(row["no_prior_tabicl_pearson"])
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "trait_slug"]).reset_index(drop=True)


def build_model_average(compare_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = {
        "triple_prior_tabicl": "triple_prior_tabicl_pearson",
        "only_triple_prior": "only_triple_prior_pearson",
        "no_prior_tabicl": "no_prior_tabicl_pearson",
        "BayesA": "BayesA_pearson",
        "BayesB": "BayesB_pearson",
        "BayesLasso": "BayesLasso_pearson",
        "RKHS": "RKHS_pearson",
        "GBLUP": "GBLUP_pearson",
    }
    rows = []
    for model_name, col in metric_cols.items():
        rows.append(
            {
                "model": model_name,
                "traits_completed": int(compare_df[col].notna().sum()),
                "pearson_mean": float(compare_df[col].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("pearson_mean", ascending=False).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    triple_root = Path(args.triple_root)
    rice_root = Path(args.rice_baseline_root)
    multi_root = Path(args.multi_baseline_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    compare_df = build_compare(triple_root=triple_root, rice_root=rice_root, multi_root=multi_root)
    compare_df.to_csv(output_dir / "trait_compare.csv", index=False)

    dataset_summary = (
        compare_df.groupby("dataset", as_index=False)[
            [
                "triple_prior_tabicl_pearson",
                "only_triple_prior_pearson",
                "no_prior_tabicl_pearson",
                "BayesA_pearson",
                "BayesB_pearson",
                "BayesLasso_pearson",
                "RKHS_pearson",
                "GBLUP_pearson",
            ]
        ]
        .mean()
        .sort_values("dataset")
    )
    dataset_summary.to_csv(output_dir / "dataset_summary.csv", index=False)

    model_average = build_model_average(compare_df)
    model_average.to_csv(output_dir / "model_average.csv", index=False)

    summary = {
        "traits_completed": int(len(compare_df)),
        "datasets_completed": int(compare_df["dataset"].nunique()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(compare_df.to_string(index=False))
    print(model_average.to_string(index=False))


if __name__ == "__main__":
    main()
