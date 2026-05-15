from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize TabICL/Attention/baseline family comparison.")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _load_experiment_summary(path: Path, variant_name: str, residual_mode: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.copy()
    frame["variant_name"] = variant_name
    frame["residual_mode"] = residual_mode
    return frame


def _load_baselines(gblup_path: Path, bayes_path: Path) -> pd.DataFrame:
    gblup = pd.read_csv(gblup_path)
    gblup = gblup[gblup["model"] == "GBLUP"].copy()
    bayes = pd.read_csv(bayes_path).copy()
    base = pd.concat([gblup, bayes], ignore_index=True)
    base["variant_name"] = base["model"]
    base["residual_mode"] = "baseline"
    return base


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path("/home/server/code/git/TabICLv2-test/outputs")
    def with_variant(frame: pd.DataFrame, variant_name: str, residual_mode: str) -> pd.DataFrame:
        x = frame.copy()
        x["variant_name"] = variant_name
        x["residual_mode"] = residual_mode
        return x

    # Build attention frames directly from fold_summary layout.
    def load_attention_variant(root_dir: Path, variant_dir: str, variant_name: str, residual_mode: str) -> pd.DataFrame:
        rows = []
        for metrics_path in sorted((root_dir / variant_dir).glob("group_size_*/*/fold_metrics.csv")):
            group_size = int(metrics_path.parent.parent.name.split("_")[-1])
            trait_slug = metrics_path.parent.name
            frame = pd.read_csv(metrics_path)
            model_metrics = frame[frame["model"].str.contains("Attention", na=False) | frame["model"].str.contains("TabICL", na=False)]
            if model_metrics.empty:
                continue
            rows.append(
                {
                    "trait_slug": trait_slug,
                    "group_size": group_size,
                    "pearson_mean": float(model_metrics["pearson"].mean()),
                    "r2_mean": float(model_metrics["r2"].mean()),
                    "mean_reduced_block_embedding_dim_mean": float(model_metrics["mean_reduced_block_embedding_dim"].mean()),
                    "stage2_input_dim_mean": float(model_metrics["stage2_input_dim"].mean()),
                    "variant_name": variant_name,
                    "residual_mode": residual_mode,
                }
            )
        return pd.DataFrame(rows)

    comparison_frames = [
        with_variant(
            pd.read_csv(root / "rice529_10traits_dynamic99_block_norm" / "experiment_summary.csv"),
            "TabICL-TabICL",
            "y",
        ),
        load_attention_variant(root / "rice529_10traits_attention_variants", "tabicl_attention", "TabICL-Attention", "y"),
        load_attention_variant(root / "rice529_10traits_attention_variants", "tabicl_pca_attention", "TabICL-PCA-Attention", "y"),
        load_attention_variant(root / "rice529_10traits_attention_variants_residual_bayesb", "tabicl_attention", "TabICL-Attention", "y_minus_BayesB"),
        load_attention_variant(root / "rice529_10traits_attention_variants_residual_bayesb", "tabicl_pca_attention", "TabICL-PCA-Attention", "y_minus_BayesB"),
    ]
    tabicl_all = pd.concat(comparison_frames, ignore_index=True)
    baselines = _load_baselines(
        root / "rice529_10traits_baseline_only" / "trait_model_summary.csv",
        root / "rice529_10traits_bayes_only_15000iter" / "trait_model_summary.csv",
    )

    overall_tabicl = (
        tabicl_all.groupby(["variant_name", "residual_mode", "group_size"], as_index=False)
        .agg(
            traits_completed=("trait_slug", "nunique"),
            pearson_mean=("pearson_mean", "mean"),
            r2_mean=("r2_mean", "mean"),
            pc99_dim=("mean_reduced_block_embedding_dim_mean", "mean"),
            stage2_input_dim=("stage2_input_dim_mean", "mean"),
        )
        .sort_values(["variant_name", "residual_mode", "group_size"])
    )
    baseline_overall = (
        baselines.groupby(["variant_name"], as_index=False)
        .agg(
            traits_completed=("trait_slug", "nunique"),
            pearson_mean=("pearson_mean", "mean"),
            r2_mean=("r2_mean", "mean"),
        )
        .sort_values("pearson_mean", ascending=False)
    )

    best_tabicl = (
        tabicl_all.sort_values(["trait_slug", "pearson_mean"], ascending=[True, False])
        .groupby("trait_slug", as_index=False)
        .first()
    )
    best_baseline = (
        baselines.sort_values(["trait_slug", "pearson_mean"], ascending=[True, False])
        .groupby("trait_slug", as_index=False)
        .first()
    )
    best_compare = best_tabicl.merge(best_baseline, on="trait_slug", suffixes=("_tabicl", "_baseline"))
    best_compare["delta_pearson"] = best_compare["pearson_mean_tabicl"] - best_compare["pearson_mean_baseline"]
    best_compare["delta_r2"] = best_compare["r2_mean_tabicl"] - best_compare["r2_mean_baseline"]
    best_compare = best_compare.sort_values("delta_pearson", ascending=False)

    tabicl_all.to_csv(out_dir / "tabicl_family_full.csv", index=False)
    overall_tabicl.to_csv(out_dir / "tabicl_family_overall.csv", index=False)
    baseline_overall.to_csv(out_dir / "baseline_overall.csv", index=False)
    best_compare.to_csv(out_dir / "best_tabicl_vs_best_baseline.csv", index=False)

    print("=== overall_tabicl ===")
    print(overall_tabicl.to_string(index=False))
    print()
    print("=== baseline_overall ===")
    print(baseline_overall.to_string(index=False))
    print()
    print("=== best_tabicl_vs_best_baseline ===")
    print(
        best_compare[
            [
                "trait_slug",
                "variant_name_tabicl",
                "residual_mode_tabicl",
                "group_size",
                "pearson_mean_tabicl",
                "r2_mean_tabicl",
                "mean_reduced_block_embedding_dim_mean",
                "variant_name_baseline",
                "pearson_mean_baseline",
                "r2_mean_baseline",
                "delta_pearson",
                "delta_r2",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
