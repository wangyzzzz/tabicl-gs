from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "outputs" / "results_support_workbook" / "main_results_non_pig_fixed.csv"
COMP = ROOT / "outputs" / "results_support_workbook" / "compare_all_41_traits.csv"
OUTDIR = ROOT / "outputs" / "figure2_inputs"


def pct_diff(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a - b) / b.abs() * 100.0


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    main_df = pd.read_csv(MAIN)
    comp_df = pd.read_csv(COMP)

    single_blocks = []
    prior_map = {
        "BayesB": "single_bayesb_two_step_ls",
        "GBLUP": "single_gblup_two_step_ls",
        "RKHS": "single_rkhs_two_step_ls",
    }
    for prior_name, single_name in prior_map.items():
        block = main_df[
            [
                "dataset",
                "trait_slug",
                "best_baseline_name",
                "best_baseline",
                "no_prior_tabicl",
                prior_name,
                single_name,
            ]
        ].copy()
        block["prior_name"] = prior_name
        block["single_name"] = single_name
        block["prior_only"] = block[prior_name]
        block["single_value"] = block[single_name]
        block["single_vs_own_prior_abs"] = block["single_value"] - block["prior_only"]
        block["single_vs_own_prior_pct"] = pct_diff(block["single_value"], block["prior_only"])
        block["single_vs_best_baseline_pct"] = pct_diff(block["single_value"], block["best_baseline"])
        block["single_vs_no_prior_pct"] = pct_diff(block["single_value"], block["no_prior_tabicl"])
        single_blocks.append(
            block[
                [
                    "dataset",
                    "trait_slug",
                    "prior_name",
                    "single_name",
                    "best_baseline_name",
                    "no_prior_tabicl",
                    "prior_only",
                    "single_value",
                    "single_vs_own_prior_abs",
                    "single_vs_own_prior_pct",
                    "single_vs_best_baseline_pct",
                    "single_vs_no_prior_pct",
                ]
            ]
        )
    single_long = pd.concat(single_blocks, ignore_index=True)

    triple_trait = main_df[
        [
            "dataset",
            "trait_slug",
            "best_baseline_name",
            "best_baseline",
            "no_prior_tabicl",
            "BayesB",
            "GBLUP",
            "RKHS",
            "single_bayesb_two_step_ls",
            "single_gblup_two_step_ls",
            "single_rkhs_two_step_ls",
            "triple_two_step_ls",
        ]
    ].copy()
    triple_trait = triple_trait.merge(
        comp_df[["dataset", "trait_slug", "only_triple"]],
        on=["dataset", "trait_slug"],
        how="left",
    )
    for ref in [
        "BayesB",
        "GBLUP",
        "RKHS",
        "best_baseline",
        "no_prior_tabicl",
        "single_bayesb_two_step_ls",
        "single_gblup_two_step_ls",
        "single_rkhs_two_step_ls",
        "only_triple",
    ]:
        triple_trait[f"triple_vs_{ref}_pct"] = pct_diff(triple_trait["triple_two_step_ls"], triple_trait[ref])

    compare_rows = []
    for base in ["BayesB", "GBLUP", "RKHS"]:
        diff = pct_diff(main_df["triple_two_step_ls"], main_df[base])
        compare_rows.append(
            {
                "block": "triple_vs_baseline",
                "comparison": base,
                "mean_pct": diff.mean(),
                "strict_win_count": int((main_df["triple_two_step_ls"] > main_df[base]).sum()),
                "nonloss_count": int((main_df["triple_two_step_ls"] >= main_df[base]).sum()),
                "n_traits": len(main_df),
            }
        )
    for comp_name, col in [
        ("best_baseline", "best_baseline"),
        ("no_prior_tabicl", "no_prior_tabicl"),
        ("only_triple", "only_triple"),
    ]:
        diff = pct_diff(triple_trait["triple_two_step_ls"], triple_trait[col])
        compare_rows.append(
            {
                "block": f"triple_vs_{comp_name}",
                "comparison": comp_name,
                "mean_pct": diff.mean(),
                "strict_win_count": int((triple_trait["triple_two_step_ls"] > triple_trait[col]).sum()),
                "nonloss_count": int((triple_trait["triple_two_step_ls"] >= triple_trait[col]).sum()),
                "n_traits": len(triple_trait),
            }
        )
    compare_sum = pd.DataFrame(compare_rows)

    triple_vs_best = main_df[
        ["dataset", "trait_slug", "best_baseline_name", "best_baseline", "triple_two_step_ls"]
    ].copy()
    triple_vs_best["triple_minus_best_abs"] = triple_vs_best["triple_two_step_ls"] - triple_vs_best["best_baseline"]
    triple_vs_best["triple_minus_best_pct"] = pct_diff(
        triple_vs_best["triple_two_step_ls"], triple_vs_best["best_baseline"]
    )

    outputs = {
        "figure2_single_long.csv": single_long,
        "figure2_compare_sum.csv": compare_sum,
        "figure2_triple_trait.csv": triple_trait,
        "figure2_triple_vs_best.csv": triple_vs_best,
    }
    for filename, df in outputs.items():
        path = OUTDIR / filename
        df.to_csv(path, index=False)
        print(path)


if __name__ == "__main__":
    main()
