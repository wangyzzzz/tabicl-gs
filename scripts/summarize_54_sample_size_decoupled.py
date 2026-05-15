from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MAIN_RESULT_COLUMNS = [
    "no_prior",
    "BayesB",
    "GBLUP",
    "RKHS",
    "single_bayesb_two_step_ls",
    "single_gblup_two_step_ls",
    "single_rkhs_two_step_ls",
    "triple_two_step_ls",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize decoupled sample-size experiment results.")
    parser.add_argument("--sample-root", required=True)
    parser.add_argument("--full-main-results-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _read_trait_rows_from_repeat(compare_dir: Path) -> dict[str, float] | None:
    compare_csv = compare_dir / "compare_main.csv"
    if not compare_csv.exists():
        return None

    frame = pd.read_csv(compare_csv)
    if frame.empty:
        return None
    row = frame.iloc[0]
    return {
        "no_prior": float(row["no_prior_tabicl"]),
        "BayesB": float(row["BayesB"]),
        "GBLUP": float(row["GBLUP"]),
        "RKHS": float(row["RKHS"]),
        "single_bayesb_two_step_ls": float(row["single_bayesb_two_step_ls"]),
        "single_gblup_two_step_ls": float(row["single_gblup_two_step_ls"]),
        "single_rkhs_two_step_ls": float(row["single_rkhs_two_step_ls"]),
        "triple_two_step_ls": float(row["triple_two_step_ls"]),
    }


def _best_baseline_name(row: dict[str, float]) -> str:
    return max(["BayesB", "GBLUP", "RKHS"], key=lambda name: row[name])


def _append_relative_fields(row: dict[str, float | str]) -> dict[str, float | str]:
    best_name = _best_baseline_name(row)  # type: ignore[arg-type]
    best_value = float(row[best_name])  # type: ignore[index]
    row["best_baseline_name"] = best_name
    row["best_baseline"] = best_value
    for col in [
        "single_bayesb_two_step_ls",
        "single_gblup_two_step_ls",
        "single_rkhs_two_step_ls",
        "triple_two_step_ls",
    ]:
        val = float(row[col])  # type: ignore[index]
        row[f"{col}_vs_no_prior_pct"] = float((val - float(row["no_prior"])) / abs(float(row["no_prior"])) * 100.0)
        row[f"{col}_vs_best_baseline_pct"] = float((val - best_value) / abs(best_value) * 100.0)
    return row


def main() -> None:
    args = parse_args()
    sample_root = Path(args.sample_root)
    full_main = pd.read_csv(args.full_main_results_csv)
    no_prior_col = "no_prior" if "no_prior" in full_main.columns else "no_prior_tabicl"

    rows: list[dict[str, float | str]] = []
    for dataset_dir in sorted([p for p in sample_root.iterdir() if p.is_dir()]):
        for trait_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
            for proportion_dir in sorted([p for p in trait_dir.iterdir() if p.is_dir() and p.name.startswith("p_")]):
                repeat_rows = []
                for repeat_dir in sorted([p for p in proportion_dir.iterdir() if p.is_dir() and p.name.startswith("repeat_")]):
                    metrics = _read_trait_rows_from_repeat(repeat_dir / "compare")
                    if metrics is None:
                        continue
                    repeat_rows.append(metrics)
                if not repeat_rows:
                    continue
                mean_row = {
                    key: float(np.mean([row[key] for row in repeat_rows]))
                    for key in MAIN_RESULT_COLUMNS
                }
                out_row: dict[str, float | str] = {
                    "dataset": dataset_dir.name,
                    "trait_slug": trait_dir.name,
                    "proportion": float(proportion_dir.name.replace("p_", "")),
                    "repeats_completed": int(len(repeat_rows)),
                    **mean_row,
                }
                rows.append(_append_relative_fields(out_row))

    if not full_main.empty:
        for _, record in full_main.iterrows():
            row = {
                "dataset": str(record["dataset"]),
                "trait_slug": str(record["trait_slug"]),
                "proportion": 1.0,
                "repeats_completed": 1,
                "no_prior": float(record[no_prior_col]),
                "BayesB": float(record["BayesB"]),
                "GBLUP": float(record["GBLUP"]),
                "RKHS": float(record["RKHS"]),
                "single_bayesb_two_step_ls": float(record["single_bayesb_two_step_ls"]),
                "single_gblup_two_step_ls": float(record["single_gblup_two_step_ls"]),
                "single_rkhs_two_step_ls": float(record["single_rkhs_two_step_ls"]),
                "triple_two_step_ls": float(record["triple_two_step_ls"]),
            }
            rows.append(_append_relative_fields(row))

    frame = pd.DataFrame(rows).sort_values(["dataset", "trait_slug", "proportion"]).reset_index(drop=True)
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)

    summary = {
        "traits_completed": int(frame[["dataset", "trait_slug"]].drop_duplicates().shape[0]) if not frame.empty else 0,
        "rows_completed": int(len(frame)),
        "proportions": sorted({float(v) for v in frame["proportion"].tolist()}) if not frame.empty else [],
        "mean_by_proportion": {
            f"{float(proportion):.2f}": {
                col: float(sub[col].mean())
                for col in [
                    "no_prior",
                    "BayesB",
                    "GBLUP",
                    "RKHS",
                    "single_bayesb_two_step_ls",
                    "single_gblup_two_step_ls",
                    "single_rkhs_two_step_ls",
                    "triple_two_step_ls",
                ]
            }
            for proportion, sub in frame.groupby("proportion")
        },
    }
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(frame.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
