from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MAIN_RESULT_COLUMNS = [
    "no_prior_tabicl",
    "BayesB",
    "GBLUP",
    "RKHS",
    "single_bayesb_two_step_ls",
    "single_gblup_two_step_ls",
    "single_rkhs_two_step_ls",
    "triple_two_step_ls",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize decoupled marker-count experiment results.")
    parser.add_argument("--marker-root", required=True)
    parser.add_argument("--full-main-results-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _read_manifest(manifest_path: Path) -> dict[str, object]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _best_baseline_name(row: dict[str, float | str | int]) -> str:
    return max(["BayesB", "GBLUP", "RKHS"], key=lambda name: float(row[name]))


def _append_relative_fields(row: dict[str, float | str | int]) -> dict[str, float | str | int]:
    best_name = _best_baseline_name(row)
    best_value = float(row[best_name])
    no_prior = float(row["no_prior_tabicl"])
    row["best_baseline_name"] = best_name
    row["best_baseline"] = best_value
    for col in [
        "single_bayesb_two_step_ls",
        "single_gblup_two_step_ls",
        "single_rkhs_two_step_ls",
        "triple_two_step_ls",
    ]:
        val = float(row[col])
        row[f"{col}_vs_no_prior_pct"] = float((val - no_prior) / abs(no_prior) * 100.0)
        row[f"{col}_vs_best_baseline_pct"] = float((val - best_value) / abs(best_value) * 100.0)
    row["single_bayesb_two_step_ls_vs_own_prior_pct"] = float(
        (float(row["single_bayesb_two_step_ls"]) - float(row["BayesB"])) / abs(float(row["BayesB"])) * 100.0
    )
    row["single_gblup_two_step_ls_vs_own_prior_pct"] = float(
        (float(row["single_gblup_two_step_ls"]) - float(row["GBLUP"])) / abs(float(row["GBLUP"])) * 100.0
    )
    row["single_rkhs_two_step_ls_vs_own_prior_pct"] = float(
        (float(row["single_rkhs_two_step_ls"]) - float(row["RKHS"])) / abs(float(row["RKHS"])) * 100.0
    )
    row["triple_two_step_ls_vs_BayesB_pct"] = float(
        (float(row["triple_two_step_ls"]) - float(row["BayesB"])) / abs(float(row["BayesB"])) * 100.0
    )
    row["triple_two_step_ls_vs_GBLUP_pct"] = float(
        (float(row["triple_two_step_ls"]) - float(row["GBLUP"])) / abs(float(row["GBLUP"])) * 100.0
    )
    row["triple_two_step_ls_vs_RKHS_pct"] = float(
        (float(row["triple_two_step_ls"]) - float(row["RKHS"])) / abs(float(row["RKHS"])) * 100.0
    )
    return row


def _read_marker_rows(marker_root: Path) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for compare_csv in sorted(marker_root.glob("*/*/maxsnps_*/compare/compare_main.csv")):
        manifest_path = compare_csv.parent.parent / "run_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = _read_manifest(manifest_path)
        frame = pd.read_csv(compare_csv)
        if frame.empty:
            continue
        row = frame.iloc[0]
        out_row: dict[str, float | str | int] = {
            "dataset": str(manifest["dataset_slug"]),
            "trait_slug": str(manifest["trait_slug"]),
            "marker_count": int(manifest["requested_marker_count"]),
            "effective_marker_count": int(manifest["effective_marker_count"]),
            "best_block_group_size": int(manifest["best_block_group_size"]),
            "source": "marker_count_run",
        }
        for col in MAIN_RESULT_COLUMNS:
            out_row[col] = float(row[col])
        rows.append(_append_relative_fields(out_row))
    return rows


def _read_main_reference_rows(full_main: pd.DataFrame, target_traits: set[tuple[str, str]]) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    if full_main.empty:
        return rows
    no_prior_col = "no_prior_tabicl" if "no_prior_tabicl" in full_main.columns else "no_prior"
    filtered = full_main[full_main[["dataset", "trait_slug"]].apply(tuple, axis=1).isin(target_traits)]
    for _, record in filtered.iterrows():
        out_row: dict[str, float | str | int] = {
            "dataset": str(record["dataset"]),
            "trait_slug": str(record["trait_slug"]),
            "marker_count": 10000,
            "effective_marker_count": 10000,
            "best_block_group_size": int(record.get("best_block_group_size", 0)) if "best_block_group_size" in record else 0,
            "source": "main_10k_reference",
            "no_prior_tabicl": float(record[no_prior_col]),
            "BayesB": float(record["BayesB"]),
            "GBLUP": float(record["GBLUP"]),
            "RKHS": float(record["RKHS"]),
            "single_bayesb_two_step_ls": float(record["single_bayesb_two_step_ls"]),
            "single_gblup_two_step_ls": float(record["single_gblup_two_step_ls"]),
            "single_rkhs_two_step_ls": float(record["single_rkhs_two_step_ls"]),
            "triple_two_step_ls": float(record["triple_two_step_ls"]),
        }
        rows.append(_append_relative_fields(out_row))
    return rows


def main() -> None:
    args = parse_args()
    marker_root = Path(args.marker_root)
    full_main = pd.read_csv(args.full_main_results_csv)

    marker_rows = _read_marker_rows(marker_root)
    target_traits = {(str(row["dataset"]), str(row["trait_slug"])) for row in marker_rows}
    all_rows = marker_rows + _read_main_reference_rows(full_main, target_traits)

    frame = pd.DataFrame(all_rows).sort_values(["dataset", "trait_slug", "marker_count"]).reset_index(drop=True)
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)

    summary = {
        "traits_completed": int(frame[["dataset", "trait_slug"]].drop_duplicates().shape[0]) if not frame.empty else 0,
        "rows_completed": int(len(frame)),
        "marker_counts": sorted({int(v) for v in frame["marker_count"].tolist()}) if not frame.empty else [],
        "mean_by_marker_count": {
            str(int(marker_count)): {
                col: float(sub[col].mean())
                for col in [
                    "no_prior_tabicl",
                    "BayesB",
                    "GBLUP",
                    "RKHS",
                    "single_bayesb_two_step_ls",
                    "single_gblup_two_step_ls",
                    "single_rkhs_two_step_ls",
                    "triple_two_step_ls",
                ]
            }
            for marker_count, sub in frame.groupby("marker_count")
        },
    }
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(frame.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
