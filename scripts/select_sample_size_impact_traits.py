from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select gain/flat traits for sample-size impact study.")
    parser.add_argument("--compare-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--datasets", nargs="+", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.compare_csv)
    if args.datasets:
        wanted = {str(x).lower() for x in args.datasets}
        frame = frame[frame["dataset"].str.lower().isin(wanted)].copy()

    selections: dict[str, dict[str, object]] = {}
    for dataset, ds_frame in frame.groupby("dataset"):
        ordered = ds_frame.sort_values("dual_minus_prior_only", ascending=False).reset_index(drop=True)
        gain_row = ordered.iloc[0]
        flat_row = ordered.iloc[(ordered["dual_minus_prior_only"].abs()).sort_values().index[0]]
        if str(gain_row["trait"]) == str(flat_row["trait"]) and len(ordered) > 1:
            flat_row = ordered.iloc[1]
        selections[str(dataset)] = {
            "gain_trait": {
                "trait": str(gain_row["trait"]),
                "trait_col": str(gain_row.get("trait_col", gain_row["trait"])),
                "dual_minus_prior_only": float(gain_row["dual_minus_prior_only"]),
                "best_baseline_model": str(gain_row["best_baseline_model"]),
            },
            "flat_trait": {
                "trait": str(flat_row["trait"]),
                "trait_col": str(flat_row.get("trait_col", flat_row["trait"])),
                "dual_minus_prior_only": float(flat_row["dual_minus_prior_only"]),
                "best_baseline_model": str(flat_row["best_baseline_model"]),
            },
        }

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selections, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(selections, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
