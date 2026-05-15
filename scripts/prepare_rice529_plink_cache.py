from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabicl_gs.data.grouping import subsample_snp_indices
from tabicl_gs.data.plink import convert_matrix_csv_to_plink, plink_num_snps, read_phenotype_table, write_subsampled_plink


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare rice529 PLINK cache and 10000-SNP subset cache.")
    parser.add_argument("--genotype-csv", default="genome/rice529/rice529_gen.csv")
    parser.add_argument("--phenotype-csv", default="genome/rice529/rice529_phe.csv")
    parser.add_argument("--plink-prefix", default="genome/rice529/plink/rice529")
    parser.add_argument("--sample-id-col", default="sample_id")
    parser.add_argument("--max-snps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_prefix = Path(args.plink_prefix)
    raw_prefix.parent.mkdir(parents=True, exist_ok=True)
    if not raw_prefix.with_suffix(".bed").exists():
        convert_matrix_csv_to_plink(
            genotype_csv=args.genotype_csv,
            phenotype_csv=args.phenotype_csv,
            plink_prefix=raw_prefix,
            sample_id_col=args.sample_id_col,
        )

    total_snps = plink_num_snps(raw_prefix)
    selected_indices = subsample_snp_indices(int(total_snps), int(args.max_snps), int(args.seed))
    subset_prefix = raw_prefix.parent / f"rice529_max{int(args.max_snps)}_seed{int(args.seed)}"
    if not subset_prefix.with_suffix(".bed").exists():
        write_subsampled_plink(raw_prefix, subset_prefix, selected_indices)

    phenotype = read_phenotype_table(args.phenotype_csv, sample_id_col=args.sample_id_col)
    trait_cols = [str(col) for col in phenotype.columns if str(col) != str(args.sample_id_col)]
    summary = {
        "dataset": "rice529",
        "raw_plink_prefix": str(raw_prefix),
        "prepared_phenotype_csv": str(Path(args.phenotype_csv)),
        "subset_plink_prefix": str(subset_prefix),
        "sample_id_col": str(args.sample_id_col),
        "trait_cols": trait_cols,
        "trait_count": len(trait_cols),
        "raw_snp_count": int(total_snps),
        "subset_snp_count": min(int(total_snps), int(args.max_snps)),
        "seed": int(args.seed),
    }
    summary_path = raw_prefix.parent / "rice529_cache_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
