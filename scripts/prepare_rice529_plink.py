from __future__ import annotations

import argparse
from pathlib import Path

from tabicl_gs.data.plink import convert_matrix_csv_to_plink


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert rice529 genotype CSV to PLINK BED/BIM/FAM.")
    parser.add_argument("--genotype-csv", default="genome/rice529/rice529_gen.csv")
    parser.add_argument("--phenotype-csv", default="genome/rice529/rice529_phe.csv")
    parser.add_argument("--plink-prefix", default="genome/rice529/plink/rice529")
    parser.add_argument("--sample-id-col", default="sample_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix = convert_matrix_csv_to_plink(
        genotype_csv=args.genotype_csv,
        phenotype_csv=args.phenotype_csv,
        plink_prefix=args.plink_prefix,
        sample_id_col=args.sample_id_col,
    )
    print(f"PLINK files written to: {Path(prefix).parent}")


if __name__ == "__main__":
    main()
