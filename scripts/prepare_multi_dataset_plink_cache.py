from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tabicl_gs.data.grouping import subsample_snp_indices
from tabicl_gs.data.plink import (
    convert_matrix_csv_to_plink,
    plink_num_snps,
    write_subsampled_plink,
)


DATASETS = {
    "Cotton1245": {
        "phenotype_csv": "genome/Cotton1245/Cotton_all.txt",
        "phenotype_sep": "\t",
        "sample_id_col": "Taxa",
        "raw_plink_prefix": "genome/Cotton1245/Cotton_all",
    },
    "Soybean951": {
        "phenotype_csv": "genome/Soybean951/Soybean_all.txt",
        "phenotype_sep": "\t",
        "sample_id_col": "Taxa",
        "raw_plink_prefix": "genome/Soybean951/Soybean_1500K",
    },
    "pig3534": {
        "phenotype_csv": "genome/pig3534/phenotypes.txt",
        "phenotype_sep": ",",
        "sample_id_col": "ID",
        "genotype_csv": "genome/pig3534/genotypes.txt",
        "genotype_sep": ",",
        "genotype_has_header": True,
        "raw_plink_prefix": "genome/pig3534/plink/pig3534_raw",
    },
    "wheat406": {
        "phenotype_csv": "genome/wheat406/wheat406_phe.csv",
        "phenotype_sep": ",",
        "sample_id_col": "sample_id",
        "genotype_csv": "genome/wheat406/wheat406_gen.csv",
        "genotype_sep": ",",
        "genotype_has_header": False,
        "raw_plink_prefix": "genome/wheat406/plink/wheat406_raw",
    },
}


def _read_header(path: str | Path, sep: str) -> list[str]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=sep)
        return next(reader)


def _write_phenotype_csv(src_path: str | Path, dst_path: str | Path, sep: str, sample_id_col: str) -> Path:
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with src_path.open("r", encoding="utf-8", newline="") as src, dst_path.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.reader(src, delimiter=sep)
        writer = csv.writer(dst)
        rows = list(reader)
        if not rows:
            raise ValueError(f"Empty phenotype file: {src_path}")
        header = list(rows[0])
        if sample_id_col not in header:
            header = [sample_id_col] + header
            writer.writerow(header)
            for index, row in enumerate(rows[1:], start=1):
                writer.writerow([f"line_{index:04d}"] + list(row))
        else:
            writer.writerow(header)
            for row in rows[1:]:
                writer.writerow(row)
    return dst_path


def _write_genotype_matrix_csv(
    src_path: str | Path,
    dst_path: str | Path,
    sep: str,
    has_header: bool,
) -> Path:
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with src_path.open("r", encoding="utf-8", newline="") as src, dst_path.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.reader(src, delimiter=sep)
        writer = csv.writer(dst)
        for row_id, row in enumerate(reader):
            if has_header and row_id == 0:
                continue
            if not row:
                continue
            writer.writerow(row[1:] if has_header else row)
    return dst_path


def _trait_columns(phenotype_csv: str | Path, sep: str, sample_id_col: str) -> list[str]:
    header = _read_header(phenotype_csv, sep)
    return [col for col in header if col != sample_id_col]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PLINK cache and 10000-SNP subset cache for multiple datasets.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), nargs="+", default=sorted(DATASETS))
    parser.add_argument("--max-snps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []
    for dataset_name in args.dataset:
        spec = DATASETS[dataset_name]
        phenotype_csv = spec["phenotype_csv"]
        phenotype_sep = spec["phenotype_sep"]
        sample_id_col = spec["sample_id_col"]
        raw_plink_prefix = Path(spec["raw_plink_prefix"])
        raw_bed = raw_plink_prefix.with_suffix(".bed")

        prepared_pheno_csv = raw_plink_prefix.parent / "prepared_phenotype.csv"
        _write_phenotype_csv(phenotype_csv, prepared_pheno_csv, sep=phenotype_sep, sample_id_col=sample_id_col)

        if not raw_bed.exists():
            genotype_csv = spec.get("genotype_csv")
            if genotype_csv is None:
                raise FileNotFoundError(f"Missing raw PLINK for {dataset_name}: {raw_bed}")
            prepared_geno_csv = raw_plink_prefix.parent / "prepared_genotype.csv"
            _write_genotype_matrix_csv(
                genotype_csv,
                prepared_geno_csv,
                sep=spec.get("genotype_sep", ","),
                has_header=bool(spec.get("genotype_has_header", False)),
            )
            convert_matrix_csv_to_plink(
                genotype_csv=prepared_geno_csv,
                phenotype_csv=prepared_pheno_csv,
                plink_prefix=raw_plink_prefix,
                sample_id_col=sample_id_col,
            )

        total_snps = plink_num_snps(raw_plink_prefix)
        selected_indices = subsample_snp_indices(total_snps, int(args.max_snps), int(args.seed))
        subset_prefix = raw_plink_prefix.parent / f"{dataset_name.lower()}_max{args.max_snps}_seed{args.seed}"
        subset_bed = subset_prefix.with_suffix(".bed")
        if not subset_bed.exists():
            write_subsampled_plink(raw_plink_prefix, subset_prefix, selected_indices)

        trait_cols = _trait_columns(phenotype_csv, phenotype_sep, sample_id_col)
        summary = {
            "dataset": dataset_name,
            "raw_plink_prefix": str(raw_plink_prefix),
            "prepared_phenotype_csv": str(prepared_pheno_csv),
            "subset_plink_prefix": str(subset_prefix),
            "sample_id_col": sample_id_col,
            "trait_cols": trait_cols,
            "trait_count": len(trait_cols),
            "raw_snp_count": int(total_snps),
            "subset_snp_count": min(int(total_snps), int(args.max_snps)),
            "seed": int(args.seed),
        }
        summary_path = raw_plink_prefix.parent / f"{dataset_name.lower()}_cache_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries.append(summary)

    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
