from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
try:
    from bed_reader import create_bed, open_bed
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight test envs
    create_bed = None
    open_bed = None


def _require_bed_reader() -> None:
    if create_bed is None or open_bed is None:
        raise ModuleNotFoundError(
            "bed_reader is required for PLINK IO. Install `bed_reader` in the active environment."
        )


def build_sample_ids(n_samples: int) -> list[str]:
    return [f"line_{index:04d}" for index in range(1, n_samples + 1)]


def read_phenotype_table(path: str | Path, sample_id_col: str = "sample_id") -> pd.DataFrame:
    frame = pd.read_csv(path, na_values=[".", "NA", "NaN", "nan", ""])
    if sample_id_col not in frame.columns:
        frame.insert(0, sample_id_col, build_sample_ids(len(frame)))
    return frame


def align_phenotype_to_sample_ids(
    phenotype: pd.DataFrame,
    sample_ids: list[str],
    sample_id_col: str = "sample_id",
) -> tuple[pd.DataFrame, list[int]]:
    if sample_id_col not in phenotype.columns:
        raise KeyError(f"{sample_id_col} not found in phenotype columns.")
    frame = phenotype.copy()
    frame[sample_id_col] = frame[sample_id_col].astype(str)
    frame = frame.drop_duplicates(subset=[sample_id_col], keep="first").set_index(sample_id_col)
    keep_ids = [sample_id for sample_id in sample_ids if sample_id in frame.index]
    keep_indices = [idx for idx, sample_id in enumerate(sample_ids) if sample_id in frame.index]
    aligned = frame.loc[keep_ids].reset_index()
    return aligned, keep_indices


def normalize_genotype_to_plink(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32).copy()
    finite = values[np.isfinite(values)]
    unique_values = set(np.unique(finite).tolist())
    if unique_values.issubset({-1.0, 1.0}):
        return values + 1.0
    if unique_values.issubset({-1.0, 0.0, 1.0}):
        return values + 1.0
    if np.nanmin(finite, initial=0.0) >= 0.0 and np.nanmax(finite, initial=2.0) <= 2.0:
        return np.clip(np.rint(values), 0.0, 2.0)
    return values


def _count_matrix_shape(genotype_csv: str | Path) -> tuple[int, int]:
    row_count = 0
    n_snps = None
    with Path(genotype_csv).open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if n_snps is None:
                n_snps = len(row)
            row_count += 1
    if n_snps is None:
        raise ValueError(f"Empty genotype CSV: {genotype_csv}")
    return row_count, n_snps


def convert_matrix_csv_to_plink(
    genotype_csv: str | Path,
    phenotype_csv: str | Path,
    plink_prefix: str | Path,
    sample_id_col: str = "sample_id",
) -> Path:
    _require_bed_reader()
    genotype_csv = Path(genotype_csv)
    phenotype_csv = Path(phenotype_csv)
    plink_prefix = Path(plink_prefix)
    plink_prefix.parent.mkdir(parents=True, exist_ok=True)

    phenotype = read_phenotype_table(phenotype_csv, sample_id_col=sample_id_col)
    n_samples, n_snps = _count_matrix_shape(genotype_csv)
    if len(phenotype) != n_samples:
        raise ValueError("Phenotype row count does not match genotype row count.")

    sample_ids = phenotype[sample_id_col].astype(str).tolist()
    snp_ids = [f"snp_{index:06d}" for index in range(1, n_snps + 1)]
    properties = {
        "fid": sample_ids,
        "iid": sample_ids,
        "father": ["0"] * n_samples,
        "mother": ["0"] * n_samples,
        "sex": [0] * n_samples,
        "pheno": [0] * n_samples,
        "chromosome": ["1"] * n_snps,
        "sid": snp_ids,
        "cm_position": [0.0] * n_snps,
        "bp_position": list(range(1, n_snps + 1)),
        "allele_1": ["A"] * n_snps,
        "allele_2": ["C"] * n_snps,
    }
    writer = create_bed(f"{plink_prefix}.bed", iid_count=n_samples, sid_count=n_snps, properties=properties, major="individual")
    try:
        with genotype_csv.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                cleaned = [np.nan if cell in {"", ".", "NA", "NaN", "nan"} else float(cell) for cell in row]
                writer.write(normalize_genotype_to_plink(np.asarray(cleaned, dtype=np.float32)))
    finally:
        writer.close()
    return plink_prefix


@dataclass
class PlinkMatrix:
    matrix: np.ndarray
    sample_ids: list[str]
    snp_ids: list[str]


def _resolve_bed_path(prefix: str | Path) -> Path:
    prefix = Path(prefix)
    return prefix if prefix.suffix == ".bed" else prefix.with_suffix(".bed")


def load_plink_matrix(prefix: str | Path, snp_indices: Iterable[int] | None = None) -> PlinkMatrix:
    _require_bed_reader()
    bed_path = _resolve_bed_path(prefix)
    bed = open_bed(bed_path)
    if snp_indices is None:
        matrix = bed.read(dtype="float32")
        snp_ids = bed.sid.tolist()
    else:
        snp_indices = list(snp_indices)
        matrix = bed.read(index=np.s_[:, snp_indices], dtype="float32")
        snp_ids = [bed.sid[index] for index in snp_indices]
    return PlinkMatrix(
        matrix=matrix,
        sample_ids=bed.iid.tolist(),
        snp_ids=snp_ids,
    )


def plink_num_snps(prefix: str | Path) -> int:
    _require_bed_reader()
    bed = open_bed(_resolve_bed_path(prefix))
    return int(bed.sid_count)


def write_plink_from_matrix(
    matrix: np.ndarray,
    plink_prefix: str | Path,
    sample_ids: list[str] | None = None,
    snp_ids: list[str] | None = None,
) -> Path:
    _require_bed_reader()
    values = normalize_genotype_to_plink(np.asarray(matrix, dtype=np.float32))
    if values.ndim != 2:
        raise ValueError("matrix must be 2D.")
    n_samples, n_snps = values.shape
    plink_prefix = Path(plink_prefix)
    plink_prefix.parent.mkdir(parents=True, exist_ok=True)

    if sample_ids is None:
        sample_ids = build_sample_ids(n_samples)
    if snp_ids is None:
        snp_ids = [f"snp_{index:06d}" for index in range(1, n_snps + 1)]
    if len(sample_ids) != n_samples:
        raise ValueError("sample_ids length does not match matrix row count.")
    if len(snp_ids) != n_snps:
        raise ValueError("snp_ids length does not match matrix column count.")

    properties = {
        "fid": sample_ids,
        "iid": sample_ids,
        "father": ["0"] * n_samples,
        "mother": ["0"] * n_samples,
        "sex": [0] * n_samples,
        "pheno": [0] * n_samples,
        "chromosome": ["1"] * n_snps,
        "sid": snp_ids,
        "cm_position": [0.0] * n_snps,
        "bp_position": list(range(1, n_snps + 1)),
        "allele_1": ["A"] * n_snps,
        "allele_2": ["C"] * n_snps,
    }
    writer = create_bed(
        f"{plink_prefix}.bed",
        iid_count=n_samples,
        sid_count=n_snps,
        properties=properties,
        major="individual",
    )
    try:
        for row in values:
            writer.write(row.astype(np.float32))
    finally:
        writer.close()
    return plink_prefix


def write_subsampled_plink(
    source_prefix: str | Path,
    target_prefix: str | Path,
    snp_indices: Iterable[int],
) -> Path:
    snp_indices = list(snp_indices)
    plink = load_plink_matrix(source_prefix, snp_indices=snp_indices)
    return write_plink_from_matrix(
        matrix=plink.matrix,
        plink_prefix=target_prefix,
        sample_ids=plink.sample_ids,
        snp_ids=plink.snp_ids,
    )


def impute_by_train_mean(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train, dtype=np.float32).copy()
    test = np.asarray(test, dtype=np.float32).copy()
    means = np.nanmean(train, axis=0)
    means = np.where(np.isnan(means), 0.0, means)
    train_nan = np.isnan(train)
    test_nan = np.isnan(test)
    if train_nan.any():
        train[train_nan] = np.take(means, np.where(train_nan)[1])
    if test_nan.any():
        test[test_nan] = np.take(means, np.where(test_nan)[1])
    return train, test
