from pathlib import Path

import numpy as np

from tabicl_gs.data.plink import (
    align_phenotype_to_sample_ids,
    build_sample_ids,
    normalize_genotype_to_plink,
    read_phenotype_table,
)


def test_normalize_genotype_to_plink_maps_minus_one_and_one():
    matrix = np.array([[-1, 1], [1, -1]], dtype=np.float32)
    normalized = normalize_genotype_to_plink(matrix)
    assert normalized.tolist() == [[0.0, 2.0], [2.0, 0.0]]


def test_build_sample_ids_creates_stable_ids():
    sample_ids = build_sample_ids(3)
    assert sample_ids == ["line_0001", "line_0002", "line_0003"]


def test_read_phenotype_table_injects_generated_sample_ids(tmp_path: Path):
    csv_path = tmp_path / "phe.csv"
    csv_path.write_text("trait\n1.2\n3.4\n", encoding="utf-8")
    frame = read_phenotype_table(csv_path, sample_id_col="sample_id")
    assert list(frame.columns) == ["sample_id", "trait"]
    assert frame["sample_id"].tolist() == ["line_0001", "line_0002"]


def test_read_phenotype_table_treats_dot_as_nan(tmp_path: Path):
    csv_path = tmp_path / "phe.csv"
    csv_path.write_text("sample_id,trait\ns1,.\ns2,3.4\n", encoding="utf-8")
    frame = read_phenotype_table(csv_path, sample_id_col="sample_id")
    assert np.isnan(frame.loc[0, "trait"])
    assert float(frame.loc[1, "trait"]) == 3.4


def test_align_phenotype_to_sample_ids_uses_intersection_and_plink_order():
    import pandas as pd

    phenotype = pd.DataFrame(
        {
            "sample_id": ["s3", "s1", "s4"],
            "trait": [3.0, 1.0, 4.0],
        }
    )
    aligned, keep_indices = align_phenotype_to_sample_ids(
        phenotype,
        ["s1", "s2", "s3", "s4"],
        sample_id_col="sample_id",
    )
    assert keep_indices == [0, 2, 3]
    assert aligned["sample_id"].tolist() == ["s1", "s3", "s4"]
    assert aligned["trait"].tolist() == [1.0, 3.0, 4.0]
