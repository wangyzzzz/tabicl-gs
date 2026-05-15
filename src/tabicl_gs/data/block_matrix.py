from __future__ import annotations

from typing import Iterable

import numpy as np


def extract_block_matrix(matrix: np.ndarray, block_columns: Iterable[int], pad_marker: int = -1, pad_value: float = 0.0) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    columns = list(block_columns)
    output = np.full((matrix.shape[0], len(columns)), pad_value, dtype=np.float32)
    valid_positions = [position for position, column in enumerate(columns) if column != pad_marker]
    valid_columns = [column for column in columns if column != pad_marker]
    if valid_columns:
        output[:, valid_positions] = matrix[:, valid_columns]
    return output
