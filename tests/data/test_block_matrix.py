import numpy as np

from tabicl_gs.data.block_matrix import extract_block_matrix


def test_extract_block_matrix_keeps_valid_columns_and_zero_pads_missing_columns():
    matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    block = extract_block_matrix(matrix, [2, -1, 0], pad_marker=-1, pad_value=0.0)
    assert block.tolist() == [[3.0, 0.0, 1.0], [6.0, 0.0, 4.0]]
