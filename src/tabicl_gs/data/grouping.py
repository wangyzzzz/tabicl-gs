from __future__ import annotations

from math import ceil
from typing import Iterable

import numpy as np


def subsample_snp_indices(n_snps: int, max_snps: int, seed: int) -> list[int]:
    if max_snps <= 0:
        raise ValueError("max_snps must be positive.")
    if n_snps <= max_snps:
        return list(range(n_snps))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n_snps, size=max_snps, replace=False).tolist())


def _chunk_indices(
    marker_indices: list[int],
    group_size: int,
    pad_incomplete: bool = False,
    pad_value: int = -1,
) -> list[list[int]]:
    if group_size <= 0:
        raise ValueError("group_size must be positive.")
    remainder = len(marker_indices) % group_size
    if remainder != 0 and not pad_incomplete:
        raise ValueError("Number of SNPs must be divisible by group_size.")
    chunks = [marker_indices[start : start + group_size] for start in range(0, len(marker_indices), group_size)]
    if remainder != 0 and pad_incomplete:
        pad_length = group_size - remainder
        chunks[-1] = chunks[-1] + [pad_value] * pad_length
    return chunks


def make_random_groups(
    marker_indices: Iterable[int],
    group_size: int,
    seed: int,
    pad_incomplete: bool = False,
    pad_value: int = -1,
) -> list[list[int]]:
    shuffled = list(marker_indices)
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled)
    return _chunk_indices(shuffled, group_size, pad_incomplete=pad_incomplete, pad_value=pad_value)


def make_window_groups(
    marker_indices: Iterable[int],
    group_size: int,
    pad_incomplete: bool = False,
    pad_value: int = -1,
) -> list[list[int]]:
    return _chunk_indices(list(marker_indices), group_size, pad_incomplete=pad_incomplete, pad_value=pad_value)


def expected_num_blocks(n_snps: int, group_size: int) -> int:
    return int(ceil(n_snps / group_size))


def build_blocks(
    snp_indices: Iterable[int],
    strategy: str,
    group_size: int,
    seed: int,
    pad_incomplete: bool = False,
    pad_value: int = -1,
) -> list[list[int]]:
    strategy = strategy.lower()
    if strategy == "random":
        return make_random_groups(
            snp_indices,
            group_size=group_size,
            seed=seed,
            pad_incomplete=pad_incomplete,
            pad_value=pad_value,
        )
    if strategy == "window":
        return make_window_groups(
            snp_indices,
            group_size=group_size,
            pad_incomplete=pad_incomplete,
            pad_value=pad_value,
        )
    raise ValueError(f"Unsupported grouping strategy: {strategy}")
