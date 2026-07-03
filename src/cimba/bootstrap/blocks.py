"""Independent and block bootstrap trajectory generators."""

import numpy as np
from numpy.typing import ArrayLike

from ._common import (
    TraceGenerator,
    _as_series,
    _check_block,
    _check_length,
    _stationary_indices,
)


def iid(data: ArrayLike, length: int) -> TraceGenerator:
    """Ordinary (Efron) bootstrap: resample single observations with
    replacement. For serially independent data only."""
    arr = _as_series(data, "iid")
    _check_length(length, "iid")

    def generate(rng: np.random.Generator) -> np.ndarray:
        return arr[rng.integers(0, arr.size, size=length)]

    return generate


def moving_block(data: ArrayLike, length: int, block: int) -> TraceGenerator:
    """Moving block bootstrap (Kunsch): concatenate fixed-length blocks
    drawn uniformly from all overlapping windows of the series."""
    arr = _as_series(data, "moving_block")
    _check_length(length, "moving_block")
    _check_block(block, arr.size, "moving_block")
    n_blocks = -(-length // block)
    offsets = np.arange(block)

    def generate(rng: np.random.Generator) -> np.ndarray:
        starts = rng.integers(0, arr.size - block + 1, size=n_blocks)
        return arr[(starts[:, None] + offsets).ravel()[:length]]

    return generate


def circular_block(data: ArrayLike, length: int,
                   block: int) -> TraceGenerator:
    """Circular block bootstrap (Politis & Romano): as moving_block, but
    blocks wrap around the end of the series, so every observation is
    equally likely to appear."""
    arr = _as_series(data, "circular_block")
    _check_length(length, "circular_block")
    _check_block(block, arr.size, "circular_block")
    n_blocks = -(-length // block)
    offsets = np.arange(block)

    def generate(rng: np.random.Generator) -> np.ndarray:
        starts = rng.integers(0, arr.size, size=n_blocks)
        idx = (starts[:, None] + offsets).ravel()[:length] % arr.size
        return arr[idx]

    return generate


def stationary(data: ArrayLike, length: int,
               mean_block: float) -> TraceGenerator:
    """Stationary bootstrap (Politis & Romano): blocks start at uniform
    positions, wrap circularly, and have geometrically distributed
    lengths with the given mean, so the resampled series is stationary."""
    arr = _as_series(data, "stationary")
    _check_length(length, "stationary")
    if mean_block < 1:
        raise ValueError("stationary: mean_block must be >= 1")
    p = 1.0 / float(mean_block)

    def generate(rng: np.random.Generator) -> np.ndarray:
        return arr[_stationary_indices(rng, arr.size, length, p)]

    return generate
