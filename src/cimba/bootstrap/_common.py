"""Shared helpers for bootstrap trajectory generators."""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

#: A per-trial trajectory generator, as accepted by Model.experiment()
#: for sim.Trace fields.
TraceGenerator = Callable[[np.random.Generator], np.ndarray]


def _as_series(data: ArrayLike, name: str) -> np.ndarray:
    arr = np.ascontiguousarray(data, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name}: data must be a non-empty 1-D series")
    return arr


def _check_length(length: int, name: str) -> None:
    if length < 1:
        raise ValueError(f"{name}: length must be >= 1")


def _check_block(block: int, n: int, name: str) -> None:
    if not 1 <= block <= n:
        raise ValueError(f"{name}: block must be in [1, {n}] "
                         "(the series length)")


def _stationary_indices(rng: np.random.Generator, n: int, length: int,
                        p: float) -> np.ndarray:
    """Index sequence of the stationary bootstrap: geometric-length runs
    from uniform starts, wrapping circularly over a series of size n."""
    idx = np.empty(length, dtype=np.intp)
    pos = 0
    while pos < length:
        start = int(rng.integers(0, n))
        run = min(int(rng.geometric(p)), length - pos)
        idx[pos:pos + run] = (start + np.arange(run)) % n
        pos += run
    return idx
