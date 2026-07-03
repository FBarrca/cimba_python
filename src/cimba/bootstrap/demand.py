"""Demand-specific bootstrap trajectory generators."""

import numpy as np
from numpy.typing import ArrayLike

from ._common import TraceGenerator, _as_series, _check_length


def intermittent(data: ArrayLike, length: int, *,
                 jitter: bool = False) -> TraceGenerator:
    """Intermittent-demand bootstrap (Willemain et al.): for
    zero-inflated series such as spare parts. Demand occurrence is a
    two-state Markov chain fitted to the zero/nonzero pattern
    (preserving the clustering of demand periods), and each occurrence
    draws a size with replacement from the observed nonzero values.

    ``jitter=True`` perturbs each drawn size by ``z * sqrt(size)``
    (``z ~ N(0, 1)``, keeping the unjittered size when the result would
    be <= 0), so sizes not present in the history can occur."""
    arr = _as_series(data, "intermittent")
    _check_length(length, "intermittent")
    occurs = arr != 0.0
    sizes = arr[occurs]
    if sizes.size < 2 or sizes.size == arr.size:
        raise ValueError("intermittent: need at least 2 nonzero and 1 "
                         "zero observation; for regular demand use iid "
                         "or the block methods")
    # Two-state occurrence chain, add-one smoothed so unobserved
    # transitions stay possible
    state = occurs.astype(np.int64)
    from_zero = state[:-1] == 0
    n0 = int(from_zero.sum())
    n1 = state.size - 1 - n0
    p01 = (float(state[1:][from_zero].sum()) + 1.0) / (n0 + 2.0)
    p11 = (float(state[1:][~from_zero].sum()) + 1.0) / (n1 + 2.0)
    p_start = sizes.size / arr.size

    def generate(rng: np.random.Generator) -> np.ndarray:
        u = rng.random(length)
        demand = np.zeros(length, dtype=bool)
        s = 1 if rng.random() < p_start else 0
        for i in range(length):
            s = 1 if u[i] < (p11 if s else p01) else 0
            demand[i] = s
        out = np.zeros(length, dtype=np.float64)
        k = int(demand.sum())
        if k:
            drawn = sizes[rng.integers(0, sizes.size, size=k)]
            if jitter:
                moved = drawn + rng.standard_normal(k) * np.sqrt(
                    np.abs(drawn))
                drawn = np.where(moved > 0.0, moved, drawn)
            out[demand] = drawn
        return out

    return generate
