"""Bootstrap trajectory generators for data-driven traces.

Each factory takes an observed 1-D series and returns a callable
``f(rng)`` that produces one resampled trajectory of ``length`` float64
values -- ready to pass as a ``sim.Trace`` field value to
``Model.experiment()``, which invokes it once per trial with that
trial's ``sim.trace_rng()`` generator, so a single experiment ``seed``
reproduces every resample::

    from cimba import bootstrap

    demand = bootstrap.stationary(history, length=horizon, mean_block=7)
    exp = model.experiment(demand=demand, replications=200, seed=42)

Choosing a method:

* ``iid`` -- observations are independent (service times, order sizes).
  Destroys autocorrelation, so it is wrong for serially dependent data.
* ``moving_block`` / ``circular_block`` -- stationary dependent series;
  fixed-length contiguous blocks preserve within-block dependence.
  The circular variant wraps around the end of the series so edge
  observations are not underweighted.
* ``stationary`` -- like the block methods but with random geometric
  block lengths (Politis & Romano), so the resampled series is itself
  stationary rather than having seams every fixed ``block`` steps.
  A good default for autocorrelated data such as demand histories.

Block length trades off dependence preservation (longer) against
resample diversity (shorter); n**(1/3) is a common starting point.
Trending or seasonal data should be decomposed first and the residuals
bootstrapped -- these factories assume stationarity.

Size ``length`` to cover the experiment's warmup + duration + cooldown:
a generator that exhausts its trace simply finishes while the trial
runs on.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["iid", "moving_block", "circular_block", "stationary"]

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
        idx = np.empty(length, dtype=np.intp)
        pos = 0
        while pos < length:
            start = int(rng.integers(0, arr.size))
            run = min(int(rng.geometric(p)), length - pos)
            idx[pos:pos + run] = (start + np.arange(run)) % arr.size
            pos += run
        return arr[idx]

    return generate
