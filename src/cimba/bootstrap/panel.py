"""Joint bootstrap trajectory generators for panels of related series."""

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike

from ._common import (
    TraceGenerator,
    _as_series,
    _check_length,
    _stationary_indices,
)


def joint(panel: "Mapping[str, ArrayLike]", length: int, *,
          name: str, mean_block: float) -> "dict[str, TraceGenerator]":
    """Jointly resample several series with the stationary bootstrap,
    preserving their cross-correlation -- e.g. demands of related SKUs
    feeding separate trace fields. Returns one generator per panel key:

        gens = bootstrap.joint({"demand_a": hist_a, "demand_b": hist_b},
                               length=400, name="demand", mean_block=7)
        exp = model.experiment(**gens, replications=200, seed=42)

    All generators carry ``trace_rng_name = "joint:<name>"``, so
    experiment() hands them identical per-trial rngs and every series
    replays the same block choices. ``name`` keys that shared stream:
    keep it unique per joint resample, and rebuild any trial's rng with
    ``sim.trace_rng(trial_seed, "joint:<name>")``."""
    if not isinstance(panel, Mapping) or not panel:
        raise ValueError("joint: panel must be a non-empty mapping of "
                         "field name -> 1-D series")
    series = {key: _as_series(value, f"joint[{key}]")
              for key, value in panel.items()}
    lengths = {arr.size for arr in series.values()}
    if len(lengths) != 1:
        raise ValueError("joint: all series must have the same length")
    n = lengths.pop()
    _check_length(length, "joint")
    if mean_block < 1:
        raise ValueError("joint: mean_block must be >= 1")
    p = 1.0 / float(mean_block)
    tag = f"joint:{name}"

    def make(column: np.ndarray) -> TraceGenerator:
        def generate(rng: np.random.Generator) -> np.ndarray:
            return column[_stationary_indices(rng, n, length, p)]

        generate.trace_rng_name = tag
        return generate

    return {key: make(column) for key, column in series.items()}
