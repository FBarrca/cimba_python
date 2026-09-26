"""Fitted time-series sources, generated outside simulation trials."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray
from statsmodels.regression.linear_model import yule_walker
from statsmodels.tsa.ar_model import ar_select_order

from .bootstrap import stationary_indices
from .decompose import _decompose
from .sources import GeneratedRows, InputError, reference, row_source


def residual(data: ArrayLike, *, trend=1, period=None,
             mean_block: float | None = None, start: int = 0,
             nonnegative: bool = False, robust: bool = False) -> GeneratedRows:
    values, _ = reference(data, "residual")
    if mean_block is not None and (not np.isfinite(mean_block) or mean_block < 1):
        raise InputError("mean_block must be finite and >= 1")

    @lru_cache(maxsize=8)
    def structure(length: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return _decompose(values, length, trend, period, robust, "residual", start)

    def draw(rng: np.random.Generator, length: int) -> NDArray[np.float64]:
        base, errors = structure(length)
        if mean_block is None:
            idx = rng.integers(0, len(errors), size=length)
        else:
            idx = stationary_indices(rng, len(errors), length, mean_block)
        out = base + errors[idx]
        return np.maximum(out, 0.0) if nonnegative else out

    return row_source("fitted.residual", values, draw,
                      parameters={"trend": trend, "period": period,
                                  "mean_block": mean_block, "start": start,
                                  "nonnegative": nonnegative, "robust": robust})


def wild(data: ArrayLike, *, trend=1, period=None,
         weights: str = "rademacher", start: int = 0,
         nonnegative: bool = False, robust: bool = False) -> GeneratedRows:
    values, _ = reference(data, "wild")
    if weights not in {"rademacher", "mammen", "normal"}:
        raise InputError("weights must be rademacher, mammen or normal")
    if start < 0 or start >= len(values):
        raise InputError("wild start must index the reference series")
    max_length = len(values) - start

    @lru_cache(maxsize=8)
    def structure(length: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return _decompose(values, length, trend, period, robust, "wild", start)

    def draw(rng: np.random.Generator, length: int) -> NDArray[np.float64]:
        base, errors = structure(length)
        if weights == "rademacher":
            w = rng.integers(0, 2, size=length) * 2.0 - 1.0
        elif weights == "mammen":
            s5 = np.sqrt(5.0)
            p_lo = (s5 + 1.0) / (2.0 * s5)
            w = np.where(rng.random(length) < p_lo,
                         (1.0 - s5) / 2.0, (1.0 + s5) / 2.0)
        else:
            w = rng.standard_normal(length)
        out = base + errors[start:start + length] * w
        return np.maximum(out, 0.0) if nonnegative else out

    return row_source("fitted.wild", values, draw,
                      parameters={"trend": trend, "period": period,
                                  "weights": weights, "start": start,
                                  "nonnegative": nonnegative, "robust": robust},
                      max_length=max_length, on_exhausted="fail")


def sieve(data: ArrayLike, *, order: int | None = None, trend=None,
          period=None, start: int = 0, nonnegative: bool = False,
          robust: bool = False) -> GeneratedRows:
    values, _ = reference(data, "sieve")
    n = len(values)
    if n < 8:
        raise InputError("sieve needs at least eight observations")
    _, errors = _decompose(values, n, trend, period, robust, "sieve", 0)
    max_order = max(1, min(int(10 * np.log10(n)), n // 4))
    if order is None:
        selected = ar_select_order(errors, maxlag=max_order, ic="aic", trend="n")
        p = max(selected.ar_lags) if selected.ar_lags else 0
    else:
        p = int(order)
        if not 0 <= p <= n // 2:
            raise InputError(f"sieve order must be in [0, {n // 2}]")
    if p:
        phi = np.asarray(yule_walker(errors, order=p, method="mle")[0],
                         dtype=np.float64)
        lagged = np.column_stack([errors[p - j - 1:n - j - 1]
                                  for j in range(p)])
        innovations = errors[p:] - lagged @ phi
    else:
        phi = np.empty(0, dtype=np.float64)
        innovations = errors
    innovations = innovations - innovations.mean()
    burn = max(50, 10 * p)

    @lru_cache(maxsize=8)
    def structure(length: int) -> NDArray[np.float64]:
        return _decompose(values, length, trend, period, robust,
                          "sieve", start)[0]

    def draw(rng: np.random.Generator, length: int) -> NDArray[np.float64]:
        noise = innovations[rng.integers(0, len(innovations),
                                         size=burn + length)]
        state = np.zeros(p + burn + length, dtype=np.float64)
        for i in range(burn + length):
            state[p + i] = noise[i]
            if p:
                state[p + i] += state[i:i + p][::-1] @ phi
        out = structure(length) + state[p + burn:]
        return np.maximum(out, 0.0) if nonnegative else out

    return row_source("fitted.sieve", values, draw,
                      parameters={"order": p, "trend": trend,
                                  "period": period, "start": start,
                                  "nonnegative": nonnegative,
                                  "robust": robust})


from .bootstrap import intermittent  # same fitted occurrence model, public here
