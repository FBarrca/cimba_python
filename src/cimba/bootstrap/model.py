"""Residual, wild, and sieve bootstrap trajectory generators."""

import numpy as np
from numpy.typing import ArrayLike
from statsmodels.regression.linear_model import yule_walker
from statsmodels.tsa.ar_model import ar_select_order

from ._common import (
    TraceGenerator,
    _as_series,
    _check_length,
    _stationary_indices,
)
from ._decompose import _decompose


def residual(data: ArrayLike, length: int, *,
             trend=1, period=None, mean_block: "float | None" = None,
             start: int = 0, nonnegative: bool = False,
             robust: bool = False) -> TraceGenerator:
    """Residual bootstrap: fit the deterministic structure of the series
    (trend and, with a period, STL seasonality), resample the residuals,
    and add them back onto the (extrapolated) structure.

    ``trend`` is a polynomial degree (None = mean only, default linear)
    and ``period`` a seasonal period in samples (None = no seasonality);
    both also accept ``"auto"`` (AICc degree selection, periodogram peak
    detection). Pin them when they are known -- detection is
    data-dependent. ``mean_block=None`` resamples
    residuals i.i.d.; give a mean block length to resample them with the
    stationary bootstrap when dependence remains after decomposition.

    ``start`` shifts the structure window: ``start=len(data)`` simulates
    the horizon after the observed history. ``nonnegative=True`` clips
    the trajectory at zero -- right for demand data, at the cost of a
    small upward mean bias where the structure is near zero."""
    arr = _as_series(data, "residual")
    _check_length(length, "residual")
    structure, resid = _decompose(arr, length, trend, period, robust,
                                  "residual", start)
    if mean_block is None:
        def draw(rng: np.random.Generator) -> np.ndarray:
            return resid[rng.integers(0, resid.size, size=length)]
    else:
        if mean_block < 1:
            raise ValueError("residual: mean_block must be >= 1")
        p = 1.0 / float(mean_block)

        def draw(rng: np.random.Generator) -> np.ndarray:
            return resid[_stationary_indices(rng, resid.size, length, p)]

    def generate(rng: np.random.Generator) -> np.ndarray:
        out = structure + draw(rng)
        return np.maximum(out, 0.0, out=out) if nonnegative else out

    return generate


def wild(data: ArrayLike, *, length: "int | None" = None,
         trend=1, period=None, weights: str = "rademacher",
         start: int = 0, nonnegative: bool = False,
         robust: bool = False) -> TraceGenerator:
    """Wild bootstrap for heteroskedastic residuals: each residual stays
    at its own time position -- preserving variance that changes over
    time -- and is multiplied by a zero-mean unit-variance random weight
    (``"rademacher"``, ``"mammen"``, or ``"normal"``).

    ``length`` defaults to the series length; longer outputs (and
    ``start`` offsets) tile the residual positions cyclically under the
    extrapolated structure, a convenience for covering warmup + duration
    + cooldown. ``nonnegative=True`` clips the trajectory at zero."""
    arr = _as_series(data, "wild")
    length = arr.size if length is None else length
    _check_length(length, "wild")
    structure, resid = _decompose(arr, length, trend, period, robust, "wild",
                                  start)
    placed = resid[(start + np.arange(length)) % arr.size]

    if weights == "rademacher":
        def draw(rng: np.random.Generator) -> np.ndarray:
            return rng.integers(0, 2, size=length) * 2.0 - 1.0
    elif weights == "mammen":
        s5 = np.sqrt(5.0)
        lo, hi = (1.0 - s5) / 2.0, (1.0 + s5) / 2.0
        p_lo = (s5 + 1.0) / (2.0 * s5)

        def draw(rng: np.random.Generator) -> np.ndarray:
            return np.where(rng.random(length) < p_lo, lo, hi)
    elif weights == "normal":
        def draw(rng: np.random.Generator) -> np.ndarray:
            return rng.standard_normal(length)
    else:
        raise ValueError("wild: weights must be 'rademacher', 'mammen', "
                         "or 'normal'")

    def generate(rng: np.random.Generator) -> np.ndarray:
        out = structure + placed * draw(rng)
        return np.maximum(out, 0.0, out=out) if nonnegative else out

    return generate


def sieve(data: ArrayLike, length: int, *,
          order: "int | None" = None, trend=None, period=None,
          start: int = 0, nonnegative: bool = False,
          robust: bool = False) -> TraceGenerator:
    """Sieve bootstrap: remove the deterministic structure (default:
    mean only -- the AR is supposed to capture the dynamics), fit an
    AR(p), and simulate the series forward with i.i.d.-resampled
    innovations, adding the structure back.

    ``order=None`` selects p by AIC (statsmodels ``ar_select_order``);
    the coefficients come from a Yule-Walker fit, so the simulated AR is
    always stationary. Innovations are the centered one-step residuals.
    ``trend``/``period`` accept the same values as ``residual`` for
    removing structure before the fit; ``start`` and ``nonnegative``
    behave as in ``residual``."""
    arr = _as_series(data, "sieve")
    _check_length(length, "sieve")
    n = arr.size
    if n < 8:
        raise ValueError("sieve: need at least 8 observations")
    structure, resid = _decompose(arr, length, trend, period, robust,
                                  "sieve", start)

    max_order = max(1, min(int(10 * np.log10(n)), n // 4))
    if order is None:
        sel = ar_select_order(resid, maxlag=max_order, ic="aic", trend="n")
        p = max(sel.ar_lags) if sel.ar_lags else 0
    else:
        p = int(order)
        if not 0 <= p <= n // 2:
            raise ValueError(f"sieve: order must be in [0, {n // 2}]")

    if p == 0:
        innov = resid - resid.mean()

        def generate(rng: np.random.Generator) -> np.ndarray:
            out = structure + innov[rng.integers(0, innov.size, size=length)]
            return np.maximum(out, 0.0, out=out) if nonnegative else out
        return generate

    phi, _sigma = yule_walker(resid, order=p, method="mle")
    phi = np.asarray(phi, dtype=np.float64)
    # One-step-ahead residuals of the fitted AR as the innovation pool
    lagged = np.column_stack([resid[p - j - 1:n - j - 1] for j in range(p)])
    innov = resid[p:] - lagged @ phi
    innov = innov - innov.mean()
    burn = max(50, 10 * p)

    def generate(rng: np.random.Generator) -> np.ndarray:
        e = innov[rng.integers(0, innov.size, size=burn + length)]
        x = np.zeros(p + burn + length, dtype=np.float64)
        for i in range(burn + length):
            x[p + i] = x[i:i + p][::-1] @ phi + e[i]
        out = structure + x[p + burn:]
        return np.maximum(out, 0.0, out=out) if nonnegative else out

    return generate
