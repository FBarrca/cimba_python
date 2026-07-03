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
* ``residual`` -- trending or seasonal data: fits the deterministic
  structure (polynomial trend and, given a ``period``, STL
  seasonality), resamples the residuals (i.i.d. or stationary-block),
  and adds them back onto the extrapolated structure.
* ``wild`` -- heteroskedastic residuals: keeps each residual at its own
  time position and multiplies it by a random weight, preserving
  variance that changes over time.
* ``sieve`` -- autoregressive dynamics: fits an AR(p) (order selected
  by AIC) and simulates forward with resampled innovations.
* ``intermittent`` -- zero-inflated series (spare parts, slow movers):
  Markov-chain demand occurrence plus resampled nonzero sizes.
* ``joint`` -- several correlated series (related SKUs): one stationary
  resample drives every series, preserving cross-correlation.

``residual``, ``wild``, and ``sieve`` use statsmodels for STL and AR
fitting. Their ``trend``/``period`` arguments also accept ``"auto"``;
``start`` shifts the structure window past the history and
``nonnegative=True`` clips at zero for demand data.

Block length trades off dependence preservation (longer) against
resample diversity (shorter); n**(1/3) is a common starting point.
The pure block factories assume stationarity.

Size ``length`` to cover the experiment's warmup + duration + cooldown:
a generator that exhausts its trace simply finishes while the trial
runs on.
"""

from collections.abc import Callable, Mapping

import numpy as np
from numpy.typing import ArrayLike
from statsmodels.regression.linear_model import yule_walker
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.seasonal import STL

__all__ = ["iid", "moving_block", "circular_block", "stationary",
           "residual", "wild", "sieve", "intermittent", "joint"]

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


# --- Residual / model-based methods ---------------------------------------

def _select_trend_degree(arr: np.ndarray) -> int:
    """Pick a polynomial trend degree from {0, 1, 2} by AICc."""
    n = arr.size
    t = np.arange(n, dtype=np.float64)
    best, best_aicc = 0, np.inf
    for d in (0, 1, 2):
        k = d + 2  # polynomial coefficients + residual variance
        if n - k - 1 <= 0:
            break
        resid = arr - np.polynomial.Polynomial.fit(t, arr, d)(t)
        sigma2 = max(float(np.mean(resid ** 2)), 1e-300)
        aicc = n * np.log(sigma2) + 2 * k + 2 * k * (k + 1) / (n - k - 1)
        if aicc < best_aicc:
            best, best_aicc = d, aicc
    return best


def _detect_period(arr: np.ndarray) -> "int | None":
    """Detect a seasonal period: dominant periodogram peak of the
    linearly detrended series, confirmed by autocorrelation at that lag.
    Returns None when no credible period exists."""
    n = arr.size
    if n < 8:
        return None
    t = np.arange(n, dtype=np.float64)
    x = arr - np.polynomial.Polynomial.fit(t, arr, 1)(t)
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    spectrum[0] = 0.0
    k = int(np.argmax(spectrum))
    if k == 0:
        return None
    period = int(round(n / k))
    if not 2 <= period <= n // 3:
        return None
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        return None
    acf = float(np.dot(x[:-period], x[period:])) / denom
    if acf < 2.0 / np.sqrt(n):
        return None
    return period


def _decompose(arr: np.ndarray, length: int, trend, period, robust: bool,
               name: str, start: int = 0) -> "tuple[np.ndarray, np.ndarray]":
    """Split a series into deterministic structure and residuals.

    Returns (structure evaluated on start..start+length-1, in-sample
    residuals). With a period, structure is an STL fit whose seasonal
    part tiles beyond the data and whose trend extrapolates linearly
    from its final two periods; without one, it is a least-squares
    polynomial of the given degree (None = mean), extrapolated by the
    polynomial."""
    n = arr.size
    if start < 0:
        raise ValueError(f"{name}: start must be >= 0")
    if isinstance(trend, str):
        if trend != "auto":
            raise ValueError(f"{name}: trend must be a degree, None, "
                             "or 'auto'")
        trend = _select_trend_degree(arr)
    if isinstance(period, str):
        if period != "auto":
            raise ValueError(f"{name}: period must be an int, None, "
                             "or 'auto'")
        period = _detect_period(arr)

    if period is not None:
        period = int(period)
        if not 2 <= period <= n // 2:
            raise ValueError(f"{name}: period must be in [2, {n // 2}] "
                             "(need at least two full cycles)")
        fit = STL(arr, period=period, robust=robust).fit()
        trend_c = np.asarray(fit.trend, dtype=np.float64)
        seasonal = np.asarray(fit.seasonal, dtype=np.float64)
        resid = np.asarray(fit.resid, dtype=np.float64)
        t_all = np.arange(start, start + length)
        structure = np.empty(length, dtype=np.float64)
        inside = t_all < n
        t_in = t_all[inside]
        structure[inside] = trend_c[t_in] + seasonal[t_in]
        if not inside.all():
            tail = np.arange(n - 2 * period, n, dtype=np.float64)
            line = np.polynomial.Polynomial.fit(tail, trend_c[-2 * period:],
                                                1)
            t_ext = t_all[~inside]
            phase = n - period + (t_ext - (n - period)) % period
            structure[~inside] = (line(t_ext.astype(np.float64))
                                  + seasonal[phase])
        return structure, resid

    degree = 0 if trend is None else int(trend)
    if not 0 <= degree < n:
        raise ValueError(f"{name}: trend degree must be in [0, {n - 1}]")
    t = np.arange(n, dtype=np.float64)
    poly = np.polynomial.Polynomial.fit(t, arr, degree)
    t_out = np.arange(start, start + length, dtype=np.float64)
    return poly(t_out), arr - poly(t)


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
