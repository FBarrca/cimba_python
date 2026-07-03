"""Deterministic structure fitting for model-based bootstraps."""

import numpy as np
from statsmodels.tsa.seasonal import STL


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
