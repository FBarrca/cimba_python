"""Fit and rank native distribution sources against observed samples."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from . import dist
from .sources import DistributionSource, InputError, reference


@dataclass(frozen=True)
class FitResult:
    source: DistributionSource
    aic: float
    ks: float
    log_likelihood: float


def fit(
    data: ArrayLike,
    candidates: Sequence[Callable[..., DistributionSource]] | None = None,
) -> tuple[FitResult, ...]:
    """Return supported maximum-likelihood fits ordered by AIC."""
    values, _ = reference(data, "fit")
    candidates = candidates or (dist.exponential, dist.gamma,
                                dist.lognormal, dist.normal, dist.weibull)
    results: list[FitResult] = []
    for candidate in candidates:
        if candidate is dist.exponential:
            if np.any(values < 0):
                continue
            params = stats.expon.fit(values, floc=0)
            source = dist.exponential(mean=params[1])
            distribution = stats.expon
            free = 1
        elif candidate is dist.gamma:
            if np.any(values <= 0):
                continue
            params = stats.gamma.fit(values, floc=0)
            source = dist.gamma(shape=params[0], scale=params[2])
            distribution = stats.gamma
            free = 2
        elif candidate is dist.lognormal:
            if np.any(values <= 0):
                continue
            params = stats.lognorm.fit(values, floc=0)
            source = dist.lognormal(mean=np.log(params[2]), sigma=params[0])
            distribution = stats.lognorm
            free = 2
        elif candidate is dist.normal:
            params = stats.norm.fit(values)
            source = dist.normal(mean=params[0], sd=max(params[1], 1e-12))
            distribution = stats.norm
            free = 2
        elif candidate is dist.weibull:
            if np.any(values <= 0):
                continue
            params = stats.weibull_min.fit(values, floc=0)
            source = dist.weibull(shape=params[0], scale=params[2])
            distribution = stats.weibull_min
            free = 2
        else:
            raise InputError(f"unsupported fit candidate: {candidate!r}")
        ll = float(np.sum(distribution.logpdf(values, *params)))
        ks = float(stats.kstest(values, distribution.cdf, args=params).statistic)
        results.append(FitResult(source, 2 * free - 2 * ll, ks, ll))
    if not results:
        raise InputError("no candidate distribution supports this data")
    return tuple(sorted(results, key=lambda x: x.aic))
