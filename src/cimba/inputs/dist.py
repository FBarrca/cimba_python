"""Parametric sources drawn in the native trial on per-input streams."""

from __future__ import annotations

from collections.abc import Iterable
from math import isfinite

from .sources import DistributionSource, InputError


def _source(method: str, **parameters: object) -> DistributionSource:
    for key, value in parameters.items():
        if isinstance(value, (int, float)) and not isfinite(float(value)):
            raise InputError(f"{method}: {key} must be finite")
    return DistributionSource(method, tuple(parameters.items()))


def exponential(*, mean: object) -> DistributionSource:
    if isinstance(mean, (int, float)) and mean <= 0:
        raise InputError("exponential: mean must be positive")
    return _source("exponential", mean=mean)


def normal(*, mean: object = 0.0, sd: object = 1.0) -> DistributionSource:
    if isinstance(sd, (int, float)) and sd <= 0:
        raise InputError("normal: sd must be positive")
    return _source("normal", mean=mean, sd=sd)


def gamma(*, shape: object, scale: object = 1.0) -> DistributionSource:
    if isinstance(shape, (int, float)) and shape <= 0:
        raise InputError("gamma: shape must be positive")
    if isinstance(scale, (int, float)) and scale <= 0:
        raise InputError("gamma: scale must be positive")
    return _source("gamma", shape=shape, scale=scale)


def lognormal(*, mean: object = 0.0, sigma: object = 1.0) -> DistributionSource:
    if isinstance(sigma, (int, float)) and sigma <= 0:
        raise InputError("lognormal: sigma must be positive")
    return _source("lognormal", mean=mean, sigma=sigma)


def weibull(*, shape: object, scale: object = 1.0) -> DistributionSource:
    if isinstance(shape, (int, float)) and shape <= 0:
        raise InputError("weibull: shape must be positive")
    if isinstance(scale, (int, float)) and scale <= 0:
        raise InputError("weibull: scale must be positive")
    return _source("weibull", shape=shape, scale=scale)


def poisson(*, mean: object) -> DistributionSource:
    if isinstance(mean, (int, float)) and mean < 0:
        raise InputError("poisson: mean must be nonnegative")
    return _source("poisson", mean=mean)


def triangular(*, low: object, mode: object, high: object) -> DistributionSource:
    if (isinstance(low, (int, float)) and isinstance(mode, (int, float))
            and isinstance(high, (int, float))):
        if not low <= mode <= high or low == high:
            raise InputError("triangular: require low <= mode <= high and low < high")
    return _source("triangular", low=low, mode=mode, high=high)


def pert(*, low: object, mode: object, high: object) -> DistributionSource:
    if (isinstance(low, (int, float)) and isinstance(mode, (int, float))
            and isinstance(high, (int, float))):
        if not low <= mode <= high or low == high:
            raise InputError("pert: require low <= mode <= high and low < high")
    return _source("pert", low=low, mode=mode, high=high)


def categorical(values: Iterable[int | float], probabilities: Iterable[int | float]) -> DistributionSource:
    choices = tuple(values)
    weights = tuple(probabilities)
    if not choices or len(choices) != len(weights):
        raise InputError("categorical: values and probabilities must have equal nonzero length")
    if any(not isinstance(x, (int, float)) or not isfinite(float(x)) for x in choices):
        raise InputError("categorical: values must be finite numbers")
    if any(not isinstance(x, (int, float)) or not isfinite(float(x)) or x < 0 for x in weights):
        raise InputError("categorical: probabilities must be finite and nonnegative")
    if abs(sum(weights) - 1.0) > 1e-9:
        raise InputError("categorical: probabilities must sum to one")
    return _source("categorical", values=choices, probabilities=weights)


def uniform(*, low: object = 0.0, high: object = 1.0) -> DistributionSource:
    if isinstance(low, (int, float)) and isinstance(high, (int, float)) and low >= high:
        raise InputError("uniform: low must be less than high")
    return _source("uniform", low=low, high=high)


def logistic(*, mean: object = 0.0, scale: object = 1.0) -> DistributionSource:
    if isinstance(scale, (int, float)) and scale <= 0:
        raise InputError("logistic: scale must be positive")
    return _source("logistic", mean=mean, scale=scale)


def cauchy(*, mode: object = 0.0, scale: object = 1.0) -> DistributionSource:
    if isinstance(scale, (int, float)) and scale <= 0:
        raise InputError("cauchy: scale must be positive")
    return _source("cauchy", mode=mode, scale=scale)


def erlang(*, k: object, mean: object) -> DistributionSource:
    if isinstance(k, (int, float)) and (k < 1 or int(k) != k):
        raise InputError("erlang: k must be a positive integer")
    if isinstance(mean, (int, float)) and mean <= 0:
        raise InputError("erlang: mean must be positive")
    return _source("erlang", k=k, mean=mean)


def beta(*, a: object, b: object, low: object = 0.0,
         high: object = 1.0) -> DistributionSource:
    if any(isinstance(x, (int, float)) and x <= 0 for x in (a, b)):
        raise InputError("beta: a and b must be positive")
    if isinstance(low, (int, float)) and isinstance(high, (int, float)) and low >= high:
        raise InputError("beta: low must be less than high")
    return _source("beta", a=a, b=b, low=low, high=high)


def pert_mod(*, low: object, mode: object, high: object,
             weight: object) -> DistributionSource:
    if (isinstance(low, (int, float)) and isinstance(mode, (int, float))
            and isinstance(high, (int, float))):
        if not low <= mode <= high or low == high:
            raise InputError("pert_mod: require low <= mode <= high and low < high")
    if isinstance(weight, (int, float)) and weight <= 0:
        raise InputError("pert_mod: weight must be positive")
    return _source("pert_mod", low=low, mode=mode, high=high, weight=weight)


def rayleigh(*, scale: object) -> DistributionSource:
    if isinstance(scale, (int, float)) and scale <= 0:
        raise InputError("rayleigh: scale must be positive")
    return _source("rayleigh", scale=scale)


def bernoulli(*, p: object) -> DistributionSource:
    if isinstance(p, (int, float)) and not 0 <= p <= 1:
        raise InputError("bernoulli: p must be in [0, 1]")
    return _source("bernoulli", p=p)


def geometric(*, p: object) -> DistributionSource:
    if isinstance(p, (int, float)) and not 0 < p <= 1:
        raise InputError("geometric: p must be in (0, 1]")
    return _source("geometric", p=p)


def binomial(*, n: object, p: object) -> DistributionSource:
    if isinstance(n, (int, float)) and (n < 0 or int(n) != n):
        raise InputError("binomial: n must be a nonnegative integer")
    if isinstance(p, (int, float)) and not 0 <= p <= 1:
        raise InputError("binomial: p must be in [0, 1]")
    return _source("binomial", n=n, p=p)


def negative_binomial(*, successes: object, p: object) -> DistributionSource:
    if isinstance(successes, (int, float)) and (
        successes < 1 or int(successes) != successes
    ):
        raise InputError("negative_binomial: successes must be a positive integer")
    if isinstance(p, (int, float)) and not 0 < p <= 1:
        raise InputError("negative_binomial: p must be in (0, 1]")
    return _source("negative_binomial", successes=successes, p=p)


def pareto(*, shape: object, mode: object = 1.0) -> DistributionSource:
    if any(isinstance(x, (int, float)) and x <= 0 for x in (shape, mode)):
        raise InputError("pareto: shape and mode must be positive")
    return _source("pareto", shape=shape, mode=mode)


def chi_squared(*, df: object) -> DistributionSource:
    if isinstance(df, (int, float)) and df <= 0:
        raise InputError("chi_squared: df must be positive")
    return _source("chi_squared", df=df)


def f_dist(*, df1: object, df2: object) -> DistributionSource:
    if any(isinstance(x, (int, float)) and x <= 0 for x in (df1, df2)):
        raise InputError("f_dist: degrees of freedom must be positive")
    return _source("f_dist", df1=df1, df2=df2)


def student_t(*, df: object, mean: object = 0.0,
              scale: object = 1.0) -> DistributionSource:
    if isinstance(df, (int, float)) and df <= 0:
        raise InputError("student_t: df must be positive")
    if isinstance(scale, (int, float)) and scale <= 0:
        raise InputError("student_t: scale must be positive")
    return _source("student_t", df=df, mean=mean, scale=scale)


def dice(*, low: object, high: object) -> DistributionSource:
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        if int(low) != low or int(high) != high or low > high:
            raise InputError("dice: require integer low <= high")
    return _source("dice", low=low, high=high)


def hypoexponential(means: Iterable[int | float]) -> DistributionSource:
    values = tuple(means)
    if not values or any(not isinstance(x, (int, float)) or
                         not isfinite(float(x)) or x <= 0 for x in values):
        raise InputError("hypoexponential: means must be positive finite numbers")
    return _source("hypoexponential", means=values)


def hyperexponential(means: Iterable[int | float], probabilities: Iterable[int | float]) -> DistributionSource:
    values = tuple(means)
    weights = tuple(probabilities)
    if not values or len(values) != len(weights):
        raise InputError("hyperexponential: means and probabilities must match")
    if any(not isinstance(x, (int, float)) or not isfinite(float(x)) or x <= 0
           for x in values):
        raise InputError("hyperexponential: means must be positive finite numbers")
    if any(not isinstance(x, (int, float)) or not isfinite(float(x)) or x < 0
           for x in weights) or abs(sum(weights) - 1.0) > 1e-9:
        raise InputError("hyperexponential: probabilities must sum to one")
    return _source("hyperexponential", means=values, probabilities=weights)
