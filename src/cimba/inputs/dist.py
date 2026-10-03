"""Parametric sources drawn in the native trial on per-input streams."""

from __future__ import annotations

from collections.abc import Iterable
from math import isfinite

from .sources import DistributionSource, InputError


def _source(method: str, *, tag: str | None = None,
            **parameters: object) -> DistributionSource:
    for key, value in parameters.items():
        if isinstance(value, (int, float)) and not isfinite(float(value)):
            raise InputError(f"{method}: {key} must be finite")
    return DistributionSource(method, tuple(parameters.items()), tag=tag)


def exponential(*, mean: object, tag: str | None = None) -> DistributionSource:
    if isinstance(mean, (int, float)) and mean <= 0:
        raise InputError("exponential: mean must be positive")
    return _source("exponential", mean=mean, tag=tag)


def normal(*, mean: object = 0.0, sd: object = 1.0,
           tag: str | None = None) -> DistributionSource:
    if isinstance(sd, (int, float)) and sd <= 0:
        raise InputError("normal: sd must be positive")
    return _source("normal", mean=mean, sd=sd, tag=tag)


def gamma(*, shape: object, scale: object = 1.0, tag: str | None = None) -> DistributionSource:
    if isinstance(shape, (int, float)) and shape <= 0:
        raise InputError("gamma: shape must be positive")
    if isinstance(scale, (int, float)) and scale <= 0:
        raise InputError("gamma: scale must be positive")
    return _source("gamma", shape=shape, scale=scale, tag=tag)


def lognormal(*, mean: object = 0.0, sigma: object = 1.0,
              tag: str | None = None) -> DistributionSource:
    if isinstance(sigma, (int, float)) and sigma <= 0:
        raise InputError("lognormal: sigma must be positive")
    return _source("lognormal", mean=mean, sigma=sigma, tag=tag)


def weibull(*, shape: object, scale: object = 1.0, tag: str | None = None) -> DistributionSource:
    if isinstance(shape, (int, float)) and shape <= 0:
        raise InputError("weibull: shape must be positive")
    if isinstance(scale, (int, float)) and scale <= 0:
        raise InputError("weibull: scale must be positive")
    return _source("weibull", shape=shape, scale=scale, tag=tag)


def poisson(*, mean: object, tag: str | None = None) -> DistributionSource:
    if isinstance(mean, (int, float)) and mean < 0:
        raise InputError("poisson: mean must be nonnegative")
    return _source("poisson", mean=mean, tag=tag)


def triangular(*, low: object, mode: object, high: object,
               tag: str | None = None) -> DistributionSource:
    if (isinstance(low, (int, float)) and isinstance(mode, (int, float))
            and isinstance(high, (int, float))):
        if not low <= mode <= high or low == high:
            raise InputError("triangular: require low <= mode <= high and low < high")
    return _source("triangular", low=low, mode=mode, high=high, tag=tag)


def pert(*, low: object, mode: object, high: object, tag: str | None = None) -> DistributionSource:
    if (isinstance(low, (int, float)) and isinstance(mode, (int, float))
            and isinstance(high, (int, float))):
        if not low <= mode <= high or low == high:
            raise InputError("pert: require low <= mode <= high and low < high")
    return _source("pert", low=low, mode=mode, high=high, tag=tag)


def categorical(values: Iterable[int | float], probabilities: Iterable[int | float],
                *, tag: str | None = None) -> DistributionSource:
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
    return _source("categorical", values=choices, probabilities=weights, tag=tag)


def uniform(*, low: object = 0.0, high: object = 1.0,
            tag: str | None = None) -> DistributionSource:
    if isinstance(low, (int, float)) and isinstance(high, (int, float)) and low >= high:
        raise InputError("uniform: low must be less than high")
    return _source("uniform", low=low, high=high, tag=tag)


def logistic(*, mean: object = 0.0, scale: object = 1.0,
             tag: str | None = None) -> DistributionSource:
    if isinstance(scale, (int, float)) and scale <= 0:
        raise InputError("logistic: scale must be positive")
    return _source("logistic", mean=mean, scale=scale, tag=tag)


def cauchy(*, mode: object = 0.0, scale: object = 1.0,
           tag: str | None = None) -> DistributionSource:
    if isinstance(scale, (int, float)) and scale <= 0:
        raise InputError("cauchy: scale must be positive")
    return _source("cauchy", mode=mode, scale=scale, tag=tag)


def erlang(*, k: object, mean: object, tag: str | None = None) -> DistributionSource:
    if isinstance(k, (int, float)) and (k < 1 or int(k) != k):
        raise InputError("erlang: k must be a positive integer")
    if isinstance(mean, (int, float)) and mean <= 0:
        raise InputError("erlang: mean must be positive")
    return _source("erlang", k=k, mean=mean, tag=tag)


def beta(*, a: object, b: object, low: object = 0.0,
         high: object = 1.0, tag: str | None = None) -> DistributionSource:
    if any(isinstance(x, (int, float)) and x <= 0 for x in (a, b)):
        raise InputError("beta: a and b must be positive")
    if isinstance(low, (int, float)) and isinstance(high, (int, float)) and low >= high:
        raise InputError("beta: low must be less than high")
    return _source("beta", a=a, b=b, low=low, high=high, tag=tag)


def pert_mod(*, low: object, mode: object, high: object,
             weight: object, tag: str | None = None) -> DistributionSource:
    if (isinstance(low, (int, float)) and isinstance(mode, (int, float))
            and isinstance(high, (int, float))):
        if not low <= mode <= high or low == high:
            raise InputError("pert_mod: require low <= mode <= high and low < high")
    if isinstance(weight, (int, float)) and weight <= 0:
        raise InputError("pert_mod: weight must be positive")
    return _source("pert_mod", low=low, mode=mode, high=high, weight=weight, tag=tag)


def rayleigh(*, scale: object, tag: str | None = None) -> DistributionSource:
    if isinstance(scale, (int, float)) and scale <= 0:
        raise InputError("rayleigh: scale must be positive")
    return _source("rayleigh", scale=scale, tag=tag)


def bernoulli(*, p: object, tag: str | None = None) -> DistributionSource:
    if isinstance(p, (int, float)) and not 0 <= p <= 1:
        raise InputError("bernoulli: p must be in [0, 1]")
    return _source("bernoulli", p=p, tag=tag)


def geometric(*, p: object, tag: str | None = None) -> DistributionSource:
    if isinstance(p, (int, float)) and not 0 < p <= 1:
        raise InputError("geometric: p must be in (0, 1]")
    return _source("geometric", p=p, tag=tag)


def binomial(*, n: object, p: object, tag: str | None = None) -> DistributionSource:
    if isinstance(n, (int, float)) and (n < 0 or int(n) != n):
        raise InputError("binomial: n must be a nonnegative integer")
    if isinstance(p, (int, float)) and not 0 <= p <= 1:
        raise InputError("binomial: p must be in [0, 1]")
    return _source("binomial", n=n, p=p, tag=tag)


def negative_binomial(*, successes: object, p: object,
                      tag: str | None = None) -> DistributionSource:
    if isinstance(successes, (int, float)) and (
        successes < 1 or int(successes) != successes
    ):
        raise InputError("negative_binomial: successes must be a positive integer")
    if isinstance(p, (int, float)) and not 0 < p <= 1:
        raise InputError("negative_binomial: p must be in (0, 1]")
    return _source("negative_binomial", successes=successes, p=p, tag=tag)


def pareto(*, shape: object, mode: object = 1.0, tag: str | None = None) -> DistributionSource:
    if any(isinstance(x, (int, float)) and x <= 0 for x in (shape, mode)):
        raise InputError("pareto: shape and mode must be positive")
    return _source("pareto", shape=shape, mode=mode, tag=tag)


def chi_squared(*, df: object, tag: str | None = None) -> DistributionSource:
    if isinstance(df, (int, float)) and df <= 0:
        raise InputError("chi_squared: df must be positive")
    return _source("chi_squared", df=df, tag=tag)


def f_dist(*, df1: object, df2: object, tag: str | None = None) -> DistributionSource:
    if any(isinstance(x, (int, float)) and x <= 0 for x in (df1, df2)):
        raise InputError("f_dist: degrees of freedom must be positive")
    return _source("f_dist", df1=df1, df2=df2, tag=tag)


def student_t(*, df: object, mean: object = 0.0,
              scale: object = 1.0, tag: str | None = None) -> DistributionSource:
    if isinstance(df, (int, float)) and df <= 0:
        raise InputError("student_t: df must be positive")
    if isinstance(scale, (int, float)) and scale <= 0:
        raise InputError("student_t: scale must be positive")
    return _source("student_t", df=df, mean=mean, scale=scale, tag=tag)


def dice(*, low: object, high: object, tag: str | None = None) -> DistributionSource:
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        if int(low) != low or int(high) != high or low > high:
            raise InputError("dice: require integer low <= high")
    return _source("dice", low=low, high=high, tag=tag)


def hypoexponential(means: Iterable[int | float], *, tag: str | None = None) -> DistributionSource:
    values = tuple(means)
    if not values or any(not isinstance(x, (int, float)) or
                         not isfinite(float(x)) or x <= 0 for x in values):
        raise InputError("hypoexponential: means must be positive finite numbers")
    return _source("hypoexponential", means=values, tag=tag)


def hyperexponential(means: Iterable[int | float], probabilities: Iterable[int | float],
                     *, tag: str | None = None) -> DistributionSource:
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
    return _source("hyperexponential", means=values, probabilities=weights, tag=tag)
