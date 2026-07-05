"""Numba-compatible helpers for ``cimba.random`` calls."""

from numba import njit

from .. import _bindings as _b

uniform = _b.random_uniform
exponential = _b.random_exponential
gamma = _b.random_gamma
normal = _b.random_normal
rayleigh = _b.random_rayleigh
pert = _b.random_pert
pert_mod = _b.random_pert_mod
bernoulli = _b.random_bernoulli
triangular = _b.random_triangular
weibull = _b.random_weibull
lognormal = _b.random_lognormal
erlang = _b.random_erlang
beta = _b.random_beta
poisson = _b.random_poisson
dice = _b.random_dice
logistic = _b.random_logistic
cauchy = _b.random_cauchy
pareto = _b.random_pareto
chi_squared = _b.random_chisquared
f_dist = _b.random_f_dist
geometric = _b.random_geometric
binomial = _b.random_binomial
negative_binomial = _b.random_negative_binomial


@njit
def student_t(v, m, s):
    """Location-scale Student's t draw using public argument order."""

    return _b.random_t(m, s, v)


@njit
def hypoexponential(means):
    """Hypoexponential draw from a non-empty sequence of means."""

    if len(means) == 0:
        raise ValueError("hypoexponential() expects at least one mean")
    x = 0.0
    for mean in means:
        x += exponential(mean)
    return x


@njit
def categorical(probabilities):
    """Return an index sampled from probabilities that sum to 1."""

    if len(probabilities) == 0:
        raise ValueError("categorical() expects at least one probability")

    total = 0.0
    for probability in probabilities:
        if probability < 0.0:
            raise ValueError("categorical() probabilities must be non-negative")
        total += probability

    tolerance = 1.0e-3
    if total < 1.0 - tolerance or total > 1.0 + tolerance:
        raise ValueError("categorical() probabilities must sum to 1.0")

    target = _b.random01() * total
    cumulative = 0.0
    last = 0
    for i, probability in enumerate(probabilities):
        cumulative += probability
        last = i
        if target < cumulative:
            return i
    return last


@njit
def hyperexponential(means, probabilities):
    """Hyperexponential draw from matching mean and probability sequences."""

    if len(means) != len(probabilities):
        raise ValueError("hyperexponential() means and probabilities must match")
    return exponential(means[categorical(probabilities)])


__all__ = [
    "bernoulli",
    "beta",
    "binomial",
    "categorical",
    "cauchy",
    "chi_squared",
    "dice",
    "erlang",
    "exponential",
    "f_dist",
    "gamma",
    "geometric",
    "hyperexponential",
    "hypoexponential",
    "logistic",
    "lognormal",
    "negative_binomial",
    "normal",
    "pareto",
    "pert",
    "pert_mod",
    "poisson",
    "rayleigh",
    "student_t",
    "triangular",
    "uniform",
    "weibull",
]
