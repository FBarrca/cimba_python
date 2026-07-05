"""Public random draw API for Cimba."""

from numba.extending import overload as _nb_overload

from .._cimba import (
    AliasSampler,
    bernoulli,
    beta as _beta,
    binomial,
    cauchy,
    chi_squared,
    current_seed,
    dice,
    erlang,
    exponential as _exponential,
    f_dist,
    fmix64,
    gamma as _gamma,
    geometric,
    hwseed,
    hyperexponential,
    hypoexponential,
    loaded_dice as _categorical,
    logistic,
    lognormal,
    negative_binomial,
    normal as _normal,
    pareto,
    pert,
    pert_mod,
    poisson,
    random_u64,
    rayleigh,
    seed,
    student_t,
    triangular,
    uniform as _uniform,
    weibull,
)


def uniform(min: float = 0.0, max: float = 1.0) -> float:
    """Draw from a continuous uniform distribution on [min, max]."""

    return _uniform(min, max)


def exponential(mean: float = 1.0) -> float:
    """Draw from an exponential distribution with the given mean."""

    return _exponential(mean)


def gamma(shape: float, scale: float = 1.0) -> float:
    """Draw from a gamma distribution with shape and scale parameters."""

    return _gamma(shape, scale)


def normal(mu: float = 0.0, sigma: float = 1.0) -> float:
    """Draw from a normal distribution with mean mu and standard deviation sigma."""

    return _normal(mu, sigma)


def beta(
    a: float,
    b: float,
    min: float = 0.0,
    max: float = 1.0,
) -> float:
    """Draw from a beta distribution scaled to [min, max]."""

    return _beta(a, b, min, max)


def categorical(probabilities) -> int:
    """Draw one index from nonnegative probabilities that sum to 1.0."""

    return _categorical(probabilities)


def _compiled_namespace():
    from . import _compiled
    return _compiled


# Standalone @njit helpers do not pass through Cimba's model callback lowering,
# so register direct Numba implementations for the same public names.
@_nb_overload(uniform)
def _ol_uniform(min=0.0, max=1.0):
    compiled = _compiled_namespace()

    def impl(min=0.0, max=1.0):
        return compiled.uniform(min, max)
    return impl


@_nb_overload(exponential)
def _ol_exponential(mean=1.0):
    compiled = _compiled_namespace()

    def impl(mean=1.0):
        return compiled.exponential(mean)
    return impl


@_nb_overload(gamma)
def _ol_gamma(shape, scale=1.0):
    compiled = _compiled_namespace()

    def impl(shape, scale=1.0):
        return compiled.gamma(shape, scale)
    return impl


@_nb_overload(normal)
def _ol_normal(mu=0.0, sigma=1.0):
    compiled = _compiled_namespace()

    def impl(mu=0.0, sigma=1.0):
        return compiled.normal(mu, sigma)
    return impl


@_nb_overload(rayleigh)
def _ol_rayleigh(s):
    compiled = _compiled_namespace()

    def impl(s):
        return compiled.rayleigh(s)
    return impl


@_nb_overload(pert)
def _ol_pert(min, mode, max):
    compiled = _compiled_namespace()

    def impl(min, mode, max):
        return compiled.pert(min, mode, max)
    return impl


@_nb_overload(pert_mod)
def _ol_pert_mod(min, mode, max, lambda_):
    compiled = _compiled_namespace()

    def impl(min, mode, max, lambda_):
        return compiled.pert_mod(min, mode, max, lambda_)
    return impl


@_nb_overload(bernoulli)
def _ol_bernoulli(p):
    compiled = _compiled_namespace()

    def impl(p):
        return compiled.bernoulli(p)
    return impl


@_nb_overload(triangular)
def _ol_triangular(min, mode, max):
    compiled = _compiled_namespace()

    def impl(min, mode, max):
        return compiled.triangular(min, mode, max)
    return impl


@_nb_overload(weibull)
def _ol_weibull(shape, scale):
    compiled = _compiled_namespace()

    def impl(shape, scale):
        return compiled.weibull(shape, scale)
    return impl


@_nb_overload(lognormal)
def _ol_lognormal(m, s):
    compiled = _compiled_namespace()

    def impl(m, s):
        return compiled.lognormal(m, s)
    return impl


@_nb_overload(erlang)
def _ol_erlang(k, mean):
    compiled = _compiled_namespace()

    def impl(k, mean):
        return compiled.erlang(k, mean)
    return impl


@_nb_overload(beta)
def _ol_beta(a, b, min=0.0, max=1.0):
    compiled = _compiled_namespace()

    def impl(a, b, min=0.0, max=1.0):
        return compiled.beta(a, b, min, max)
    return impl


@_nb_overload(poisson)
def _ol_poisson(r):
    compiled = _compiled_namespace()

    def impl(r):
        return compiled.poisson(r)
    return impl


@_nb_overload(dice)
def _ol_dice(min, max):
    compiled = _compiled_namespace()

    def impl(min, max):
        return compiled.dice(min, max)
    return impl


@_nb_overload(logistic)
def _ol_logistic(m, s):
    compiled = _compiled_namespace()

    def impl(m, s):
        return compiled.logistic(m, s)
    return impl


@_nb_overload(cauchy)
def _ol_cauchy(mode, scale):
    compiled = _compiled_namespace()

    def impl(mode, scale):
        return compiled.cauchy(mode, scale)
    return impl


@_nb_overload(pareto)
def _ol_pareto(shape, mode):
    compiled = _compiled_namespace()

    def impl(shape, mode):
        return compiled.pareto(shape, mode)
    return impl


@_nb_overload(chi_squared)
def _ol_chi_squared(k):
    compiled = _compiled_namespace()

    def impl(k):
        return compiled.chi_squared(k)
    return impl


@_nb_overload(f_dist)
def _ol_f_dist(a, b):
    compiled = _compiled_namespace()

    def impl(a, b):
        return compiled.f_dist(a, b)
    return impl


@_nb_overload(student_t)
def _ol_student_t(v, m=0.0, s=1.0):
    compiled = _compiled_namespace()

    def impl(v, m=0.0, s=1.0):
        return compiled.student_t(v, m, s)
    return impl


@_nb_overload(geometric)
def _ol_geometric(p):
    compiled = _compiled_namespace()

    def impl(p):
        return compiled.geometric(p)
    return impl


@_nb_overload(binomial)
def _ol_binomial(n, p):
    compiled = _compiled_namespace()

    def impl(n, p):
        return compiled.binomial(n, p)
    return impl


@_nb_overload(negative_binomial)
def _ol_negative_binomial(m, p):
    compiled = _compiled_namespace()

    def impl(m, p):
        return compiled.negative_binomial(m, p)
    return impl


@_nb_overload(hypoexponential)
def _ol_hypoexponential(means):
    compiled = _compiled_namespace()

    def impl(means):
        return compiled.hypoexponential(means)
    return impl


@_nb_overload(hyperexponential)
def _ol_hyperexponential(means, probabilities):
    compiled = _compiled_namespace()

    def impl(means, probabilities):
        return compiled.hyperexponential(means, probabilities)
    return impl


@_nb_overload(categorical)
def _ol_categorical(probabilities):
    compiled = _compiled_namespace()

    def impl(probabilities):
        return compiled.categorical(probabilities)
    return impl


__all__ = [
    "AliasSampler",
    "bernoulli",
    "beta",
    "binomial",
    "categorical",
    "cauchy",
    "chi_squared",
    "current_seed",
    "dice",
    "erlang",
    "exponential",
    "f_dist",
    "fmix64",
    "gamma",
    "geometric",
    "hwseed",
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
    "random_u64",
    "rayleigh",
    "seed",
    "student_t",
    "triangular",
    "uniform",
    "weibull",
]
