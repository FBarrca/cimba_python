"""Random draws for compiled model callbacks and their Numba helpers."""

from collections.abc import Iterable

from numba.extending import overload as _nb_overload

_MODEL_ONLY = (
    "cimba.random draws require a compiled model callback. "
    "Use model.experiment(seed=...) for reproducibility; "
    "use numpy.random.default_rng() outside models."
)


def uniform(min: float = 0.0, max: float = 1.0) -> float:
    raise RuntimeError(_MODEL_ONLY)


def triangular(min: float, mode: float, max: float) -> float:
    raise RuntimeError(_MODEL_ONLY)


def normal(mu: float = 0.0, sigma: float = 1.0) -> float:
    raise RuntimeError(_MODEL_ONLY)


def lognormal(m: float, s: float) -> float:
    raise RuntimeError(_MODEL_ONLY)


def logistic(m: float, s: float) -> float:
    raise RuntimeError(_MODEL_ONLY)


def cauchy(mode: float, scale: float) -> float:
    raise RuntimeError(_MODEL_ONLY)


def exponential(mean: float = 1.0) -> float:
    raise RuntimeError(_MODEL_ONLY)


def erlang(k: int, mean: float) -> float:
    raise RuntimeError(_MODEL_ONLY)


def hypoexponential(means: Iterable[float]) -> float:
    raise RuntimeError(_MODEL_ONLY)


def hyperexponential(means: Iterable[float], probabilities: Iterable[float]) -> float:
    raise RuntimeError(_MODEL_ONLY)


def gamma(shape: float, scale: float = 1.0) -> float:
    raise RuntimeError(_MODEL_ONLY)


def beta(a: float, b: float, min: float = 0.0, max: float = 1.0) -> float:
    raise RuntimeError(_MODEL_ONLY)


def pert(min: float, mode: float, max: float) -> float:
    raise RuntimeError(_MODEL_ONLY)


def pert_mod(min: float, mode: float, max: float, lambda_: float) -> float:
    raise RuntimeError(_MODEL_ONLY)


def weibull(shape: float, scale: float) -> float:
    raise RuntimeError(_MODEL_ONLY)


def pareto(shape: float, mode: float) -> float:
    raise RuntimeError(_MODEL_ONLY)


def chi_squared(k: float) -> float:
    raise RuntimeError(_MODEL_ONLY)


def f_dist(a: float, b: float) -> float:
    raise RuntimeError(_MODEL_ONLY)


def student_t(v: float, m: float = 0.0, s: float = 1.0) -> float:
    raise RuntimeError(_MODEL_ONLY)


def rayleigh(s: float) -> float:
    raise RuntimeError(_MODEL_ONLY)


def dice(min: int, max: int) -> int:
    raise RuntimeError(_MODEL_ONLY)


def bernoulli(p: float) -> bool:
    raise RuntimeError(_MODEL_ONLY)


def geometric(p: float) -> int:
    raise RuntimeError(_MODEL_ONLY)


def binomial(n: int, p: float) -> int:
    raise RuntimeError(_MODEL_ONLY)


def negative_binomial(m: int, p: float) -> int:
    raise RuntimeError(_MODEL_ONLY)


def poisson(r: float) -> int:
    raise RuntimeError(_MODEL_ONLY)


def categorical(probabilities: Iterable[float]) -> int:
    raise RuntimeError(_MODEL_ONLY)


def _compiled_namespace():
    from . import _compiled
    return _compiled


# Model callbacks and the @njit helpers they call share these implementations.
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
