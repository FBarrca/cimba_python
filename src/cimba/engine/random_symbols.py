"""Native random calls used by compiled model code.

The standalone runtime owns symbol registration.  No extension module or
legacy binding is needed to make these functions visible to Numba.
"""

from numba import types

from .symbols import register


def _extern(name, result, *arguments):
    register(name)
    return types.ExternalFunction(name, result(*arguments))


f64 = types.float64
u64 = types.uint64
i64 = types.int64

random01 = _extern("cpy_random01", f64)
random_uniform = _extern("cpy_random_uniform", f64, f64, f64)
random_exponential = _extern("cpy_random_exponential", f64, f64)
random_gamma = _extern("cpy_random_gamma", f64, f64, f64)
random_normal = _extern("cpy_random_normal", f64, f64, f64)
random_rayleigh = _extern("cpy_random_rayleigh", f64, f64)
random_pert = _extern("cpy_random_PERT", f64, f64, f64, f64)
random_pert_mod = _extern("cpy_random_PERT_mod", f64, f64, f64, f64, f64)
random_bernoulli = _extern("cpy_random_bernoulli", u64, f64)
random_triangular = _extern("cpy_random_triangular", f64, f64, f64, f64)
random_weibull = _extern("cpy_random_weibull", f64, f64, f64)
random_lognormal = _extern("cpy_random_lognormal", f64, f64, f64)
random_erlang = _extern("cpy_random_erlang", f64, u64, f64)
random_beta = _extern("cpy_random_beta", f64, f64, f64, f64, f64)
random_poisson = _extern("cpy_random_poisson", u64, f64)
random_dice = _extern("cpy_random_dice", i64, i64, i64)
random_logistic = _extern("cpy_random_logistic", f64, f64, f64)
random_cauchy = _extern("cpy_random_cauchy", f64, f64, f64)
random_pareto = _extern("cpy_random_pareto", f64, f64, f64)
random_chisquared = _extern("cpy_random_chisquared", f64, f64)
random_f_dist = _extern("cpy_random_F_dist", f64, f64, f64)
random_t = _extern("cpy_random_t_dist", f64, f64, f64, f64)
random_geometric = _extern("cpy_random_geometric", u64, f64)
random_binomial = _extern("cpy_random_binomial", u64, u64, f64)
random_negative_binomial = _extern("cpy_random_negative_binomial", u64, u64, f64)
