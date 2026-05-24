# This file is included by ../_cimba.pyx.

cdef double _PROBABILITY_SUM_TOLERANCE = 1.0e-3


cdef inline double _random_real_value(object value, str name) except *:
    cdef double result
    if PyFloat_CheckExact(value):
        result = PyFloat_AS_DOUBLE(value)
    elif PyLong_CheckExact(value):
        result = PyLong_AsDouble(value)
    else:
        try:
            result = float(value)
        except (TypeError, ValueError):
            raise TypeError(f"{name} must be a real number") from None
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


cdef inline double _random_positive_value(object value, str name) except *:
    cdef double result = _random_real_value(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


cdef inline double _random_probability_value(object value, str name, bint allow_zero) except *:
    cdef double result = _random_real_value(value, name)
    if result > 1.0 or result < 0.0 or (not allow_zero and result == 0.0):
        if allow_zero:
            raise ValueError(f"{name} must be in [0.0, 1.0]")
        raise ValueError(f"{name} must be in (0.0, 1.0]")
    return result


cdef inline unsigned _random_unsigned_value(object value, str name) except *:
    cdef uint64_t result = _u64_value(value, name, 1)
    if result > <uint64_t>UINT_MAX:
        raise OverflowError(f"{name} must fit in unsigned int")
    return <unsigned>result


cdef double *_random_double_array(
    object values,
    str name,
    bint require_positive,
    bint probabilities,
    unsigned *n_out,
) except NULL:
    cdef object items
    cdef Py_ssize_t n
    cdef Py_ssize_t i
    cdef double *array = NULL
    cdef double value
    try:
        items = list(values)
    except TypeError:
        raise TypeError(f"{name} must be a sequence") from None
    n = len(items)
    if n == 0:
        raise ValueError(f"{name} must not be empty")
    if n > <Py_ssize_t>UINT_MAX:
        raise OverflowError(f"{name} length must fit in unsigned int")
    array = <double *>PyMem_Malloc(<size_t>n * sizeof(double))
    if array == NULL:
        raise MemoryError()
    try:
        for i in range(n):
            value = _random_real_value(items[i], f"{name} entries")
            if require_positive and value <= 0.0:
                raise ValueError(f"{name} entries must be positive")
            if probabilities and value < 0.0:
                raise ValueError(f"{name} entries must be non-negative")
            array[i] = value
    except:
        PyMem_Free(array)
        raise
    n_out[0] = <unsigned>n
    return array


cdef void _validate_probability_sum(double *probabilities, unsigned n, str name) except *:
    cdef unsigned i
    cdef double total = 0.0
    for i in range(n):
        total += probabilities[i]
    if (
        total < 1.0 - _PROBABILITY_SUM_TOLERANCE
        or total > 1.0 + _PROBABILITY_SUM_TOLERANCE
    ):
        raise ValueError(f"{name} must sum to 1.0")


cdef double *_random_probability_array(object probabilities, str name, unsigned *n_out) except NULL:
    cdef double *array = _random_double_array(probabilities, name, False, True, n_out)
    try:
        _validate_probability_sum(array, n_out[0], name)
    except:
        PyMem_Free(array)
        raise
    return array


cdef class AliasSampler:
    """Reusable Vose alias sampler for non-uniform discrete probabilities."""

    cdef cmb_random_alias *_ptr
    cdef unsigned _n
    cdef public bint _closed

    def __cinit__(self):
        self._ptr = NULL
        self._n = 0
        self._closed = True

    def __init__(self, object probabilities):
        cdef unsigned n
        cdef double *array = _random_probability_array(probabilities, "probabilities", &n)
        try:
            self._ptr = cmb_random_alias_create(n, array)
        finally:
            PyMem_Free(array)
        if self._ptr == NULL:
            raise MemoryError()
        self._n = n
        self._closed = False

    def __dealloc__(self):
        if not self._closed:
            self.close()

    def __enter__(self):
        _raise_if_closed(self)
        return self

    def __exit__(self, object exc_type, object exc, object tb):
        self.close()
        return False

    def __len__(self):
        return <object>self._n

    def sample(self) -> int:
        _raise_if_closed(self)
        return <object>cmb_random_alias_sample(self._ptr)

    def close(self) -> None:
        if self._closed:
            return
        if self._ptr != NULL:
            cmb_random_alias_destroy(self._ptr)
            self._ptr = NULL
        self._closed = True

def hwseed() -> int:
    """Return a hardware-derived random seed."""
    return <object>cmb_random_hwseed()


def seed(object value=None) -> int:
    """Initialize the thread-local random generator and return the seed used."""
    cdef uint64_t seed_value = cmb_random_hwseed() if value is None else _seed_to_u64(value)
    cmb_random_initialize(seed_value)
    return <object>seed_value


def current_seed() -> int:
    """Return the seed currently used by Cimba in this thread."""
    return <object>cmb_random_curseed()


def random() -> float:
    return cmb_random()


def random_u64() -> int:
    return <object>cmb_random_sfc64()


def fmix64(object seed, object nonce) -> int:
    return <object>cmb_random_fmix64(_seed_to_u64(seed), _u64_value(nonce, "nonce", 0))


def uniform(double min, double max) -> float:
    return cmb_random_uniform(min, max)


def triangular(double min, double mode, double max) -> float:
    return cmb_random_triangular(min, mode, max)


def normal(double mu=0.0, double sigma=1.0) -> float:
    return cmb_random_normal(mu, sigma)


def lognormal(object m, object s) -> float:
    return cmb_random_lognormal(_random_real_value(m, "m"), _random_positive_value(s, "s"))


def logistic(object m, object s) -> float:
    return cmb_random_logistic(_random_real_value(m, "m"), _random_positive_value(s, "s"))


def cauchy(object mode, object scale) -> float:
    return cmb_random_cauchy(_random_real_value(mode, "mode"), _random_positive_value(scale, "scale"))


def exponential(double mean) -> float:
    return cmb_random_exponential(mean)


def erlang(object k, object mean) -> float:
    return cmb_random_erlang(_random_unsigned_value(k, "k"), _random_positive_value(mean, "mean"))


def hypoexponential(object means) -> float:
    cdef unsigned n
    cdef double *array = _random_double_array(means, "means", True, False, &n)
    try:
        return cmb_random_hypoexponential(n, array)
    finally:
        PyMem_Free(array)


def hyperexponential(object means, object probabilities) -> float:
    cdef unsigned means_n
    cdef unsigned probabilities_n
    cdef double *means_array = _random_double_array(means, "means", True, False, &means_n)
    cdef double *probabilities_array = NULL
    try:
        probabilities_array = _random_probability_array(
            probabilities,
            "probabilities",
            &probabilities_n,
        )
        if means_n != probabilities_n:
            raise ValueError("means and probabilities must have the same length")
        return cmb_random_hyperexponential(means_n, means_array, probabilities_array)
    finally:
        if probabilities_array != NULL:
            PyMem_Free(probabilities_array)
        PyMem_Free(means_array)


def gamma(double shape, double scale=1.0) -> float:
    return cmb_random_gamma(shape, scale)


def beta(double a, double b, double min=0.0, double max=1.0) -> float:
    return cmb_random_beta(a, b, min, max)


def pert(double min, double mode, double max) -> float:
    return cmb_random_PERT(min, mode, max)


def pert_mod(double min, double mode, double max, double lambda_) -> float:
    return cmb_random_PERT_mod(min, mode, max, lambda_)


def weibull(object shape, object scale) -> float:
    return cmb_random_weibull(_random_positive_value(shape, "shape"), _random_positive_value(scale, "scale"))


def pareto(object shape, object mode) -> float:
    return cmb_random_pareto(_random_positive_value(shape, "shape"), _random_positive_value(mode, "mode"))


def chi_squared(object k) -> float:
    return cmb_random_chisquared(_random_positive_value(k, "k"))


def f_dist(object a, object b) -> float:
    return cmb_random_F_dist(_random_positive_value(a, "a"), _random_positive_value(b, "b"))


def student_t(object v, object m=0.0, object s=1.0) -> float:
    return cmb_random_t_dist(
        _random_real_value(m, "m"),
        _random_positive_value(s, "s"),
        _random_positive_value(v, "v"),
    )


def rayleigh(object s) -> float:
    return cmb_random_rayleigh(_random_positive_value(s, "s"))


def dice(object min, object max) -> int:
    return cmb_random_dice(_i64_value(min, "min"), _i64_value(max, "max"))


def flip() -> bool:
    return True if cmb_random_flip() else False


def bernoulli(double p) -> bool:
    return True if cmb_random_bernoulli(p) else False


def geometric(object p) -> int:
    cdef double probability = _random_probability_value(p, "p", False)
    if probability == 1.0:
        return 1
    return <object>cmb_random_geometric(probability)


def binomial(object n, object p) -> int:
    return <object>cmb_random_binomial(
        _random_unsigned_value(n, "n"),
        _random_probability_value(p, "p", False),
    )


def negative_binomial(object m, object p) -> int:
    cdef unsigned successes = _random_unsigned_value(m, "m")
    cdef double probability = _random_probability_value(p, "p", False)
    if probability == 1.0:
        return 0
    return <object>cmb_random_negative_binomial(
        successes,
        probability,
    )


def pascal(object m, object p) -> int:
    cdef unsigned successes = _random_unsigned_value(m, "m")
    cdef double probability = _random_probability_value(p, "p", False)
    if probability == 1.0:
        return 0
    return <object>cmb_random_pascal(
        successes,
        probability,
    )


def poisson(object r) -> int:
    return <object>cmb_random_poisson(_random_positive_value(r, "r"))


def loaded_dice(object probabilities) -> int:
    cdef unsigned n
    cdef double *array = _random_probability_array(probabilities, "probabilities", &n)
    try:
        return <object>cmb_random_loaded_dice(n, array)
    finally:
        PyMem_Free(array)
