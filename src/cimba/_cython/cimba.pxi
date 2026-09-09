# This file is included by ../_cimba_native.pyx.

import operator
import threading

cimport cython
from cpython.bool cimport PyBool_Check
from cpython.float cimport PyFloat_AS_DOUBLE, PyFloat_CheckExact
from cpython.long cimport (
    PY_LONG_LONG,
    PyLong_AsDouble,
    PyLong_AsLongLong,
    PyLong_AsUnsignedLongLong,
    PyLong_CheckExact,
    uPY_LONG_LONG,
)
from cpython.mem cimport PyMem_Free, PyMem_Malloc
from cpython.pycapsule cimport (
    PyCapsule_CheckExact,
    PyCapsule_GetPointer,
    PyCapsule_IsValid,
    PyCapsule_New,
)
from libc.limits cimport UINT_MAX
from libc.math cimport isfinite
from libc.stddef cimport size_t
from libc.stdint cimport (
    int64_t,
    uint32_t,
    uint64_t,
)


cdef extern from "stdbool.h":
    ctypedef bint bool


cdef extern from "cimba.h":
    cdef struct cmb_random_alias:
        pass

    ctypedef void cimba_trial_func(void *trial_struct) noexcept nogil
    ctypedef void *cimba_thread_init_func(uint64_t tid, void *usrarg) noexcept nogil
    ctypedef void cimba_thread_exit_func(void *thrctx) noexcept nogil

    const char *cimba_version()
    uint64_t cimba_run(
        void *your_experiment_array,
        uint64_t num_trials,
        size_t trial_struct_size,
        cimba_trial_func *your_trial_func,
    ) noexcept nogil
    void cimba_thread_hooks_set(
        cimba_thread_init_func *initfunc,
        void *usrarg,
        cimba_thread_exit_func *exitfunc,
    ) noexcept nogil

    void cmb_random_initialize(uint64_t seed)
    uint64_t cmb_random_hwseed()
    uint64_t cmb_random_curseed()
    uint64_t cmb_random_sfc64()
    uint64_t cmb_random_fmix64(uint64_t seed, uint64_t nonce)
    double cmb_random_uniform(double min, double max)
    double cmb_random_triangular(double min, double mode, double max)
    double cmb_random_normal(double mu, double sigma)
    double cmb_random_lognormal(double m, double s)
    double cmb_random_logistic(double m, double s)
    double cmb_random_cauchy(double mode, double scale)
    double cmb_random_exponential(double mean)
    double cmb_random_erlang(unsigned k, double mean)
    double cmb_random_hypoexponential(unsigned n, const double *means)
    double cmb_random_hyperexponential(unsigned n, const double *means, const double *probabilities)
    double cmb_random_gamma(double shape, double scale)
    double cmb_random_beta(double a, double b, double min, double max)
    double cmb_random_PERT(double min, double mode, double max)
    double cmb_random_PERT_mod(double min, double mode, double max, double lambd)
    double cmb_random_weibull(double shape, double scale)
    double cmb_random_pareto(double shape, double mode)
    double cmb_random_chisquared(double k)
    double cmb_random_F_dist(double a, double b)
    double cmb_random_t_dist(double m, double s, double v)
    double cmb_random_rayleigh(double s)
    long cmb_random_dice(long min, long max)
    bool cmb_random_bernoulli(double p)
    unsigned cmb_random_geometric(double p)
    unsigned cmb_random_binomial(unsigned n, double p)
    unsigned cmb_random_negative_binomial(unsigned m, double p)
    unsigned cmb_random_poisson(double r)
    uint64_t cmb_random_discrete_nonuniform(uint64_t n, const double *probabilities)
    cmb_random_alias *cmb_random_alias_create(unsigned n, const double *probabilities)
    unsigned cmb_random_alias_sample(const cmb_random_alias *alias)
    void cmb_random_alias_destroy(cmb_random_alias *alias)


cdef object _UINT64_MAX_OBJ = (1 << 64) - 1
cdef object _INT64_MIN_OBJ = -(1 << 63)
cdef object _INT64_MAX_OBJ = (1 << 63) - 1

UNLIMITED = _UINT64_MAX_OBJ

SUCCESS = 0
PREEMPTED = -1
INTERRUPTED = -2
STOPPED = -3
CANCELLED = -4
TIMEOUT = -5

PROCESS_CREATED = 0
PROCESS_RUNNING = 1
PROCESS_FINISHED = 2

LOGGER_FATAL = 0x80000000
LOGGER_ERROR = 0x40000000
LOGGER_WARNING = 0x20000000
LOGGER_INFO = 0x10000000

cdef bytes _TRIAL_FUNC_CAPSULE_NAME = b"cimba.trial_func"
cdef bytes _THREAD_INIT_CAPSULE_NAME = b"cimba.thread_init_func"
cdef bytes _THREAD_EXIT_CAPSULE_NAME = b"cimba.thread_exit_func"
cdef bytes _USER_CONTEXT_CAPSULE_NAME = b"cimba.user_context"


cdef void _raise_if_closed(object obj):
    if obj._closed:
        raise RuntimeError(f"{obj.__class__.__name__} is closed")


cdef inline object _index_value(object value, str name):
    if PyBool_Check(value):
        raise TypeError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None


cdef inline uint64_t _u64_value(object value, str name, uint64_t min_value) except *:
    cdef uPY_LONG_LONG exact_value
    cdef object ivalue = _index_value(value, name)
    if PyLong_CheckExact(ivalue):
        if ivalue < <object>min_value:
            raise ValueError(f"{name} must be at least {min_value}")
        exact_value = PyLong_AsUnsignedLongLong(ivalue)
        return <uint64_t>exact_value
    if ivalue < <object>min_value:
        raise ValueError(f"{name} must be at least {min_value}")
    if ivalue > _UINT64_MAX_OBJ:
        raise OverflowError(f"{name} must fit in uint64")
    return <uint64_t>ivalue


cdef inline int64_t _i64_value(object value, str name) except *:
    cdef PY_LONG_LONG exact_value
    cdef object ivalue = _index_value(value, name)
    if PyLong_CheckExact(ivalue):
        exact_value = PyLong_AsLongLong(ivalue)
        return <int64_t>exact_value
    if ivalue < _INT64_MIN_OBJ or ivalue > _INT64_MAX_OBJ:
        raise OverflowError(f"{name} must fit in int64")
    return <int64_t>ivalue


cdef inline uint64_t _seed_to_u64(object seed) except *:
    return _u64_value(seed, "seed", 0)


def native_version() -> str:
    """Return the version of the underlying Cimba C library."""
    cdef const char *v = cimba_version()
    return v.decode("utf-8")


cdef void *_capsule_pointer(object capsule, const char *name, str arg_name) except *:
    cdef str expected = (<bytes>name).decode("ascii")
    if callable(capsule):
        raise TypeError(f"{arg_name} must be a native PyCapsule, not a Python callable")
    if not PyCapsule_CheckExact(capsule):
        raise TypeError(f"{arg_name} must be a PyCapsule named {expected}")
    if not PyCapsule_IsValid(capsule, name):
        raise TypeError(f"{arg_name} must be a PyCapsule named {expected}")
    return PyCapsule_GetPointer(capsule, name)


def run_native_experiment(object experiment_buffer, object trial_struct_size, object trial_func_capsule) -> None:
    """Run a native Cimba experiment using a writable C-contiguous trial buffer."""
    cdef uint64_t struct_size = _u64_value(trial_struct_size, "trial_struct_size", 1)
    cdef cimba_trial_func *trial_func = <cimba_trial_func *>_capsule_pointer(
        trial_func_capsule,
        _TRIAL_FUNC_CAPSULE_NAME,
        "trial_func_capsule",
    )
    cdef object mv
    cdef object byte_mv
    cdef unsigned char[::1] view
    cdef uint64_t nbytes
    cdef uint64_t num_trials

    try:
        mv = memoryview(experiment_buffer)
    except TypeError:
        raise TypeError("experiment_buffer must support the buffer protocol") from None
    if mv.readonly:
        raise TypeError("experiment_buffer must be writable")
    if not mv.c_contiguous:
        raise TypeError("experiment_buffer must be C-contiguous")
    nbytes = <uint64_t>mv.nbytes
    if nbytes == 0:
        raise ValueError("experiment_buffer must not be empty")
    if nbytes % struct_size != 0:
        raise ValueError("experiment_buffer byte length must be an exact multiple of trial_struct_size")
    num_trials = nbytes // struct_size
    byte_mv = mv.cast("B")
    view = byte_mv
    with cython.boundscheck(False):
        with nogil:
            cimba_run(<void *>&view[0], num_trials, <size_t>struct_size, trial_func)


def set_native_thread_hooks(object init_capsule=None, object user_arg_capsule=None, object exit_capsule=None) -> None:
    """Set native Cimba pthread hooks from fixed-name PyCapsules."""
    cdef cimba_thread_init_func *initfunc = NULL
    cdef cimba_thread_exit_func *exitfunc = NULL
    cdef void *usrarg = NULL
    if init_capsule is not None:
        initfunc = <cimba_thread_init_func *>_capsule_pointer(
            init_capsule,
            _THREAD_INIT_CAPSULE_NAME,
            "init_capsule",
        )
    if user_arg_capsule is not None:
        usrarg = _capsule_pointer(
            user_arg_capsule,
            _USER_CONTEXT_CAPSULE_NAME,
            "user_arg_capsule",
        )
    if exit_capsule is not None:
        exitfunc = <cimba_thread_exit_func *>_capsule_pointer(
            exit_capsule,
            _THREAD_EXIT_CAPSULE_NAME,
            "exit_capsule",
        )
    with nogil:
        cimba_thread_hooks_set(initfunc, usrarg, exitfunc)


cdef uint64_t _test_user_context = 0


cdef void _test_trial_increment_u64(void *trial_struct) noexcept nogil:
    cdef uint64_t *fields = <uint64_t *>trial_struct
    fields[0] += 1


cdef void *_test_thread_init(uint64_t tid, void *usrarg) noexcept nogil:
    return usrarg


cdef void _test_thread_exit(void *thrctx) noexcept nogil:
    return


def _test_trial_func_capsule():
    return PyCapsule_New(<void *>_test_trial_increment_u64, _TRIAL_FUNC_CAPSULE_NAME, NULL)


def _test_thread_init_capsule():
    return PyCapsule_New(<void *>_test_thread_init, _THREAD_INIT_CAPSULE_NAME, NULL)


def _test_thread_exit_capsule():
    return PyCapsule_New(<void *>_test_thread_exit, _THREAD_EXIT_CAPSULE_NAME, NULL)


def _test_user_context_capsule():
    return PyCapsule_New(<void *>&_test_user_context, _USER_CONTEXT_CAPSULE_NAME, NULL)
