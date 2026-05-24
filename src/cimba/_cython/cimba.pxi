# This file is included by ../_cimba.pyx.

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
from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF
from libc.limits cimport UINT_MAX
from libc.math cimport isfinite
from libc.stddef cimport size_t
from libc.stdint cimport (
    INT64_MAX,
    INT64_MIN,
    UINT64_MAX,
    int64_t,
    uint16_t,
    uint32_t,
    uint64_t,
    uintptr_t,
)


cdef extern from "stdbool.h":
    ctypedef bint bool


cdef extern from "cmb_priorityqueue.h":
    pass


cdef extern from "cimba.h":
    cdef struct cmb_process:
        int64_t priority
        char name[32]

    cdef struct cmb_buffer:
        uint64_t capacity
        uint64_t level

    cdef struct cmb_objectqueue:
        uint64_t capacity
        uint64_t length

    cdef struct cmb_resource:
        pass

    cdef struct cmb_resourcepool:
        uint64_t capacity
        uint64_t in_use

    cdef struct cmb_condition:
        pass

    cdef struct cmb_random_alias:
        pass

    cdef struct cmb_dataset:
        uint64_t count
        double min
        double max
        double *xa

    cdef struct cmb_datasummary:
        uint64_t count
        double min
        double max
        double m1
        double m2
        double m3
        double m4

    cdef struct cmb_wtdsummary:
        cmb_datasummary ds
        double wsum

    cdef struct cmb_timeseries:
        cmb_dataset ds
        double *ta
        double *wa

    ctypedef void *(*cmb_process_func)(cmb_process *cp, void *context)
    ctypedef void (*cmb_event_func)(void *subject, void *object)
    ctypedef void cimba_trial_func(void *trial_struct) noexcept nogil
    ctypedef void *cimba_thread_init_func(void *usrarg, uint64_t tid) noexcept nogil
    ctypedef void cimba_thread_exit_func(void *thrctx) noexcept nogil
    ctypedef bool (*cmb_condition_demand_func)(
        const cmb_condition *cnd,
        const cmb_process *prc,
        const void *ctx,
    )

    const char *cimba_version()
    void cimba_run_experiment(
        void *your_experiment_array,
        uint64_t num_trials,
        size_t trial_struct_size,
        cimba_trial_func *your_trial_func,
    ) noexcept nogil
    void cimba_set_thread_hooks(
        cimba_thread_init_func *initfunc,
        void *usrarg,
        cimba_thread_exit_func *exitfunc,
    ) noexcept nogil
    void *cimba_thread_context() noexcept nogil

    void cmb_logger_flags_on(uint32_t flags)
    void cmb_logger_flags_off(uint32_t flags)

    void cmb_random_initialize(uint64_t seed)
    void cmb_random_terminate()
    uint64_t cmb_random_hwseed()
    uint64_t cmb_random_curseed()
    uint64_t cmb_random_sfc64()
    uint64_t cmb_random_fmix64(uint64_t seed, uint64_t nonce)
    double cmb_random()
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
    bool cmb_random_flip()
    bool cmb_random_bernoulli(double p)
    unsigned cmb_random_geometric(double p)
    unsigned cmb_random_binomial(unsigned n, double p)
    unsigned cmb_random_negative_binomial(unsigned m, double p)
    unsigned cmb_random_pascal(unsigned m, double p)
    unsigned cmb_random_poisson(double r)
    unsigned cmb_random_loaded_dice(unsigned n, const double *probabilities)
    cmb_random_alias *cmb_random_alias_create(unsigned n, const double *probabilities)
    unsigned cmb_random_alias_sample(const cmb_random_alias *alias)
    void cmb_random_alias_destroy(cmb_random_alias *alias)

    double cmb_time()
    void cmb_event_queue_initialize(double start_time)
    void cmb_event_queue_terminate()
    void cmb_event_queue_clear()
    bool cmb_event_queue_is_empty()
    uint64_t cmb_event_queue_count()
    uint64_t cmb_event_schedule(
        cmb_event_func action,
        void *subject,
        void *object,
        double time,
        int64_t priority,
    )
    bool cmb_event_execute_next()
    void cmb_event_queue_execute()
    uint64_t cmb_event_current()
    bool cmb_event_is_scheduled(uint64_t handle)
    double cmb_event_time(uint64_t handle)
    int64_t cmb_event_priority(uint64_t handle)
    bool cmb_event_cancel(uint64_t handle)
    bool cmb_event_reschedule(uint64_t handle, double time)
    bool cmb_event_reprioritize(uint64_t handle, int64_t priority)
    cmb_event_func CMB_ANY_ACTION
    void *CMB_ANY_OBJECT
    uint64_t cmb_event_pattern_count(
        cmb_event_func action,
        const void *subject,
        const void *object,
    )
    uint64_t cmb_event_pattern_cancel(
        cmb_event_func action,
        const void *subject,
        const void *object,
    )

    cmb_process *cmb_process_create()
    void cmb_process_initialize(
        cmb_process *pp,
        const char *name,
        cmb_process_func procfunc,
        void *context,
        int64_t priority,
    )
    void cmb_process_terminate(cmb_process *pp)
    void cmb_process_destroy(cmb_process *pp)
    void cmb_process_start(cmb_process *pp)
    cmb_process *cmb_process_current()
    int64_t cmb_process_yield()
    void cmb_process_resume(cmb_process *pp, int64_t sig)
    void cmb_process_timers_clear(cmb_process *pp)
    uint64_t cmb_process_timer_add(cmb_process *pp, double dur, int64_t sig)
    uint64_t cmb_process_timer_set(cmb_process *pp, double dur, int64_t sig)
    bool cmb_process_timer_cancel(cmb_process *pp, uint64_t handle)
    int64_t cmb_process_hold(double dur)
    int64_t cmb_process_wait_process(cmb_process *awaited)
    int64_t cmb_process_wait_event(uint64_t ev_handle)
    void cmb_process_exit(void *retval)
    void cmb_process_interrupt(cmb_process *pp, int64_t sig, int64_t pri)
    int64_t cmb_process_stop(cmb_process *tgt, void *retval)
    const char *cmb_process_name(const cmb_process *pp)
    void cmb_process_name_set(cmb_process *pp, const char *name)
    void *cmb_process_context(const cmb_process *pp)
    int64_t cmb_process_priority(const cmb_process *pp)
    void cmb_process_priority_set(cmb_process *pp, int64_t pri)
    int cmb_process_status(const cmb_process *pp)
    void *cmb_process_exit_value(const cmb_process *pp)

    cmb_buffer *cmb_buffer_create()
    void cmb_buffer_initialize(cmb_buffer *bp, const char *name, uint64_t capacity)
    void cmb_buffer_destroy(cmb_buffer *bp)
    int64_t cmb_buffer_get(cmb_buffer *bp, uint64_t *amntp)
    int64_t cmb_buffer_put(cmb_buffer *bp, uint64_t *amntp)
    const char *cmb_buffer_get_name(cmb_buffer *bp)
    uint64_t cmb_buffer_level(cmb_buffer *bp)
    uint64_t cmb_buffer_space(cmb_buffer *bp)
    void cmb_buffer_recording_start(cmb_buffer *bp)
    void cmb_buffer_recording_stop(cmb_buffer *bp)
    cmb_timeseries *cmb_buffer_history(cmb_buffer *bp)

    cmb_objectqueue *cmb_objectqueue_create()
    void cmb_objectqueue_initialize(cmb_objectqueue *oqp, const char *name, uint64_t capacity)
    void cmb_objectqueue_destroy(cmb_objectqueue *oqp)
    int64_t cmb_objectqueue_get(cmb_objectqueue *oqp, void **objectloc)
    int64_t cmb_objectqueue_put(cmb_objectqueue *oqp, void *object)
    const char *cmb_objectqueue_name(cmb_objectqueue *oqp)
    uint64_t cmb_objectqueue_length(cmb_objectqueue *oqp)
    uint64_t cmb_objectqueue_space(cmb_objectqueue *oqp)
    uint64_t cmb_objectqueue_position(cmb_objectqueue *oqp, void *object)
    void cmb_objectqueue_recording_start(cmb_objectqueue *oqp)
    void cmb_objectqueue_recording_stop(cmb_objectqueue *oqp)
    cmb_timeseries *cmb_objectqueue_history(cmb_objectqueue *oqp)

    cdef struct cmb_priorityqueue:
        uint64_t capacity

    cmb_priorityqueue *cmb_priorityqueue_create()
    void cmb_priorityqueue_initialize(cmb_priorityqueue *pqp, const char *name, uint64_t capacity)
    void cmb_priorityqueue_destroy(cmb_priorityqueue *pqp)
    int64_t cmb_priorityqueue_get(cmb_priorityqueue *pqp, void **objectloc)
    int64_t cmb_priorityqueue_put(
        cmb_priorityqueue *pqp,
        void *object,
        int64_t priority,
        uint64_t *handleloc,
    )
    uint64_t cmb_priorityqueue_position(cmb_priorityqueue *pqp, uint64_t handle)
    bool cmb_priorityqueue_cancel(cmb_priorityqueue *pqp, uint64_t handle)
    void cmb_priorityqueue_reprioritize(cmb_priorityqueue *pqp, uint64_t handle, int64_t priority)
    const char *cmb_priorityqueue_name(cmb_priorityqueue *pqp)
    uint64_t cmb_priorityqueue_length(cmb_priorityqueue *pqp)
    uint64_t cmb_priorityqueue_space(cmb_priorityqueue *pqp)
    void cmb_priorityqueue_recording_start(cmb_priorityqueue *pqp)
    void cmb_priorityqueue_recording_stop(cmb_priorityqueue *pqp)
    cmb_timeseries *cmb_priorityqueue_history(cmb_priorityqueue *pqp)

    cmb_resource *cmb_resource_create()
    void cmb_resource_initialize(cmb_resource *rp, const char *name)
    void cmb_resource_destroy(cmb_resource *rp)
    int64_t cmb_resource_acquire(cmb_resource *rp)
    void cmb_resource_release(cmb_resource *rp)
    int64_t cmb_resource_preempt(cmb_resource *rp)
    const char *cmb_resource_name(const cmb_resource *rp)
    uint64_t cmb_resource_in_use(const cmb_resource *rp)
    uint64_t cmb_resource_available(const cmb_resource *rp)
    uint64_t cmb_resource_held_by_process(const cmb_resource *rp, const cmb_process *pp)
    void cmb_resource_start_recording(cmb_resource *rp)
    void cmb_resource_stop_recording(cmb_resource *rp)
    cmb_timeseries *cmb_resource_history(cmb_resource *rp)

    cmb_resourcepool *cmb_resourcepool_create()
    void cmb_resourcepool_initialize(cmb_resourcepool *rpp, const char *name, uint64_t capacity)
    void cmb_resourcepool_destroy(cmb_resourcepool *rpp)
    uint64_t cmb_resourcepool_held_by_process(cmb_resourcepool *rpp, const cmb_process *pp)
    int64_t cmb_resourcepool_acquire(cmb_resourcepool *rpp, uint64_t req_amount)
    int64_t cmb_resourcepool_preempt(cmb_resourcepool *rpp, uint64_t req_amount)
    void cmb_resourcepool_release(cmb_resourcepool *rpp, uint64_t rel_amount)
    const char *cmb_resourcepool_get_name(cmb_resourcepool *rsp)
    uint64_t cmb_resourcepool_in_use(cmb_resourcepool *rsp)
    uint64_t cmb_resourcepool_available(cmb_resourcepool *rsp)
    void cmb_resourcepool_start_recording(cmb_resourcepool *rsp)
    void cmb_resourcepool_stop_recording(cmb_resourcepool *rsp)
    cmb_timeseries *cmb_resourcepool_get_history(cmb_resourcepool *rsp)

    cmb_condition *cmb_condition_create()
    void cmb_condition_initialize(cmb_condition *cvp, const char *name)
    void cmb_condition_destroy(cmb_condition *cvp)
    int64_t cmb_condition_wait(
        cmb_condition *cvp,
        cmb_condition_demand_func dmnd,
        const void *ctx,
    )
    uint64_t cmb_condition_signal(cmb_condition *cvp)

    cmb_dataset *cmb_dataset_create()
    void cmb_dataset_reset(cmb_dataset *dsp)
    uint64_t cmb_dataset_copy(cmb_dataset *tgt, const cmb_dataset *src)
    void cmb_dataset_destroy(cmb_dataset *dsp)
    void cmb_dataset_sort(const cmb_dataset *dsp)
    uint64_t cmb_dataset_add(cmb_dataset *dsp, double x)
    uint64_t cmb_dataset_summarize(const cmb_dataset *dsp, cmb_datasummary *dsump)
    uint64_t cmb_dataset_count(const cmb_dataset *dsp)
    double cmb_dataset_min(const cmb_dataset *dsp)
    double cmb_dataset_max(const cmb_dataset *dsp)
    double cmb_dataset_median(const cmb_dataset *dsp)
    void cmb_dataset_ACF(const cmb_dataset *dsp, unsigned n, double *acf)
    void cmb_dataset_PACF(const cmb_dataset *dsp, unsigned n, double *pacf, double *acf)

    cmb_datasummary *cmb_datasummary_create()
    void cmb_datasummary_reset(cmb_datasummary *dsp)
    uint64_t cmb_datasummary_merge(cmb_datasummary *tgt, const cmb_datasummary *dsp1, const cmb_datasummary *dsp2)
    void cmb_datasummary_destroy(cmb_datasummary *dsp)
    uint64_t cmb_datasummary_add(cmb_datasummary *dsp, double y)
    uint64_t cmb_datasummary_count(const cmb_datasummary *dsp)
    double cmb_datasummary_min(const cmb_datasummary *dsp)
    double cmb_datasummary_max(const cmb_datasummary *dsp)
    double cmb_datasummary_mean(const cmb_datasummary *dsp)
    double cmb_datasummary_variance(const cmb_datasummary *dsp)
    double cmb_datasummary_stddev(const cmb_datasummary *dsp)
    double cmb_datasummary_skewness(const cmb_datasummary *dsp)
    double cmb_datasummary_kurtosis(const cmb_datasummary *dsp)

    cmb_wtdsummary *cmb_wtdsummary_create()
    void cmb_wtdsummary_reset(cmb_wtdsummary *wsp)
    uint64_t cmb_wtdsummary_merge(cmb_wtdsummary *tgt, const cmb_wtdsummary *ws1, const cmb_wtdsummary *ws2)
    void cmb_wtdsummary_destroy(cmb_wtdsummary *wsp)
    uint64_t cmb_wtdsummary_add(cmb_wtdsummary *wsp, double x, double w)
    uint64_t cmb_wtdsummary_count(const cmb_wtdsummary *wsp)
    double cmb_wtdsummary_min(const cmb_wtdsummary *wsp)
    double cmb_wtdsummary_max(const cmb_wtdsummary *wsp)
    double cmb_wtdsummary_mean(const cmb_wtdsummary *wsp)
    double cmb_wtdsummary_variance(const cmb_wtdsummary *wsp)
    double cmb_wtdsummary_stddev(const cmb_wtdsummary *wsp)
    double cmb_wtdsummary_skewness(const cmb_wtdsummary *wsp)
    double cmb_wtdsummary_kurtosis(const cmb_wtdsummary *wsp)

    cmb_timeseries *cmb_timeseries_create()
    void cmb_timeseries_reset(cmb_timeseries *tsp)
    void cmb_timeseries_destroy(cmb_timeseries *tsp)
    uint64_t cmb_timeseries_copy(cmb_timeseries *tgt, const cmb_timeseries *src)
    uint64_t cmb_timeseries_add(cmb_timeseries *tsp, double x, double t)
    uint64_t cmb_timeseries_finalize(cmb_timeseries *tsp, double t)
    void cmb_timeseries_sort_x(cmb_timeseries *tsp)
    void cmb_timeseries_sort_t(cmb_timeseries *tsp)
    uint64_t cmb_timeseries_summarize(const cmb_timeseries *tsp, cmb_wtdsummary *wsp)
    uint64_t cmb_timeseries_count(const cmb_timeseries *tsp)
    double cmb_timeseries_min(const cmb_timeseries *tsp)
    double cmb_timeseries_max(const cmb_timeseries *tsp)
    double cmb_timeseries_median(const cmb_timeseries *tsp)
    void cmb_timeseries_ACF(const cmb_timeseries *tsp, uint16_t n, double *acf)
    void cmb_timeseries_PACF(const cmb_timeseries *tsp, uint16_t n, double *pacf, double *acf)


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

cdef uint64_t _UNLIMITED_U64 = <uint64_t>-1
cdef int64_t _PROCESS_CANCEL_SIGNAL = INT64_MIN
cdef int64_t _PROCESS_CANCEL_PRIORITY = INT64_MAX
cdef int64_t _CLEAR_AFTER_CANCEL_PRIORITY = INT64_MIN
cdef object _thread_state = threading.local()
cdef object _MISSING = object()
cdef object _ZERO = 0
cdef object _ONE = 1
cdef bytes _TRIAL_FUNC_CAPSULE_NAME = b"cimba.trial_func"
cdef bytes _THREAD_INIT_CAPSULE_NAME = b"cimba.thread_init_func"
cdef bytes _THREAD_EXIT_CAPSULE_NAME = b"cimba.thread_exit_func"
cdef bytes _USER_CONTEXT_CAPSULE_NAME = b"cimba.user_context"


class _ProcessCancelled(BaseException):
    pass


class _ProcessExit(BaseException):
    def __init__(self, object value=None):
        self.value = value


cdef object _active_simulation_get():
    return getattr(_thread_state, "active_simulation", None)


cdef void _active_simulation_set(object sim):
    if sim is None:
        if hasattr(_thread_state, "active_simulation"):
            delattr(_thread_state, "active_simulation")
    else:
        _thread_state.active_simulation = sim


cdef dict _process_registry_get():
    cdef object registry = getattr(_thread_state, "process_registry", None)
    if registry is None:
        registry = {}
        _thread_state.process_registry = registry
    return <dict>registry


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


cdef inline uint64_t _capacity_to_u64(object capacity) except *:
    if capacity is None:
        return _UNLIMITED_U64
    return _u64_value(capacity, "capacity", 1)


cdef inline uint64_t _amount_to_u64(object amount) except *:
    return _u64_value(amount, "amount", 1)


cdef inline uint64_t _handle_to_u64(object handle, str name="handle") except *:
    return _u64_value(handle, name, 1)


cdef inline uint64_t _seed_to_u64(object seed) except *:
    return _u64_value(seed, "seed", 0)


cdef inline int64_t _priority_to_i64(object priority) except *:
    return _i64_value(priority, "priority")


cdef inline int64_t _signal_to_i64(object signal, bint allow_cancel=False) except *:
    cdef int64_t sig = _i64_value(signal, "signal")
    if sig == _PROCESS_CANCEL_SIGNAL and not allow_cancel:
        raise ValueError("signal value is reserved for internal process cancellation")
    return sig


cdef inline double _duration_to_double(object duration, str name, bint allow_zero=True) except *:
    cdef double value
    if PyFloat_CheckExact(duration):
        value = PyFloat_AS_DOUBLE(duration)
    elif PyLong_CheckExact(duration):
        value = PyLong_AsDouble(duration)
    else:
        try:
            value = float(duration)
        except (TypeError, ValueError):
            raise TypeError(f"{name} must be a real number") from None
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0.0 or (not allow_zero and value == 0.0):
        if allow_zero:
            raise ValueError(f"{name} must be non-negative")
        raise ValueError(f"{name} must be positive")
    return value


cdef inline uint16_t _lags_to_u16(object lags) except *:
    cdef uint64_t value = _u64_value(lags, "lags", 0)
    if value > 65535:
        raise OverflowError("lags must fit in uint16")
    return <uint16_t>value


cdef bytes _name_bytes(str name):
    cdef bytes encoded = name.encode("utf-8")
    if len(encoded) == 0:
        raise ValueError("name must not be empty")
    if len(encoded) >= 32:
        raise ValueError("Cimba names must be shorter than 32 UTF-8 bytes")
    return encoded


cdef void _register_with_active_simulation(object obj):
    cdef object sim = _active_simulation_get()
    if sim is not None:
        sim._owned.append(obj)


cdef void _cancel_active_processes(object skip):
    cdef object obj
    cdef object stop
    cdef object sim = _active_simulation_get()
    if sim is None:
        return
    for obj in list(sim._owned):
        if obj is skip:
            continue
        stop = getattr(obj, "stop", None)
        if stop is not None:
            stop()


cdef void _clear_after_cancellation_event(void *subject, void *object) noexcept with gil:
    cmb_event_queue_clear()


cdef void _record_exception(BaseException exc):
    cdef cmb_process *pp = cmb_process_current()
    cdef object skip = None
    cdef object sim = _active_simulation_get()
    if pp != NULL:
        skip = _process_registry_get().get(<uintptr_t>pp)
    if sim is not None:
        sim._exception = exc
    _cancel_active_processes(skip)
    cmb_event_schedule(
        _clear_after_cancellation_event,
        NULL,
        NULL,
        cmb_time(),
        _CLEAR_AFTER_CANCEL_PRIORITY,
    )


cdef object _object_from_owned_pointer(void *ptr):
    cdef PyObject *op = <PyObject *>ptr
    cdef object obj
    if op == NULL:
        return None
    obj = <object>op
    Py_DECREF(<object>op)
    return obj


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
            cimba_run_experiment(<void *>&view[0], num_trials, <size_t>struct_size, trial_func)


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
        cimba_set_thread_hooks(initfunc, usrarg, exitfunc)


cdef void _test_trial_increment_u64(void *trial_struct) noexcept nogil:
    cdef uint64_t *fields = <uint64_t *>trial_struct
    fields[0] += 1


cdef void *_test_thread_init(void *usrarg, uint64_t tid) noexcept nogil:
    return usrarg


cdef void _test_thread_exit(void *thrctx) noexcept nogil:
    return


cdef uint64_t _test_user_context = 0


def _test_trial_func_capsule():
    return PyCapsule_New(<void *>_test_trial_increment_u64, _TRIAL_FUNC_CAPSULE_NAME, NULL)


def _test_thread_init_capsule():
    return PyCapsule_New(<void *>_test_thread_init, _THREAD_INIT_CAPSULE_NAME, NULL)


def _test_thread_exit_capsule():
    return PyCapsule_New(<void *>_test_thread_exit, _THREAD_EXIT_CAPSULE_NAME, NULL)


def _test_user_context_capsule():
    return PyCapsule_New(<void *>&_test_user_context, _USER_CONTEXT_CAPSULE_NAME, NULL)
