# cython: language_level=3
"""Cython bindings for the Cimba simulation library.

The binding layer follows Cimba's create/initialize/terminate/destroy model:
Python wrapper objects own their native pointer and release it from ``close()``
or ``__dealloc__``.  ``Simulation`` owns the thread-local random generator and
event queue and keeps objects created while it is active alive until shutdown.
"""

from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF
from libc.stddef cimport size_t
from libc.stdint cimport int64_t, uint16_t, uint32_t, uint64_t, uintptr_t


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
    ctypedef bool (*cmb_condition_demand_func)(
        const cmb_condition *cnd,
        const cmb_process *prc,
        const void *ctx,
    )

    const char *cimba_version()

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
    double cmb_random_exponential(double mean)
    double cmb_random_gamma(double shape, double scale)
    double cmb_random_beta(double a, double b, double min, double max)
    double cmb_random_PERT(double min, double mode, double max)
    double cmb_random_PERT_mod(double min, double mode, double max, double lambd)
    long cmb_random_dice(long min, long max)
    bool cmb_random_flip()
    bool cmb_random_bernoulli(double p)

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
    void cmb_dataset_destroy(cmb_dataset *dsp)
    uint64_t cmb_dataset_add(cmb_dataset *dsp, double x)
    uint64_t cmb_dataset_summarize(const cmb_dataset *dsp, cmb_datasummary *dsump)
    uint64_t cmb_dataset_count(const cmb_dataset *dsp)
    double cmb_dataset_min(const cmb_dataset *dsp)
    double cmb_dataset_max(const cmb_dataset *dsp)
    double cmb_dataset_median(const cmb_dataset *dsp)

    cmb_datasummary *cmb_datasummary_create()
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
    void cmb_timeseries_destroy(cmb_timeseries *tsp)
    uint64_t cmb_timeseries_copy(cmb_timeseries *tgt, const cmb_timeseries *src)
    uint64_t cmb_timeseries_add(cmb_timeseries *tsp, double x, double t)
    uint64_t cmb_timeseries_finalize(cmb_timeseries *tsp, double t)
    uint64_t cmb_timeseries_summarize(const cmb_timeseries *tsp, cmb_wtdsummary *wsp)
    uint64_t cmb_timeseries_count(const cmb_timeseries *tsp)
    double cmb_timeseries_min(const cmb_timeseries *tsp)
    double cmb_timeseries_max(const cmb_timeseries *tsp)
    double cmb_timeseries_median(const cmb_timeseries *tsp)


UNLIMITED = (1 << 64) - 1

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
cdef object _active_simulation = None
cdef dict _process_registry = {}
cdef object _MISSING = object()


cdef void _raise_if_closed(object obj):
    if obj._closed:
        raise RuntimeError(f"{obj.__class__.__name__} is closed")


cdef uint64_t _capacity_to_u64(object capacity):
    cdef object value = UNLIMITED if capacity is None else capacity
    if value < 1 or value > UNLIMITED:
        raise ValueError("capacity must be in the range [1, UNLIMITED]")
    return <uint64_t>value


cdef bytes _name_bytes(str name):
    cdef bytes encoded = name.encode("utf-8")
    if len(encoded) == 0:
        raise ValueError("name must not be empty")
    if len(encoded) >= 32:
        raise ValueError("Cimba names must be shorter than 32 UTF-8 bytes")
    return encoded


cdef void _register_with_active_simulation(object obj):
    if _active_simulation is not None:
        _active_simulation._owned.append(obj)


cdef void _stop_active_processes(object skip):
    cdef object obj
    cdef object stop
    if _active_simulation is None:
        return
    for obj in list(_active_simulation._owned):
        if obj is skip:
            continue
        stop = getattr(obj, "stop", None)
        if stop is not None:
            stop()


cdef void _record_exception(BaseException exc):
    cdef cmb_process *pp = cmb_process_current()
    cdef object skip = None
    if pp != NULL:
        skip = _process_registry.get(<uintptr_t>pp)
    if _active_simulation is not None:
        _active_simulation._exception = exc
    _stop_active_processes(skip)
    cmb_event_queue_clear()


cdef object _object_from_owned_pointer(void *ptr):
    cdef PyObject *op = <PyObject *>ptr
    cdef object obj
    if op == NULL:
        return None
    obj = <object>op
    Py_DECREF(<object>op)
    return obj


cdef void _clear_event(void *subject, void *object) noexcept with gil:
    _stop_active_processes(None)
    cmb_event_queue_clear()


cdef void *_process_trampoline(cmb_process *me, void *ctx) noexcept with gil:
    cdef Process proc = <Process><object><PyObject *>ctx
    cdef object result
    cdef PyObject *op

    try:
        result = proc._func(proc, proc._context)
        if result is None:
            return NULL
        op = <PyObject *>result
        Py_INCREF(<object>op)
        proc._owns_exit_value = True
        return <void *>op
    except BaseException as exc:
        _record_exception(exc)
        return NULL


cdef bool _condition_demand_trampoline(
    const cmb_condition *cnd,
    const cmb_process *prc,
    const void *ctx,
) noexcept with gil:
    cdef object demand_ctx = <object><PyObject *>ctx
    cdef object predicate = demand_ctx[0]
    cdef object user_ctx = demand_ctx[1]
    cdef Process proc = _process_registry.get(<uintptr_t>prc)

    try:
        return True if predicate(proc, user_ctx) else False
    except BaseException as exc:
        _record_exception(exc)
        return False


def native_version() -> str:
    """Return the version of the underlying Cimba C library."""
    cdef const char *v = cimba_version()
    return v.decode("utf-8")


def logger_flags_on(int flags) -> None:
    """Turn on Cimba logger flags in the current thread."""
    cmb_logger_flags_on(<uint32_t>flags)


def logger_flags_off(int flags) -> None:
    """Turn off Cimba logger flags in the current thread."""
    cmb_logger_flags_off(<uint32_t>flags)


def hwseed() -> int:
    """Return a hardware-derived random seed."""
    return <object>cmb_random_hwseed()


def seed(object value=None) -> int:
    """Initialize the thread-local random generator and return the seed used."""
    cdef uint64_t seed_value = cmb_random_hwseed() if value is None else <uint64_t>value
    cmb_random_initialize(seed_value)
    return <object>seed_value


def current_seed() -> int:
    """Return the seed currently used by Cimba in this thread."""
    return <object>cmb_random_curseed()


def random() -> float:
    return cmb_random()


def random_u64() -> int:
    return <object>cmb_random_sfc64()


def fmix64(int seed, int nonce) -> int:
    return <object>cmb_random_fmix64(<uint64_t>seed, <uint64_t>nonce)


def uniform(double min, double max) -> float:
    return cmb_random_uniform(min, max)


def triangular(double min, double mode, double max) -> float:
    return cmb_random_triangular(min, mode, max)


def normal(double mu=0.0, double sigma=1.0) -> float:
    return cmb_random_normal(mu, sigma)


def exponential(double mean) -> float:
    return cmb_random_exponential(mean)


def gamma(double shape, double scale=1.0) -> float:
    return cmb_random_gamma(shape, scale)


def beta(double a, double b, double min=0.0, double max=1.0) -> float:
    return cmb_random_beta(a, b, min, max)


def pert(double min, double mode, double max) -> float:
    return cmb_random_PERT(min, mode, max)


def pert_mod(double min, double mode, double max, double lambda_) -> float:
    return cmb_random_PERT_mod(min, mode, max, lambda_)


def dice(int min, int max) -> int:
    return cmb_random_dice(min, max)


def flip() -> bool:
    return True if cmb_random_flip() else False


def bernoulli(double p) -> bool:
    return True if cmb_random_bernoulli(p) else False


def time() -> float:
    """Return the current simulation time."""
    return cmb_time()


def hold(double duration) -> int:
    """Suspend the current process for ``duration`` simulated time units."""
    return cmb_process_hold(duration)


def yield_process() -> int:
    """Yield the current process until another process or timer resumes it."""
    return cmb_process_yield()


def process_exit(object value=None):
    """Exit the current process with an optional Python exit value."""
    cdef Process proc = current_process()
    cdef PyObject *op = NULL
    if value is not None:
        op = <PyObject *>value
        Py_INCREF(<object>op)
        if proc is not None:
            proc._owns_exit_value = True
    cmb_process_exit(<void *>op)


def current_process():
    """Return the Python wrapper for the currently running process, or ``None``."""
    cdef cmb_process *pp = cmb_process_current()
    if pp == NULL:
        return None
    return _process_registry.get(<uintptr_t>pp)


cdef class Simulation:
    """Own the thread-local Cimba random generator and event queue."""

    cdef public bint _closed
    cdef bint _event_initialized
    cdef bint _random_initialized
    cdef public object seed_used
    cdef public object _owned
    cdef public object _exception

    def __init__(self, double start_time=0.0, object seed=None, bint log_info=False):
        global _active_simulation
        if _active_simulation is not None:
            raise RuntimeError("only one active Simulation is supported per Python thread")

        self._closed = False
        self._event_initialized = False
        self._random_initialized = False
        self._owned = []
        self._exception = None

        cdef uint64_t seed_value = cmb_random_hwseed() if seed is None else <uint64_t>seed
        cmb_random_initialize(seed_value)
        self._random_initialized = True
        self.seed_used = <object>seed_value

        cmb_event_queue_initialize(start_time)
        self._event_initialized = True
        if log_info:
            cmb_logger_flags_on(LOGGER_INFO)
        else:
            cmb_logger_flags_off(LOGGER_INFO)

        _active_simulation = self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __dealloc__(self):
        if not self._closed:
            self.close()

    property closed:
        def __get__(self):
            return True if self._closed else False

    property now:
        def __get__(self):
            return cmb_time()

    property event_count:
        def __get__(self):
            return <object>cmb_event_queue_count()

    def stop_at(self, double when, int priority=0) -> int:
        """Schedule the event queue to clear at absolute simulation time ``when``."""
        _raise_if_closed(self)
        if when < cmb_time():
            raise ValueError("stop time cannot be before the current simulation time")
        return <object>cmb_event_schedule(_clear_event, NULL, NULL, when, priority)

    def clear(self) -> None:
        """Clear all scheduled events, stopping the simulation run."""
        _raise_if_closed(self)
        cmb_event_queue_clear()

    def execute_next(self) -> bool:
        """Execute one scheduled event."""
        _raise_if_closed(self)
        cdef bool ok = cmb_event_execute_next()
        if self._exception is not None:
            exc = self._exception
            self._exception = None
            raise exc
        return True if ok else False

    def execute(self) -> None:
        """Run the event queue until it is empty."""
        _raise_if_closed(self)
        cmb_event_queue_execute()
        if self._exception is not None:
            exc = self._exception
            self._exception = None
            raise exc

    def close(self) -> None:
        """Stop owned processes and release the thread-local Cimba state."""
        global _active_simulation
        if self._closed:
            return

        cdef object obj
        for obj in reversed(list(self._owned)):
            close = getattr(obj, "close", None)
            if close is not None:
                close()
        self._owned.clear()

        if self._event_initialized:
            cmb_event_queue_terminate()
            self._event_initialized = False
        if self._random_initialized:
            cmb_random_terminate()
            self._random_initialized = False

        if _active_simulation is self:
            _active_simulation = None
        self._closed = True


cdef class Process:
    """A Cimba process backed by a Python callable."""

    cdef cmb_process *_ptr
    cdef public bint _closed
    cdef bint _initialized
    cdef bint _keepalive
    cdef bint _owns_exit_value
    cdef object _func
    cdef object _context
    cdef object _name

    def __cinit__(self):
        self._ptr = NULL
        self._closed = True
        self._initialized = False
        self._keepalive = False
        self._owns_exit_value = False
        self._func = None
        self._context = None
        self._name = None

    def __init__(self, str name, object func, object context=None, int priority=0):
        if not callable(func):
            raise TypeError("func must be callable")
        cdef bytes bname = _name_bytes(name)
        self._ptr = cmb_process_create()
        self._func = func
        self._context = context
        self._name = name
        cmb_process_initialize(
            self._ptr,
            bname,
            _process_trampoline,
            <void *><PyObject *>self,
            priority,
        )
        self._initialized = True
        self._closed = False
        _process_registry[<uintptr_t>self._ptr] = self
        _register_with_active_simulation(self)

    def __dealloc__(self):
        if not self._closed:
            self.close()

    property name:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_process_name(self._ptr).decode("utf-8")

    property priority:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_process_priority(self._ptr)
        def __set__(self, int value):
            _raise_if_closed(self)
            cmb_process_priority_set(self._ptr, value)

    property status:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_process_status(self._ptr)

    def start(self):
        """Schedule the process to start at the current simulation time."""
        _raise_if_closed(self)
        if not self._keepalive:
            Py_INCREF(<object><PyObject *>self)
            self._keepalive = True
        cmb_process_start(self._ptr)
        return self

    def stop(self) -> int:
        """Stop the process if it is running."""
        _raise_if_closed(self)
        if cmb_process_status(self._ptr) == PROCESS_RUNNING:
            return cmb_process_stop(self._ptr, NULL)
        return STOPPED

    def interrupt(self, int signal=INTERRUPTED, int priority=0) -> None:
        _raise_if_closed(self)
        if signal == SUCCESS:
            raise ValueError("interrupt signal cannot be SUCCESS")
        cmb_process_interrupt(self._ptr, signal, priority)

    def resume(self, int signal=SUCCESS) -> None:
        _raise_if_closed(self)
        cmb_process_resume(self._ptr, signal)

    def wait(self) -> int:
        _raise_if_closed(self)
        return cmb_process_wait_process(self._ptr)

    def timer_add(self, double duration, int signal=TIMEOUT) -> int:
        _raise_if_closed(self)
        return <object>cmb_process_timer_add(self._ptr, duration, signal)

    def timer_set(self, double duration, int signal=TIMEOUT) -> int:
        _raise_if_closed(self)
        return <object>cmb_process_timer_set(self._ptr, duration, signal)

    def timer_cancel(self, int handle) -> bool:
        _raise_if_closed(self)
        return True if cmb_process_timer_cancel(self._ptr, <uint64_t>handle) else False

    def timers_clear(self) -> None:
        _raise_if_closed(self)
        cmb_process_timers_clear(self._ptr)

    def exit_value(self):
        _raise_if_closed(self)
        cdef void *ptr = cmb_process_exit_value(self._ptr)
        if ptr == NULL:
            return None
        cdef object obj = <object><PyObject *>ptr
        return obj

    def close(self) -> None:
        cdef void *ptr = NULL
        if self._closed:
            return
        if self._ptr != NULL:
            if cmb_process_status(self._ptr) == PROCESS_RUNNING:
                cmb_process_stop(self._ptr, NULL)
            cmb_event_pattern_cancel(
                CMB_ANY_ACTION,
                <const void *>self._ptr,
                <const void *>CMB_ANY_OBJECT,
            )
            if self._owns_exit_value:
                ptr = cmb_process_exit_value(self._ptr)
                if ptr != NULL:
                    Py_DECREF(<object><PyObject *>ptr)
                self._owns_exit_value = False
            if self._initialized:
                cmb_process_terminate(self._ptr)
                self._initialized = False
            _process_registry.pop(<uintptr_t>self._ptr, None)
            cmb_process_destroy(self._ptr)
            self._ptr = NULL
        if self._keepalive:
            self._keepalive = False
            Py_DECREF(<object><PyObject *>self)
        self._closed = True


cdef class Buffer:
    """Numeric Cimba buffer with put/get semantics."""

    cdef cmb_buffer *_ptr
    cdef public bint _closed

    def __cinit__(self):
        self._ptr = NULL
        self._closed = True

    def __init__(self, str name, object capacity=None):
        cdef bytes bname = _name_bytes(name)
        self._ptr = cmb_buffer_create()
        cmb_buffer_initialize(self._ptr, bname, _capacity_to_u64(capacity))
        self._closed = False
        _register_with_active_simulation(self)

    def __dealloc__(self):
        if not self._closed:
            self.close()

    property name:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_buffer_get_name(self._ptr).decode("utf-8")

    property capacity:
        def __get__(self):
            _raise_if_closed(self)
            return <object>self._ptr.capacity

    property level:
        def __get__(self):
            _raise_if_closed(self)
            return <object>cmb_buffer_level(self._ptr)

    property space:
        def __get__(self):
            _raise_if_closed(self)
            return <object>cmb_buffer_space(self._ptr)

    def put(self, int amount=1):
        """Put ``amount`` into the buffer. Returns ``(signal, remaining)``."""
        _raise_if_closed(self)
        cdef uint64_t n = <uint64_t>amount
        cdef int64_t sig = cmb_buffer_put(self._ptr, &n)
        return sig, <object>n

    def get(self, int amount=1):
        """Get ``amount`` from the buffer. Returns ``(signal, obtained)``."""
        _raise_if_closed(self)
        cdef uint64_t n = <uint64_t>amount
        cdef int64_t sig = cmb_buffer_get(self._ptr, &n)
        return sig, <object>n

    def start_recording(self) -> None:
        _raise_if_closed(self)
        cmb_buffer_recording_start(self._ptr)

    def stop_recording(self) -> None:
        _raise_if_closed(self)
        cmb_buffer_recording_stop(self._ptr)

    def history(self):
        _raise_if_closed(self)
        return _timeseries_copy(cmb_buffer_history(self._ptr))

    def close(self) -> None:
        if self._closed:
            return
        if self._ptr != NULL:
            cmb_buffer_destroy(self._ptr)
            self._ptr = NULL
        self._closed = True


cdef class ObjectQueue:
    """FIFO queue for Python objects."""

    cdef cmb_objectqueue *_ptr
    cdef public bint _closed

    def __cinit__(self):
        self._ptr = NULL
        self._closed = True

    def __init__(self, str name, object capacity=None):
        cdef bytes bname = _name_bytes(name)
        self._ptr = cmb_objectqueue_create()
        cmb_objectqueue_initialize(self._ptr, bname, _capacity_to_u64(capacity))
        self._closed = False
        _register_with_active_simulation(self)

    def __dealloc__(self):
        if not self._closed:
            self.close()

    property name:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_objectqueue_name(self._ptr).decode("utf-8")

    property capacity:
        def __get__(self):
            _raise_if_closed(self)
            return <object>self._ptr.capacity

    property length:
        def __get__(self):
            _raise_if_closed(self)
            return <object>cmb_objectqueue_length(self._ptr)

    property space:
        def __get__(self):
            _raise_if_closed(self)
            return <object>cmb_objectqueue_space(self._ptr)

    def put(self, object obj) -> int:
        _raise_if_closed(self)
        cdef PyObject *op = <PyObject *>obj
        Py_INCREF(<object>op)
        cdef int64_t sig = cmb_objectqueue_put(self._ptr, <void *>op)
        if sig != SUCCESS:
            Py_DECREF(<object>op)
        return sig

    def get(self):
        """Get one object. Returns ``(signal, object)``."""
        _raise_if_closed(self)
        cdef void *ptr = NULL
        cdef int64_t sig = cmb_objectqueue_get(self._ptr, &ptr)
        if sig != SUCCESS:
            return sig, None
        return sig, _object_from_owned_pointer(ptr)

    def position(self, object obj) -> int:
        _raise_if_closed(self)
        return <object>cmb_objectqueue_position(self._ptr, <void *><PyObject *>obj)

    def start_recording(self) -> None:
        _raise_if_closed(self)
        cmb_objectqueue_recording_start(self._ptr)

    def stop_recording(self) -> None:
        _raise_if_closed(self)
        cmb_objectqueue_recording_stop(self._ptr)

    def history(self):
        _raise_if_closed(self)
        return _timeseries_copy(cmb_objectqueue_history(self._ptr))

    cdef void _decref_queued_objects(self):
        cdef void *ptr = NULL
        while self._ptr != NULL and cmb_objectqueue_length(self._ptr) > 0:
            ptr = NULL
            if cmb_objectqueue_get(self._ptr, &ptr) == SUCCESS and ptr != NULL:
                Py_DECREF(<object><PyObject *>ptr)

    def close(self) -> None:
        if self._closed:
            return
        if self._ptr != NULL:
            self._decref_queued_objects()
            cmb_objectqueue_destroy(self._ptr)
            self._ptr = NULL
        self._closed = True


cdef class PriorityQueue:
    """Priority queue for Python objects."""

    cdef cmb_priorityqueue *_ptr
    cdef public bint _closed
    cdef object _objects

    def __cinit__(self):
        self._ptr = NULL
        self._closed = True
        self._objects = None

    def __init__(self, str name, object capacity=None):
        cdef bytes bname = _name_bytes(name)
        self._ptr = cmb_priorityqueue_create()
        cmb_priorityqueue_initialize(self._ptr, bname, _capacity_to_u64(capacity))
        self._objects = {}
        self._closed = False
        _register_with_active_simulation(self)

    def __dealloc__(self):
        if not self._closed:
            self.close()

    property name:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_priorityqueue_name(self._ptr).decode("utf-8")

    property capacity:
        def __get__(self):
            _raise_if_closed(self)
            return <object>self._ptr.capacity

    property length:
        def __get__(self):
            _raise_if_closed(self)
            return <object>cmb_priorityqueue_length(self._ptr)

    property space:
        def __get__(self):
            _raise_if_closed(self)
            return <object>cmb_priorityqueue_space(self._ptr)

    def put(self, object obj, int priority=0):
        """Put an object and return ``(signal, handle)``."""
        _raise_if_closed(self)
        cdef PyObject *op = <PyObject *>obj
        cdef uint64_t handle = 0
        Py_INCREF(<object>op)
        cdef int64_t sig = cmb_priorityqueue_put(self._ptr, <void *>op, priority, &handle)
        if sig != SUCCESS:
            Py_DECREF(<object>op)
        else:
            self._objects[<object>handle] = obj
        return sig, <object>handle

    def get(self):
        """Get the highest-priority object. Returns ``(signal, object)``."""
        _raise_if_closed(self)
        cdef void *ptr = NULL
        cdef int64_t sig = cmb_priorityqueue_get(self._ptr, &ptr)
        cdef object obj
        cdef object key
        if sig != SUCCESS:
            return sig, None
        obj = _object_from_owned_pointer(ptr)
        for key in list(self._objects):
            if self._objects[key] is obj:
                del self._objects[key]
                break
        return sig, obj

    def position(self, int handle) -> int:
        _raise_if_closed(self)
        return <object>cmb_priorityqueue_position(self._ptr, <uint64_t>handle)

    def cancel(self, int handle) -> bool:
        _raise_if_closed(self)
        cdef bint found = cmb_priorityqueue_cancel(self._ptr, <uint64_t>handle)
        cdef object obj
        if found:
            obj = self._objects.pop(handle, _MISSING)
            if obj is not _MISSING:
                Py_DECREF(<object>obj)
        return True if found else False

    def reprioritize(self, int handle, int priority) -> None:
        _raise_if_closed(self)
        cmb_priorityqueue_reprioritize(self._ptr, <uint64_t>handle, priority)

    def start_recording(self) -> None:
        _raise_if_closed(self)
        cmb_priorityqueue_recording_start(self._ptr)

    def stop_recording(self) -> None:
        _raise_if_closed(self)
        cmb_priorityqueue_recording_stop(self._ptr)

    def history(self):
        _raise_if_closed(self)
        return _timeseries_copy(cmb_priorityqueue_history(self._ptr))

    cdef void _decref_queued_objects(self):
        cdef void *ptr = NULL
        while self._ptr != NULL and cmb_priorityqueue_length(self._ptr) > 0:
            ptr = NULL
            if cmb_priorityqueue_get(self._ptr, &ptr) == SUCCESS and ptr != NULL:
                Py_DECREF(<object><PyObject *>ptr)

    def close(self) -> None:
        if self._closed:
            return
        if self._ptr != NULL:
            self._decref_queued_objects()
            self._objects.clear()
            cmb_priorityqueue_destroy(self._ptr)
            self._ptr = NULL
        self._closed = True


cdef class Resource:
    """Binary semaphore resource."""

    cdef cmb_resource *_ptr
    cdef public bint _closed

    def __cinit__(self):
        self._ptr = NULL
        self._closed = True

    def __init__(self, str name):
        cdef bytes bname = _name_bytes(name)
        self._ptr = cmb_resource_create()
        cmb_resource_initialize(self._ptr, bname)
        self._closed = False
        _register_with_active_simulation(self)

    def __dealloc__(self):
        if not self._closed:
            self.close()

    property name:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_resource_name(self._ptr).decode("utf-8")

    property in_use:
        def __get__(self):
            _raise_if_closed(self)
            return <object>cmb_resource_in_use(self._ptr)

    property available:
        def __get__(self):
            _raise_if_closed(self)
            return <object>cmb_resource_available(self._ptr)

    def acquire(self) -> int:
        _raise_if_closed(self)
        return cmb_resource_acquire(self._ptr)

    def preempt(self) -> int:
        _raise_if_closed(self)
        return cmb_resource_preempt(self._ptr)

    def release(self) -> None:
        _raise_if_closed(self)
        cmb_resource_release(self._ptr)

    def held_by(self, Process process) -> int:
        _raise_if_closed(self)
        _raise_if_closed(process)
        return <object>cmb_resource_held_by_process(self._ptr, process._ptr)

    def start_recording(self) -> None:
        _raise_if_closed(self)
        cmb_resource_start_recording(self._ptr)

    def stop_recording(self) -> None:
        _raise_if_closed(self)
        cmb_resource_stop_recording(self._ptr)

    def history(self):
        _raise_if_closed(self)
        return _timeseries_copy(cmb_resource_history(self._ptr))

    def close(self) -> None:
        if self._closed:
            return
        if self._ptr != NULL:
            cmb_resource_destroy(self._ptr)
            self._ptr = NULL
        self._closed = True


cdef class ResourcePool:
    """Counting semaphore resource pool."""

    cdef cmb_resourcepool *_ptr
    cdef public bint _closed

    def __cinit__(self):
        self._ptr = NULL
        self._closed = True

    def __init__(self, str name, int capacity):
        cdef bytes bname = _name_bytes(name)
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._ptr = cmb_resourcepool_create()
        cmb_resourcepool_initialize(self._ptr, bname, <uint64_t>capacity)
        self._closed = False
        _register_with_active_simulation(self)

    def __dealloc__(self):
        if not self._closed:
            self.close()

    property name:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_resourcepool_get_name(self._ptr).decode("utf-8")

    property capacity:
        def __get__(self):
            _raise_if_closed(self)
            return <object>self._ptr.capacity

    property in_use:
        def __get__(self):
            _raise_if_closed(self)
            return <object>cmb_resourcepool_in_use(self._ptr)

    property available:
        def __get__(self):
            _raise_if_closed(self)
            return <object>cmb_resourcepool_available(self._ptr)

    def acquire(self, int amount=1) -> int:
        _raise_if_closed(self)
        return cmb_resourcepool_acquire(self._ptr, <uint64_t>amount)

    def preempt(self, int amount=1) -> int:
        _raise_if_closed(self)
        return cmb_resourcepool_preempt(self._ptr, <uint64_t>amount)

    def release(self, int amount=1) -> None:
        _raise_if_closed(self)
        cmb_resourcepool_release(self._ptr, <uint64_t>amount)

    def held_by(self, Process process) -> int:
        _raise_if_closed(self)
        _raise_if_closed(process)
        return <object>cmb_resourcepool_held_by_process(self._ptr, process._ptr)

    def start_recording(self) -> None:
        _raise_if_closed(self)
        cmb_resourcepool_start_recording(self._ptr)

    def stop_recording(self) -> None:
        _raise_if_closed(self)
        cmb_resourcepool_stop_recording(self._ptr)

    def history(self):
        _raise_if_closed(self)
        return _timeseries_copy(cmb_resourcepool_get_history(self._ptr))

    def close(self) -> None:
        if self._closed:
            return
        if self._ptr != NULL:
            cmb_resourcepool_destroy(self._ptr)
            self._ptr = NULL
        self._closed = True


cdef class Condition:
    """Condition variable with a Python demand predicate."""

    cdef cmb_condition *_ptr
    cdef public bint _closed

    def __cinit__(self):
        self._ptr = NULL
        self._closed = True

    def __init__(self, str name):
        cdef bytes bname = _name_bytes(name)
        self._ptr = cmb_condition_create()
        cmb_condition_initialize(self._ptr, bname)
        self._closed = False
        _register_with_active_simulation(self)

    def __dealloc__(self):
        if not self._closed:
            self.close()

    def wait(self, object predicate, object context=None) -> int:
        """Wait until ``predicate(process, context)`` returns true."""
        _raise_if_closed(self)
        if not callable(predicate):
            raise TypeError("predicate must be callable")
        cdef object demand_ctx = (predicate, context)
        return cmb_condition_wait(
            self._ptr,
            _condition_demand_trampoline,
            <const void *><PyObject *>demand_ctx,
        )

    def signal(self) -> int:
        _raise_if_closed(self)
        return <object>cmb_condition_signal(self._ptr)

    def close(self) -> None:
        if self._closed:
            return
        if self._ptr != NULL:
            cmb_condition_destroy(self._ptr)
            self._ptr = NULL
        self._closed = True


cdef DataSummary _datasummary_owner(cmb_datasummary *ptr):
    cdef DataSummary summary = DataSummary.__new__(DataSummary)
    summary._ptr = ptr
    summary._owner = True
    summary._closed = False
    return summary


cdef WeightedSummary _wtdsummary_owner(cmb_wtdsummary *ptr):
    cdef WeightedSummary summary = WeightedSummary.__new__(WeightedSummary)
    summary._ptr = ptr
    summary._owner = True
    summary._closed = False
    return summary


cdef TimeSeries _timeseries_copy(cmb_timeseries *ptr):
    cdef TimeSeries ts = TimeSeries.__new__(TimeSeries)
    ts._ptr = cmb_timeseries_create()
    cmb_timeseries_copy(ts._ptr, ptr)
    ts._owner = True
    ts._closed = False
    return ts


cdef class DataSummary:
    """Running unweighted sample summary."""

    cdef cmb_datasummary *_ptr
    cdef bint _owner
    cdef public bint _closed

    def __cinit__(self):
        self._ptr = NULL
        self._owner = False
        self._closed = True

    def __init__(self):
        if self._ptr == NULL:
            self._ptr = cmb_datasummary_create()
            self._owner = True
            self._closed = False

    def __dealloc__(self):
        if not self._closed:
            self.close()

    def add(self, double value) -> int:
        _raise_if_closed(self)
        return <object>cmb_datasummary_add(self._ptr, value)

    property count:
        def __get__(self):
            _raise_if_closed(self)
            return <object>cmb_datasummary_count(self._ptr)

    property min:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_datasummary_min(self._ptr)

    property max:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_datasummary_max(self._ptr)

    property mean:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_datasummary_mean(self._ptr)

    property variance:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_datasummary_variance(self._ptr)

    property stddev:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_datasummary_stddev(self._ptr)

    property skewness:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_datasummary_skewness(self._ptr)

    property kurtosis:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_datasummary_kurtosis(self._ptr)

    def close(self) -> None:
        if self._closed:
            return
        if self._owner and self._ptr != NULL:
            cmb_datasummary_destroy(self._ptr)
        self._ptr = NULL
        self._closed = True


cdef class WeightedSummary:
    """Running weighted sample summary."""

    cdef cmb_wtdsummary *_ptr
    cdef bint _owner
    cdef public bint _closed

    def __cinit__(self):
        self._ptr = NULL
        self._owner = False
        self._closed = True

    def __init__(self):
        if self._ptr == NULL:
            self._ptr = cmb_wtdsummary_create()
            self._owner = True
            self._closed = False

    def __dealloc__(self):
        if not self._closed:
            self.close()

    def add(self, double value, double weight=1.0) -> int:
        _raise_if_closed(self)
        return <object>cmb_wtdsummary_add(self._ptr, value, weight)

    property count:
        def __get__(self):
            _raise_if_closed(self)
            return <object>cmb_wtdsummary_count(self._ptr)

    property weight_sum:
        def __get__(self):
            _raise_if_closed(self)
            return self._ptr.wsum

    property min:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_wtdsummary_min(self._ptr)

    property max:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_wtdsummary_max(self._ptr)

    property mean:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_wtdsummary_mean(self._ptr)

    property variance:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_wtdsummary_variance(self._ptr)

    property stddev:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_wtdsummary_stddev(self._ptr)

    property skewness:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_wtdsummary_skewness(self._ptr)

    property kurtosis:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_wtdsummary_kurtosis(self._ptr)

    def close(self) -> None:
        if self._closed:
            return
        if self._owner and self._ptr != NULL:
            cmb_wtdsummary_destroy(self._ptr)
        self._ptr = NULL
        self._closed = True


cdef class Dataset:
    """Resizable unweighted data set of doubles."""

    cdef cmb_dataset *_ptr
    cdef public bint _closed

    def __cinit__(self):
        self._ptr = NULL
        self._closed = True

    def __init__(self):
        self._ptr = cmb_dataset_create()
        self._closed = False

    def __dealloc__(self):
        if not self._closed:
            self.close()

    def add(self, double value) -> int:
        _raise_if_closed(self)
        return <object>cmb_dataset_add(self._ptr, value)

    def values(self):
        _raise_if_closed(self)
        return [self._ptr.xa[i] for i in range(self._ptr.count)]

    def summary(self):
        _raise_if_closed(self)
        cdef cmb_datasummary *summary = cmb_datasummary_create()
        cmb_dataset_summarize(self._ptr, summary)
        return _datasummary_owner(summary)

    property count:
        def __get__(self):
            _raise_if_closed(self)
            return <object>cmb_dataset_count(self._ptr)

    property min:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_dataset_min(self._ptr)

    property max:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_dataset_max(self._ptr)

    property median:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_dataset_median(self._ptr)

    def close(self) -> None:
        if self._closed:
            return
        if self._ptr != NULL:
            cmb_dataset_destroy(self._ptr)
            self._ptr = NULL
        self._closed = True


cdef class TimeSeries:
    """Time-stamped data series. Resource histories return non-owning views."""

    cdef cmb_timeseries *_ptr
    cdef bint _owner
    cdef public bint _closed

    def __cinit__(self):
        self._ptr = NULL
        self._owner = False
        self._closed = True

    def __init__(self):
        if self._ptr == NULL:
            self._ptr = cmb_timeseries_create()
            self._owner = True
            self._closed = False

    def __dealloc__(self):
        if not self._closed:
            self.close()

    def add(self, double value, double time) -> int:
        _raise_if_closed(self)
        return <object>cmb_timeseries_add(self._ptr, value, time)

    def finalize(self, double time) -> int:
        _raise_if_closed(self)
        return <object>cmb_timeseries_finalize(self._ptr, time)

    def values(self):
        """Return ``(time, value, weight)`` tuples."""
        _raise_if_closed(self)
        return [
            (self._ptr.ta[i], self._ptr.ds.xa[i], self._ptr.wa[i])
            for i in range(self._ptr.ds.count)
        ]

    def summary(self):
        _raise_if_closed(self)
        cdef cmb_wtdsummary *summary = cmb_wtdsummary_create()
        cmb_timeseries_summarize(self._ptr, summary)
        return _wtdsummary_owner(summary)

    property count:
        def __get__(self):
            _raise_if_closed(self)
            return <object>cmb_timeseries_count(self._ptr)

    property min:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_timeseries_min(self._ptr)

    property max:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_timeseries_max(self._ptr)

    property median:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_timeseries_median(self._ptr)

    def close(self) -> None:
        if self._closed:
            return
        if self._owner and self._ptr != NULL:
            cmb_timeseries_destroy(self._ptr)
        self._ptr = NULL
        self._closed = True
