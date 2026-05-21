# This file is included by ../_cimba.pyx.

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


def native_version() -> str:
    """Return the version of the underlying Cimba C library."""
    cdef const char *v = cimba_version()
    return v.decode("utf-8")
