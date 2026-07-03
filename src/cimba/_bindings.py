"""Numba bindings for the native cimba symbols.

The compiled ``_cimba`` extension embeds libcimba plus the nbshim.c wrappers
(``cpy_*``, re-exporting upstream's static-inline helpers). Loading the
extension into LLVM makes those symbols visible to the JIT linker; each
``types.ExternalFunction`` below declares one symbol and its signature.
They are callable only from nopython-compiled code.
"""

import llvmlite.binding as _llvm
from numba import types

from . import _cimba_native
from ._cimba import ffi as _ffi

_llvm.load_library_permanently(_cimba_native.__file__)

_extern = types.ExternalFunction
_intp = types.intp
_void = types.void
_i64 = types.int64
_u32 = types.uint32
_u64 = types.uint64
_f64 = types.float64

# --- Event queue and simulation clock --------------------------------------
event_queue_initialize = _extern("cmb_event_queue_initialize", _void(_f64))
event_queue_execute = _extern("cmb_event_queue_execute", _void())
event_queue_terminate = _extern("cmb_event_queue_terminate", _void())
event_queue_clear = _extern("cmb_event_queue_clear", _void())
event_queue_count = _extern("cmb_event_queue_count", _u64())
event_schedule = _extern(
    "cmb_event_schedule", _u64(_intp, _intp, _intp, _f64, _i64))
event_current = _extern("cmb_event_current", _u64())
event_time = _extern("cmb_event_time", _f64(_u64))
event_priority = _extern("cmb_event_priority", _i64(_u64))
event_cancel = _extern("cpy_event_cancel", _u64(_u64))
event_reschedule = _extern("cpy_event_reschedule", _u64(_u64, _f64))
event_reprioritize = _extern("cpy_event_reprioritize", _u64(_u64, _i64))
event_is_scheduled = _extern("cpy_event_is_scheduled", _u64(_u64))
time = _extern("cmb_time", _f64())

# --- Random streams ---------------------------------------------------------
random_initialize = _extern("cmb_random_initialize", _void(_u64))
random_terminate = _extern("cmb_random_terminate", _void())
random_exponential = _extern("cpy_random_exponential", _f64(_f64))
random_gamma = _extern("cpy_random_gamma", _f64(_f64, _f64))
random01 = _extern("cpy_random01", _f64())
random_uniform = _extern("cpy_random_uniform", _f64(_f64, _f64))
random_normal = _extern("cpy_random_normal", _f64(_f64, _f64))
random_rayleigh = _extern("cpy_random_rayleigh", _f64(_f64))
random_pert = _extern("cpy_random_PERT", _f64(_f64, _f64, _f64))
random_pert_mod = _extern(
    "cpy_random_PERT_mod", _f64(_f64, _f64, _f64, _f64))
random_bernoulli = _extern("cpy_random_bernoulli", _u64(_f64))
random_flip = _extern("cpy_random_flip", _u64())
random_triangular = _extern("cpy_random_triangular", _f64(_f64, _f64, _f64))
random_weibull = _extern("cpy_random_weibull", _f64(_f64, _f64))
random_lognormal = _extern("cpy_random_lognormal", _f64(_f64, _f64))
random_erlang = _extern("cpy_random_erlang", _f64(_u64, _f64))
random_beta = _extern("cpy_random_beta", _f64(_f64, _f64, _f64, _f64))
random_poisson = _extern("cpy_random_poisson", _u64(_f64))
random_dice = _extern("cpy_random_dice", _i64(_i64, _i64))
random_std_normal = _extern("cpy_random_std_normal", _f64())
random_std_exponential = _extern("cpy_random_std_exponential", _f64())
random_std_gamma = _extern("cpy_random_std_gamma", _f64(_f64))
random_std_beta = _extern("cpy_random_std_beta", _f64(_f64, _f64))
random_logistic = _extern("cpy_random_logistic", _f64(_f64, _f64))
random_cauchy = _extern("cpy_random_cauchy", _f64(_f64, _f64))
random_pareto = _extern("cpy_random_pareto", _f64(_f64, _f64))
random_chisquared = _extern("cpy_random_chisquared", _f64(_f64))
random_f_dist = _extern("cpy_random_F_dist", _f64(_f64, _f64))
random_std_t = _extern("cpy_random_std_t_dist", _f64(_f64))
random_t = _extern("cpy_random_t_dist", _f64(_f64, _f64, _f64))
random_geometric = _extern("cpy_random_geometric", _u64(_f64))
random_binomial = _extern("cpy_random_binomial", _u64(_u64, _f64))
random_negative_binomial = _extern(
    "cpy_random_negative_binomial", _u64(_u64, _f64))
random_pascal = _extern("cpy_random_pascal", _u64(_u64, _f64))

# --- Processes ---------------------------------------------------------------
process_create = _extern("cmb_process_create", _intp())
process_create_sized = _extern("cpy_process_create_sized", _intp(_u64))
spawned_register = _extern("cpy_spawned_register", _void(_intp))
spawned_unregister = _extern("cpy_spawned_unregister", _u64(_intp))
spawned_stop_all = _extern("cpy_spawned_stop_all", _void())
spawned_reclaim = _extern("cpy_spawned_reclaim", _void())
process_initialize = _extern(
    "cmb_process_initialize", _void(_intp, _intp, _intp, _intp, _i64))
process_start = _extern("cmb_process_start", _void(_intp))
process_stop = _extern("cmb_process_stop", _i64(_intp, _intp))
process_terminate = _extern("cmb_process_terminate", _void(_intp))
process_destroy = _extern("cmb_process_destroy", _void(_intp))
process_hold = _extern("cmb_process_hold", _i64(_f64))
process_interrupt = _extern("cmb_process_interrupt", _void(_intp, _i64, _i64))
process_wait_process = _extern("cmb_process_wait_process", _i64(_intp))
process_wait_event = _extern("cmb_process_wait_event", _i64(_u64))
process_resume = _extern("cmb_process_resume", _void(_intp, _i64))
process_current = _extern("cpy_process_current", _intp())
process_status = _extern("cpy_process_status", _i64(_intp))
process_yield = _extern("cpy_process_yield", _i64())
process_priority_set = _extern("cmb_process_priority_set",
                               _void(_intp, _i64))
process_timer_set = _extern("cpy_process_timer_set", _u64(_intp, _f64, _i64))
process_timer_add = _extern("cmb_process_timer_add", _u64(_intp, _f64, _i64))
process_timer_cancel = _extern(
    "cpy_process_timer_cancel", _u64(_intp, _u64))
process_timers_clear = _extern("cmb_process_timers_clear", _void(_intp))

# --- Buffers: counted amounts -------------------------------------------------
buffer_create = _extern("cmb_buffer_create", _intp())
buffer_initialize = _extern("cmb_buffer_initialize", _void(_intp, _intp, _u64))
buffer_destroy = _extern("cmb_buffer_destroy", _void(_intp))
buffer_recording_start = _extern("cmb_buffer_recording_start", _void(_intp))
buffer_recording_stop = _extern("cmb_buffer_recording_stop", _void(_intp))
buffer_put = _extern("cpy_buffer_put", _i64(_intp, _u64))
buffer_get = _extern("cpy_buffer_get", _i64(_intp, _u64))
buffer_mean_level = _extern("cpy_buffer_mean_level", _f64(_intp))
buffer_history = _extern("cpy_buffer_history", _intp(_intp))
buffer_report_file = _extern("cpy_buffer_report_file",
                             _u64(_intp, _intp, _u64))
buffer_level = _extern("cpy_buffer_level", _u64(_intp))
buffer_space = _extern("cpy_buffer_space", _u64(_intp))

# --- Resources: single holder, priority-aware ----------------------------------
resource_create = _extern("cmb_resource_create", _intp())
resource_initialize = _extern("cmb_resource_initialize", _void(_intp, _intp))
resource_destroy = _extern("cmb_resource_destroy", _void(_intp))
resource_acquire = _extern("cmb_resource_acquire", _i64(_intp))
resource_release = _extern("cmb_resource_release", _void(_intp))
resource_preempt = _extern("cmb_resource_preempt", _i64(_intp))
resource_recording_start = _extern("cmb_resource_start_recording", _void(_intp))
resource_recording_stop = _extern("cmb_resource_stop_recording", _void(_intp))
resource_available = _extern("cpy_resource_available", _u64(_intp))
resource_in_use = _extern("cpy_resource_in_use", _u64(_intp))
resource_held = _extern("cpy_resource_held_by_process", _u64(_intp, _intp))
resource_mean_in_use = _extern("cpy_resource_mean_in_use", _f64(_intp))
resource_history = _extern("cpy_resource_history", _intp(_intp))
resource_report_file = _extern("cpy_resource_report_file",
                               _u64(_intp, _intp, _u64))

# --- Resource pools: capacity > 1 ----------------------------------------------
resourcepool_create = _extern("cmb_resourcepool_create", _intp())
resourcepool_initialize = _extern(
    "cmb_resourcepool_initialize", _void(_intp, _intp, _u64))
resourcepool_destroy = _extern("cmb_resourcepool_destroy", _void(_intp))
resourcepool_acquire = _extern("cmb_resourcepool_acquire", _i64(_intp, _u64))
resourcepool_preempt = _extern("cmb_resourcepool_preempt", _i64(_intp, _u64))
resourcepool_release = _extern("cmb_resourcepool_release", _void(_intp, _u64))
resourcepool_recording_start = _extern(
    "cmb_resourcepool_start_recording", _void(_intp))
resourcepool_recording_stop = _extern(
    "cmb_resourcepool_stop_recording", _void(_intp))
resourcepool_available = _extern("cpy_resourcepool_available", _u64(_intp))
resourcepool_held = _extern("cmb_resourcepool_held_by_process",
                            _u64(_intp, _intp))
resourcepool_in_use = _extern("cpy_resourcepool_in_use", _u64(_intp))
resourcepool_mean_in_use = _extern("cpy_resourcepool_mean_in_use", _f64(_intp))
resourcepool_history = _extern("cpy_resourcepool_history", _intp(_intp))
resourcepool_report_file = _extern("cpy_resourcepool_report_file",
                                   _u64(_intp, _intp, _u64))

# --- Object queues: FIFO of opaque int64 objects --------------------------------
objectqueue_create = _extern("cmb_objectqueue_create", _intp())
objectqueue_initialize = _extern(
    "cmb_objectqueue_initialize", _void(_intp, _intp, _u64))
objectqueue_destroy = _extern("cmb_objectqueue_destroy", _void(_intp))
objectqueue_put = _extern("cpy_objectqueue_put", _i64(_intp, _intp))
objectqueue_get = _extern("cpy_objectqueue_get", _i64(_intp, _intp))
objectqueue_take = _extern("cpy_objectqueue_take", _intp(_intp))
objectqueue_length = _extern("cpy_objectqueue_length", _u64(_intp))
objectqueue_space = _extern("cpy_objectqueue_space", _u64(_intp))
objectqueue_position = _extern("cmb_objectqueue_position", _u64(_intp, _intp))
objectqueue_recording_start = _extern(
    "cmb_objectqueue_recording_start", _void(_intp))
objectqueue_recording_stop = _extern(
    "cmb_objectqueue_recording_stop", _void(_intp))
objectqueue_mean_length = _extern("cpy_objectqueue_mean_length", _f64(_intp))
objectqueue_history = _extern("cpy_objectqueue_history", _intp(_intp))
objectqueue_report_file = _extern("cpy_objectqueue_report_file",
                                  _u64(_intp, _intp, _u64))

# --- Priority queues: entries ordered by priority, with cancellation -------------
priorityqueue_create = _extern("cmb_priorityqueue_create", _intp())
priorityqueue_initialize = _extern(
    "cmb_priorityqueue_initialize", _void(_intp, _intp, _u64))
priorityqueue_terminate = _extern("cmb_priorityqueue_terminate", _void(_intp))
priorityqueue_destroy = _extern("cmb_priorityqueue_destroy", _void(_intp))
priorityqueue_put = _extern("cpy_priorityqueue_put", _u64(_intp, _intp, _i64))
priorityqueue_take = _extern("cpy_priorityqueue_take", _intp(_intp))
priorityqueue_length = _extern("cpy_priorityqueue_length", _u64(_intp))
priorityqueue_space = _extern("cpy_priorityqueue_space", _u64(_intp))
priorityqueue_position = _extern(
    "cmb_priorityqueue_position", _u64(_intp, _u64))
priorityqueue_reprioritize = _extern(
    "cpy_priorityqueue_reprioritize", _void(_intp, _u64, _i64))
priorityqueue_cancel = _extern("cpy_priorityqueue_cancel", _u64(_intp, _u64))
priorityqueue_recording_start = _extern(
    "cmb_priorityqueue_recording_start", _void(_intp))
priorityqueue_recording_stop = _extern(
    "cmb_priorityqueue_recording_stop", _void(_intp))
priorityqueue_mean_length = _extern("cpy_priorityqueue_mean_length",
                                    _f64(_intp))
priorityqueue_history = _extern("cpy_priorityqueue_history", _intp(_intp))
priorityqueue_report_file = _extern("cpy_priorityqueue_report_file",
                                    _u64(_intp, _intp, _u64))

# --- Datasets: tally statistics --------------------------------------------------
dataset_create = _extern("cmb_dataset_create", _intp())
dataset_initialize = _extern("cmb_dataset_initialize", _void(_intp))
dataset_destroy = _extern("cmb_dataset_destroy", _void(_intp))
dataset_add = _extern("cmb_dataset_add", _u64(_intp, _f64))
dataset_reset = _extern("cmb_dataset_reset", _void(_intp))
dataset_mean = _extern("cpy_dataset_mean", _f64(_intp))
dataset_count = _extern("cpy_dataset_count", _u64(_intp))
dataset_min = _extern("cpy_dataset_min", _f64(_intp))
dataset_max = _extern("cpy_dataset_max", _f64(_intp))
dataset_std = _extern("cpy_dataset_stddev", _f64(_intp))
dataset_median = _extern("cpy_dataset_median", _f64(_intp))
dataset_quantile = _extern("cpy_dataset_quantile", _f64(_intp, _f64))
dataset_print_file = _extern("cpy_dataset_print_file",
                             _u64(_intp, _intp, _u64))
dataset_fivenum_file = _extern("cpy_dataset_fivenum_file",
                               _u64(_intp, _intp, _u64))
dataset_histogram_file = _extern("cpy_dataset_histogram_file",
                                 _u64(_intp, _intp, _u64, _u64, _f64, _f64))
dataset_correlogram_file = _extern("cpy_dataset_correlogram_file",
                                   _u64(_intp, _intp, _u64, _u64))
dataset_pacf_correlogram_file = _extern(
    "cpy_dataset_pacf_correlogram_file", _u64(_intp, _intp, _u64, _u64))

# --- Timeseries histories ---------------------------------------------------------
timeseries_count = _extern("cpy_timeseries_count", _u64(_intp))
timeseries_min = _extern("cpy_timeseries_min", _f64(_intp))
timeseries_max = _extern("cpy_timeseries_max", _f64(_intp))
timeseries_mean = _extern("cpy_timeseries_mean", _f64(_intp))
timeseries_std = _extern("cpy_timeseries_stddev", _f64(_intp))
timeseries_median = _extern("cpy_timeseries_median", _f64(_intp))
timeseries_print_file = _extern("cpy_timeseries_print_file",
                                _u64(_intp, _intp, _u64))
timeseries_fivenum_file = _extern("cpy_timeseries_fivenum_file",
                                  _u64(_intp, _intp, _u64))
timeseries_histogram_file = _extern(
    "cpy_timeseries_histogram_file", _u64(_intp, _intp, _u64, _u64, _f64, _f64))
timeseries_correlogram_file = _extern("cpy_timeseries_correlogram_file",
                                      _u64(_intp, _intp, _u64, _u64))
timeseries_pacf_correlogram_file = _extern(
    "cpy_timeseries_pacf_correlogram_file", _u64(_intp, _intp, _u64, _u64))

# --- Conditions -------------------------------------------------------------------
condition_create = _extern("cmb_condition_create", _intp())
condition_initialize = _extern("cmb_condition_initialize", _void(_intp, _intp))
condition_destroy = _extern("cmb_condition_destroy", _void(_intp))
condition_wait = _extern("cmb_condition_wait", _i64(_intp, _intp, _intp))
condition_signal = _extern("cmb_condition_signal", _u64(_intp))

# --- Logging --------------------------------------------------------------------
logger_apply_flags = _extern("cpy_logger_apply_flags", _void())
logger_user_msg = _extern("cpy_logger_user_msg", _void(_u32, _intp))
logger_user_i64 = _extern("cpy_logger_user_i64", _void(_u32, _intp, _i64))
logger_user_f64 = _extern("cpy_logger_user_f64", _void(_u32, _intp, _f64))


_keepalive: list[object] = []


def cstring(s: str) -> int:
    """Address of a NUL-terminated copy of ``s``, kept alive forever."""
    buf = _ffi.new("char[]", s.encode())
    _keepalive.append(buf)
    return int(_ffi.cast("intptr_t", buf))
