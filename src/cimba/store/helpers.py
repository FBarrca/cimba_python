"""Grouped entity-method helpers for the ``env.<entity>.method(...)`` sugar.

Each function is a thin njit-callable wrapper over the matching native
binding, named ``<kind>_<method>``; ``methods.py`` looks these up by name
to build the lowered calls. ``report()``/``get()``/``take()`` forms that
the raw bindings don't expose directly (stdout report, tuple-returning
get) are composed here exactly like their ``cimba.sim`` counterparts.
"""

import numpy as _np

from numba import njit

from .. import _bindings as _b
from .._declarations import Handle
from .._intrinsics import pq_get as _pq_get
from .._intrinsics import record_addr as _record_addr
from .._intrinsics import store_get as _store_get

# --- Queue (cmb_buffer) -------------------------------------------------------
queue_put = _b.buffer_put
queue_get = _b.buffer_get
queue_level = _b.buffer_level
queue_space = _b.buffer_space
queue_mean_level = _b.buffer_mean_level
queue_report_file = _b.buffer_report_file


@njit
def queue_report(queue: Handle) -> int:
    return queue_report_file(queue, 0, _np.uint64(1))


# --- Resource (cmb_resource) ---------------------------------------------------
resource_acquire = _b.resource_acquire
resource_release = _b.resource_release
resource_preempt = _b.resource_preempt
resource_available = _b.resource_available
resource_in_use = _b.resource_in_use
resource_held = _b.resource_held
resource_mean_in_use = _b.resource_mean_in_use
resource_report_file = _b.resource_report_file


@njit
def resource_report(resource: Handle) -> int:
    return resource_report_file(resource, 0, _np.uint64(1))


# --- Pool (cmb_resourcepool) ---------------------------------------------------
pool_acquire = _b.resourcepool_acquire
pool_release = _b.resourcepool_release
pool_preempt = _b.resourcepool_preempt
pool_available = _b.resourcepool_available
pool_held = _b.resourcepool_held
pool_in_use = _b.resourcepool_in_use
pool_mean_in_use = _b.resourcepool_mean_in_use
pool_report_file = _b.resourcepool_report_file


@njit
def pool_report(pool: Handle) -> int:
    return pool_report_file(pool, 0, _np.uint64(1))


# --- Store (cmb_objectqueue) ---------------------------------------------------
store_put = _b.objectqueue_put
store_take = _b.objectqueue_take
store_length = _b.objectqueue_length
store_space = _b.objectqueue_space
store_position = _b.objectqueue_position
store_mean_length = _b.objectqueue_mean_length
store_report_file = _b.objectqueue_report_file


@njit
def store_get(store: Handle) -> tuple:
    return _store_get(store)


@njit
def store_report(store: Handle) -> int:
    return store_report_file(store, 0, _np.uint64(1))


# --- PQueues element (cmb_priorityqueue) ---------------------------------------
pq_put = _b.priorityqueue_put
pq_take = _b.priorityqueue_take
pq_length = _b.priorityqueue_length
pq_space = _b.priorityqueue_space
pq_position = _b.priorityqueue_position
pq_reprioritize = _b.priorityqueue_reprioritize
pq_cancel = _b.priorityqueue_cancel
pq_mean_length = _b.priorityqueue_mean_length
pq_report_file = _b.priorityqueue_report_file


@njit
def pq_get(pqueue: Handle) -> tuple:
    return _pq_get(pqueue)


@njit
def pq_report(pqueue: Handle) -> int:
    return pq_report_file(pqueue, 0, _np.uint64(1))


# --- Condition (cmb_condition) --------------------------------------------------
condition_signal = _b.condition_signal
_condition_wait = _b.condition_wait


@njit
def condition_wait(condition: Handle, predicate: int, env) -> int:
    return _condition_wait(condition, predicate, _record_addr(env))
