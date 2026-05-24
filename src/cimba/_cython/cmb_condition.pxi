# This file is included by ../_cimba.pyx.

cdef bool _condition_demand_trampoline(
    const cmb_condition *cnd,
    const cmb_process *prc,
    const void *ctx,
) noexcept with gil:
    cdef object demand_ctx = <object><PyObject *>ctx
    cdef object predicate = demand_ctx[0]
    cdef object user_ctx = demand_ctx[1]
    cdef Process proc = _process_registry_get().get(<uintptr_t>prc)

    try:
        return True if predicate(proc, user_ctx) else False
    except BaseException as exc:
        _record_exception(exc)
        return False


cdef int _CONDITION_GUARD_SINGLE = 0
cdef int _CONDITION_GUARD_FRONT = 1
cdef int _CONDITION_GUARD_REAR = 2


cdef object _condition_guard_codes(object source, object on):
    if isinstance(source, Resource) or isinstance(source, ResourcePool):
        if on is not None:
            raise ValueError('on is only supported for Buffer, ObjectQueue, and PriorityQueue')
        return (_CONDITION_GUARD_SINGLE,)

    if (
        isinstance(source, Buffer)
        or isinstance(source, ObjectQueue)
        or isinstance(source, PriorityQueue)
    ):
        if on is None:
            return (_CONDITION_GUARD_FRONT, _CONDITION_GUARD_REAR)
        if on == "front":
            return (_CONDITION_GUARD_FRONT,)
        if on == "rear":
            return (_CONDITION_GUARD_REAR,)
        raise ValueError('on must be None, "front", or "rear"')

    raise TypeError(
        "source must be a Resource, ResourcePool, Buffer, ObjectQueue, or PriorityQueue"
    )


cdef cmb_resourceguard *_condition_tracked_guard(object source, int guard) noexcept:
    cdef Resource resource
    cdef ResourcePool pool
    cdef Buffer buffer
    cdef ObjectQueue object_queue
    cdef PriorityQueue priority_queue

    if isinstance(source, Resource):
        resource = <Resource>source
        if resource._closed or resource._ptr == NULL or guard != _CONDITION_GUARD_SINGLE:
            return NULL
        return &resource._ptr.guard

    if isinstance(source, ResourcePool):
        pool = <ResourcePool>source
        if pool._closed or pool._ptr == NULL or guard != _CONDITION_GUARD_SINGLE:
            return NULL
        return &pool._ptr.guard

    if isinstance(source, Buffer):
        buffer = <Buffer>source
        if buffer._closed or buffer._ptr == NULL:
            return NULL
        if guard == _CONDITION_GUARD_FRONT:
            return &buffer._ptr.front_guard
        if guard == _CONDITION_GUARD_REAR:
            return &buffer._ptr.rear_guard
        return NULL

    if isinstance(source, ObjectQueue):
        object_queue = <ObjectQueue>source
        if object_queue._closed or object_queue._ptr == NULL:
            return NULL
        if guard == _CONDITION_GUARD_FRONT:
            return &object_queue._ptr.front_guard
        if guard == _CONDITION_GUARD_REAR:
            return &object_queue._ptr.rear_guard
        return NULL

    if isinstance(source, PriorityQueue):
        priority_queue = <PriorityQueue>source
        if priority_queue._closed or priority_queue._ptr == NULL:
            return NULL
        if guard == _CONDITION_GUARD_FRONT:
            return &priority_queue._ptr.front_guard
        if guard == _CONDITION_GUARD_REAR:
            return &priority_queue._ptr.rear_guard

    return NULL


cdef cmb_resourceguard *_condition_source_guard(object source, int guard) except NULL:
    cdef cmb_resourceguard *rgp
    _raise_if_closed(source)
    rgp = _condition_tracked_guard(source, guard)
    if rgp == NULL:
        raise ValueError("invalid condition subscription guard")
    return rgp


cdef class Condition:
    """Condition variable with a Python demand predicate."""

    cdef cmb_condition *_ptr
    cdef public bint _closed
    cdef object _subscriptions

    def __cinit__(self):
        self._ptr = NULL
        self._closed = True
        self._subscriptions = None

    def __init__(self, str name):
        cdef bytes bname = _name_bytes(name)
        self._ptr = cmb_condition_create()
        cmb_condition_initialize(self._ptr, bname)
        self._subscriptions = set()
        self._closed = False
        _register_with_active_simulation(self)

    def __dealloc__(self):
        if not self._closed:
            self.close()

    def wait(self, object predicate, object context=None) -> int:
        """Wait until ``predicate(process, context)`` returns true."""
        if self._closed:
            raise RuntimeError("Condition is closed")
        if not callable(predicate):
            raise TypeError("predicate must be callable")
        cdef object demand_ctx = (predicate, context)
        cdef int64_t sig = cmb_condition_wait(
            self._ptr,
            _condition_demand_trampoline,
            <const void *><PyObject *>demand_ctx,
        )
        if sig == _PROCESS_CANCEL_SIGNAL:
            raise _ProcessCancelled()
        return <object>sig

    def signal(self) -> int:
        _raise_if_closed(self)
        return <object>cmb_condition_signal(self._ptr)

    def subscribe(self, *sources, on=None):
        """Forward native resource-guard signals from ``sources`` to this condition."""
        cdef object source
        cdef object guard
        cdef object key
        cdef object codes
        cdef cmb_resourceguard *source_guard
        cdef cmb_resourceguard *condition_guard

        _raise_if_closed(self)
        condition_guard = &self._ptr.guard
        for source in sources:
            codes = _condition_guard_codes(source, on)
            for guard in codes:
                key = (source, guard)
                if key in self._subscriptions:
                    continue
                source_guard = _condition_source_guard(source, <int>guard)
                cmb_resourceguard_register(source_guard, condition_guard)
                self._subscriptions.add(key)
        return self

    def unsubscribe(self, *sources, on=None) -> int:
        """Stop forwarding native resource-guard signals from ``sources``."""
        cdef object source
        cdef object guard
        cdef object key
        cdef object codes
        cdef list keys

        _raise_if_closed(self)
        if len(sources) == 0:
            if on is not None:
                raise ValueError("sources are required when on is provided")
            keys = list(self._subscriptions)
        else:
            keys = []
            for source in sources:
                codes = _condition_guard_codes(source, on)
                for guard in codes:
                    key = (source, guard)
                    if key in self._subscriptions:
                        keys.append(key)
        return <object>self._unsubscribe_keys(keys)

    def cancel(self, Process process) -> bool:
        """Remove ``process`` from this condition and wake it with ``CANCELLED``."""
        cdef uint64_t removed
        _raise_if_closed(self)
        _raise_if_closed(process)
        removed = cmi_hashheap_pattern_cancel(
            <cmi_hashheap *>&self._ptr.guard,
            <const void *>process._ptr,
            <const void *>CMI_ANY_ITEM,
            <const void *>CMI_ANY_ITEM,
            <const void *>CMI_ANY_ITEM,
        )
        if removed != 0:
            cmb_process_resume(process._ptr, CANCELLED)
            return True
        return False

    def remove(self, Process process) -> bool:
        """Remove ``process`` from this condition without waking it."""
        cdef uint64_t removed
        _raise_if_closed(self)
        _raise_if_closed(process)
        removed = cmi_hashheap_pattern_cancel(
            <cmi_hashheap *>&self._ptr.guard,
            <const void *>process._ptr,
            <const void *>CMI_ANY_ITEM,
            <const void *>CMI_ANY_ITEM,
            <const void *>CMI_ANY_ITEM,
        )
        return True if removed != 0 else False

    cdef int _unsubscribe_keys(self, object keys) except *:
        cdef int count = 0
        cdef int guard
        cdef object key
        cdef object source
        cdef cmb_resourceguard *source_guard

        if self._subscriptions is None:
            return 0
        for key in keys:
            if key not in self._subscriptions:
                continue
            self._subscriptions.remove(key)
            count += 1
            if self._ptr == NULL:
                continue
            source = key[0]
            guard = <int>key[1]
            source_guard = _condition_tracked_guard(source, guard)
            if source_guard != NULL:
                cmb_resourceguard_unregister(source_guard, &self._ptr.guard)
        return count

    def close(self) -> None:
        if self._closed:
            return
        if self._subscriptions is not None:
            self._unsubscribe_keys(list(self._subscriptions))
        if self._ptr != NULL:
            cmb_condition_destroy(self._ptr)
            self._ptr = NULL
        self._closed = True
