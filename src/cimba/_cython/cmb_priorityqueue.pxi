# This file is included by ../_cimba.pyx.

@cython.freelist(1024)
cdef class _PriorityQueueEntry:
    cdef uint64_t handle
    cdef object obj

    def __init__(self, object obj):
        self.handle = 0
        self.obj = obj


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

    def put(self, object obj, object priority=0):
        """Put an object and return ``(signal, handle)``."""
        if self._closed:
            raise RuntimeError("PriorityQueue is closed")
        cdef _PriorityQueueEntry entry = _PriorityQueueEntry(obj)
        cdef PyObject *op = <PyObject *>entry
        cdef uint64_t handle = 0
        cdef int64_t pri = 0 if priority is _ZERO else _priority_to_i64(priority)
        Py_INCREF(<object>op)
        cdef int64_t sig = cmb_priorityqueue_put(
            self._ptr,
            <void *>op,
            pri,
            &handle,
        )
        if sig != SUCCESS:
            Py_DECREF(<object>op)
        else:
            entry.handle = handle
            self._objects[<object>handle] = entry
        if sig == _PROCESS_CANCEL_SIGNAL:
            raise _ProcessCancelled()
        return <object>sig, <object>handle

    def get(self):
        """Get the highest-priority object. Returns ``(signal, object)``."""
        if self._closed:
            raise RuntimeError("PriorityQueue is closed")
        cdef void *ptr = NULL
        cdef int64_t sig = cmb_priorityqueue_get(self._ptr, &ptr)
        cdef _PriorityQueueEntry entry
        if sig != SUCCESS:
            if sig == _PROCESS_CANCEL_SIGNAL:
                raise _ProcessCancelled()
            return sig, None
        entry = <_PriorityQueueEntry>_object_from_owned_pointer(ptr)
        self._objects.pop(<object>entry.handle, None)
        return sig, entry.obj

    def position(self, object handle) -> int:
        _raise_if_closed(self)
        return <object>cmb_priorityqueue_position(self._ptr, _handle_to_u64(handle))

    def cancel(self, object handle) -> bool:
        _raise_if_closed(self)
        cdef uint64_t h = _handle_to_u64(handle)
        cdef bint found = cmb_priorityqueue_cancel(self._ptr, h)
        cdef object entry
        if found:
            entry = self._objects.pop(<object>h, _MISSING)
            if entry is not _MISSING:
                Py_DECREF(<object>entry)
        return True if found else False

    def reprioritize(self, object handle, object priority) -> None:
        _raise_if_closed(self)
        cmb_priorityqueue_reprioritize(self._ptr, _handle_to_u64(handle), _priority_to_i64(priority))

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
        cdef _PriorityQueueEntry entry
        while self._ptr != NULL and cmb_priorityqueue_length(self._ptr) > 0:
            ptr = NULL
            if cmb_priorityqueue_get(self._ptr, &ptr) == SUCCESS and ptr != NULL:
                entry = <_PriorityQueueEntry>_object_from_owned_pointer(ptr)
                self._objects.pop(<object>entry.handle, None)

    def close(self) -> None:
        if self._closed:
            return
        if self._ptr != NULL:
            self._decref_queued_objects()
            self._objects.clear()
            cmb_priorityqueue_destroy(self._ptr)
            self._ptr = NULL
        self._closed = True
