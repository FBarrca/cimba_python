# This file is included by ../_cimba.pyx.

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

