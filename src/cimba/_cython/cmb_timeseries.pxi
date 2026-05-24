# This file is included by ../_cimba.pyx.

cdef TimeSeries _timeseries_copy(cmb_timeseries *ptr):
    cdef TimeSeries ts = TimeSeries.__new__(TimeSeries)
    ts._ptr = cmb_timeseries_create()
    cdef cmb_timeseries *tgt = ts._ptr
    with nogil:
        cmb_timeseries_copy(tgt, ptr)
    ts._owner = True
    ts._closed = False
    return ts


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
        cdef Py_ssize_t count = <Py_ssize_t>self._ptr.ds.count
        cdef Py_ssize_t i
        cdef list result = [None] * count
        for i in range(count):
            result[i] = (self._ptr.ta[i], self._ptr.ds.xa[i], self._ptr.wa[i])
        return result

    def summary(self):
        _raise_if_closed(self)
        cdef cmb_wtdsummary *summary = cmb_wtdsummary_create()
        cdef cmb_timeseries *src = self._ptr
        with nogil:
            cmb_timeseries_summarize(src, summary)
        return _wtdsummary_owner(summary)

    def reset(self) -> None:
        _raise_if_closed(self)
        cmb_timeseries_reset(self._ptr)

    def copy(self):
        _raise_if_closed(self)
        cdef TimeSeries copied = TimeSeries()
        cdef cmb_timeseries *src = self._ptr
        cdef cmb_timeseries *tgt = copied._ptr
        with nogil:
            cmb_timeseries_copy(tgt, src)
        return copied

    def sort_by_value(self) -> None:
        _raise_if_closed(self)
        cdef cmb_timeseries *src = self._ptr
        with nogil:
            cmb_timeseries_sort_x(src)

    def sort_by_time(self) -> None:
        _raise_if_closed(self)
        cdef cmb_timeseries *src = self._ptr
        with nogil:
            cmb_timeseries_sort_t(src)

    def acf(self, object lags):
        _raise_if_closed(self)
        cdef uint16_t n = _lags_to_u16(lags)
        cdef double *values
        cdef list result
        cdef uint16_t i
        cdef cmb_timeseries *src = self._ptr
        if self._ptr.ds.count < 2:
            raise ValueError("at least two samples are required")
        if n == 0:
            return [1.0]
        if n >= self._ptr.ds.count:
            raise ValueError("lags must be smaller than sample count")
        values = <double *>PyMem_Malloc((<size_t>n + 1) * sizeof(double))
        if values == NULL:
            raise MemoryError()
        try:
            with nogil:
                cmb_timeseries_ACF(src, n, values)
            result = [0.0] * (n + 1)
            with cython.boundscheck(False):
                for i in range(n + 1):
                    result[i] = values[i]
            return result
        finally:
            PyMem_Free(values)

    def pacf(self, object lags):
        _raise_if_closed(self)
        cdef uint16_t n = _lags_to_u16(lags)
        cdef double *values
        cdef list result
        cdef uint16_t i
        cdef cmb_timeseries *src = self._ptr
        if self._ptr.ds.count < 3:
            raise ValueError("at least three samples are required")
        if n == 0:
            return [1.0]
        if n >= self._ptr.ds.count - 1:
            raise ValueError("lags must be smaller than sample count minus one")
        values = <double *>PyMem_Malloc((<size_t>n + 1) * sizeof(double))
        if values == NULL:
            raise MemoryError()
        try:
            with nogil:
                cmb_timeseries_PACF(src, n, values, NULL)
            result = [0.0] * (n + 1)
            with cython.boundscheck(False):
                for i in range(n + 1):
                    result[i] = values[i]
            return result
        finally:
            PyMem_Free(values)

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
