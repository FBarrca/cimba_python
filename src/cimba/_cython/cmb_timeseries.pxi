# This file is included by ../_cimba.pyx.

cdef TimeSeries _timeseries_copy(cmb_timeseries *ptr):
    cdef TimeSeries ts = TimeSeries.__new__(TimeSeries)
    ts._ptr = cmb_timeseries_create()
    cmb_timeseries_copy(ts._ptr, ptr)
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
