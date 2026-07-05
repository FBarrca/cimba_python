# This file is included by ../_cimba.pyx.

import os as _os


cdef bytes _dataset_path_bytes(object path):
    cdef bytes encoded = _os.fsencode(path)
    if len(encoded) == 0:
        raise ValueError("path must not be empty")
    if b"\0" in encoded:
        raise ValueError("path must not contain NUL bytes")
    return encoded


cdef inline uint64_t _append_to_u64(object append) except *:
    return 1 if append else 0


cdef inline uint64_t _bins_to_u64(object bins) except *:
    return _u64_value(bins, "bins", 1)


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
        cdef Py_ssize_t count = <Py_ssize_t>self._ptr.count
        cdef Py_ssize_t i
        cdef list result = [None] * count
        for i in range(count):
            result[i] = self._ptr.xa[i]
        return result

    def summary(self):
        _raise_if_closed(self)
        cdef cmb_datasummary *summary = cmb_datasummary_create()
        cdef cmb_dataset *src = self._ptr
        with nogil:
            cmb_dataset_summarize(src, summary)
        return _datasummary_owner(summary)

    def reset(self) -> None:
        _raise_if_closed(self)
        cmb_dataset_reset(self._ptr)

    def copy(self):
        _raise_if_closed(self)
        cdef Dataset copied = Dataset()
        cdef cmb_dataset *src = self._ptr
        cdef cmb_dataset *tgt = copied._ptr
        with nogil:
            cmb_dataset_copy(tgt, src)
        return copied

    def merge(self, Dataset other):
        _raise_if_closed(self)
        _raise_if_closed(other)
        cdef Dataset merged = Dataset()
        cdef cmb_dataset *left = self._ptr
        cdef cmb_dataset *right = other._ptr
        cdef cmb_dataset *tgt = merged._ptr
        cdef uint64_t i
        with nogil:
            cmb_dataset_copy(tgt, left)
            for i in range(right.count):
                cmb_dataset_add(tgt, right.xa[i])
        return merged

    def sort(self) -> None:
        _raise_if_closed(self)
        cdef cmb_dataset *src = self._ptr
        with nogil:
            cmb_dataset_sort(src)

    def acf(self, object lags):
        _raise_if_closed(self)
        cdef uint16_t n = _lags_to_u16(lags)
        cdef double *values
        cdef list result
        cdef uint16_t i
        cdef cmb_dataset *src = self._ptr
        if self._ptr.count < 2:
            raise ValueError("at least two samples are required")
        if n == 0:
            return [1.0]
        if n >= self._ptr.count:
            raise ValueError("lags must be smaller than sample count")
        values = <double *>PyMem_Malloc((<size_t>n + 1) * sizeof(double))
        if values == NULL:
            raise MemoryError()
        try:
            with nogil:
                cmb_dataset_ACF(src, <unsigned int>n, values)
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
        cdef cmb_dataset *src = self._ptr
        if self._ptr.count < 3:
            raise ValueError("at least three samples are required")
        if n == 0:
            return [1.0]
        if n >= self._ptr.count - 1:
            raise ValueError("lags must be smaller than sample count minus one")
        values = <double *>PyMem_Malloc((<size_t>n + 1) * sizeof(double))
        if values == NULL:
            raise MemoryError()
        try:
            with nogil:
                cmb_dataset_PACF(src, <unsigned int>n, values, NULL)
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
            return <object>cmb_dataset_count(self._ptr)

    property min:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_dataset_min(self._ptr)

    property max:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_dataset_max(self._ptr)

    property mean:
        def __get__(self):
            _raise_if_closed(self)
            return cpy_dataset_mean(self._ptr)

    property std:
        def __get__(self):
            _raise_if_closed(self)
            return cpy_dataset_stddev(self._ptr)

    property stddev:
        def __get__(self):
            _raise_if_closed(self)
            return cpy_dataset_stddev(self._ptr)

    property median:
        def __get__(self):
            _raise_if_closed(self)
            return cmb_dataset_median(self._ptr)

    def quantile(self, double q) -> float:
        _raise_if_closed(self)
        return cpy_dataset_quantile(self._ptr, q)

    def print(self) -> int:
        _raise_if_closed(self)
        return <object>cpy_dataset_print_file(self._ptr, <intptr_t>0, 1)

    def print_file(self, object path, object append=True) -> int:
        _raise_if_closed(self)
        cdef bytes path_bytes = _dataset_path_bytes(path)
        cdef const char *path_ptr = path_bytes
        return <object>cpy_dataset_print_file(
            self._ptr,
            <intptr_t>path_ptr,
            _append_to_u64(append),
        )

    def fivenum(self) -> int:
        _raise_if_closed(self)
        return <object>cpy_dataset_fivenum_file(self._ptr, <intptr_t>0, 1)

    def fivenum_file(self, object path, object append=True) -> int:
        _raise_if_closed(self)
        cdef bytes path_bytes = _dataset_path_bytes(path)
        cdef const char *path_ptr = path_bytes
        return <object>cpy_dataset_fivenum_file(
            self._ptr,
            <intptr_t>path_ptr,
            _append_to_u64(append),
        )

    def histogram(self, object bins=20, double low=0.0, double high=0.0) -> int:
        _raise_if_closed(self)
        return <object>cpy_dataset_histogram_file(
            self._ptr,
            <intptr_t>0,
            1,
            _bins_to_u64(bins),
            low,
            high,
        )

    def histogram_file(self, object path, object append=True, object bins=20,
                       double low=0.0, double high=0.0) -> int:
        _raise_if_closed(self)
        cdef bytes path_bytes = _dataset_path_bytes(path)
        cdef const char *path_ptr = path_bytes
        return <object>cpy_dataset_histogram_file(
            self._ptr,
            <intptr_t>path_ptr,
            _append_to_u64(append),
            _bins_to_u64(bins),
            low,
            high,
        )

    def correlogram(self, object lags=20) -> int:
        _raise_if_closed(self)
        return <object>cpy_dataset_correlogram_file(
            self._ptr,
            <intptr_t>0,
            1,
            _lags_to_u16(lags),
        )

    def correlogram_file(self, object path, object append=True,
                         object lags=20) -> int:
        _raise_if_closed(self)
        cdef bytes path_bytes = _dataset_path_bytes(path)
        cdef const char *path_ptr = path_bytes
        return <object>cpy_dataset_correlogram_file(
            self._ptr,
            <intptr_t>path_ptr,
            _append_to_u64(append),
            _lags_to_u16(lags),
        )

    def pacf_correlogram(self, object lags=20) -> int:
        _raise_if_closed(self)
        return <object>cpy_dataset_pacf_correlogram_file(
            self._ptr,
            <intptr_t>0,
            1,
            _lags_to_u16(lags),
        )

    def pacf_correlogram_file(self, object path, object append=True,
                              object lags=20) -> int:
        _raise_if_closed(self)
        cdef bytes path_bytes = _dataset_path_bytes(path)
        cdef const char *path_ptr = path_bytes
        return <object>cpy_dataset_pacf_correlogram_file(
            self._ptr,
            <intptr_t>path_ptr,
            _append_to_u64(append),
            _lags_to_u16(lags),
        )

    def close(self) -> None:
        if self._closed:
            return
        if self._ptr != NULL:
            cmb_dataset_destroy(self._ptr)
            self._ptr = NULL
        self._closed = True
