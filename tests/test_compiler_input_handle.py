"""A compiled class method uses the same typed input handle for every source."""

import ctypes
from typing import Any, cast

import numpy as np
from numba import njit

import cimba.compiler.handles  # register the modeling language with Numba
from cimba.engine.abi import INPUT_SLOT
from cimba.engine.library import load
from cimba.modeling import Input


class _Consumer:
    source: Input[float]

    def consume(self):
        return self.source.next() + self.source.next()


def test_compiled_input_handle_is_source_agnostic():
    record = np.zeros(1, dtype=np.dtype([("source", INPUT_SLOT)], align=True))
    row = np.asarray([2.0, 3.0], dtype=np.float64)
    record["source"]["kind"] = 2
    record["source"]["data"] = row.ctypes.data
    record["source"]["length"] = 2
    compiled = cast(Any, njit(_Consumer.consume))
    assert compiled(record[0]) == 5.0
    assert len(compiled.signatures) == 1

    record["source"] = np.zeros(1, dtype=INPUT_SLOT)[0]
    record["source"]["kind"] = 1
    record["source"]["distribution"] = 1
    record["source"]["parameters"][0, 0] = 2.0
    library = load()
    library.cpy_input_seed.argtypes = (ctypes.c_void_p, ctypes.c_uint64)
    assert record.dtype.fields is not None
    address = record.ctypes.data + record.dtype.fields["source"][1]
    library.cpy_input_seed(address, 42)
    assert compiled(record[0]) > 0
    assert len(compiled.signatures) == 1
