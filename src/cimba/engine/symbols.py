"""JIT symbol registration; native addresses stay in the engine boundary."""

from __future__ import annotations

import ctypes
from functools import lru_cache

from llvmlite import binding

from .library import load


@lru_cache(maxsize=None)
def register(name: str) -> None:
    symbol = getattr(load(), name)
    binding.add_symbol(name, ctypes.cast(symbol, ctypes.c_void_p).value)
