"""Typed re-export surface for the private Cython extension."""

from typing import Final

from .cimba import (
    gil_enabled as gil_enabled,
    native_version as native_version,
    run_experiment as run_experiment,
    run_native_experiment as run_native_experiment,
    set_native_thread_hooks as set_native_thread_hooks,
)
from .cmb_logger import (
    LOGGER_ERROR as LOGGER_ERROR,
    LOGGER_FATAL as LOGGER_FATAL,
    LOGGER_INFO as LOGGER_INFO,
    LOGGER_WARNING as LOGGER_WARNING,
    logger_flags_off as logger_flags_off,
    logger_flags_on as logger_flags_on,
)
from . import random as random

__version__: Final[str]

def version() -> str: ...
def use_threads(n: int) -> int: ...
