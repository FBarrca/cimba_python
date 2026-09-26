"""Object-addressed discrete-event models and source-agnostic inputs."""

from . import analysis, inputs, random
from .modeling import (
    Condition, Container, Dataset, Input, Model, Output, Param, PriorityStore,
    Process, Ref, Resource, Scheduled, Series, State, Store, end_trial, event,
    hold, log, now, on_end, on_start, predicate, process, release, schedule,
    spawn, suspend, sweep, sweeps, this_process,
)
from .experiments import Experiment, Window
from .results import Results, Samples, Signal

__version__ = "0.7.0"

__all__ = [
    "Model", "Param", "State", "Output", "Input", "Series", "Ref",
    "Container", "Store", "PriorityStore", "Resource", "Condition",
    "Dataset", "Process", "Scheduled", "process", "on_start", "on_end",
    "predicate", "event", "hold", "now", "suspend", "spawn", "release",
    "schedule", "this_process", "log", "end_trial", "sweep", "sweeps",
    "Experiment", "Window", "Results", "Samples", "Signal", "inputs",
    "random", "analysis", "engine_version", "set_engine_log_level",
    "cache_info", "clear_cache", "__version__",
]


def engine_version() -> str:
    """Version of the bundled Cimba engine."""
    import ctypes
    from .engine.library import load

    library = load()
    library.cimba_version.restype = ctypes.c_char_p
    return library.cimba_version().decode("utf-8")


def set_engine_log_level(flags: int) -> None:
    """Set the native engine's category bitmask for subsequent runs."""
    import ctypes
    from .engine.library import load

    if not 0 <= flags <= 0xFFFFFFFF:
        raise ValueError("log flags must fit a 32-bit unsigned integer")
    library = load()
    library.cpy_logger_flags_off.argtypes = (ctypes.c_uint32,)
    library.cpy_logger_flags_on.argtypes = (ctypes.c_uint32,)
    library.cpy_logger_flags_off(0xFFFFFFFF)
    if flags:
        library.cpy_logger_flags_on(flags)


def cache_info():
    """Report in-memory class compilation cache usage."""
    from .compiler.classes import ensure

    return ensure.cache_info()


def clear_cache() -> None:
    """Discard compiled class callbacks after current runs have finished."""
    from .compiler.classes import ensure

    ensure.cache_clear()
