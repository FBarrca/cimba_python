"""Configuration for the compiled model runtime."""

from typing import Final

from . import random as random

LOGGER_FATAL: Final[int]
LOGGER_ERROR: Final[int]
LOGGER_WARNING: Final[int]
LOGGER_INFO: Final[int]
__version__: Final[str]

def native_version() -> str: ...
def version() -> str: ...
def logger_flags_on(flags: int) -> None: ...
def logger_flags_off(flags: int) -> None: ...
def use_threads(n: int) -> int: ...
