"""Typed surface for Cimba's cmb_logger module."""

from typing import Final

from ._types import _LoggerFlags

LOGGER_FATAL: Final[int]
"""Fatal logger flag; fatal messages abort the program."""
LOGGER_ERROR: Final[int]
"""Error logger flag."""
LOGGER_WARNING: Final[int]
"""Warning logger flag."""
LOGGER_INFO: Final[int]
"""Internal informational logger flag."""

def logger_flags_on(flags: _LoggerFlags) -> None:
    """Enable one or more Cimba logger flags in the current thread."""
    ...

def logger_flags_off(flags: _LoggerFlags) -> None:
    """Disable one or more Cimba logger flags in the current thread."""
    ...
