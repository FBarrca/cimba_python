"""Typed surface for Cimba's cmb_logger module."""

from typing import Final

LOGGER_FATAL: Final[int]
"""Fatal logger flag; fatal messages abort the program."""
LOGGER_ERROR: Final[int]
"""Error logger flag."""
LOGGER_WARNING: Final[int]
"""Warning logger flag."""
LOGGER_INFO: Final[int]
"""Internal informational logger flag."""

def logger_flags_on(flags: int) -> None:
    """Enable logger flags in this thread and subsequent model trials."""
    ...

def logger_flags_off(flags: int) -> None:
    """Disable logger flags in this thread and subsequent model trials."""
    ...
