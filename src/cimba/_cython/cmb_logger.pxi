# This file is included by ../_cimba_native.pyx.

cdef extern from "nbshim.h":
    void cpy_logger_flags_on(uint32_t flags)
    void cpy_logger_flags_off(uint32_t flags)


def logger_flags_on(uint32_t flags) -> None:
    """Enable logger flags in this thread and subsequent model trials."""
    cpy_logger_flags_on(flags)


def logger_flags_off(uint32_t flags) -> None:
    """Disable logger flags in this thread and subsequent model trials."""
    cpy_logger_flags_off(flags)
