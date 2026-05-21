# This file is included by ../_cimba.pyx.

def logger_flags_on(int flags) -> None:
    """Turn on Cimba logger flags in the current thread."""
    cmb_logger_flags_on(<uint32_t>flags)


def logger_flags_off(int flags) -> None:
    """Turn off Cimba logger flags in the current thread."""
    cmb_logger_flags_off(<uint32_t>flags)
