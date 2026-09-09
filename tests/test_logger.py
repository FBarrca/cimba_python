import pytest

import cimba


@pytest.mark.parametrize("flags", [
    cimba.LOGGER_FATAL,
    cimba.LOGGER_ERROR | cimba.LOGGER_WARNING | cimba.LOGGER_INFO,
    0x00000001,
])
def test_logger_flag_helpers_accept_native_and_user_masks(flags):
    cimba.logger_flags_off(flags)
    cimba.logger_flags_on(flags)


@pytest.mark.parametrize("flags", [-1, 1 << 32])
def test_logger_flag_helpers_reject_masks_outside_uint32(flags):
    with pytest.raises(OverflowError):
        cimba.logger_flags_off(flags)
    with pytest.raises(OverflowError):
        cimba.logger_flags_on(flags)
