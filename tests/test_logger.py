import cimba


def test_logger_flag_helpers_accept_native_and_user_masks():
    user_flag = 0x00000001

    cimba.logger_flags_off(cimba.LOGGER_INFO)
    cimba.logger_flags_on(cimba.LOGGER_INFO)
    cimba.logger_flags_off(user_flag)
    cimba.logger_flags_on(user_flag)
