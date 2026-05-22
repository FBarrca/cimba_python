import cimba
from tutorial import tut_1_3


def test_tut_1_3_logger_flags_can_be_configured_for_tutorial_runs():
    trial = tut_1_3.run(seed=13)

    assert cimba.LOGGER_INFO == 0x10000000
    assert trial.avg_queue_length >= 0.0
