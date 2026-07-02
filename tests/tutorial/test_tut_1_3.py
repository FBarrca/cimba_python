import cimba
from tutorial import tut_1_3


def test_tut_1_3_logger_flags_can_be_configured_for_tutorial_runs():
    cimba.logger_flags_off(cimba.LOGGER_INFO)
    cimba.logger_flags_on(tut_1_3.USERFLAG1)
    exp = tut_1_3.model.experiment(
        utilization=[0.75],
        replications=1,
        duration=25.0,
        warmup=0.0,
        seed=13,
    )
    try:
        assert exp.run() == 0
    finally:
        cimba.logger_flags_off(tut_1_3.USERFLAG1)

    assert cimba.LOGGER_INFO == 0x10000000
    assert exp["avg_queue_length"][0] >= 0.0
