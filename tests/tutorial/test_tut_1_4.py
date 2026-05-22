import pytest

from tutorial import tut_1_4


def test_tut_1_4_buffer_history_estimates_mm1_queue_length():
    trial = tut_1_4.run(seed=14)
    expected = tut_1_4.theoretical_queue_length(trial.arr_rate, trial.srv_rate)

    assert trial.avg_queue_length == pytest.approx(expected, abs=1.1)
