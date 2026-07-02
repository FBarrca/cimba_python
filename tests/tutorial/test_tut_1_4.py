import pytest

from tutorial import tut_1_4


def test_tut_1_4_buffer_history_estimates_mm1_queue_length():
    exp = tut_1_4.model.experiment(
        utilization=[0.75],
        replications=1,
        duration=5000.0,
        warmup=100.0,
        seed=14,
    )
    expected = 0.75 * 0.75 / (1.0 - 0.75)

    assert exp.run() == 0
    assert exp["avg_queue_length"][0] == pytest.approx(expected, abs=1.1)
