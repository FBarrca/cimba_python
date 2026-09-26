import pytest

import cimba as cb
from tutorial.tut_1_4 import MM1


def test_long_run_queue_mean_is_plausible():
    model = MM1()
    results = cb.Experiment(
        model, window=cb.Window(warmup=100, duration=5_000), seed=14,
    ).run()
    assert not results.failed.any()
    assert results[model].avg_queue_length[0, 0] == pytest.approx(
        0.75**2 / (1 - 0.75), abs=1.1)
