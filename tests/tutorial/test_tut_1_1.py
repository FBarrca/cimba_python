import cimba as cb
from tutorial.tut_1_1 import MM1


def test_first_queue_runs():
    model = MM1()
    results = cb.Experiment(model, window=cb.Window(duration=25), seed=11).run()
    assert not results.failed.any()
    assert results[model].avg_queue_length[0, 0] >= 0.0
