import cimba as cb
from tutorial.tut_1_2 import MM1


def test_queue_history_and_arrival_dataset():
    model = MM1()
    model.queue = cb.Container()
    model.queue.capture()
    results = cb.Experiment(model, window=cb.Window(duration=25), seed=12).run()
    assert not results.failed.any()
    time, level = results[model].queue.trial(0, 0)
    assert len(time) == len(level) > 0
    assert results[model].avg_interarrival_time[0, 0] > 0
