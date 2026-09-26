import cimba as cb
from tutorial.tut_1_3 import MM1


def test_compiled_logging_does_not_change_trial_result():
    cb.set_engine_log_level(0)
    model = MM1()
    results = cb.Experiment(model, window=cb.Window(duration=25), seed=13).run()
    assert not results.failed.any()
    assert results[model].avg_queue_length[0, 0] >= 0
