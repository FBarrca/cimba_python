import cimba as cb
from tutorial.tut_4_0 import HarborTemplate


def test_empty_harbor_template():
    model = HarborTemplate()
    results = cb.Experiment(model, window=cb.Window(duration=10)).run()
    assert results[model].result[0, 0] == 0.0
