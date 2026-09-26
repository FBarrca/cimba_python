"""Input handle type controls the value seen by compiled model code."""

import cimba as cb
from cimba import inputs


class TypedInputs(cb.Model):
    count: cb.Input[int] = inputs.dist.poisson(mean=3)
    enabled: cb.Series[bool] = cb.Series(step=1.0)
    count_out: cb.Output[int]
    enabled_out: cb.Output[bool]

    @cb.process
    def consume(self):
        self.count_out = self.count.next()
        self.enabled_out = self.enabled.now()


def test_typed_input_and_series():
    model = TypedInputs()
    model.enabled = inputs.trace([1, 0, 1], on_exhausted="wrap")
    result = cb.Experiment(model, replications=4).run(workers=1)
    assert result[model].count_out.values.min() >= 0
    assert result[model].enabled_out.values.tolist() == [[1.0] * 4]
