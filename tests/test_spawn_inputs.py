"""Spawned models may consume an input cloned from a configured parent."""

import numpy as np

import cimba as cb
from cimba import inputs


class Child(cb.Model):
    owner: cb.Ref["Parent"]
    value: cb.Input[float]

    @cb.process
    def use_input(self):
        self.owner.total += self.value.next()
        cb.release(self)


class Parent(cb.Model):
    value: cb.Input[float]
    total: cb.State[float] = 0.0
    outcome: cb.Output[float]

    @cb.process
    def make_children(self):
        for _ in range(3):
            cb.spawn(Child, owner=self, value=self.value)

    @cb.on_end
    def measure(self):
        self.outcome = self.total


def test_spawned_inputs_have_independent_cursors():
    model = Parent()
    model.value = inputs.trace([2.0, 5.0], on_exhausted="wrap")
    result = cb.Experiment(model, replications=2).run(workers=2)
    assert not result.failed.any(), result.failure_reasons
    np.testing.assert_array_equal(result[model].outcome.values,
                                  [[6.0, 6.0]])


def test_spawned_distribution_streams_are_worker_invariant():
    model = Parent()
    model.value = inputs.dist.exponential(mean=2.0)
    experiment = cb.Experiment(model, replications=6, seed=91)
    one_result = experiment.run(workers=1)
    many_result = experiment.run(workers=3)
    assert not one_result.failed.any(), one_result.failure_reasons
    assert not many_result.failed.any(), many_result.failure_reasons
    one = one_result[model].outcome.values
    many = many_result[model].outcome.values
    np.testing.assert_array_equal(one, many)
    assert np.std(one) > 0.0


class SeriesChild(cb.Model):
    owner: cb.Ref["SeriesParent"]
    demand: cb.Series[float] = cb.Series(step=1.0)

    @cb.process
    def use_series(self):
        cb.hold(1.0)
        self.owner.total += self.demand.now()
        cb.release(self)


class SeriesParent(cb.Model):
    demand: cb.Series[float] = cb.Series(step=1.0)
    total: cb.State[float] = 0.0
    outcome: cb.Output[float]

    @cb.process
    def make_children(self):
        for _ in range(3):
            cb.spawn(SeriesChild, owner=self, demand=self.demand)

    @cb.on_end
    def measure(self):
        self.outcome = self.total


def test_spawned_series_preserves_time_indexing():
    model = SeriesParent()
    model.demand = inputs.trace([10.0, 20.0], on_exhausted="wrap")
    result = cb.Experiment(model, window=cb.Window(duration=2.0)).run()
    assert not result.failed.any(), result.failure_reasons
    assert result[model].outcome[0, 0] == 60.0


class LongChild(cb.Model):
    owner: cb.Ref["LongParent"]
    value: cb.Input[float]

    @cb.process
    def consume(self):
        for _ in range(1050):
            self.owner.total += self.value.next()
        cb.release(self)


class LongParent(cb.Model):
    value: cb.Input[float]
    total: cb.State[float] = 0.0
    outcome: cb.Output[float]

    @cb.process
    def make_child(self):
        cb.spawn(LongChild, owner=self, value=self.value)

    @cb.on_end
    def measure(self):
        self.outcome = self.total


def test_spawned_row_input_extends_and_replays():
    model = LongParent()
    model.value = inputs.bootstrap.iid([1.0])
    result = cb.Experiment(model, replications=2).run(workers=2)
    assert not result.failed.any(), result.failure_reasons
    np.testing.assert_array_equal(result[model].outcome.values,
                                  [[1050.0, 1050.0]])
    assert result.meta.extensions > 0
