"""Captured native series survive entity teardown as read-only results."""

import cimba as cb
from cimba import inputs


class Captured(cb.Model):
    stock: cb.Container
    observations: cb.Dataset

    @cb.process
    def act(self):
        cb.hold(1.0)
        self.stock.put(2)
        self.observations.record(4.0)
        cb.hold(1.0)
        self.stock.get(1)
        self.observations.record(6.0)


def test_entity_capture():
    model = Captured()
    model.stock = cb.Container()
    model.observations = cb.Dataset()
    model.stock.capture()
    model.observations.capture()
    result = cb.Experiment(model, replications=2,
                           window=cb.Window(duration=3)).run(workers=1)
    times, values = result[model].stock.trial(0, 0)
    assert 2.0 in values
    assert times.shape == values.shape
    assert not values.flags.writeable
    dataset_times, dataset_values = result[model].observations.trial(0, 1)
    assert dataset_times.size == 0
    assert dataset_values.tolist() == [4.0, 6.0]
    assert not dataset_values.flags.writeable


class ChunkedCapture(cb.Model):
    draw: cb.Input[float] = inputs.bootstrap.iid([2.0, 4.0])
    observations: cb.Dataset

    @cb.process
    def act(self):
        self.observations.record(self.draw.next())


def test_capture_survives_input_chunks():
    model = ChunkedCapture()
    model.observations = cb.Dataset()
    model.observations.capture()
    result = cb.Experiment(model, replications=3).run(
        workers=1, input_memory=8192)
    assert result.meta["chunks"] == 3
    for replication in range(3):
        _, values = result[model].observations.trial(0, replication)
        assert values.size == 1
        assert values[0] in (2.0, 4.0)
