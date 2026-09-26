"""Native end-to-end behavior of the new object-addressed experiment path."""

import numpy as np

from cimba import inputs
from cimba.experiments import Experiment, Window
from cimba.modeling import (
    Container, Dataset, Input, Model, Output, Param, Resource, Series,
    hold, on_end, process, sweep,
)


class _Queue(Model):
    gaps: Input[float] = inputs.dist.exponential(mean=1.25)
    service: Input[float] = inputs.dist.exponential(mean=1.0)
    queue: Container
    mean: Output[float]

    @process
    def arrivals(self):
        while True:
            hold(self.gaps.next())
            self.queue.put(1)

    @process
    def server(self):
        while True:
            self.queue.get(1)
            hold(self.service.next())

    @on_end
    def measure(self):
        self.mean = self.queue.mean_level()


def test_native_trial_is_worker_invariant_and_source_agnostic():
    model = _Queue()
    experiment = Experiment(model, replications=6,
                            window=Window(warmup=10, duration=100), seed=123)
    one = experiment.run(workers=1)
    many = experiment.run(workers=3)
    np.testing.assert_array_equal(one[model].mean.values,
                                  many[model].mean.values)
    np.testing.assert_array_equal(one[model].gaps.consumed,
                                  many[model].gaps.consumed)
    assert not one.failed.any()
    assert len(one[model].gaps.rows(0, 0)) == one[model].gaps.consumed[0, 0]

    model.gaps = inputs.trace([1.0, 2.0, 1.0], on_exhausted="wrap")
    replay = Experiment(model, replications=2,
                        window=Window(duration=20), seed=123).run(workers=2)
    assert not replay.failed.any()
    assert replay[model].gaps.source[0]["method"] == "trace"


class _Finite(Model):
    source: Input[float]
    total: Output[float]

    @process
    def consume(self):
        total = 0.0
        for _ in range(10):
            value = self.source.next()
            total += value
            hold(value)
        self.total = total


def test_prefix_stable_extension_replays_only_exhausted_trial():
    model = _Finite()
    model.source = inputs.row_source(
        "test.constant", [0.1, 0.2],
        lambda rng, length: np.full(length, 0.1), length_hint=2)
    result = Experiment(model, replications=3,
                        window=Window(duration=2.0), seed=9).run(workers=2)
    np.testing.assert_allclose(result[model].total.values, [[1.0] * 3])
    np.testing.assert_array_equal(result[model].source.consumed, [[10] * 3])
    np.testing.assert_array_equal(result[model].source.extended, [[3] * 3])
    assert not result.failed.any()
    chunked = Experiment(model, replications=3,
                         window=Window(duration=2.0), seed=9).run(
                             workers=1, input_memory=32)
    np.testing.assert_array_equal(result[model].total.values,
                                  chunked[model].total.values)
    np.testing.assert_array_equal(result[model].source.consumed,
                                  chunked[model].source.consumed)
    assert chunked.meta["chunks"] == 2


class _Failing(Model):
    source: Input[float] = inputs.trace([1.0])

    @process
    def consume(self):
        while True:
            hold(self.source.next())


def test_exhausted_trace_fails_one_trial_without_poisoning_next():
    model = _Failing()
    result = Experiment(model, replications=4,
                        window=Window(duration=5.0)).run(workers=1)
    assert result.failed.all()
    assert all("source exhausted" in reason for reason in result.failure_reasons[0])


class _Station(Model):
    servers: Resource = Resource(capacity=2)
    observations: Dataset
    count: Output[int]
    utilization: Output[float]

    @process(copies=3)
    def visit(self):
        self.servers.acquire()
        hold(2.0)
        self.observations.record(2.0)
        self.servers.release()

    @on_end
    def collect(self):
        self.count = self.observations.sample_count()
        self.utilization = self.servers.mean_in_use()


def test_resource_and_dataset_handles_run_natively():
    model = _Station()
    result = Experiment(model, replications=2,
                        window=Window(duration=5)).run(workers=2)
    np.testing.assert_array_equal(result[model].count.values, [[3, 3]])
    np.testing.assert_array_equal(result[model].utilization.values, [[1.2, 1.2]])


class _SeriesReader(Model):
    daily: Series[float] = Series(step=1.0)
    sum: Output[float]

    @process
    def read(self):
        first = self.daily.now()
        hold(2.0)
        self.sum = first + self.daily.now() + self.daily.at(0.0)


def test_distribution_series_can_revisit_earlier_bucket():
    model = _SeriesReader()
    model.daily = inputs.dist.normal(mean=2.0, sd=0.1)
    result = Experiment(model, replications=2,
                        window=Window(duration=4), seed=12).run(workers=2)
    assert not result.failed.any()
    np.testing.assert_array_equal(result[model].daily.consumed, [[3, 3]])
    rows = result[model].daily.rows(0, 0)
    assert result[model].sum[0, 0] == rows[0] * 2 + rows[2]


class _StreamProbe(Model):
    consume_extra: Param[int] = 0
    first: Input[float] = inputs.dist.exponential(mean=1.0)
    second: Input[float] = inputs.dist.exponential(mean=1.0)
    observed: Output[float]

    @process
    def run(self):
        if self.consume_extra:
            for _ in range(10):
                self.first.next()
        self.observed = self.second.next()


def test_common_random_numbers_use_independent_input_streams():
    model = _StreamProbe()
    setattr(model, "consume_extra", sweep(0, 1))
    result = Experiment(model, replications=8,
                        window=Window(duration=1), seed=44).run(workers=3)
    np.testing.assert_array_equal(result[model].observed.values[0],
                                  result[model].observed.values[1])
    assert result[model].first.consumed[0, 0] == 0
    assert result[model].first.consumed[1, 0] == 10
