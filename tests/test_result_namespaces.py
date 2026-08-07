"""Tests for typed and discoverable experiment result namespaces."""

import numpy as np
import pytest

import cimba.sim as sim


def test_nested_output_namespace_preserves_flattened_array():
    class Counter(sim.Component):
        mean_queue_length: sim.Output

    class QueueModel(sim.Model):
        counters: list[Counter] = [Counter(), Counter()]
        customers_served: sim.Output

        @sim.process
        def idle(env: "QueueModel"):
            sim.suspend()

        @sim.collect
        def collect(env: "QueueModel"):
            env.customers_served = 3.0
            env.counters[0].mean_queue_length = 1.0
            env.counters[1].mean_queue_length = 2.0

    model = QueueModel()



    experiment = model.experiment(replications=3, duration=10.0,
                                  warmup=0.0, seed=1)
    assert experiment.run() == 0

    root = experiment.results.customers_served
    nested = experiment.results.counters.mean_queue_length
    assert np.array_equal(root, experiment["customers_served"])
    assert np.array_equal(nested, experiment["counters__mean_queue_length"])
    assert root.shape == (3,)
    assert nested.shape == (3, 2)
    assert root.dtype == experiment["customers_served"].dtype
    assert np.shares_memory(root, experiment.trials["customers_served"])
    assert np.shares_memory(
        nested, experiment.trials["counters__mean_queue_length"])
    assert "customers_served" in dir(experiment.results)
    assert "mean_queue_length" in dir(experiment.results.counters)

    with pytest.raises(AttributeError, match="unknown result"):
        _ = experiment.results.counters.missing
    with pytest.raises(AttributeError, match="read-only"):
        experiment.results.customers_served = root


def test_dynamic_output_names_remain_available_through_string_api():
    class ModelWithLegacyOutput(sim.Model):
        pass

        @sim.collect
        def collect(env: "ModelWithLegacyOutput"):
            env.dynamic_value = 2.0

        @sim.process
        def idle(env: "ModelWithLegacyOutput"):
            sim.suspend()

    model = ModelWithLegacyOutput(outputs=["dynamic_value"])



    experiment = model.experiment(duration=1.0, warmup=0.0, seed=2)
    assert experiment.run() == 0
    assert "dynamic_value" in experiment.model.outputs
    assert np.array_equal(
        experiment["dynamic_value"], experiment.trials["dynamic_value"])
    assert np.array_equal(
        experiment.results.dynamic_value,
        experiment["dynamic_value"],
    )


def test_dataset_namespace_matches_existing_plural_accessor():
    class Samples(sim.Model):
        waits: sim.Dataset
        count: sim.Output

        @sim.process
        def driver(env: "Samples"):
            env.waits.add(1.0)
            env.waits.add(2.0)

        @sim.collect
        def collect(env: "Samples"):
            env.count = float(env.waits.count())
            env.waits.capture()

    model = Samples()



    experiment = model.experiment(replications=2, duration=1.0,
                                  warmup=0.0, seed=3)
    with pytest.raises(RuntimeError, match=r"run\(\)"):
        _ = experiment.results.waits
    assert experiment.run() == 0
    assert experiment.results.waits is experiment.datasets("waits")
    assert len(experiment.results.waits) == 2
    assert experiment.results.waits[0].tolist() == [1.0, 2.0]


def test_component_results_merge_outputs_datasets_and_histories():
    class Station(sim.Component):
        score: sim.Output
        samples: sim.Dataset
        queue: sim.Queue = sim.capacity(5)

    class Clinic(sim.Model):
        station: Station = Station()

        @sim.process
        def driver(env: "Clinic"):
            env.station.samples.add(4.0)
            env.station.queue.put(1)
            sim.suspend()

        @sim.collect
        def collect(env: "Clinic"):
            env.station.score = 4.0
            env.station.samples.capture()
            env.station.queue.history().capture()

    model = Clinic()



    experiment = model.experiment(duration=1.0, warmup=0.0, seed=4)
    assert experiment.run() == 0
    assert np.array_equal(
        experiment.results.station.score,
        experiment["station__score"],
    )
    assert experiment.results.station.samples is experiment.datasets(
        "station__samples")
    assert experiment.results.station.samples[0].tolist() == [4.0]
    assert experiment.results.station.queue is experiment.histories(
        "station__queue")


def test_history_namespace_preserves_scalar_and_collection_shapes():
    class Counter(sim.Component):
        line: sim.Queue = sim.capacity(5)

    class QueueModel(sim.Model):
        q: sim.Queue = sim.capacity(5)
        counters: list[Counter] = [Counter(), Counter()]

        @sim.process
        def driver(env: "QueueModel"):
            env.q.put(1)
            env.counters[0].line.put(1)
            env.counters[1].line.put(2)
            sim.hold(1.0)

        @sim.collect
        def collect(env: "QueueModel"):
            env.q.history().capture()
            env.counters[0].line.history().capture()
            env.counters[1].line.history().capture()

    model = QueueModel()



    experiment = model.experiment(replications=2, duration=1.0,
                                  warmup=0.0, seed=5)
    with pytest.raises(RuntimeError, match="run"):
        _ = experiment.results.q
    assert experiment.run() == 0

    root = experiment.results.q
    nested = experiment.results.counters.line
    assert root is experiment.histories("q")
    assert nested is experiment.histories("counters__line")
    assert len(root) == len(experiment) == 2
    assert len(nested) == len(experiment) == 2
    assert all(len(trial) == 2 for trial in nested)
    assert all(rows.ndim == 2 and rows.shape[1] == 3
               for rows in root)
    assert all(rows.ndim == 2 and rows.shape[1] == 3
               for trial in nested for rows in trial)
