"""Smoke tests: confirm the package built and links the native C library."""

import numpy as np
import pytest

import cimba
import cimba.sim as sim


def test_wrapper_version():
    assert cimba.__version__ == "0.1.0"


def test_native_version_is_linked():
    v = cimba.native_version()
    assert isinstance(v, str)
    assert v
    assert v.startswith("3.")


class MM1(sim.Model):
    utilization: sim.Param
    avg_queue_length: sim.Output
    queue: sim.Queue


def test_sim_model_run():
    model = MM1("smoke")

    @model.process
    def arrivals(env: MM1):
        while True:
            sim.hold(sim.exponential(1.0 / env.utilization))
            sim.put(env.queue, 1)

    @model.process
    def service(env: MM1):
        while True:
            sim.hold(1.0)
            sim.get(env.queue, 1)

    @model.collect
    def collect_stats(env: MM1):
        env.avg_queue_length = sim.mean_level(env.queue)

    exp = model.experiment(
        utilization=[0.5],
        replications=1,
        duration=1000.0,
        warmup=100.0,
        seed=42,
    )
    assert exp.trials.size == 1
    failures = exp.run()
    assert failures == 0
    assert np.isfinite(exp["avg_queue_length"][0])


def test_class_declarations():
    class Shop(sim.Model):
        rho: sim.Param
        out: sim.Output
        q: sim.Queue
        crew: sim.Pool = 3
        jobs: sim.Store = sim.capacity("rho")
        done: sim.Condition
        count: sim.State
        level: sim.FloatState
        ready: sim.Predicate

    model = Shop()
    assert model.name == "Shop"
    assert model.params == ["rho"]
    assert model.outputs == ["out"]
    assert model.queues == ["q"]
    assert model.pools == {"crew": 3}
    assert model.stores == {"jobs": "rho"}
    assert model.conditions == ["done"]
    assert model.state == ["count"]
    assert model.float_state == ["level"]
    assert model._predicate_fields == ["ready"]
    # all declared fields land in the trial record
    for field in ("rho", "out", "q", "crew", "jobs", "done", "count",
                  "level", "ready"):
        assert field in model.dtype.fields
    assert model.dtype.fields["count"][0] == np.dtype("<i8")
    assert model.dtype.fields["level"][0] == np.dtype("<f8")


def test_unbound_predicate_field_rejected():
    class Gate(sim.Model):
        x: sim.Param
        ready: sim.Predicate

    model = Gate()

    @model.process
    def proc(env: Gate):
        sim.hold(1.0)

    with pytest.raises(ValueError, match="ready"):
        model.experiment(x=1.0)


def test_kwargs_model_still_works():
    model = sim.Model("legacy", params=["rho"], outputs=["out"],
                      queues=["q"])
    assert model.name == "legacy"
    assert model.params == ["rho"]
    assert model.queues == ["q"]
