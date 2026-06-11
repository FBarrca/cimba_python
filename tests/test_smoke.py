"""Smoke tests: confirm the package built and links the native C library."""

import numpy as np

import cimba
import cimba.sim as sim


def test_wrapper_version():
    assert cimba.__version__ == "0.1.0"


def test_native_version_is_linked():
    v = cimba.native_version()
    assert isinstance(v, str)
    assert v
    assert v.startswith("3.")


def test_sim_model_run():
    model = sim.Model(
        "smoke",
        params=["utilization"],
        outputs=["avg_queue_length"],
        queues=["queue"],
    )

    @model.process
    def arrivals(env):
        while True:
            sim.hold(sim.exponential(1.0 / env.utilization))
            sim.put(env.queue, 1)

    @model.process
    def service(env):
        while True:
            sim.hold(1.0)
            sim.get(env.queue, 1)

    @model.collect
    def collect_stats(env):
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


def test_sim_model_import():
    model = sim.Model("smoke", params=["rho"], outputs=["out"], queues=["q"])
    assert model.name == "smoke"
