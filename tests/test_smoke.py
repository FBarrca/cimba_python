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
        dock: sim.Queue = 4
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
    assert model.queues == {"q": None, "dock": 4}
    assert model.pools == {"crew": 3}
    assert model.stores == {"jobs": "rho"}
    assert model.conditions == ["done"]
    assert model.state == ["count"]
    assert model.float_state == ["level"]
    assert model._predicate_fields == ["ready"]
    # all declared fields land in the trial record
    for field in ("rho", "out", "q", "dock", "crew", "jobs", "done",
                  "count", "level", "ready"):
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


def test_bounded_queue_and_dataset_stats():
    class Bounded(sim.Model):
        max_level: sim.Output
        space_ok: sim.Output
        d_min: sim.Output
        d_max: sim.Output
        d_std: sim.Output
        q: sim.Queue = 5
        d: sim.Dataset

    model = Bounded()

    @model.process
    def producer(env: Bounded):
        env.max_level = 0.0
        env.space_ok = 1.0
        while True:
            sim.put(env.q, 1)       # blocks while the queue is full
            lvl = sim.level(env.q)
            if lvl > env.max_level:
                env.max_level = lvl
            if sim.space(env.q) + lvl != 5:
                env.space_ok = 0.0
            sim.tally(env.d, 1.0 * lvl)
            sim.hold(0.5)

    @model.process
    def consumer(env: Bounded):
        while True:
            sim.hold(1.0)
            sim.get(env.q, 1)

    @model.collect
    def stats(env: Bounded):
        env.d_min = sim.dataset_min(env.d)
        env.d_max = sim.dataset_max(env.d)
        env.d_std = sim.dataset_std(env.d)

    exp = model.experiment(replications=1, duration=100.0, warmup=10.0,
                           seed=1)
    assert exp.run() == 0
    assert exp["space_ok"][0] == 1.0
    assert 1 <= exp["max_level"][0] <= 5
    assert 0 <= exp["d_min"][0] <= exp["d_max"][0] <= 5
    assert exp["d_std"][0] >= 0


def test_random_draws_and_suspend():
    class Draws(sim.Model):
        tri: sim.Output
        wei: sim.Output
        lgn: sim.Output
        erl: sim.Output
        bet: sim.Output
        poi: sim.Output
        die: sim.Output

    model = Draws()

    @model.process
    def draw(env: Draws):
        env.tri = sim.triangular(0.0, 1.0, 2.0)
        env.wei = sim.weibull(1.5, 2.0)
        env.lgn = sim.lognormal(0.0, 0.5)
        env.erl = sim.erlang(3, 2.0)
        env.bet = sim.beta(2.0, 3.0, 0.0, 1.0)
        env.poi = sim.poisson(4.0)
        env.die = sim.dice(1, 6)
        sim.suspend()           # idle until the trial ends

    exp = model.experiment(replications=1, duration=10.0, warmup=0.0,
                           seed=3)
    assert exp.run() == 0
    assert 0.0 <= exp["tri"][0] <= 2.0
    assert exp["wei"][0] > 0.0
    assert exp["lgn"][0] > 0.0
    assert exp["erl"][0] > 0.0
    assert 0.0 <= exp["bet"][0] <= 1.0
    assert exp["poi"][0] >= 0.0
    assert 1.0 <= exp["die"][0] <= 6.0


def test_process_handles_and_interrupt():
    class Game(sim.Model):
        got_sig: sim.Output
        worker: sim.Processes

    model = Game()

    @model.process(copies=2)
    def worker(env: Game, idx: int):
        if idx == 0:
            env.got_sig = sim.hold(1000.0)  # interrupted by the poker
        else:
            while True:
                sim.hold(1000.0)

    @model.process
    def poker(env: Game):
        sim.hold(1.0)
        sim.interrupt(env.worker[0], 42, 0)
        while True:
            sim.hold(1000.0)

    assert model.dtype["worker"].shape == (2,)
    exp = model.experiment(replications=1, duration=10.0, warmup=0.0,
                           seed=5)
    assert exp.run() == 0
    assert exp["got_sig"][0] == 42.0


def test_pqueues_and_timers():
    class Shop(sim.Model):
        served_first: sim.Output    # object taken first (priority order)
        timed_out: sim.Output       # signal a waiter got from its timer
        qs: sim.PQueues = sim.count(2)

    model = Shop()

    @model.process
    def producer(env: Shop):
        sim.pq_put(env.qs[0], 7, 0)     # low priority first
        sim.pq_put(env.qs[0], 8, 5)     # high priority second
        sim.hold(1.0)
        env.served_first = sim.pq_take(env.qs[0])  # leftover entry

    @model.process
    def consumer(env: Shop):
        sim.hold(0.5)
        env.served_first = sim.pq_take(env.qs[0])

    @model.process
    def waiter(env: Shop):
        me = sim.current()
        sim.timer_set(me, 2.0, 99)
        env.timed_out = sim.suspend()
        while True:
            sim.hold(1000.0)

    exp = model.experiment(replications=1, duration=10.0, warmup=0.0,
                           seed=11)
    assert exp.run() == 0
    # the consumer at t=0.5 must get the priority-5 object
    assert exp["served_first"][0] == 7.0  # producer drained the leftover
    assert exp["timed_out"][0] == 99.0


def test_kwargs_model_still_works():
    model = sim.Model("legacy", params=["rho"], outputs=["out"],
                      queues=["q"])
    assert model.name == "legacy"
    assert model.params == ["rho"]
    assert model.queues == {"q": None}
