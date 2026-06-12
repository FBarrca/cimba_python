"""Smoke tests: confirm the package built and links the native C library."""

import numpy as np
import pytest

import cimba
import cimba.sim as sim

CAT_WEIGHTS = np.array([2.0, 3.0, 5.0], dtype=np.float64)


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


def test_extra_random_distributions():
    class ExtraDraws(sim.Model):
        sn: sim.Output
        se: sim.Output
        sg: sim.Output
        sb: sim.Output
        pm: sim.Output
        lg: sim.Output
        cy: sim.Output
        pr: sim.Output
        cs: sim.Output
        fd: sim.Output
        st: sim.Output
        td: sim.Output
        geo: sim.Output
        binom: sim.Output
        nbinom: sim.Output
        pas: sim.Output
        hypo: sim.Output
        hyper: sim.Output
        loaded: sim.Output
        cat: sim.Output

    model = ExtraDraws()

    @model.process
    def draw(env: ExtraDraws):
        env.sn = sim.std_normal()
        env.se = sim.std_exponential()
        env.sg = sim.std_gamma(2.5)
        env.sb = sim.std_beta(2.0, 3.0)
        env.pm = sim.pert_mod(0.0, 4.0, 10.0, 6.0)
        env.lg = sim.logistic(0.0, 1.0)
        env.cy = sim.cauchy(0.0, 1.0)
        env.pr = sim.pareto(2.5, 1.0)
        env.cs = sim.chisquared(4.0)
        env.fd = sim.f_dist(5.0, 8.0)
        env.st = sim.std_t(7.0)
        env.td = sim.t_dist(1.0, 2.0, 7.0)
        env.geo = sim.geometric(0.4)
        env.binom = sim.binomial(10, 0.4)
        env.nbinom = sim.negative_binomial(3, 0.4)
        env.pas = sim.pascal(3, 0.4)
        env.hypo = sim.hypoexponential((1.0, 2.0, 4.0, 8.0))
        env.hyper = sim.hyperexponential(
            (1.0, 2.0, 4.0, 8.0), (1.0, 2.0, 3.0, 4.0))
        env.loaded = sim.loaded_dice([0.2, 0.3, 0.5])
        env.cat = sim.categorical(CAT_WEIGHTS)
        sim.suspend()

    exp = model.experiment(replications=1, duration=10.0, warmup=0.0,
                           seed=13)
    assert exp.run() == 0
    for field in ("sn", "lg", "cy", "st", "td"):
        assert np.isfinite(exp[field][0])
    assert exp["se"][0] >= 0.0
    assert exp["sg"][0] >= 0.0
    assert 0.0 <= exp["sb"][0] <= 1.0
    assert 0.0 <= exp["pm"][0] <= 10.0
    assert exp["pr"][0] >= 1.0
    assert exp["cs"][0] >= 0.0
    assert exp["fd"][0] >= 0.0
    assert exp["geo"][0] >= 1.0
    assert 0.0 <= exp["binom"][0] <= 10.0
    assert exp["nbinom"][0] >= 0.0
    assert exp["pas"][0] >= 0.0
    assert exp["hypo"][0] >= 0.0
    assert exp["hyper"][0] >= 0.0
    assert 0.0 <= exp["loaded"][0] <= 2.0
    assert 0.0 <= exp["cat"][0] <= 2.0


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


def test_process_timeout_bindings():
    class Timeout(sim.Model):
        cancel_first: sim.Output
        cancel_second: sim.Output
        waited: sim.Output
        target_signal: sim.Output
        constants_ok: sim.Output
        target: sim.Processes

    model = Timeout()

    @model.process
    def target(env: Timeout):
        env.target_signal = sim.suspend()
        while True:
            sim.hold(1000.0)

    @model.process
    def controller(env: Timeout):
        me = sim.current()
        cancelled = sim.timer_add(me, 5.0, sim.TIMEOUT)
        env.cancel_first = sim.timer_cancel(me, cancelled)
        env.cancel_second = sim.timer_cancel(me, cancelled)
        env.constants_ok = 1.0
        if sim.TIMEOUT != -5 or sim.CANCELLED != -4:
            env.constants_ok = 0.0

        sim.hold(0.1)
        target_timer = sim.timer_add(env.target[0], 1.0, sim.TIMEOUT)
        env.waited = sim.wait_event(target_timer)
        while True:
            sim.hold(1000.0)

    exp = model.experiment(replications=1, duration=10.0, warmup=0.0,
                           seed=17)
    assert exp.run() == 0
    assert exp["cancel_first"][0] == 1.0
    assert exp["cancel_second"][0] == 0.0
    assert exp["waited"][0] == sim.SUCCESS
    assert exp["target_signal"][0] == sim.TIMEOUT
    assert exp["constants_ok"][0] == 1.0


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


def test_pqueue_get_status_and_object():
    class PQGet(sim.Model):
        first_status: sim.Output
        first_obj: sim.Output
        take_obj: sim.Output
        timeout_status: sim.Output
        timeout_obj: sim.Output
        qs: sim.PQueues = sim.count(1)

    model = PQGet()

    @model.process
    def actor(env: PQGet):
        q = env.qs[0]
        sim.pq_put(q, 7, 0)
        sim.pq_put(q, 8, 5)

        status, obj = sim.pq_get(q)
        env.first_status = status
        env.first_obj = obj
        env.take_obj = sim.pq_take(q)

        me = sim.current()
        sim.timer_set(me, 1.0, sim.TIMEOUT)
        status, obj = sim.pq_get(q)
        env.timeout_status = status
        env.timeout_obj = obj
        while True:
            sim.hold(1000.0)

    exp = model.experiment(replications=1, duration=10.0, warmup=0.0,
                           seed=29)
    assert exp.run() == 0
    assert exp["first_status"][0] == sim.SUCCESS
    assert exp["first_obj"][0] == 8.0
    assert exp["take_obj"][0] == 7.0
    assert exp["timeout_status"][0] == sim.TIMEOUT
    assert exp["timeout_obj"][0] == 0.0


def test_pqueue_space_reprioritize_and_mean_length():
    class PQStats(sim.Model):
        space_ok: sim.Output
        pos_before: sim.Output
        pos_after: sim.Output
        first: sim.Output
        mean_len: sim.Output
        qs: sim.PQueues = sim.count(1)

    model = PQStats()

    @model.process
    def actor(env: PQStats):
        q = env.qs[0]
        low = sim.pq_put(q, 10, 0)
        sim.pq_put(q, 20, 5)
        env.space_ok = 0.0
        if sim.pq_space(q) > sim.pq_length(q):
            env.space_ok = 1.0
        env.pos_before = sim.pq_position(q, low)
        sim.pq_reprioritize(q, low, 10)
        env.pos_after = sim.pq_position(q, low)
        sim.hold(1.0)
        env.first = sim.pq_take(q)
        sim.hold(1.0)
        sim.pq_take(q)
        sim.suspend()

    @model.collect
    def collect(env: PQStats):
        env.mean_len = sim.pq_mean_length(env.qs[0])

    exp = model.experiment(replications=1, duration=5.0, warmup=0.0,
                           seed=19)
    assert exp.run() == 0
    assert exp["space_ok"][0] == 1.0
    assert exp["pos_before"][0] == 2.0
    assert exp["pos_after"][0] == 1.0
    assert exp["first"][0] == 10.0
    assert np.isfinite(exp["mean_len"][0])
    assert exp["mean_len"][0] > 0.0


def test_store_get_position_and_resource_held():
    class StoreResource(sim.Model):
        zero_status: sim.Output
        zero_obj: sim.Output
        pos: sim.Output
        timeout_status: sim.Output
        timeout_obj: sim.Output
        held_before: sim.Output
        held_after: sim.Output
        store: sim.Store
        resource: sim.Resource

    model = StoreResource()

    @model.process
    def actor(env: StoreResource):
        me = sim.current()
        sim.acquire(env.resource)
        env.held_before = sim.held(env.resource, me)
        sim.release(env.resource)
        env.held_after = sim.held(env.resource, me)

        sim.store_put(env.store, 0)
        status, obj = sim.store_get(env.store)
        env.zero_status = status
        env.zero_obj = obj

        sim.store_put(env.store, 41)
        sim.store_put(env.store, 42)
        env.pos = sim.store_position(env.store, 42)
        sim.store_take(env.store)
        sim.store_take(env.store)

        sim.timer_set(me, 1.0, sim.TIMEOUT)
        status, obj = sim.store_get(env.store)
        env.timeout_status = status
        env.timeout_obj = obj
        while True:
            sim.hold(1000.0)

    exp = model.experiment(replications=1, duration=10.0, warmup=0.0,
                           seed=23)
    assert exp.run() == 0
    assert exp["held_before"][0] == 1.0
    assert exp["held_after"][0] == 0.0
    assert exp["zero_status"][0] == sim.SUCCESS
    assert exp["zero_obj"][0] == 0.0
    assert exp["pos"][0] == 2.0
    assert exp["timeout_status"][0] == sim.TIMEOUT
    assert exp["timeout_obj"][0] == 0.0


def test_kwargs_model_still_works():
    model = sim.Model("legacy", params=["rho"], outputs=["out"],
                      queues=["q"])
    assert model.name == "legacy"
    assert model.params == ["rho"]
    assert model.queues == {"q": None}
