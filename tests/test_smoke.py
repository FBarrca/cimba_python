"""Smoke tests: confirm the package built and links the native C library."""

import os
import numpy as np
import pytest
import time

import cimba
import cimba.sim as sim
from cimba import Buffer, Process, Simulation

CAT_PROBABILITIES = np.array([0.2, 0.3, 0.5], dtype=np.float64)


def capture_native_stdout(fn):
    read_fd, write_fd = os.pipe()
    saved_fd = os.dup(1)
    try:
        os.dup2(write_fd, 1)
        os.close(write_fd)
        fn()
        os.dup2(saved_fd, 1)
        chunks = []
        while True:
            chunk = os.read(read_fd, 8192)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks).decode()
    finally:
        try:
            os.dup2(saved_fd, 1)
        except OSError:
            pass
        os.close(saved_fd)
        try:
            os.close(read_fd)
        except OSError:
            pass


def test_c_module_shaped_imports_are_available():
    assert Buffer is cimba.Buffer
    assert Simulation is cimba.Simulation
    assert Process is cimba.Process
    assert not hasattr(sim, "random")


def test_native_version_is_linked():
    v = cimba.native_version()
    assert isinstance(v, str)
    assert v
    assert v.startswith("3.")


def test_logger_flag_controls():
    cimba.logger_flags_off(cimba.LOGGER_INFO)
    cimba.logger_flags_on(cimba.LOGGER_INFO)


def test_sim_logging_helpers():
    userflag = 0x00000001
    msg = sim.log_text("python logging smoke")
    label_i = sim.log_text("count")
    label_f = sim.log_text("value")

    class LogModel(sim.Model):
        done: sim.Output

    model = LogModel()

    @model.process
    def actor(env: LogModel):
        sim.log_user(userflag, msg)
        sim.log_user_i64(userflag, label_i, 7)
        sim.log_user_f64(userflag, label_f, 2.5)
        env.done = 1.0
        sim.suspend()

    cimba.logger_flags_on(userflag)
    exp = model.experiment(replications=1, duration=1.0, warmup=0.0, seed=7)
    out = capture_native_stdout(lambda: exp.run())
    assert "python logging smoke" in out
    assert "count 7" in out
    assert "value 2.500000" in out
    assert exp.failures == 0
    assert exp["done"][0] == 1.0


def test_sim_logging_suppressed_in_trial_threads():
    userflag = 0x00000002
    msg = sim.log_text("this should be suppressed")

    class LogModel(sim.Model):
        done: sim.Output

    model = LogModel()

    @model.process
    def actor(env: LogModel):
        sim.log_user(userflag, msg)
        env.done = 1.0
        sim.suspend()

    cimba.logger_flags_off(userflag)
    exp = model.experiment(replications=1, duration=1.0, warmup=0.0, seed=8)
    out = capture_native_stdout(lambda: exp.run())
    assert "this should be suppressed" not in out
    assert exp.failures == 0
    assert exp["done"][0] == 1.0
    cimba.logger_flags_on(userflag)


def test_disabled_logging_overhead_is_bounded():
    userflag = 0x00000004
    msg = sim.log_text("disabled")
    cimba.logger_flags_off(userflag)

    class NoLog(sim.Model):
        done: sim.Output

    no_log = NoLog()

    @no_log.process
    def no_log_actor(env: NoLog):
        for _ in range(200):
            sim.hold(0.0)
        env.done = 1.0

    class DisabledLog(sim.Model):
        done: sim.Output

    disabled_log = DisabledLog()

    @disabled_log.process
    def disabled_actor(env: DisabledLog):
        for _ in range(200):
            sim.log_user(userflag, msg)
            sim.hold(0.0)
        env.done = 1.0

    no_log_exp = no_log.experiment(replications=1, duration=1.0, warmup=0.0,
                                   seed=9)
    disabled_exp = disabled_log.experiment(replications=1, duration=1.0,
                                           warmup=0.0, seed=9)
    no_log_exp.run()
    disabled_exp.run()

    t0 = time.perf_counter()
    assert no_log_exp.run() == 0
    no_log_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    assert disabled_exp.run() == 0
    disabled_time = time.perf_counter() - t0

    assert no_log_exp["done"][0] == 1.0
    assert disabled_exp["done"][0] == 1.0
    assert disabled_time < max(0.25, no_log_time * 20.0)
    cimba.logger_flags_on(userflag)


class MM1(sim.Model):
    utilization: sim.Param
    avg_queue_length: sim.Output
    queue: sim.Queue


def test_sim_model_run():
    model = MM1("smoke")

    @model.process
    def arrivals(env: MM1):
        while True:
            sim.hold(cimba.random.exponential(1.0 / env.utilization))
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
            env.d.add(1.0 * lvl)
            sim.hold(0.5)

    @model.process
    def consumer(env: Bounded):
        while True:
            sim.hold(1.0)
            sim.get(env.q, 1)

    @model.collect
    def stats(env: Bounded):
        env.d_min = env.d.min()
        env.d_max = env.d.max()
        env.d_std = env.d.std()

    exp = model.experiment(replications=1, duration=100.0, warmup=10.0,
                           seed=1)
    assert exp.run() == 0
    assert exp["space_ok"][0] == 1.0
    assert 1 <= exp["max_level"][0] <= 5
    assert 0 <= exp["d_min"][0] <= exp["d_max"][0] <= 5
    assert exp["d_std"][0] >= 0


def test_dataset_median_and_quantile():
    class Quant(sim.Model):
        med: sim.Output
        q0: sim.Output
        q25: sim.Output
        q100: sim.Output
        med_empty: sim.Output
        d: sim.Dataset
        d_empty: sim.Dataset

    model = Quant()

    @model.process
    def feed(env: Quant):
        sim.hold(2.0)           # tally inside the measurement window
        for i in range(1, 7):
            env.d.add(1.0 * i)
        sim.suspend()

    @model.collect
    def stats(env: Quant):
        env.med = env.d.median()
        env.q0 = env.d.quantile(0.0)
        env.q25 = env.d.quantile(0.25)
        env.q100 = env.d.quantile(1.0)
        env.med_empty = env.d_empty.median()

    exp = model.experiment(replications=1, duration=10.0, warmup=1.0,
                           seed=1)
    assert exp.run() == 0
    assert exp["med"][0] == 3.5           # 1..6, even count interpolates
    assert exp["q0"][0] == 1.0
    assert exp["q25"][0] == 2.25          # h = 0.25 * 5 between 2 and 3
    assert exp["q100"][0] == 6.0
    assert exp["med_empty"][0] == 0.0


def test_experiment_summary():
    class Sweep(sim.Model):
        x: sim.Param
        y: sim.Output
        z: sim.Output

    model = Sweep()

    @model.process
    def p(env: Sweep):
        env.y = env.x * 2.0
        env.z = cimba.random.uniform(0.0, 1.0)
        sim.suspend()

    exp = model.experiment(x=[1.0, 2.0, 3.0], replications=5,
                           duration=10.0, warmup=0.0, seed=7)
    with pytest.raises(RuntimeError, match="run"):
        exp.summary()
    assert exp.run() == 0

    s = exp.summary()
    assert s.shape == (3,)
    assert exp.swept == ("x",)
    assert list(s["x"]) == [1.0, 2.0, 3.0]
    assert np.allclose(s["y"], [2.0, 4.0, 6.0])   # deterministic in x
    assert np.allclose(s["y_hw"], 0.0)
    assert ((0.0 <= s["z"]) & (s["z"] <= 1.0)).all()
    assert (s["z_hw"] > 0.0).all()                # random, 5 reps

    only_y = exp.summary("y", confidence=0.99)
    assert only_y.dtype.names == ("x", "y", "y_hw")
    with pytest.raises(ValueError, match="unknown"):
        exp.summary("nope")


def test_experiment_summary_single_point():
    class Single(sim.Model):
        y: sim.Output

    model = Single()

    @model.process
    def p(env: Single):
        env.y = cimba.random.uniform(0.0, 1.0)
        sim.suspend()

    exp = model.experiment(replications=1, duration=10.0, warmup=0.0,
                           seed=7)
    assert exp.run() == 0
    s = exp.summary()
    assert s.shape == (1,)
    assert 0.0 <= s["y"][0] <= 1.0
    assert np.isnan(s["y_hw"][0])   # one replication: no CI


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
        env.tri = cimba.random.triangular(0.0, 1.0, 2.0)
        env.wei = cimba.random.weibull(1.5, 2.0)
        env.lgn = cimba.random.lognormal(0.0, 0.5)
        env.erl = cimba.random.erlang(3, 2.0)
        env.bet = cimba.random.beta(2.0, 3.0, 0.0, 1.0)
        env.poi = cimba.random.poisson(4.0)
        env.die = cimba.random.dice(1, 6)
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
        env.sn = cimba.random.normal()
        env.se = cimba.random.exponential()
        env.sg = cimba.random.gamma(2.5)
        env.sb = cimba.random.beta(2.0, 3.0)
        env.pm = cimba.random.pert_mod(0.0, 4.0, 10.0, 6.0)
        env.lg = cimba.random.logistic(0.0, 1.0)
        env.cy = cimba.random.cauchy(0.0, 1.0)
        env.pr = cimba.random.pareto(2.5, 1.0)
        env.cs = cimba.random.chi_squared(4.0)
        env.fd = cimba.random.f_dist(5.0, 8.0)
        env.st = cimba.random.student_t(7.0)
        env.td = cimba.random.student_t(7.0, 1.0, 2.0)
        env.geo = cimba.random.geometric(0.4)
        env.binom = cimba.random.binomial(10, 0.4)
        env.nbinom = cimba.random.negative_binomial(3, 0.4)
        env.pas = cimba.random.negative_binomial(3, 0.4)
        env.hypo = cimba.random.hypoexponential((1.0, 2.0, 4.0, 8.0))
        env.hyper = cimba.random.hyperexponential(
            (1.0, 2.0, 4.0, 8.0), (0.1, 0.2, 0.3, 0.4))
        env.loaded = cimba.random.categorical([0.2, 0.3, 0.5])
        env.cat = cimba.random.categorical(CAT_PROBABILITIES)
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


def test_random_namespace_defaults_keywords_and_aliases_compile():
    class RandomAPI(sim.Model):
        u: sim.Output
        n: sim.Output
        e: sim.Output
        g: sim.Output
        cat: sim.Output
        hyper: sim.Output
        td: sim.Output
        chi: sim.Output

    model = RandomAPI()

    @model.process
    def draw(env: RandomAPI):
        env.u = cimba.random.uniform()
        env.n = cimba.random.normal(mu=0.0, sigma=1.0)
        env.e = cimba.random.exponential()
        env.g = cimba.random.gamma(shape=2.0)
        env.cat = cimba.random.categorical((0.2, 0.3, 0.5))
        env.hyper = cimba.random.hyperexponential(
            (1.0, 2.0), probabilities=(0.25, 0.75))
        env.td = cimba.random.student_t(v=7.0, m=1.0, s=2.0)
        env.chi = cimba.random.chi_squared(k=4.0)
        sim.suspend()

    exp = model.experiment(replications=1, duration=1.0, warmup=0.0, seed=19)
    assert exp.run() == 0
    assert 0.0 <= exp["u"][0] <= 1.0
    assert exp["e"][0] >= 0.0
    assert exp["g"][0] >= 0.0
    assert 0.0 <= exp["cat"][0] <= 2.0
    assert exp["hyper"][0] >= 0.0
    for field in ("n", "td", "chi"):
        assert np.isfinite(exp[field][0])


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


def test_low_level_events():
    class Evented(sim.Model):
        fired_at: sim.Output
        payload: sim.Output
        sched_ok: sim.Output
        t_sched: sim.Output
        t_resched: sim.Output
        prio_after: sim.Output
        wait_status: sim.Output
        cancel_first: sim.Output
        cancel_second: sim.Output
        n_bumps: sim.Output
        count_ok: sim.Output
        cur_ok: sim.Output
        ring: sim.Event
        counter: sim.State

    model = Evented()

    @model.event
    def ring(env: Evented, data: int):
        env.fired_at = sim.now()
        env.payload = data
        env.cur_ok = 1.0 if sim.current_event() != 0 else 0.0

    @model.event
    def bump(env: Evented):
        env.counter += 1

    @model.process
    def driver(env: Evented):
        h = sim.schedule(env.ring, env, 2.0, 42, 7)
        env.sched_ok = sim.event_scheduled(h)
        env.t_sched = sim.event_time(h)
        env.count_ok = 1.0 if sim.event_count() >= 1 else 0.0
        sim.event_reschedule(h, sim.now() + 3.0)
        env.t_resched = sim.event_time(h)
        sim.event_reprioritize(h, 9)
        env.prio_after = sim.event_priority(h)
        env.wait_status = sim.wait_event(h)

        h2 = sim.schedule(env._ev_bump, env, 1.0)  # defaults: data/priority
        env.cancel_first = sim.event_cancel(h2)
        env.cancel_second = sim.event_cancel(h2)
        sim.schedule_at(env._ev_bump, env, sim.now() + 1.0)
        sim.hold(2.0)
        env.n_bumps = env.counter
        while True:
            sim.hold(1000.0)

    exp = model.experiment(replications=1, duration=20.0, warmup=0.0,
                           seed=37)
    assert exp.run() == 0
    assert exp["sched_ok"][0] == 1.0
    assert exp["t_sched"][0] == 2.0
    assert exp["t_resched"][0] == 3.0
    assert exp["prio_after"][0] == 9.0
    assert exp["fired_at"][0] == 3.0
    assert exp["payload"][0] == 42.0
    assert exp["wait_status"][0] == sim.SUCCESS
    assert exp["cancel_first"][0] == 1.0
    assert exp["cancel_second"][0] == 0.0
    assert exp["n_bumps"][0] == 1.0  # cancelled bump never fired
    assert exp["count_ok"][0] == 1.0
    assert exp["cur_ok"][0] == 1.0


def test_clear_events_ends_trial():
    class Clearer(sim.Model):
        ended_at: sim.Output
        had_events: sim.Output

    model = Clearer()

    @model.process
    def runner(env: Clearer):
        sim.hold(1.0)
        env.ended_at = sim.now()
        env.had_events = 1.0 if sim.event_count() > 0 else 0.0
        sim.clear_events()
        sim.suspend()

    exp = model.experiment(replications=1, duration=100.0, warmup=10.0,
                           seed=31)
    assert exp.run() == 0
    assert exp["ended_at"][0] == 1.0
    assert exp["had_events"][0] == 1.0


def test_unbound_event_field_rejected():
    class Gate(sim.Model):
        x: sim.Param
        ring: sim.Event

    model = Gate()

    @model.process
    def proc(env: Gate):
        sim.hold(1.0)

    with pytest.raises(ValueError, match="ring"):
        model.experiment(x=1.0)


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


def test_struct_declaration_and_inheritance():
    class Base(sim.Struct):
        a: float

    class Derived(Base):
        b: int

    assert Base._dtype == np.dtype([("a", "<f8")])
    assert Derived._dtype == np.dtype([("a", "<f8"), ("b", "<i8")])
    assert Derived._alloc_size > Base._alloc_size

    with pytest.raises(TypeError, match="model code"):
        Base(0)     # views exist only inside compiled trials

    with pytest.raises(TypeError, match="only float and int"):
        class Bad(sim.Struct):
            s: str

    with pytest.raises(ValueError, match="no fields"):
        class Empty(sim.Struct):
            pass

    model = sim.Model("nostruct")
    with pytest.raises(ValueError, match="Struct subclass"):
        @model.process(struct=int)
        def proc(env):
            sim.hold(1.0)

    with pytest.raises(ValueError, match="last parameter"):
        @model.process
        def misplaced(env, tag: Base, idx: int):
            sim.hold(1.0)

    with pytest.raises(ValueError, match="disagree"):
        @model.process(struct=Derived)
        def mismatched(env, tag: Base):
            sim.hold(1.0)


def test_process_struct_cross_access():
    # The derived-struct pattern of tut_3_1.c: per-process fields,
    # injected into the owner as a view parameter, that other processes
    # read and write through the process handle.
    class Tag(sim.Struct):
        ticket: int
        stamp: float

    class Office(sim.Model):
        zero_ok: sim.Output
        t0: sim.Output
        t1: sim.Output
        s0: sim.Output
        s1: sim.Output
        clerk: sim.Processes

    model = Office()

    @model.process(copies=2)
    def clerk(env: Office, idx: int, tag: Tag):
        ok = 1.0 if (tag.ticket == 0 and tag.stamp == 0.0) else 0.0
        if idx == 0:
            env.zero_ok = ok
        tag.ticket = 10 + idx
        sim.hold(2.0)           # the stamper writes our stamp at t=1
        if idx == 0:
            env.s0 = tag.stamp
        else:
            env.s1 = tag.stamp
        sim.suspend()

    @model.process
    def stamper(env: Office, own: Tag):
        own.ticket = 99         # plain (env, view) form, own fields
        sim.hold(1.0)
        env.t0 = 1.0 * Tag(env.clerk[0]).ticket
        env.t1 = 1.0 * Tag(env.clerk[1]).ticket
        Tag(env.clerk[0]).stamp = 0.5
        Tag(env.clerk[1]).stamp = 1.5 + 0.01 * own.ticket
        sim.suspend()

    exp = model.experiment(replications=1, duration=10.0, warmup=0.0,
                           seed=99)
    assert exp.run() == 0
    assert exp["zero_ok"][0] == 1.0    # fields start zeroed
    assert exp["t0"][0] == 10.0        # each copy has its own record
    assert exp["t1"][0] == 11.0
    assert exp["s0"][0] == 0.5         # writes through the handle stick
    assert exp["s1"][0] == 1.5 + 0.01 * 99


def test_spawn_and_despawn():
    # Dynamic process creation, the tut_3_1.c visitor lifecycle:
    # spawn, initialize the struct before it runs, join, despawn.
    class Item(sim.Struct):
        weight: float

    class Factory(sim.Model):
        made: sim.Output
        total: sim.Output
        distinct: sim.Output
        done: sim.State
        acc: sim.FloatState
        worker: sim.Spawnable

    model = Factory()

    @model.process
    def worker(env: Factory, it: Item):
        sim.hold(1.0)
        env.done += 1
        env.acc += it.weight

    @model.process
    def spawner(env: Factory):
        h1 = sim.spawn(env.worker, env)
        Item(h1).weight = 2.5      # runs only once we block: init first
        h2 = sim.spawn(env.worker, env, 3)
        Item(h2).weight = 4.0
        env.distinct = 1.0 if h1 != h2 else 0.0
        sim.wait_process(h1)
        sim.wait_process(h2)
        env.made = 1.0 * env.done
        env.total = env.acc
        sim.despawn(h1)
        sim.despawn(h2)
        sim.suspend()

    exp = model.experiment(replications=1, duration=10.0, warmup=0.0,
                           seed=5)
    assert exp.run() == 0
    assert exp["distinct"][0] == 1.0
    assert exp["made"][0] == 2.0
    assert exp["total"][0] == 6.5


def test_spawned_leftovers_reclaimed():
    # Spawned processes still alive at trial end are stopped and
    # reclaimed like the static ones, and despawn is idempotent.
    class Hive(sim.Model):
        spawned: sim.Output
        redespawn_ok: sim.Output
        drone: sim.Spawnable

    model = Hive()

    @model.process
    def drone(env: Hive):
        sim.suspend()       # blocks forever; never despawned

    @model.process
    def queen(env: Hive):
        for _ in range(50):
            sim.spawn(env.drone, env)
        h = sim.spawn(env.drone, env)
        sim.hold(1.0)
        sim.despawn(h)
        sim.despawn(h)      # double despawn must be a no-op
        env.redespawn_ok = 1.0
        env.spawned = 51.0
        sim.suspend()

    exp = model.experiment(replications=20, duration=10.0, warmup=0.0,
                           seed=11)
    assert exp.run() == 0
    assert (exp["spawned"] == 51.0).all()
    assert (exp["redespawn_ok"] == 1.0).all()
    assert exp.run() == 0   # rerun on the same compiled trial


def test_spawnable_declaration_errors():
    class Loose(sim.Model):
        x: sim.Param
        ghost: sim.Spawnable

    model = Loose()

    with pytest.raises(ValueError, match="cannot take copies"):
        @model.process(copies=3)
        def ghost(env):
            sim.hold(1.0)

    with pytest.raises(ValueError, match="copy index"):
        @model.process
        def ghost(env, idx):  # noqa: F811
            sim.hold(1.0)

    @model.process
    def lonely(env):
        sim.hold(1.0)

    # the Spawnable field never got its @process
    with pytest.raises(ValueError, match="ghost"):
        model.experiment(x=1.0)


def test_kwargs_model_still_works():
    model = sim.Model("legacy", params=["rho"], outputs=["out"],
                      queues=["q"])
    assert model.name == "legacy"
    assert model.params == ["rho"]
    assert model.queues == {"q": None}


def test_native_timeseries_and_text_reports(tmp_path):
    report = tmp_path / "native_report.txt"
    report_handle = sim.log_text(str(report))

    class Reports(sim.Model):
        ok: sim.Output
        n: sim.Output
        mean: sim.Output
        q: sim.Queue = sim.capacity(5)
        d: sim.Dataset

    model = Reports()

    @model.process
    def driver(env: Reports):
        for i in range(30):
            env.d.add(float(i % 7))
            sim.put(env.q, 1)
            sim.hold(0.5)
            sim.get(env.q, 1)
            sim.hold(0.5)
        sim.suspend()

    @model.collect
    def collect(env: Reports):
        ts = sim.queue_history(env.q)
        env.n = float(sim.timeseries_count(ts))
        env.mean = sim.timeseries_mean(ts)
        ok = sim.queue_report_file(env.q, report_handle, 0)
        ok += sim.timeseries_histogram_file(ts, report_handle, 1, 5, 0.0, 5.0)
        ok += sim.timeseries_pacf_correlogram_file(ts, report_handle, 1, 3)
        ok += env.d.histogram_file(report_handle, 1, 5, 0.0, 0.0)
        ok += env.d.pacf_correlogram_file(report_handle, 1, 3)
        env.ok = float(ok)

    exp = model.experiment(replications=1, duration=40.0, warmup=0.0,
                           seed=17)
    assert exp.run() == 0
    assert exp["ok"][0] == 5.0
    assert exp["n"][0] > 10.0
    assert 0.0 < exp["mean"][0] < 1.0

    text = report.read_text()
    assert "Buffer levels for q" in text
    assert "-1.0" in text and "1.0" in text
    assert "#" in text


def test_native_reports_print_to_stdout():
    class ConsoleReports(sim.Model):
        ok: sim.Output
        q: sim.Queue = sim.capacity(3)
        d: sim.Dataset

    model = ConsoleReports()

    @model.process
    def driver(env: ConsoleReports):
        for i in range(12):
            env.d.add(float(i % 3))
            sim.put(env.q, 1)
            sim.hold(0.25)
            sim.get(env.q, 1)
            sim.hold(0.25)
        sim.suspend()

    @model.collect
    def collect(env: ConsoleReports):
        ts = sim.queue_history(env.q)
        ok = sim.queue_report(env.q)
        ok += sim.timeseries_histogram(ts, 3, 0.0, 3.0)
        ok += env.d.histogram(bins=3, low=0.0, high=0.0)
        env.ok = float(ok)

    exp = model.experiment(replications=1, duration=10.0, warmup=0.0,
                           seed=23)
    text = capture_native_stdout(exp.run)
    assert exp["ok"][0] == 3.0
    assert "Buffer levels for q" in text
    assert "#" in text
