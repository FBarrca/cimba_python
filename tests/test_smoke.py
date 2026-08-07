"""Smoke tests: confirm the package built and links the native C library."""

import os
import numpy as np
import pytest
import time

import cimba
import cimba.sim as sim

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

        @sim.process
        def actor(self):
            sim.log_user(userflag, msg)
            sim.log_user_i64(userflag, label_i, 7)
            sim.log_user_f64(userflag, label_f, 2.5)
            self.done = 1.0
            sim.suspend()

    model = LogModel()


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

        @sim.process
        def actor(self):
            sim.log_user(userflag, msg)
            self.done = 1.0
            sim.suspend()

    model = LogModel()


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

        @sim.process
        def no_log_actor(self):
            for _ in range(200):
                sim.hold(0.0)
            self.done = 1.0

    no_log = NoLog()


    class DisabledLog(sim.Model):
        done: sim.Output

        @sim.process
        def disabled_actor(self):
            for _ in range(200):
                sim.log_user(userflag, msg)
                sim.hold(0.0)
            self.done = 1.0

    disabled_log = DisabledLog()


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

    @sim.process
    def arrivals(self):
        while True:
            sim.hold(cimba.random.exponential(1.0 / self.utilization))
            self.queue.put(1)

    @sim.process
    def service(self):
        while True:
            sim.hold(1.0)
            self.queue.get(1)

    @sim.collect
    def collect_stats(self):
        self.avg_queue_length = self.queue.mean_level()


def test_sim_model_run():
    model = MM1("smoke")




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


def test_param_defaults_are_optional_visible_and_overridable():
    class Base(sim.Model):
        rate: sim.Param = 2.5

    class Defaults(Base):
        required: sim.Param
        scale: sim.Param = 4
        result: sim.Output

        @sim.process
        def run(self):
            self.result = self.rate * self.required * self.scale

    model = Defaults()
    assert model.params == ["rate", "required", "scale"]
    assert model.param_defaults == {"rate": 2.5, "scale": 4.0}


    defaulted = model.experiment(
        required=3.0, replications=1, duration=1.0)
    assert defaulted.run() == 0
    assert defaulted["result"][0] == 30.0
    assert defaulted["rate"][0] == 2.5
    assert defaulted["scale"][0] == 4.0

    swept = model.experiment(
        required=3.0,
        rate=[1.0, 2.0],
        scale=5.0,
        replications=1,
        duration=1.0,
    )
    assert swept.run() == 0
    assert swept["result"].tolist() == [15.0, 30.0]
    assert model.trial_seeds(
        seed=7, required=3.0, replications=2).shape == (2,)

    with pytest.raises(ValueError, match="missing parameter.*required"):
        model.experiment(replications=1, duration=1.0)


def test_param_defaults_reject_non_scalar_values():
    class BadString(sim.Model):
        value: sim.Param = "fast"

    class BadBool(sim.Model):
        value: sim.Param = True

    for cls in (BadString, BadBool):
        with pytest.raises(TypeError, match="default must be a real scalar"):
            cls()


def test_unbound_predicate_field_rejected():
    class Gate(sim.Model):
        x: sim.Param
        ready: sim.Predicate

        @sim.process
        def proc(self):
            sim.hold(1.0)

    model = Gate()


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

        @sim.process
        def producer(self):
            self.max_level = 0.0
            self.space_ok = 1.0
            while True:
                self.q.put(1)       # blocks while the queue is full
                lvl = self.q.level()
                if lvl > self.max_level:
                    self.max_level = lvl
                if self.q.space() + lvl != 5:
                    self.space_ok = 0.0
                self.d.add(1.0 * lvl)
                sim.hold(0.5)

        @sim.process
        def consumer(self):
            while True:
                sim.hold(1.0)
                self.q.get(1)

        @sim.collect
        def stats(self):
            self.d_min = self.d.min()
            self.d_max = self.d.max()
            self.d_std = self.d.std()

    model = Bounded()




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

        @sim.process
        def feed(self):
            sim.hold(2.0)           # tally inside the measurement window
            for i in range(1, 7):
                self.d.add(1.0 * i)
            sim.suspend()

        @sim.collect
        def stats(self):
            self.med = self.d.median()
            self.q0 = self.d.quantile(0.0)
            self.q25 = self.d.quantile(0.25)
            self.q100 = self.d.quantile(1.0)
            self.med_empty = self.d_empty.median()

    model = Quant()



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

        @sim.process
        def p(self):
            self.y = self.x * 2.0
            self.z = cimba.random.uniform(0.0, 1.0)
            sim.suspend()

    model = Sweep()


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


def test_many_scalar_params_do_not_hit_a_mesh_dimension_limit():
    # Scalar params are held fixed, so declaring more than numpy's
    # 32-dimension meshgrid limit must still build a single design point.
    n = 40
    names = [f"p{i}" for i in range(n)]
    Many = type("Many", (sim.Model,), {
        "__annotations__": {**{p: sim.Param for p in names},
                            "y": sim.Output},
    })
    def p(self):
        self.y = self.p0 + self.p39
        sim.suspend()
    Many.p = sim.process(p)
    model = Many()

    fixed = {name: float(i) for i, name in enumerate(names)}
    exp = model.experiment(duration=10.0, warmup=0.0, seed=3, **fixed)
    assert exp.trials.size == 1
    assert exp.swept == ()
    for i, name in enumerate(names):
        assert exp.trials[name][0] == float(i)
    assert exp.run() == 0
    assert exp["y"][0] == 39.0

    # sweeping a couple of them keeps the cross product in params order:
    # the last-declared param varies fastest.
    swept = {**fixed, "p38": [1.0, 2.0], "p39": [10.0, 20.0, 30.0]}
    exp = model.experiment(duration=10.0, warmup=0.0, seed=3, **swept)
    assert exp.trials.size == 6
    assert exp.swept == ("p38", "p39")
    assert list(exp.trials["p38"]) == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    assert list(exp.trials["p39"]) == [10.0, 20.0, 30.0, 10.0, 20.0, 30.0]
    assert list(exp.trials["p7"]) == [7.0] * 6
    assert len(model.trial_seeds(seed=3, **swept)) == 6


def test_param_cross_product_order_and_degenerate_axes():
    class Grid(sim.Model):
        x: sim.Param
        y: sim.Param
        out: sim.Output

        @sim.process
        def p(self):
            self.out = self.x
            sim.suspend()

    model = Grid()


    # design-point-major with replications innermost, x the outer axis
    exp = model.experiment(x=[1.0, 2.0], y=[10.0, 20.0], replications=2,
                           duration=1.0, warmup=0.0)
    assert list(exp.trials["x"]) == [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]
    assert list(exp.trials["y"]) == [10.0, 10.0, 20.0, 20.0,
                                     10.0, 10.0, 20.0, 20.0]

    # an empty axis still collapses the design to no trials
    empty = model.experiment(x=[], y=[1.0, 2.0], duration=1.0, warmup=0.0)
    assert empty.trials.size == 0


def test_shaped_param_held_fixed_while_a_scalar_param_sweeps():
    class Leg(sim.Component):
        rate: sim.Param

        @sim.process
        def idle(self, env):
            sim.suspend()

    class Net(sim.Model):
        x: sim.Param
        legs: list[Leg] = [Leg(), Leg()]
        y: sim.Output

        @sim.process
        def p(self):
            self.y = self.legs[1].rate * self.x
            sim.suspend()

    model = Net()


    # a shaped param given one row is a singleton axis, not a mesh dimension
    exp = model.experiment(x=[1.0, 2.0], legs__rate=[3.0, 4.0],
                           duration=10.0, warmup=0.0, seed=5)
    assert exp.trials.size == 2
    assert exp.swept == ("x",)
    assert np.array_equal(exp.trials["legs__rate"], [[3.0, 4.0]] * 2)
    assert exp.run() == 0
    assert np.allclose(exp["y"], [4.0, 8.0])

    # given design rows it sweeps, and x (declared first) is the outer axis
    exp = model.experiment(x=[1.0, 3.0],
                           legs__rate=[[1.0, 1.0], [1.0, 2.0]],
                           duration=10.0, warmup=0.0, seed=5)
    assert exp.trials.size == 4
    assert exp.swept == ("x", "legs__rate")
    assert list(exp.trials["x"]) == [1.0, 1.0, 3.0, 3.0]
    assert np.array_equal(exp.trials["legs__rate"][:, 1], [1.0, 2.0, 1.0, 2.0])
    assert exp.run() == 0
    assert np.allclose(exp["y"], [1.0, 2.0, 3.0, 6.0])


def test_experiment_summary_single_point():
    class Single(sim.Model):
        y: sim.Output

        @sim.process
        def p(self):
            self.y = cimba.random.uniform(0.0, 1.0)
            sim.suspend()

    model = Single()


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

        @sim.process
        def draw(self):
            self.tri = cimba.random.triangular(0.0, 1.0, 2.0)
            self.wei = cimba.random.weibull(1.5, 2.0)
            self.lgn = cimba.random.lognormal(0.0, 0.5)
            self.erl = cimba.random.erlang(3, 2.0)
            self.bet = cimba.random.beta(2.0, 3.0, 0.0, 1.0)
            self.poi = cimba.random.poisson(4.0)
            self.die = cimba.random.dice(1, 6)
            sim.suspend()           # idle until the trial ends

    model = Draws()


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

        @sim.process
        def draw(self):
            self.sn = cimba.random.normal()
            self.se = cimba.random.exponential()
            self.sg = cimba.random.gamma(2.5)
            self.sb = cimba.random.beta(2.0, 3.0)
            self.pm = cimba.random.pert_mod(0.0, 4.0, 10.0, 6.0)
            self.lg = cimba.random.logistic(0.0, 1.0)
            self.cy = cimba.random.cauchy(0.0, 1.0)
            self.pr = cimba.random.pareto(2.5, 1.0)
            self.cs = cimba.random.chi_squared(4.0)
            self.fd = cimba.random.f_dist(5.0, 8.0)
            self.st = cimba.random.student_t(7.0)
            self.td = cimba.random.student_t(7.0, 1.0, 2.0)
            self.geo = cimba.random.geometric(0.4)
            self.binom = cimba.random.binomial(10, 0.4)
            self.nbinom = cimba.random.negative_binomial(3, 0.4)
            self.pas = cimba.random.negative_binomial(3, 0.4)
            self.hypo = cimba.random.hypoexponential((1.0, 2.0, 4.0, 8.0))
            self.hyper = cimba.random.hyperexponential(
                (1.0, 2.0, 4.0, 8.0), (0.1, 0.2, 0.3, 0.4))
            self.loaded = cimba.random.categorical([0.2, 0.3, 0.5])
            self.cat = cimba.random.categorical(CAT_PROBABILITIES)
            sim.suspend()

    model = ExtraDraws()


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

        @sim.process
        def draw(self):
            self.u = cimba.random.uniform()
            self.n = cimba.random.normal(mu=0.0, sigma=1.0)
            self.e = cimba.random.exponential()
            self.g = cimba.random.gamma(shape=2.0)
            self.cat = cimba.random.categorical((0.2, 0.3, 0.5))
            self.hyper = cimba.random.hyperexponential(
                (1.0, 2.0), probabilities=(0.25, 0.75))
            self.td = cimba.random.student_t(v=7.0, m=1.0, s=2.0)
            self.chi = cimba.random.chi_squared(k=4.0)
            sim.suspend()

    model = RandomAPI()


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

        @sim.process(copies=2, field="worker")
        def worker_process(self, idx: int):
            if idx == 0:
                self.got_sig = sim.hold(1000.0)  # interrupted by the poker
            else:
                while True:
                    sim.hold(1000.0)

        @sim.process
        def poker(self):
            sim.hold(1.0)
            sim.interrupt(self.worker[0], 42, 0)
            while True:
                sim.hold(1000.0)

    model = Game()



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

        @sim.process(field="target")
        def target_process(self):
            self.target_signal = sim.suspend()
            while True:
                sim.hold(1000.0)

        @sim.process
        def controller(self):
            me = sim.current()
            cancelled = sim.timer_add(me, 5.0, sim.TIMEOUT)
            self.cancel_first = sim.timer_cancel(me, cancelled)
            self.cancel_second = sim.timer_cancel(me, cancelled)
            self.constants_ok = 1.0
            if sim.TIMEOUT != -5 or sim.CANCELLED != -4:
                self.constants_ok = 0.0

            sim.hold(0.1)
            target_timer = sim.timer_add(self.target[0], 1.0, sim.TIMEOUT)
            self.waited = sim.wait_event(target_timer)
            while True:
                sim.hold(1000.0)

    model = Timeout()



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

        @sim.event(field="ring")
        def on_ring(self, data: int):
            self.fired_at = sim.now()
            self.payload = data
            self.cur_ok = 1.0 if sim.current_event() != 0 else 0.0

        @sim.event
        def bump(self):
            self.counter += 1

        @sim.process
        def driver(self):
            h = self.ring.schedule(2.0, 42, 7)
            self.sched_ok = h.scheduled()
            self.t_sched = h.time()
            self.count_ok = 1.0 if sim.event_count() >= 1 else 0.0
            h.reschedule(sim.now() + 3.0)
            self.t_resched = h.time()
            h.reprioritize(9)
            self.prio_after = h.priority()
            self.wait_status = h.wait_event()

            h2 = self._ev_bump.schedule(1.0)  # defaults: data/priority
            self.cancel_first = h2.cancel()
            self.cancel_second = h2.cancel()
            self._ev_bump.schedule_at(sim.now() + 1.0)
            sim.hold(2.0)
            self.n_bumps = self.counter
            while True:
                sim.hold(1000.0)

    model = Evented()




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

        @sim.process
        def runner(self):
            sim.hold(1.0)
            self.ended_at = sim.now()
            self.had_events = 1.0 if sim.event_count() > 0 else 0.0
            sim.clear_events()
            sim.suspend()

    model = Clearer()


    exp = model.experiment(replications=1, duration=100.0, warmup=10.0,
                           seed=31)
    assert exp.run() == 0
    assert exp["ended_at"][0] == 1.0
    assert exp["had_events"][0] == 1.0


def test_unbound_event_field_rejected():
    class Gate(sim.Model):
        x: sim.Param
        ring: sim.Event

        @sim.process
        def proc(self):
            sim.hold(1.0)

    model = Gate()


    with pytest.raises(ValueError, match="ring"):
        model.experiment(x=1.0)


def test_pqueues_and_timers():
    class Shop(sim.Model):
        served_first: sim.Output    # object taken first (priority order)
        timed_out: sim.Output       # signal a waiter got from its timer
        qs: sim.PQueues = sim.count(2)

        @sim.process
        def producer(self):
            self.qs[0].put(7, 0)     # low priority first
            self.qs[0].put(8, 5)     # high priority second
            sim.hold(1.0)
            self.served_first = self.qs[0].take()  # leftover entry

        @sim.process
        def consumer(self):
            sim.hold(0.5)
            self.served_first = self.qs[0].take()

        @sim.process
        def waiter(self):
            me = sim.current()
            sim.timer_set(me, 2.0, 99)
            self.timed_out = sim.suspend()
            while True:
                sim.hold(1000.0)

    model = Shop()




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

        @sim.process
        def actor(self):
            q = self.qs[0]
            q.put(7, 0)
            q.put(8, 5)

            status, obj = q.get()
            self.first_status = status
            self.first_obj = obj
            self.take_obj = q.take()

            me = sim.current()
            sim.timer_set(me, 1.0, sim.TIMEOUT)
            status, obj = q.get()
            self.timeout_status = status
            self.timeout_obj = obj
            while True:
                sim.hold(1000.0)

    model = PQGet()


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

        @sim.process
        def actor(self):
            q = self.qs[0]
            low = q.put(10, 0)
            q.put(20, 5)
            self.space_ok = 0.0
            if q.space() > q.length():
                self.space_ok = 1.0
            self.pos_before = q.position(low)
            q.reprioritize(low, 10)
            self.pos_after = q.position(low)
            sim.hold(1.0)
            self.first = q.take()
            sim.hold(1.0)
            q.take()
            sim.suspend()

        @sim.collect
        def collect(self):
            self.mean_len = self.qs[0].mean_length()

    model = PQStats()



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

        @sim.process
        def actor(self):
            me = sim.current()
            self.resource.acquire()
            self.held_before = self.resource.held(me)
            self.resource.release()
            self.held_after = self.resource.held(me)

            self.store.put(0)
            status, obj = self.store.get()
            self.zero_status = status
            self.zero_obj = obj

            self.store.put(41)
            self.store.put(42)
            self.pos = self.store.position(42)
            self.store.take()
            self.store.take()

            sim.timer_set(me, 1.0, sim.TIMEOUT)
            status, obj = self.store.get()
            self.timeout_status = status
            self.timeout_obj = obj
            while True:
                sim.hold(1000.0)

    model = StoreResource()


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

    class InvalidStruct(sim.Model):
        @sim.process(struct=int)
        def proc(self):
            sim.hold(1.0)

    with pytest.raises(ValueError, match="Struct subclass"):
        InvalidStruct()

    class MisplacedStruct(sim.Model):
        @sim.process
        def misplaced(self, tag: Base, idx: int):
            sim.hold(1.0)

    with pytest.raises(ValueError, match="last parameter"):
        MisplacedStruct()

    class MismatchedStruct(sim.Model):
        @sim.process(struct=Derived)
        def mismatched(self, tag: Base):
            sim.hold(1.0)

    with pytest.raises(ValueError, match="disagree"):
        MismatchedStruct()


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

        @sim.process(copies=2, field="clerk")
        def clerk_process(self, idx: int, tag: Tag):
            ok = 1.0 if (tag.ticket == 0 and tag.stamp == 0.0) else 0.0
            if idx == 0:
                self.zero_ok = ok
            tag.ticket = 10 + idx
            sim.hold(2.0)           # the stamper writes our stamp at t=1
            if idx == 0:
                self.s0 = tag.stamp
            else:
                self.s1 = tag.stamp
            sim.suspend()

        @sim.process
        def stamper(self, own: Tag):
            own.ticket = 99         # plain (env, view) form, own fields
            sim.hold(1.0)
            self.t0 = 1.0 * Tag(self.clerk[0]).ticket
            self.t1 = 1.0 * Tag(self.clerk[1]).ticket
            Tag(self.clerk[0]).stamp = 0.5
            Tag(self.clerk[1]).stamp = 1.5 + 0.01 * own.ticket
            sim.suspend()

    model = Office()



    exp = model.experiment(replications=1, duration=10.0, warmup=0.0,
                           seed=99)
    assert exp.run() == 0
    assert exp["zero_ok"][0] == 1.0    # fields start zeroed
    assert exp["t0"][0] == 10.0        # each copy has its own record
    assert exp["t1"][0] == 11.0
    assert exp["s0"][0] == 0.5         # writes through the handle stick
    assert exp["s1"][0] == 1.5 + 0.01 * 99


def test_legacy_spawnable_declaration_explains_decorator_migration():
    def declare_legacy_model():
        class Legacy(sim.Model):
            worker: sim.Spawnable

        return Legacy()

    with pytest.raises(ValueError, match="spawnable=True"):
        declare_legacy_model()


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

        @sim.process(spawnable=True)
        def worker(self, it: Item):
            sim.hold(1.0)
            self.done += 1
            self.acc += it.weight

        @sim.process
        def spawner(self):
            h1 = sim.spawn(self.worker, self)
            Item(h1).weight = 2.5      # runs only once we block: init first
            h2 = sim.spawn(self.worker, self, 3)
            Item(h2).weight = 4.0
            self.distinct = 1.0 if h1 != h2 else 0.0
            sim.wait_process(h1)
            sim.wait_process(h2)
            self.made = 1.0 * self.done
            self.total = self.acc
            sim.despawn(h1)
            sim.despawn(h2)
            sim.suspend()

    model = Factory()



    exp = model.experiment(replications=1, duration=10.0, warmup=0.0,
                           seed=5)
    assert exp.run() == 0
    assert exp["distinct"][0] == 1.0
    assert exp["made"][0] == 2.0
    assert exp["total"][0] == 6.5


def test_decorated_model_process_is_spawnable():
    class Agent(sim.Struct):
        value: int

    class Model(sim.Model):
        finished: sim.Output

        @sim.process(spawnable=True)
        def agent(self, item: Agent):
            self.finished = item.value

        @sim.process
        def start(self):
            item = Agent(sim.spawn(self.agent, self))
            item.value = 3
            sim.hold(0.0)

    model = Model()



    experiment = model.experiment()
    assert experiment.run() == 0
    assert experiment.trials["finished"][0] == 3


def test_spawned_leftovers_reclaimed():
    # Spawned processes still alive at trial end are stopped and
    # reclaimed like the static ones, and despawn is idempotent.
    class Hive(sim.Model):
        spawned: sim.Output
        redespawn_ok: sim.Output

        @sim.process(spawnable=True)
        def drone(self):
            sim.suspend()       # blocks forever; never despawned

        @sim.process
        def queen(self):
            for _ in range(50):
                sim.spawn(self.drone, self)
            h = sim.spawn(self.drone, self)
            sim.hold(1.0)
            sim.despawn(h)
            sim.despawn(h)      # double despawn must be a no-op
            self.redespawn_ok = 1.0
            self.spawned = 51.0
            sim.suspend()

    model = Hive()



    exp = model.experiment(replications=20, duration=10.0, warmup=0.0,
                           seed=11)
    assert exp.run() == 0
    assert (exp["spawned"] == 51.0).all()
    assert (exp["redespawn_ok"] == 1.0).all()
    assert exp.run() == 0   # rerun on the same compiled trial


def test_spawnable_decorator_rejects_multiple_copies():
    with pytest.raises(ValueError, match="spawnable.*copies"):
        class Invalid(sim.Model):
            @sim.process(copies=3, spawnable=True)
            def ghost(self):
                sim.hold(1.0)


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

        @sim.process
        def driver(self):
            for i in range(30):
                self.d.add(float(i % 7))
                self.q.put(1)
                sim.hold(0.5)
                self.q.get(1)
                sim.hold(0.5)
            sim.suspend()

        @sim.collect
        def collect(self):
            self.n = float(self.q.history().count())
            self.mean = self.q.history().mean()
            ok = self.q.report_file(report_handle, 0)
            ok += self.q.history().histogram_file(report_handle, 1, 5, 0.0, 5.0)
            ok += self.q.history().pacf_correlogram_file(report_handle, 1, 3)
            ok += self.d.histogram_file(report_handle, 1, 5, 0.0, 0.0)
            ok += self.d.pacf_correlogram_file(report_handle, 1, 3)
            self.ok = float(ok)

    model = Reports()



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

        @sim.process
        def driver(self):
            for i in range(12):
                self.d.add(float(i % 3))
                self.q.put(1)
                sim.hold(0.25)
                self.q.get(1)
                sim.hold(0.25)
            sim.suspend()

        @sim.collect
        def collect(self):
            ok = self.q.report()
            ok += self.q.history().histogram(3, 0.0, 3.0)
            ok += self.d.histogram(bins=3, low=0.0, high=0.0)
            self.ok = float(ok)

    model = ConsoleReports()



    exp = model.experiment(replications=1, duration=10.0, warmup=0.0,
                           seed=23)
    text = capture_native_stdout(exp.run)
    assert exp["ok"][0] == 3.0
    assert "Buffer levels for q" in text
    assert "#" in text
