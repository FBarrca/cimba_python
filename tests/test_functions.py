"""@cb.function: methods callable from compiled code, with dynamic dispatch."""

import numpy as np
import pytest

import cimba as cb
from cimba import inputs
from cimba.compiler import ModelCompileError
from cimba.schema import ClassSchema, ModelDefinitionError


class Policy(cb.Model):
    @cb.function
    def order_quantity(self, on_hand: float, position: float) -> float:
        return 0.0

    @cb.function
    def level(self, verbose: bool = False) -> int:
        return 0


class BaseStock(Policy):
    target: cb.Param[float] = 100.0

    @cb.function
    def order_quantity(self, on_hand: float, position: float) -> float:
        return max(0.0, self.target - position)

    @cb.function
    def level(self, verbose: bool = False) -> int:
        return 2 if verbose else 1


class MinMax(Policy):
    low: cb.Param[float] = 20.0
    high: cb.Param[float] = 120.0

    @cb.function
    def order_quantity(self, on_hand: float, position: float) -> float:
        return self.high - position if position <= self.low else 0.0


class Shop(cb.Model):
    policy: cb.Ref[Policy]
    policies: list[Policy]
    stock: cb.Container = cb.Container(initial=3)
    through_ref: cb.Output[float]
    through_list: cb.Output[float]
    inherited: cb.Output[int]
    flag: cb.Output[int]
    chained: cb.Output[float]
    recursive: cb.Output[int]
    blocked_until: cb.Output[float]
    picked: cb.Output[float]

    @cb.function
    def twice(self, x: float, k: int = 2) -> float:
        return k * x

    @cb.function
    def factorial(self, n: int) -> int:
        return 1 if n <= 1 else n * self.factorial(n - 1)

    @cb.function
    def take_after(self, delay: float) -> None:
        cb.hold(delay)
        self.stock.get(1)

    @cb.function
    def pick(self, first: Policy, second: Policy, use_first: bool) -> Policy:
        return first if use_first else second

    @cb.process
    def run(self):
        self.through_ref = self.policy.order_quantity(10.0, 15.0)
        self.through_list = self.policies[1].order_quantity(10, 15)
        self.inherited = self.policies[1].level()          # MinMax inherits Policy's
        self.flag = self.policies[0].level(True)
        self.chained = self.twice(self.policies[0].order_quantity(1.0, 2.0))
        self.recursive = self.factorial(5)
        self.take_after(2.5)
        self.blocked_until = cb.now()
        chosen = self.pick(self.policies[0], self.policies[1], False)
        self.picked = chosen.order_quantity(0.0, 10.0)


def make_shop():
    shop = Shop()
    shop.policies = [BaseStock(), MinMax()]
    shop.policy = shop.policies[0]
    return shop


def test_dynamic_dispatch_defaults_recursion_and_blocking():
    shop = make_shop()
    results = cb.Experiment(shop).run(workers=1)
    assert not results.failed.any(), results.failure_reasons
    got = {name: results[shop].__getattr__(name)[0, 0] for name in (
        "through_ref", "through_list", "inherited", "flag", "chained",
        "recursive", "blocked_until", "picked")}
    assert got == {"through_ref": 85.0, "through_list": 105.0, "inherited": 0.0,
                   "flag": 2.0, "chained": 196.0, "recursive": 120.0,
                   "blocked_until": 2.5, "picked": 110.0}


def test_host_calls_are_plain_python():
    assert BaseStock().order_quantity(0.0, 40.0) == 60.0


def test_swapping_implementations_never_recompiles_the_caller():
    first = make_shop()
    cb.Experiment(first).run(workers=1)
    second = make_shop()
    second.policy = second.policies[1]            # MinMax instead of BaseStock
    results = cb.Experiment(second).run(workers=1)
    assert results.meta.compile.misses == 0
    assert results[second].through_ref[0, 0] == 105.0


class Worker(cb.Model):
    crew: cb.Ref["Crew"]
    speed: cb.Param[float] = 1.0

    @cb.function
    def duration(self, size: float) -> float:
        return size / self.speed

    @cb.process
    def work(self):
        job = self.crew.jobs.get()
        cb.hold(self.duration(job.size))
        self.crew.finished += 1
        cb.release(job)


class FastWorker(Worker):
    @cb.function
    def duration(self, size: float) -> float:
        return 0.5 * size / self.speed


class Job(cb.Model):
    size: cb.State[float] = 1.0

    @cb.function
    def weight(self) -> float:
        return 2.0 * self.size


class Crew(cb.Model):
    workers: list[Worker]
    jobs: cb.Store[Job]
    finished: cb.State[int] = 0
    total_weight: cb.State[float] = 0.0
    done_at: cb.Output[float]
    weight: cb.Output[float]

    def __init__(self):
        self.workers = [Worker(), FastWorker()]
        for worker in self.workers:
            worker.crew = self

    @cb.process
    def dispatch(self):
        for _ in range(2):
            job = cb.spawn(Job, size=4.0)
            self.total_weight += job.weight()          # function on a spawned model
            self.jobs.put(job)

    @cb.on_end
    def finish(self):
        self.done_at = cb.now()
        self.weight = self.total_weight


def test_functions_in_list_items_and_spawned_models():
    crew = Crew()
    results = cb.Experiment(crew).run(workers=1)
    assert not results.failed.any(), results.failure_reasons
    assert results[crew].done_at[0, 0] == 4.0      # slow worker: 4.0, fast: 2.0
    assert results[crew].weight[0, 0] == 16.0


def test_reserved_names_are_rejected():
    class Shadow(cb.Model):
        @cb.function
        def describe(self) -> int:  # pyright: ignore[reportIncompatibleMethodOverride]
            return 0

    with pytest.raises(ModelDefinitionError, match="reserved"):
        ClassSchema.of(Shadow)


def test_missing_annotations_are_rejected():
    class Bad(cb.Model):
        @cb.function
        def f(self, x) -> float:
            return x

    with pytest.raises(ModelDefinitionError, match="annotate parameter 'x'"):
        ClassSchema.of(Bad)

    class NoReturn(cb.Model):
        @cb.function
        def f(self, x: float):
            return x

    with pytest.raises(ModelDefinitionError, match="return type"):
        ClassSchema.of(NoReturn)


def test_overrides_must_keep_signature_and_decorator():
    class Changed(Policy):
        @cb.function
        def order_quantity(self, on_hand: int, position: float) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]
            return 0.0

    with pytest.raises(ModelDefinitionError, match="keep the signature"):
        ClassSchema.of(Changed)

    class Undecorated(Policy):
        def order_quantity(self, on_hand: float, position: float) -> float:
            return 0.0

    with pytest.raises(ModelDefinitionError, match="must also be decorated"):
        ClassSchema.of(Undecorated)


def test_calling_an_undecorated_method_explains_the_fix():
    class Plain(cb.Model):
        value: cb.Output[float]

        def helper(self) -> float:
            return 1.0

        @cb.process
        def run(self):
            self.value = self.helper()

    with pytest.raises(ModelCompileError, match="not a @cb.function"):
        cb.Experiment(Plain()).run(workers=1)


def test_wrong_argument_type_is_a_compile_error():
    class Caller(cb.Model):
        policy: cb.Ref[Policy]
        out: cb.Output[float]

        @cb.process
        def run(self):
            self.out = self.policy.order_quantity(1.0)  # pyright: ignore[reportCallIssue]

    caller = Caller()
    caller.policy = BaseStock()
    holder = type("Holder", (cb.Model,), {"__annotations__": {
        "caller": Caller, "policy": BaseStock}})()
    holder.caller, holder.policy = caller, caller.policy
    with pytest.raises(ModelCompileError, match="takes 2 to 2 arguments"):
        cb.Experiment(holder).run(workers=1)


def test_distribution_inputs_inside_functions():
    class Sampler(cb.Model):
        draws: cb.Input[float] = inputs.trace([1.0, 2.0, 3.0])
        total: cb.Output[float]

        @cb.function
        def next_two(self) -> float:
            return self.draws.next() + self.draws.next()

        @cb.process
        def run(self):
            self.total = self.next_two()

    sampler = Sampler()
    results = cb.Experiment(sampler).run(workers=1)
    assert results[sampler].total[0, 0] == 3.0


class Store(cb.Model):
    policy: Policy
    demand: cb.Input[float] = inputs.dist.exponential(mean=10.0)
    on_hand: cb.State[float] = 50.0
    position: cb.State[float] = 50.0
    ordered: cb.State[float] = 0.0
    total_ordered: cb.Output[float]
    first_demand: cb.Output[float]

    @cb.process
    def operate(self):
        for day in range(30):
            cb.hold(1.0)
            demand = self.demand.next()
            if day == 0:
                self.first_demand = demand
            self.on_hand -= demand
            self.position -= demand
            quantity = self.policy.order_quantity(self.on_hand, self.position)
            self.position += quantity
            self.ordered += quantity

    @cb.on_end
    def finish(self):
        self.total_ordered = self.ordered


class Audited(Policy):
    """A policy with its own process: it must only run where it is selected."""

    audits: cb.State[int] = 0
    audit_count: cb.Output[int]

    @cb.process
    def audit(self):
        while True:
            cb.hold(7.0)
            self.audits += 1

    @cb.on_end
    def report(self):
        self.audit_count = self.audits


def test_sweeping_a_child_model_runs_one_option_per_trial():
    store = Store()
    options = [BaseStock(), MinMax(), Audited()]
    store.policy = cb.sweep(options)  # pyright: ignore[reportAttributeAccessIssue]  (a list works too)
    results = cb.Experiment(store, replications=4, seed=3,
                            window=cb.Window(duration=40.0)).run(workers=2)
    assert not results.failed.any(), results.failure_reasons
    assert results.levels(store.policy) == tuple(options)
    assert results.meta.variants == 3
    ordered = results[store].total_ordered.values
    assert ordered.shape == (3, 4)
    assert (ordered[0] != ordered[1]).all()                 # different policies
    first = results[store].first_demand.values
    assert (first[0] == first[1]).all() and (first[1] == first[2]).all()   # CRN
    audits = results[options[2]].audit_count.values
    assert (audits[2] == 5).all()                           # runs where selected...
    assert np.isnan(audits[:2]).all()                       # ...and nowhere else
    comparison = cb.analysis.compare(results[store].total_ordered, a=0, b=1)
    assert comparison.n == 4


def test_child_sweeps_cross_with_other_sweeps_and_rerun_single_trials():
    store = Store()
    options = [BaseStock(), Audited()]
    store.policy = cb.sweep(*options)  # pyright: ignore[reportAttributeAccessIssue]
    options[0].target = cb.sweep(80.0, 120.0)
    experiment = cb.Experiment(store, replications=2, seed=5,
                               window=cb.Window(duration=40.0))
    results = experiment.run(workers=1)
    assert results.failed.shape == (4, 2)                  # 2 policies x 2 targets
    again = experiment.only(trials=[5])                    # point 2, replication 1
    assert again[store].total_ordered[0, 0] == results[store].total_ordered[2, 1]


def test_child_sweep_options_must_match_the_field_type():
    store = Store()
    with pytest.raises(TypeError, match="must contain Policy models"):
        store.policy = cb.sweep(1.0, 2.0)  # pyright: ignore[reportAttributeAccessIssue]


def test_function_names_do_not_shadow_entity_methods():
    class Named(cb.Model):
        box: cb.Store[float]
        level_now: cb.Container = cb.Container(initial=4)
        got: cb.Output[float]
        via_function: cb.Output[float]
        level_out: cb.Output[int]

        @cb.function
        def get(self, x: float) -> float:
            return 10.0 * x

        @cb.function
        def level(self) -> int:
            return -1

        @cb.process
        def run(self):
            self.box.put(2.5)
            self.got = self.box.get()                 # Store.get, not Named.get
            self.via_function = self.get(1.5)         # Named.get
            self.level_out = self.level_now.level()   # Container.level

    named = Named()
    results = cb.Experiment(named).run(workers=1)
    assert not results.failed.any(), results.failure_reasons
    assert results[named].got[0, 0] == 2.5
    assert results[named].via_function[0, 0] == 15.0
    assert results[named].level_out[0, 0] == 4.0
