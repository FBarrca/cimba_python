"""cb.Array fields: configuration-sized, per-instance tables (issue #24)."""

import sys
import time

import numpy as np
import pytest

import cimba as cb
from cimba.compiler.classes import ModelCompileError
from cimba.schema import ModelDefinitionError


def run(model, **kwargs):
    result = cb.Experiment(model, **kwargs).run(workers=1)
    assert not result.failed.any(), result.failure_reasons
    return result[model]


def value(output):
    return output.values.tolist()[0][0]


def failure(model, **kwargs):
    result = cb.Experiment(model, **kwargs).run(workers=1)
    assert result.failed.all()
    return str(result.failure_reasons[0][0])


# ---------------------------------------------------------------- the issue
class Case(cb.Model):
    case_id: int = 0
    joins: cb.Array[int]                    # length set per spawn


class Walker(cb.Model):
    join_slots: cb.Param[int] = 0           # known when the model is built
    fired: cb.Output[int]
    counted: cb.Output[int]
    length: cb.Output[int]

    @cb.function
    def start_case(self, case_id: int) -> Case:
        case = cb.spawn(Case, case_id=case_id, joins=self.join_slots)
        case.joins[3] += 1                  # plain indexing in compiled code
        return case

    @cb.function
    def all_arrived(self, case: Case, first: int, count: int) -> bool:
        for slot in range(first, first + count):
            if case.joins[slot] == 0:
                return False
        return True

    @cb.process
    def run(self) -> None:
        # A barrier: join 0 has branches 0..2, join 1 has branches 3..5
        self.fired = 0
        for case_id in range(10):
            case = self.start_case(case_id)
            for branch in (0, 1, 2, 5, 4):
                case.joins[branch] += 1
                if self.all_arrived(case, 0, 3) and self.all_arrived(case, 3, 3):
                    self.fired += 1
            total = 0
            for count in case.joins:
                total += count
            self.counted = total
            self.length = len(case.joins)
            cb.release(case)
            cb.hold(1.0)


def test_barrier_counters_sized_from_a_param():
    walker = Walker()
    walker.join_slots = 8
    out = run(walker)
    assert value(out.fired) == 10          # each case fires once, on its last arrival
    assert value(out.counted) == 6
    assert value(out.length) == 8


def test_spawn_length_can_be_swept_without_recompiling():
    walker = Walker()
    walker.join_slots = cb.sweep(6, 8, 64)  # pyright: ignore[reportAttributeAccessIssue]
    misses = cb.cache_info().misses
    out = run(walker)
    assert out.length.values.tolist() == [[6.0], [8.0], [64.0]]
    walker.join_slots = cb.sweep(7, 9)  # pyright: ignore[reportAttributeAccessIssue]
    assert run(walker).length.values.tolist() == [[7.0], [9.0]]
    assert cb.cache_info().misses - misses <= 2      # Walker and Case, once


# ---------------------------------------------------------------- bounds
class Indexed(cb.Model):
    length: cb.Param[int] = 4
    index: cb.Param[int] = 0
    write: cb.Param[bool] = False
    out: cb.Output[int]

    @cb.process
    def run(self):
        case = cb.spawn(Case, joins=self.length)
        if self.write:
            case.joins[self.index] = 1
        self.out = case.joins[self.index]


@pytest.mark.parametrize("index", [4, 5, -1, 1 << 40])
@pytest.mark.parametrize("write", [False, True])
def test_out_of_range_index_fails_the_trial(index, write):
    model = Indexed()
    model.index = index
    model.write = write
    assert failure(model) == f"Case.joins[{index}]: index out of range for length 4"


def test_negative_spawn_length_fails_the_trial():
    model = Indexed()
    model.length = -3
    assert failure(model) == "Case.joins: array length -3 is negative"


def test_zero_length_array():
    class Empty(cb.Model):
        out: cb.Output[int]

        @cb.process
        def run(self):
            case = cb.spawn(Case, joins=0)
            total = 0
            for count in case.joins:
                total += count
            self.out = len(case.joins) + total

    assert value(run(Empty()).out) == 0
    model = Indexed()
    model.length = 0
    assert failure(model) == "Case.joins[0]: index out of range for length 0"


# ---------------------------------------------------------------- spawned lifecycle
class Defaults(cb.Model):
    counts: cb.Array[int] = cb.Array(3)
    weights: cb.Array[float] = cb.Array([0.5, 1.5])
    seen: cb.Array[bool] = cb.Array(2)


class Reuse(cb.Model):
    dirty: cb.Output[int]
    lengths: cb.Output[int]
    weights: cb.Output[float]

    @cb.process
    def run(self):
        self.dirty = 0
        self.lengths = 0
        for round in range(50):
            item = cb.spawn(Defaults, counts=round % 7)
            for i in range(len(item.counts)):
                self.dirty += item.counts[i]
                item.counts[i] = 99
            if item.seen[0] or item.seen[1]:
                self.dirty += 1
            item.seen[1] = True
            self.lengths += len(item.counts)
            self.weights = item.weights[0] + item.weights[1]
            item.weights[0] = 100.0
            cb.release(item)
            cb.hold(1.0)


def test_each_spawn_gets_fresh_contents_of_its_own_length():
    out = run(Reuse())
    assert value(out.dirty) == 0
    assert value(out.lengths) == sum(round % 7 for round in range(50))
    assert value(out.weights) == 2.0       # the class default's values, every time


def test_class_default_length_and_values_for_spawn():
    class Uses(cb.Model):
        out: cb.Output[float]

        @cb.process
        def run(self):
            item = cb.spawn(Defaults)
            item.seen[1] = True
            self.out = (len(item.counts) * 100 + len(item.weights) * 10 +
                        item.weights[1] + (1.0 if item.seen[1] else 0.0))

    assert value(run(Uses()).out) == 300 + 20 + 1.5 + 1.0


def test_released_model_frees_its_arrays():
    class Keeps(cb.Model):
        length: cb.Output[int]
        after: cb.Output[int]

        @cb.process
        def run(self):
            case = cb.spawn(Case, case_id=7, joins=4)
            joins = case.joins
            cb.release(case)
            self.length = len(joins)        # a released model's arrays are gone
            self.after = case.case_id       # its record and scalars remain
            self.after += joins[0]

    result = cb.Experiment(Keeps()).run(workers=1)
    assert result.failed.all()
    assert result.failure_reasons[0][0] == \
        "Case.joins[0]: the model was released, so its arrays are gone"


def test_iteration_ends_when_the_model_is_released():
    class Loop(cb.Model):
        steps: cb.Output[int]

        @cb.process
        def run(self):
            case = cb.spawn(Case, joins=5)
            self.steps = 0
            for _ in case.joins:
                self.steps += 1
                if self.steps == 2:
                    cb.release(case)

    assert value(run(Loop()).steps) == 2


def test_spawn_requires_an_int_length_or_a_class_default():
    class Missing(cb.Model):
        @cb.process
        def run(self):
            cb.spawn(Case)

    with pytest.raises(ModelCompileError, match=r"Case.joins: spawn length required"):
        run(Missing())

    class FloatLength(cb.Model):
        @cb.process
        def run(self):
            cb.spawn(Case, joins=2.0)

    with pytest.raises(ModelCompileError, match="array length as an int"):
        run(FloatLength())


class Spawner(cb.Model):
    count: cb.Param[int] = 100_000
    bad: cb.Param[bool] = False
    total: cb.Output[int]

    @cb.process
    def run(self):
        self.total = 0
        case = cb.spawn(Case, joins=16)
        for i in range(self.count):
            case = cb.spawn(Case, case_id=i, joins=16)
            case.joins[i % 16] += 1
            self.total += case.joins[i % 16]
            if not self.bad:
                cb.release(case)
        if self.bad:
            self.total += case.joins[16]


def test_many_spawned_arrays():
    started = time.monotonic()
    out = run(Spawner())
    assert value(out.total) == 100_000
    assert time.monotonic() - started < 30


@pytest.mark.skipif(sys.platform == "win32", reason="uses the resource module")
def test_abandoned_trials_free_spawned_arrays():
    import resource
    model = Spawner()
    model.count = 200          # 200 live cases of 16 counters in every trial
    model.bad = True
    cb.Experiment(model, replications=5).run(workers=1)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = cb.Experiment(model, replications=2000).run(workers=1)
    assert result.failed.all()
    assert result.failure_reasons[0][0] == "Case.joins[16]: index out of range for length 16"
    growth = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before
    scale = 1 if sys.platform == "darwin" else 1024   # bytes on macOS, KiB on Linux
    # A leak would keep 2000 x 200 x 128 bytes of elements alone (51 MB).
    assert growth * scale < 40 * 2**20


# ---------------------------------------------------------------- static models
class Table(cb.Model):
    counts: cb.Array[int] = cb.Array(4)
    weights: cb.Array[float] = cb.Array([1.0, 2.0, 3.0])
    flags: cb.Array[bool] = cb.Array([True, False])
    index: cb.Param[int] = 0
    total: cb.Output[float]
    length: cb.Output[int]

    @cb.process
    def run(self):
        self.counts[self.index] += 5
        self.weights[0] *= 10.0
        self.flags[1] = not self.flags[1]
        total = 0.0
        for weight in self.weights:
            total += weight
        for count in self.counts:
            total += count
        self.total = (total + (100.0 if self.flags[0] else 0.0) +
                      (1000.0 if self.flags[1] else 0.0))
        self.length = len(self.counts)


def test_static_array_class_default():
    out = run(Table(), replications=3)
    # Every trial starts from the configured contents
    assert out.total.values.tolist() == [[1120.0] * 3]
    assert value(out.length) == 4


def test_static_array_configured_on_the_host():
    table = Table()
    table.counts = cb.Array(10)
    table.weights = cb.Array([0.5])
    table.index = 9
    out = run(table)
    assert value(out.length) == 10
    assert value(out.total) == 5 + 5.0 + 1100


def test_static_array_length_can_be_swept():
    table = Table()
    table.counts = cb.sweep(cb.Array(1), cb.Array(6),  # pyright: ignore[reportAttributeAccessIssue]
                            cb.Array([1, 1, 1]))
    table.index = 0
    out = run(table, replications=2)
    assert out.length.values.tolist() == [[1.0, 1.0], [6.0, 6.0], [3.0, 3.0]]
    assert out.total.values.tolist() == [[1120.0] * 2, [1120.0] * 2, [1123.0] * 2]
    table.index = 2
    result = cb.Experiment(table).run(workers=1)
    assert result.failed[:, 0].tolist() == [True, False, False]
    assert result.failure_reasons[0][0] == "Table.counts[2]: index out of range for length 1"


def test_static_array_bound_through_children_and_functions():
    class Holder(cb.Model):
        values: cb.Array[int] = cb.Array([3, 4])

        @cb.function
        def at(self, i: int) -> int:
            return self.values[i]

    class Owner(cb.Model):
        holders: list[Holder]
        one: Holder
        out: cb.Output[int]

        @cb.function
        def read(self, holder: Holder, i: int) -> int:
            return holder.values[i]

        @cb.process
        def run(self):
            self.holders[1].values[0] += 10
            self.out = (self.holders[1].values[0] + self.read(self.holders[0], 1) +
                        self.one.at(0) * 100)

    owner = Owner()
    owner.holders = [Holder(), Holder()]
    owner.one = Holder()
    owner.one.values = cb.Array([7])
    assert value(run(owner).out) == 13 + 4 + 700


def test_static_arrays_survive_input_extension_reruns():
    class Extends(cb.Model):
        counts: cb.Array[int] = cb.Array(2)
        demand: cb.Input[float]
        out: cb.Output[int]

        @cb.process
        def run(self):
            for _ in range(10):
                self.demand.next()
                self.counts[1] += 1
            self.out = self.counts[1]

    model = Extends()
    model.demand = cb.inputs.row_source(
        "test.constant", [0.1, 0.2], lambda rng, length: np.full(length, 0.1), length_hint=2)
    result = cb.Experiment(model).run(workers=1)
    assert result[model].demand.extended.tolist() == [[3]]   # the trial ran four times
    assert value(result[model].out) == 10


def test_array_configuration_errors():
    table = Table()
    with pytest.raises(TypeError, match="Table.counts expects a cb.Array"):
        table.counts = 4  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(ValueError, match="nonnegative"):
        cb.Array(-1)
    table.weights = cb.Array(["x"])
    with pytest.raises(ModelDefinitionError, match=r"table.weights\[0\]: expected a float"):
        cb.Experiment(table)

    class Unsized(cb.Model):
        counts: cb.Array[int]

    with pytest.raises(ModelDefinitionError, match="unsized.counts: array length missing"):
        cb.Experiment(Unsized())

    class Strings(cb.Model):
        names: cb.Array[str]

    with pytest.raises(ModelDefinitionError, match="elements must be int, float or bool"):
        cb.Experiment(Strings())


def test_array_typing_errors():
    class Replace(cb.Model):
        counts: cb.Array[int] = cb.Array(2)

        @cb.process
        def run(self):
            self.counts = self.counts

    with pytest.raises(ModelCompileError,
                       match=r"(?s)test_arrays.py:\d+: Replace.run: .*Replace.counts is an array "
                             r"field: assign its elements.*test_arrays.py \(\d+\)"):
        run(Replace())

    class Truncates(cb.Model):
        counts: cb.Array[int] = cb.Array(2)

        @cb.process
        def run(self):
            self.counts[0] = 1.5  # pyright: ignore[reportArgumentType]

    with pytest.raises(ModelCompileError, match="Truncates.counts holds int values, got float64"):
        run(Truncates())


def test_entity_on_spawned_model_points_to_arrays_and_refs():
    class WithStore(cb.Model):
        arrivals: cb.Store[int] = cb.Store()

    class Spawns(cb.Model):
        @cb.process
        def run(self):
            cb.spawn(WithStore)

    with pytest.raises(ModelCompileError,
                       match="do not support these fields: arrivals.*cb.Array.*cb.Ref"):
        run(Spawns())
