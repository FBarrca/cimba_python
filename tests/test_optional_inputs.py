"""An optional input is always a native handle, even when unbound."""

import pytest

import cimba as cb
from cimba import inputs
from cimba.experiments import TrialsFailed
from cimba.schema import ModelDefinitionError


@pytest.mark.parametrize("scalar", [float, int, bool])
@pytest.mark.parametrize("explicit_none", [False, True])
def test_optional_input_is_empty_or_consumes_its_bound_source(scalar, explicit_none):
    class Reader(cb.Model):
        values: cb.Input[scalar] | None
        count: cb.Output[int]
        total: cb.Output[float]

        @cb.process
        def read(self):
            self.total = 0.0
            while self.values.remaining() != 0:
                self.total += self.values.next()
                self.count += 1

    model = Reader()
    if explicit_none:
        model.values = None
    result = cb.Experiment(model, replications=2).run(workers=2, on_failure="raise")
    assert result[model].count.values.tolist() == [[0.0, 0.0]]
    assert result[model].values.source[0]["method"] == "empty"
    assert result[model].values.consumed.tolist() == [[0, 0]]
    assert result[model].values.rows(0, 0).size == 0
    model.values = inputs.trace([1.0, 2.0])
    result = cb.Experiment(model).run(workers=1, on_failure="raise")
    assert result[model].count[0, 0] == 2
    assert result[model].total[0, 0] == sum(scalar(value) for value in [1.0, 2.0])
    assert result[model].values.rows(0, 0).tolist() == [1.0, 2.0]


def test_optional_input_next_fails_with_field_specific_message():
    class Reader(cb.Model):
        values: cb.Input[float] | None

        @cb.process
        def read(self):
            self.values.next()

    experiment = cb.Experiment(Reader(), replications=3)
    result = experiment.run(workers=2)
    assert result.failed.all()
    assert all("reader.values: optional input is unbound" in reason
               for reason in result.failure_reasons[0])
    with pytest.raises(TrialsFailed, match="optional input is unbound"):
        experiment.run(workers=1, on_failure="raise")


def test_required_inputs_and_optional_assignment_validation():
    class Required(cb.Model):
        values: cb.Input[float]

    with pytest.raises(ModelDefinitionError, match="input source missing"):
        cb.Experiment(Required())
    with pytest.raises(TypeError, match="input source"):
        Required().values = None

    class Optional(cb.Model):
        values: cb.Input[float] | None

    with pytest.raises(TypeError, match="input source"):
        Optional().values = 123


def test_optional_input_can_sweep_between_unbound_and_bound():
    class Reader(cb.Model):
        values: cb.Input[float] | None
        count: cb.Output[int]

        @cb.process
        def read(self):
            while self.values.remaining() != 0:
                self.values.next()
                self.count += 1

    model = Reader()
    model.values = cb.sweep(None, inputs.trace([1.0, 2.0]))
    result = cb.Experiment(model).run(workers=1, on_failure="raise")
    assert result[model].count.values.tolist() == [[0.0], [2.0]]
    assert [source["method"] for source in result[model].values.source] == ["empty", "trace"]


class OptionalChild(cb.Model):
    values: cb.Input[float] | None

    @cb.process
    def read(self):
        if self.values.remaining() != 0:
            self.values.next()


class UnboundChild(OptionalChild):
    @cb.process
    def read(self):
        self.values.next()


@pytest.mark.parametrize("child_type", [OptionalChild, UnboundChild])
@pytest.mark.parametrize("explicit_none", [False, True])
def test_spawned_optional_input_can_be_omitted(child_type, explicit_none):
    class Parent(cb.Model):
        @cb.process
        def create(self):
            if explicit_none:
                cb.spawn(child_type, values=None)
            else:
                cb.spawn(child_type)

    result = cb.Experiment(Parent()).run(workers=1)
    assert bool(result.failed[0, 0]) == (child_type is UnboundChild)
    if child_type is UnboundChild:
        assert "UnboundChild.values: optional input is unbound" in result.failure_reasons[0, 0]


@pytest.mark.parametrize("scalar", [float, int, bool])
def test_spawned_optional_input_accepts_a_matching_typed_handle(scalar):
    class Child(cb.Model):
        values: cb.Input[scalar] | None

    class Parent(cb.Model):
        values: cb.Input[scalar] = inputs.trace([2.5])
        observed: cb.Output[float]

        @cb.process
        def create(self):
            child = cb.spawn(Child, values=self.values)
            self.observed = child.values.next()

    model = Parent()
    result = cb.Experiment(model).run(workers=1, on_failure="raise")
    assert result[model].observed[0, 0] == scalar(2.5)
