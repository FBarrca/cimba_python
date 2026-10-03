"""Declared views must be ready before a caller is compiled."""

import pytest
from numba import from_dtype

import cimba as cb
from cimba.compiler.numba_compat import cpu_target
from cimba.compiler.views import ModelViewAttributes, register_views
from cimba.layout import RecordLayout
from cimba.schema import ClassSchema


@pytest.mark.parametrize("via_list", [False, True])
@pytest.mark.parametrize("caller_first", [False, True])
def test_nested_reference_access_is_independent_of_tree_order(via_list, caller_first):
    # Fresh classes for each order: a prior run must not supply cached views.
    class Part(cb.Model):
        taken: cb.Output[int]

        @cb.function
        def take(self, n: int) -> int:
            self.taken += n
            return n

    class Item(cb.Model):
        part: cb.Ref[Part]

    if via_list:
        class Caller(cb.Model):
            items: list[Item]

            @cb.process
            def go(self):
                self.items[0].part.take(3)
    else:
        class Caller(cb.Model):
            item: Item

            @cb.process
            def go(self):
                self.item.part.take(3)

    fields = {"caller": Caller, "part": Part}
    if not caller_first:
        fields = dict(reversed(fields.items()))
    root = type("Root", (cb.Model,), {"__annotations__": fields})()
    root.part = Part()
    root.caller = Caller()
    item = Item()
    item.part = root.part
    if via_list:
        root.caller.items = [item]
    else:
        root.caller.item = item
    result = cb.Experiment(root).run(workers=1, on_failure="raise")
    assert result[root.part].taken[0, 0] == 3


def test_base_reference_reads_inherited_fields_without_base_instances():
    class Part(cb.Model):
        total: cb.Output[int]

        @cb.function
        def add(self, amount: int) -> None:
            self.total += amount

    class Base(cb.Model):
        flag: cb.State[bool] = True
        part: cb.Ref[Part]
        quantity: cb.State[int] = 7

    class Derived(Base):
        extra: cb.State[float] = 99.0

    class Caller(cb.Model):
        owner: cb.Ref[Base]

        @cb.process
        def go(self):
            if self.owner.flag:
                self.owner.part.add(self.owner.quantity)

    base = RecordLayout.of(ClassSchema.of(Base))
    derived = RecordLayout.of(ClassSchema.of(Derived))
    for field in base.schema.fields:
        assert derived.offset(field.name) == base.offset(field.name)
    assert derived.offset("extra") >= base.dtype.itemsize
    root = type("Root", (cb.Model,), {"__annotations__": {
        "caller": Caller, "owner": Derived, "part": Part}})()
    root.caller, root.owner, root.part = Caller(), Derived(), Part()
    root.owner.part = root.part
    root.caller.owner = root.owner
    result = cb.Experiment(root).run(workers=1, on_failure="raise")
    assert result[root.part].total[0, 0] == 7


def test_missing_view_reports_the_model_field():
    class Unregistered(cb.Model):
        child: cb.Ref[cb.Model]

    record = from_dtype(RecordLayout.of(ClassSchema.of(Unregistered)).dtype)
    template = ModelViewAttributes(cpu_target.typing_context)
    with pytest.raises(TypeError, match="no view registered for Unregistered.child"):
        template.generic_resolve(record, "child")


def test_inheritance_cannot_change_the_native_type_of_a_base_field():
    class Base(cb.Model):
        value: cb.State[bool] = True

    class Changed(Base):
        value: cb.State[float] = 1.0

    with pytest.raises(TypeError, match="Changed.value: native layout must preserve"):
        RecordLayout.of(ClassSchema.of(Changed))


def test_incompatible_multiple_inheritance_cannot_shift_base_field_offsets():
    class Left(cb.Model):
        left: cb.State[int] = 1

    class Right(cb.Model):
        right: cb.State[int] = 2

    class Combined(Left, Right):
        pass

    with pytest.raises(TypeError, match="native layout must preserve"):
        RecordLayout.of(ClassSchema.of(Combined))


def test_model_argument_base_views_are_ready_before_compiling_functions():
    class Part(cb.Model):
        total: cb.Output[int]

        @cb.function
        def take(self) -> int:
            self.total += 1
            return 1

    class Base(cb.Model):
        part: cb.Ref[Part]

    class Derived(Base):
        extra: cb.State[int] = 5

    class Caller(cb.Model):
        item: Derived
        part: Part

        @cb.function
        def take(self, item: Base) -> int:
            return item.part.take()

        @cb.process
        def run(self):
            self.take(self.item)

    model = Caller()
    model.part, model.item = Part(), Derived()
    model.item.part = model.part
    result = cb.Experiment(model).run(workers=1, on_failure="raise")
    assert result[model.part].total[0, 0] == 1


def test_reference_on_spawn_result_is_ready_during_typing():
    class Part(cb.Model):
        total: cb.Output[int]

        @cb.function
        def take(self) -> None:
            self.total += 1

    class Dynamic(cb.Model):
        part: cb.Ref[Part]

    class Caller(cb.Model):
        part: Part

        @cb.process
        def run(self):
            dynamic = cb.spawn(Dynamic, part=self.part)
            dynamic.part.take()

    model = Caller()
    model.part = Part()
    result = cb.Experiment(model).run(workers=1, on_failure="raise")
    assert result[model.part].total[0, 0] == 1


def test_callback_views_can_be_registered_again_after_cache_eviction():
    class Callback(cb.Model):
        ready: cb.State[bool] = False
        observed: cb.Output[bool]

        @cb.predicate
        def is_ready(self):
            return self.ready

        @cb.event
        def activate(self):
            self.ready = True

        @cb.process
        def run(self):
            cb.schedule(self.activate, 1.0)
            cb.hold(2.0)
            self.observed = self.ready

    model = Callback()
    experiment = cb.Experiment(model)
    assert experiment.run(workers=1, on_failure="raise")[model].observed[0, 0] == 1
    cb.clear_cache()
    register_views.cache_clear()
    assert experiment.run(workers=1, on_failure="raise")[model].observed[0, 0] == 1
