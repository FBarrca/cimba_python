"""Assigning cb.Ref fields in compiled model code (issue #24)."""

import pytest

import cimba as cb
from cimba.compiler.classes import ModelCompileError


def run(model, **kwargs):
    result = cb.Experiment(model, **kwargs).run(workers=1)
    assert not result.failed.any(), result.failure_reasons
    return result[model]


def value(output):
    return output.values.tolist()[0][0]


# ---------------------------------------------------------------- the issue's repro
class Link(cb.Model):  # dynamic: reassigning a Ref
    value: int = 0
    next: cb.Ref["Link"]


class Root(cb.Model):
    nil: Link
    first: cb.Output[int]
    second: cb.Output[int]

    @cb.process
    def run(self) -> None:
        head = cb.spawn(Link, value=1, next=self.nil)
        node = cb.spawn(Link, value=2, next=self.nil)
        node.next = head                  # was: Cannot cast Record(...) to uint64
        self.first = node.next.value
        self.second = node.next.next.value


def test_issue_repro_assigns_a_spawned_reference():
    root = Root()
    root.nil = Link()
    root.nil.next = root.nil
    root.nil.value = -1
    out = run(root)
    assert (value(out.first), value(out.second)) == (1, -1)


# ---------------------------------------------------------------- optional references
class Node(cb.Model):
    slot: int = 0
    count: int = 0
    next: cb.Ref["Node"] | None = None


class Chain(cb.Model):
    head: cb.Ref[Node] | None = None

    @cb.function
    def push(self, slot: int) -> None:
        link = cb.spawn(Node, slot=slot)
        head = self.head
        if head is not None:
            link.next = head              # was: Cannot cast OptionalType(...) to uint64
        self.head = link


class Builder(cb.Model):
    chain: Chain
    length: cb.Output[int]
    slots: cb.Output[int]
    cleared: cb.Output[int]

    @cb.process
    def run(self):
        for slot in range(1, 101):
            self.chain.push(slot)
        self.length = 0
        self.slots = 0
        node = self.chain.head
        while node is not None:
            self.length += 1
            if self.length == 1:
                self.slots = node.slot
            node = node.next
        # Unlink the second node: head.next = head.next.next
        head = self.chain.head
        if head is not None:
            second = head.next
            if second is not None:
                head.next = second.next
        node = self.chain.head
        self.cleared = 0
        while node is not None:
            self.cleared += node.slot
            node = node.next
        # Assign None to a static model's optional reference
        self.chain.head = None
        if self.chain.head is None:
            self.cleared += 1_000_000


def test_linked_list_of_spawned_models_is_built_and_walked():
    builder = Builder()
    builder.chain = Chain()
    out = run(builder, replications=2)
    assert value(out.length) == 100
    assert value(out.slots) == 100                    # the last pushed is the head
    assert value(out.cleared) == sum(range(1, 101)) - 99 + 1_000_000


def test_optional_reference_passed_at_spawn():
    class Pushes(cb.Model):
        head: cb.Ref[Node] | None = None
        total: cb.Output[int]

        @cb.process
        def run(self):
            for slot in range(5):
                self.head = cb.spawn(Node, slot=slot, next=self.head)
            self.total = 0
            node = self.head
            while node is not None:
                self.total = self.total * 10 + node.slot
                node = node.next

    assert value(run(Pushes()).total) == 43210


# ---------------------------------------------------------------- classes
class Policy(cb.Model):
    @cb.function
    def score(self) -> int:
        return 1


class Bonus(Policy):
    extra: int = 0

    @cb.function
    def score(self) -> int:
        return 100 + self.extra


class Other(cb.Model):
    @cb.function
    def score(self) -> int:
        return -1


class Desk(cb.Model):
    policy: cb.Ref[Policy]
    spare: cb.Ref[Policy] | None = None
    base: Policy
    out: cb.Output[int]

    @cb.process
    def run(self):
        self.out = self.policy.score()
        self.policy = cb.spawn(Bonus, extra=5)          # a subclass of the declared class
        self.out = self.out * 1000 + self.policy.score()
        self.spare = cb.spawn(Bonus, extra=7)
        spare = self.spare
        if spare is not None:
            self.out = self.out * 1000 + spare.score()
        self.policy = self.base                          # back to a static model


def test_reference_to_a_subclass_dispatches_on_the_actual_class():
    desk = Desk()
    desk.base = Policy()
    desk.policy = desk.base
    assert value(run(desk).out) == 1_105_107


def test_incompatible_class_is_a_compile_error_with_a_line():
    class Wrong(cb.Model):
        policy: cb.Ref[Policy]

        @cb.process
        def run(self):
            self.policy = cb.spawn(Other)  # pyright: ignore[reportAttributeAccessIssue]

    wrong = Wrong()
    wrong.policy = Policy()

    class Holder(cb.Model):
        wrong: Wrong
        policy: Policy

    holder = Holder()
    holder.policy = wrong.policy
    holder.wrong = wrong
    with pytest.raises(ModelCompileError,
                       match=r"(?s)test_ref_assignment.py:\d+: Wrong.run: .*cannot use a "
                             r"Other model where a Policy is expected.*"
                             r"test_ref_assignment.py \(\d+\)"):
        cb.Experiment(holder).run(workers=1)


def test_none_into_a_required_reference_is_a_compile_error():
    class Required(cb.Model):
        nil: Link
        node: cb.Ref[Link]

        @cb.process
        def run(self):
            self.node = None  # pyright: ignore[reportAttributeAccessIssue]

    model = Required()
    model.nil = Link()
    model.nil.next = model.nil
    model.node = model.nil
    with pytest.raises(ModelCompileError, match=r"None is not a Link model; declare the "
                                                r"reference cb.Ref\[Link\] \| None"):
        run(model)


def test_optional_into_a_required_reference_checks_at_run_time():
    class Narrows(cb.Model):
        nil: Link
        maybe: cb.Ref[Link] | None = None
        present: cb.Param[bool] = True
        out: cb.Output[int]

        @cb.process
        def run(self):
            node = cb.spawn(Link, value=3, next=self.nil)
            if self.present:
                self.maybe = node
            self.nil.next = self.maybe  # pyright: ignore[reportAttributeAccessIssue]  (None fails)
            self.out = self.nil.next.value

    model = Narrows()
    model.nil = Link()
    model.nil.next = model.nil
    assert value(run(model).out) == 3
    model.present = False
    result = cb.Experiment(model).run(workers=1)
    assert result.failed.all()
    assert result.failure_reasons[0][0] == "a None reference was used as a Link model"


def test_unchecked_none_reference_fails_the_trial():
    class Reads(cb.Model):
        maybe: cb.Ref[Node] | None = None
        out: cb.Output[int]

        @cb.process
        def run(self):
            self.out = 1
            self.out = self.maybe.slot  # pyright: ignore[reportOptionalMemberAccess]  (unchecked)

    result = cb.Experiment(Reads()).run(workers=1)
    assert result.failed.all()
    assert result.failure_reasons[0][0] == "a None reference was used as a Node model"


def test_children_are_structure_and_cannot_be_reassigned():
    class Swaps(cb.Model):
        a: Policy
        b: Policy

        @cb.process
        def run(self):
            self.a = self.b

    model = Swaps()
    model.a, model.b = Policy(), Policy()
    with pytest.raises(ModelCompileError, match="Swaps.a is a child model field.*cb.Ref"):
        run(model)


# ---------------------------------------------------------------- released models
class Counted(cb.Model):
    count: int = 0
    joins: cb.Array[int] = cb.Array(2)


def test_reference_to_a_released_model_stays_valid():
    """A released spawned model's record lives until the trial ends: its
    fields stay readable and writable, its processes never run again and its
    arrays are freed at once."""
    class Keeps(cb.Model):
        held: cb.Ref[Counted] | None = None
        count: cb.Output[int]
        length: cb.Output[int]

        @cb.process
        def run(self):
            item = cb.spawn(Counted, count=4)
            self.held = item
            cb.release(item)
            cb.hold(10.0)
            held = self.held
            if held is not None:
                held.count += 1
                self.count = held.count
                self.length = len(held.joins)

    out = run(Keeps())
    assert (value(out.count), value(out.length)) == (5, 0)


def test_optional_passed_at_spawn_for_a_required_reference():
    class Spawns(cb.Model):
        nil: Link
        maybe: cb.Ref[Link] | None = None
        present: cb.Param[bool] = True
        out: cb.Output[int]

        @cb.process
        def run(self):
            if self.present:
                self.maybe = self.nil
            node = cb.spawn(Link, value=1, next=self.maybe)
            self.out = node.next.value

    model = Spawns()
    model.nil = Link()
    model.nil.next = model.nil
    model.nil.value = 8
    assert value(run(model).out) == 8
    model.present = False
    result = cb.Experiment(model).run(workers=1)
    assert result.failure_reasons[0][0] == "a None reference was used as a Link model"
