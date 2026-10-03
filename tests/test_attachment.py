"""Owned children bind their owner through a host hook, including sweep options."""

import numpy as np
import pytest

import cimba as cb


class Policy(cb.Model):
    owner: cb.Ref["Planner"]
    increment: cb.Param[int] = 1
    own_count: cb.Output[int]

    def __init__(self):
        self.attachments = []

    def attached(self, owner, field):
        super().attached(owner, field)
        assert getattr(owner, field) is not None
        self.owner = owner
        self.attachments.append((owner, field))

    @cb.process
    def apply(self):
        self.owner.count += self.increment
        self.own_count += 1


class OtherPolicy(Policy):
    increment: cb.Param[int] = 3


class Planner(cb.Model):
    method: Policy
    count: cb.State[int] = 0
    total: cb.Output[int]

    @cb.on_end
    def report(self):
        self.total = self.count


def test_child_attachment_is_immediate_and_inherited():
    owner = Planner()
    child = OtherPolicy()
    owner.method = child
    assert child.owner is owner
    assert child.attachments == [(owner, "method")]
    cb.Experiment(owner).describe()
    cb.Experiment(owner).run(workers=1, on_failure="raise")
    assert child.attachments == [(owner, "method")]


def test_every_sweep_option_is_attached_and_runs_only_when_selected():
    owner = Planner()
    policies = [Policy(), OtherPolicy()]
    owner.method = cb.sweep(policies)
    for policy in policies:
        assert policy.owner is owner
        assert policy.attachments == [(owner, "method")]
    result = cb.Experiment(owner, replications=2).run(workers=2, on_failure="raise")
    assert result[owner].total.values.tolist() == [[1.0, 1.0], [3.0, 3.0]]
    for index, policy in enumerate(policies):
        assert (result[policy].own_count.values[index] == 1).all()
        assert np.isnan(result[policy].own_count.values[1 - index]).all()
        assert policy.attachments == [(owner, "method")]


def test_optional_and_collection_children_attach_but_references_do_not():
    class Owner(Planner):
        method: Policy | None = None
        children: list[Policy]
        selected: cb.Ref[Policy] | None
        references: list[cb.Ref[Policy]]

    owner = Owner()
    children = [Policy(), OtherPolicy()]
    owner.method = None
    owner.children = children
    owner.selected = children[0]
    owner.references = children
    for child in children:
        assert child.attachments == [(owner, "children")]
    cb.Experiment(owner).run(workers=1, on_failure="raise")


@pytest.mark.parametrize("swept", [False, True])
def test_default_children_materialize_and_attach_once_per_owner(swept):
    template = cb.sweep(Policy(), OtherPolicy()) if swept else Policy()

    class Owner(Planner):
        method: Policy = template

    first, second = Owner(), Owner()
    for owner in (first, second):
        cb.Experiment(owner).run(workers=1, on_failure="raise")
        cb.Experiment(owner).describe()
        for child in cb.options(owner.method):
            assert child.owner is owner
            assert child.attachments == [(owner, "method")]
    assert first.method is not second.method
    assert all(not child.attachments for child in cb.options(template))


def test_options_returns_only_the_sweep_alternatives():
    value = Policy()
    assert cb.options(value) == (value,)
    assert cb.options(None) == (None,)
    values = [1, 2]
    assert cb.options(values) == (values,)
    assert cb.options(cb.sweep(1, 2)) == (1, 2)


def test_invalid_child_assignment_does_not_invoke_hooks():
    owner = Planner()
    child = Policy()
    with pytest.raises(TypeError, match="sweep must contain Policy models"):
        owner.method = cb.sweep(child, cb.Model())
    assert child.attachments == []
    with pytest.raises(TypeError, match="expects a Policy child"):
        owner.method = 3
