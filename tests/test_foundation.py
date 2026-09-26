"""Architecture invariants that do not require a compiled trial."""

import numpy as np
import pytest

from cimba import inputs
from cimba.experiments import Design, trial_seed
from cimba.layout import RecordLayout, TrialImageLayout
from cimba.modeling import Input, Model, Output, Param, Ref, Series, State, sweep, sweeps
from cimba.schema import Assembly, ClassSchema, ModelDefinitionError


@pytest.mark.parametrize("factory", [
    lambda x: inputs.bootstrap.iid(x),
    lambda x: inputs.bootstrap.moving_block(x, block=4),
    lambda x: inputs.bootstrap.circular_block(x, block=4),
    lambda x: inputs.bootstrap.stationary(x, mean_block=5),
    lambda x: inputs.fitted.residual(x, trend=1),
    lambda x: inputs.fitted.wild(x, trend=1),
    lambda x: inputs.fitted.sieve(x, order=1),
])
def test_row_sources_are_prefix_stable_and_batch_equivalent(factory):
    source = factory(np.linspace(1, 20, 40) + np.sin(np.arange(40)))
    short = source.generate([inputs.trace_rng(123, "a")], 17)
    long = source.generate([inputs.trace_rng(123, "a")], 39)
    np.testing.assert_array_equal(short[0], long[0, :17])
    batch = source.generate([inputs.trace_rng(123, "a"),
                             inputs.trace_rng(123, "b")], 17)
    np.testing.assert_array_equal(batch[0], short[0])
    np.testing.assert_array_equal(
        batch[1], source.generate([inputs.trace_rng(123, "b")], 17)[0])
    assert not batch.flags.writeable
    assert source.describe()["version"] == 1


def test_intermittent_is_prefix_stable():
    source = inputs.bootstrap.intermittent([0, 1, 0, 2, 0, 3] * 8,
                                           jitter=True)
    short = source.generate([inputs.trace_rng(45, "demand")], 11)
    long = source.generate([inputs.trace_rng(45, "demand")], 61)
    np.testing.assert_array_equal(short[0], long[0, :11])


def test_joint_members_preserve_alignment():
    panel = inputs.bootstrap.joint([[1, 2, 3, 4] * 10,
                                    [10, 20, 30, 40] * 10], mean_block=3)
    a = panel[0].generate([inputs.trace_rng(12, panel.tag)], 25)[0]
    b = panel[1].generate([inputs.trace_rng(12, panel.tag)], 25)[0]
    np.testing.assert_array_equal(b, a * 10)


class Base(Model):
    level: Param[float] = 1.0
    total: State[float] = 0.0


class Node(Base):
    demand: Series[float] = Series(step=1.0)
    lead_time: Input[float] = inputs.dist.exponential(mean=2.0)
    served: Output[float]


class Network(Model):
    nodes: list[Node]

    def __init__(self):
        self.nodes = [Node(), Node()]


def test_assembly_and_record_prefix_are_object_based():
    root = Network()
    root.nodes[0].demand = inputs.trace([1, 2, 3])
    root.nodes[1].demand = inputs.bootstrap.iid([1, 2, 3])
    level_sweep = sweep(1.0, 2.0)
    setattr(root.nodes[0], "level", level_sweep)
    assembly = Assembly.of(root)
    image = TrialImageLayout.of(assembly)
    assert image.by_object[root.nodes[1]].label == "network.nodes[1]"
    assert image.by_object[root.nodes[0]].offset != image.by_object[root.nodes[1]].offset
    base = RecordLayout.of(ClassSchema.of(Base))
    child = RecordLayout.of(ClassSchema.of(Node))
    assert base.dtype.names is not None
    for name in base.dtype.names:
        assert base.offset(name) == child.offset(name)
    design = Design.of(assembly)
    assert design.levels(level_sweep) == (1.0, 2.0)
    assert design.points[0].bindings[root.nodes[1], "demand"] is root.nodes[1].demand


def test_linked_sweeps_and_common_seeding():
    a, b = sweeps([1, 2], [10, 20])
    root = Network()
    for node in root.nodes:
        node.demand = inputs.trace([1, 2])
    setattr(root.nodes[0], "level", a)
    setattr(root.nodes[1], "level", b)
    design = Design.of(Assembly.of(root))
    assert len(design.points) == 2
    assert trial_seed(4, 5, 0) == trial_seed(4, 5, 1)
    assert trial_seed(4, 5, 0, seeding="independent") != trial_seed(
        4, 5, 1, seeding="independent")


def test_unbound_input_rejected_at_assembly():
    root = Network()
    with pytest.raises(ModelDefinitionError, match="network.nodes\\[0\\].demand"):
        Assembly.of(root)


class LinkedNode(Model):
    parent: Ref["LinkedRoot"] | None


class LinkedRoot(Model):
    nodes: list[LinkedNode]
    links: list[Ref[LinkedNode]]

    def __init__(self):
        self.nodes = [LinkedNode(), LinkedNode()]
        self.links = list(reversed(self.nodes))
        for node in self.nodes:
            node.parent = self


def test_optional_and_collection_refs_are_links_not_children():
    root = LinkedRoot()
    assembly = Assembly.of(root)
    assert len(assembly.instances) == 3
    assert ClassSchema.of(LinkedNode).fields[0].kind == "ref"
    assert assembly.by_object[root.nodes[0]].values["parent"] is root
