"""enum.IntEnum members as compile-time constants, field types and function types."""

import enum

import numpy as np
import pytest

import cimba as cb
from cimba.compiler import ModelCompileError
from cimba.schema import ClassSchema, ModelDefinitionError


class Kind(enum.IntEnum):
    TASK = 1
    GATEWAY = 2


class NodeKind(enum.IntEnum):
    START = 0
    END = 1
    TASK = 2
    EXCLUSIVE = 3


class Color(enum.Enum):
    RED = 1
    BLUE = 2


def run(model, **kwargs):
    results = cb.Experiment(model, **kwargs).run(workers=1, on_failure="raise")
    return results


# ------------------------------------------------------------------ constants
class Root(cb.Model):
    out: cb.Output[float]

    @cb.process
    def run(self) -> None:
        kind = Kind.GATEWAY
        self.out = float(kind) if kind == Kind.GATEWAY else 0.0


def test_issue_repro():
    root = Root()
    assert run(root)[root].out[0, 0] == 2.0


def classify(kind: NodeKind) -> int:
    match kind:
        case NodeKind.START:
            return 10
        case NodeKind.TASK:
            return 20
        case _:
            return 30


class Constants(cb.Model):
    as_int: cb.Output[int]
    as_float: cb.Output[float]
    arithmetic: cb.Output[float]
    scaled: cb.Output[float]
    comparisons: cb.Output[int]
    matched: cb.Output[int]
    chained: cb.Output[int]
    indexed: cb.Output[float]
    implicit: cb.Output[float]
    truthy: cb.Output[bool]
    samples: cb.Dataset = cb.Dataset()

    @cb.process
    def run(self) -> None:
        kind = NodeKind.EXCLUSIVE
        self.as_int = int(kind)
        self.as_float = float(kind)
        self.arithmetic = kind + 1 - NodeKind.TASK * 2
        self.scaled = kind * 0.5
        count = 0
        if kind == 3:
            count += 1
        if 3 == kind:
            count += 1
        if kind != NodeKind.TASK:
            count += 1
        if kind == NodeKind.EXCLUSIVE:
            count += 1
        if NodeKind.START < kind <= NodeKind.EXCLUSIVE:
            count += 1
        self.comparisons = count
        match kind:
            case NodeKind.TASK:
                self.matched = 1
            case NodeKind.EXCLUSIVE:
                self.matched = 2
            case _:
                self.matched = 3
        if kind == NodeKind.START:
            self.chained = 0
        elif kind == NodeKind.TASK:
            self.chained = 1
        else:
            self.chained = 2
        table = np.zeros(4)
        table[NodeKind.TASK] = 7.0
        self.indexed = table[NodeKind.TASK] + table[kind]
        self.implicit = kind              # a member is accepted where a float goes
        self.truthy = bool(NodeKind.START)
        self.samples.record(float(kind))


def test_members_compare_convert_and_dispatch():
    model = Constants()
    model.samples.capture()
    got = run(model)[model]
    assert {name: got.__getattr__(name)[0, 0] for name in (
        "as_int", "as_float", "arithmetic", "scaled", "comparisons", "matched",
        "chained", "indexed", "implicit", "truthy")} == {
        "as_int": 3.0, "as_float": 3.0, "arithmetic": 0.0, "scaled": 1.5,
        "comparisons": 5.0, "matched": 2.0, "chained": 2.0, "indexed": 7.0,
        "implicit": 3.0, "truthy": 0.0}
    _, values = got.samples.trial(0, 0)
    assert values.tolist() == [3.0]


# ------------------------------------------------------------------ functions
class Router(cb.Model):
    @cb.function
    def route(self, kind: NodeKind) -> NodeKind:
        if kind == NodeKind.START:
            return NodeKind.TASK
        return NodeKind.END

    @cb.function
    def code(self, value: int) -> int:
        return value * 10

    @cb.function
    def half(self, value: float) -> float:
        return value / 2

    @cb.function
    def or_default(self, kind: NodeKind = NodeKind.EXCLUSIVE) -> int:
        return int(kind)


class FastRouter(Router):
    @cb.function
    def route(self, kind: NodeKind) -> NodeKind:
        return NodeKind.EXCLUSIVE


class Caller(cb.Model):
    router: cb.Ref[Router]
    routed: cb.Output[NodeKind]
    routed_twice: cb.Output[int]
    coded: cb.Output[int]
    halved: cb.Output[float]
    defaulted: cb.Output[int]

    @cb.process
    def run(self) -> None:
        next_kind = self.router.route(NodeKind.START)
        self.routed = next_kind
        self.routed_twice = int(self.router.route(next_kind))
        self.coded = self.router.code(NodeKind.TASK)
        self.halved = self.router.half(NodeKind.EXCLUSIVE)
        self.defaulted = self.router.or_default()


class Pair(cb.Model):
    caller: Caller
    router: Router


def make_pair(router: Router) -> Pair:
    pair = Pair()
    pair.router = router
    pair.caller = Caller()
    pair.caller.router = router
    return pair


def test_members_pass_to_and_return_from_functions():
    pair = make_pair(Router())
    got = run(pair)[pair.caller]
    assert (got.routed[0, 0], got.routed_twice[0, 0], got.coded[0, 0],
            got.halved[0, 0], got.defaulted[0, 0]) == (2.0, 1.0, 20.0, 1.5, 3.0)
    assert ClassSchema.of(Router).signature("route").types == (NodeKind,)


def test_enum_functions_dispatch_on_the_class():
    pair = make_pair(FastRouter())
    assert run(pair)[pair.caller].routed[0, 0] == NodeKind.EXCLUSIVE


class WrongArgument(cb.Model):
    router: Router

    @cb.process
    def run(self) -> None:
        self.router.route(2)  # pyright: ignore[reportArgumentType]


class OtherEnumArgument(cb.Model):
    router: Router

    @cb.process
    def run(self) -> None:
        self.router.route(Kind.TASK)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("model", [WrongArgument, OtherEnumArgument])
def test_enum_parameters_require_members_of_their_enum(model):
    instance = model()
    instance.router = Router()
    with pytest.raises(ModelCompileError, match="expects a NodeKind member"):
        run(instance)


# ------------------------------------------------------------------ fields and params
class Node(cb.Model):
    kind: cb.Param[NodeKind] = NodeKind.TASK
    state: NodeKind = NodeKind.START
    seen: cb.Output[NodeKind]
    code: cb.Output[int]
    branch: cb.Output[int]
    state_after: cb.Output[NodeKind]

    @cb.process
    def run(self) -> None:
        self.seen = self.kind
        self.code = int(self.kind) * 100 + int(self.state)
        if self.kind == NodeKind.TASK:
            self.branch = 1
        elif self.kind == NodeKind.EXCLUSIVE:
            self.branch = 2
        self.state = NodeKind.END
        self.state_after = self.state


def test_param_default_and_plain_field_in_compiled_code():
    node = Node()
    got = run(node)[node]
    assert (got.seen[0, 0], got.code[0, 0], got.branch[0, 0],
            got.state_after[0, 0]) == (2.0, 200.0, 1.0, 1.0)
    assert NodeKind(int(got.seen[0, 0])) is NodeKind.TASK


def test_host_assignment_is_validated_and_reads_back_members():
    node = Node()
    assert node.kind is NodeKind.TASK
    node.kind = NodeKind.EXCLUSIVE
    assert node.kind is NodeKind.EXCLUSIVE
    # A valid value becomes its member.
    node.kind = 0  # pyright: ignore[reportAttributeAccessIssue]
    assert node.kind is NodeKind.START
    node.state = np.int64(3)  # pyright: ignore[reportAttributeAccessIssue]
    assert node.state is NodeKind.EXCLUSIVE
    for bad, message in ((7, "7 is not a NodeKind value"),
                         (Kind.TASK, "expected a NodeKind member"),
                         (2.0, "expected a NodeKind member"),
                         (True, "expected a NodeKind member"),
                         ("TASK", "expected a NodeKind member")):
        with pytest.raises(ModelDefinitionError, match=f"Node.kind: {message}"):
            node.kind = bad  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(ModelDefinitionError, match="Node.kind: 9 is not"):
        node.kind = cb.sweep(NodeKind.TASK, 9)  # pyright: ignore[reportAttributeAccessIssue]
    assert node.kind is NodeKind.START


def test_class_defaults_are_validated():
    class Coerced(cb.Model):
        kind: cb.Param[NodeKind] = 3  # pyright: ignore[reportAssignmentType]

    assert ClassSchema.of(Coerced).fields[0].default is NodeKind.EXCLUSIVE

    class Invalid(cb.Model):
        kind: cb.Param[NodeKind] = 9  # pyright: ignore[reportAssignmentType]

    with pytest.raises(ModelDefinitionError, match="Invalid.kind: 9 is not a NodeKind"):
        ClassSchema.of(Invalid)


def test_sweep_over_members():
    node = Node()
    node.kind = cb.sweep(NodeKind.TASK, NodeKind.EXCLUSIVE, 0)  # pyright: ignore[reportAttributeAccessIssue]
    results = run(node, replications=2)
    assert results.design.levels(node.kind) == (
        NodeKind.TASK, NodeKind.EXCLUSIVE, NodeKind.START)
    np.testing.assert_array_equal(results[node].seen.values, [[2, 2], [3, 3], [0, 0]])
    np.testing.assert_array_equal(results[node].branch.values, [[1, 1], [2, 2], [0, 0]])


class WritesInt(cb.Model):
    state: NodeKind = NodeKind.START

    @cb.process
    def run(self) -> None:
        self.state = 2  # pyright: ignore[reportAttributeAccessIssue]


class WritesOtherEnum(cb.Model):
    state: NodeKind = NodeKind.START

    @cb.process
    def run(self) -> None:
        self.state = Kind.TASK  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize("model", [WritesInt, WritesOtherEnum])
def test_compiled_writes_require_members(model):
    with pytest.raises(ModelCompileError,
                       match=rf"{model.__name__}.run: cannot store .* as NodeKind: "
                             "assign a NodeKind member"):
        run(model())


class Spawned(cb.Model):
    parent: cb.Ref["Spawner"]
    kind: cb.Param[NodeKind] = NodeKind.START
    other: NodeKind = NodeKind.END

    @cb.process
    def act(self) -> None:
        self.parent.total += int(self.kind) * 10 + int(self.other)


class Spawner(cb.Model):
    total: cb.Output[int]

    @cb.process
    def create(self) -> None:
        cb.spawn(Spawned, parent=self, kind=NodeKind.EXCLUSIVE)
        cb.spawn(Spawned, parent=self)
        cb.hold(1.0)


def test_spawn_arguments_of_enum_type():
    spawner = Spawner()
    assert run(spawner)[spawner].total[0, 0] == 31 + 1


class SpawnsInt(Spawner):
    @cb.process
    def create(self) -> None:
        cb.spawn(Spawned, parent=self, kind=3)


def test_spawn_arguments_require_members():
    with pytest.raises(ModelCompileError, match="expected a NodeKind member"):
        run(SpawnsInt())


def test_decisions_over_enum_params_are_rejected():
    node = Node()
    node.kind = cb.decision(0, 3)  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(ModelDefinitionError, match="node.kind: decisions require.*IntEnum"):
        cb.Optimization(node, minimize=lambda r: r[node].code)


# ------------------------------------------------------------------ plain Enum
def test_plain_enum_fields_and_signatures_name_intenum():
    class Painted(cb.Model):
        color: cb.Param[Color] = Color.RED

    with pytest.raises(ModelDefinitionError, match="Painted.color: Color is not an enum.IntEnum"):
        ClassSchema.of(Painted)

    class Painter(cb.Model):
        @cb.function
        def paint(self, color: Color) -> None:
            pass

    with pytest.raises(ModelDefinitionError, match="parameter 'color': Color is not an enum.IntEnum"):
        ClassSchema.of(Painter)

    class Holder(cb.Model):
        color: cb.Param[Color]

    with pytest.raises(ModelDefinitionError, match="Holder.color: Color is not an enum.IntEnum"):
        Holder().color = Color.BLUE


class PlainEquality(cb.Model):
    same: cb.Output[bool]

    @cb.process
    def run(self) -> None:
        color = Color.BLUE
        self.same = color == Color.BLUE and color != Color.RED


class PlainNumber(cb.Model):
    out: cb.Output[float]

    @cb.process
    def run(self) -> None:
        self.out = float(Color.RED)  # pyright: ignore[reportArgumentType]


def test_plain_enum_members_compare_but_are_not_numbers():
    model = PlainEquality()
    assert run(model)[model].same[0, 0] == 1.0
    with pytest.raises(ModelCompileError,
                       match=r"test_enums.py:\d+: PlainNumber.run: .*Color is not an enum.IntEnum"):
        run(PlainNumber())


class Lane(cb.Model):
    weight: cb.Param[float] = 1.0


class Lanes(cb.Model):
    lanes: list[Lane]
    picked: cb.Output[float]

    @cb.process
    def run(self) -> None:
        self.picked = self.lanes[NodeKind.TASK].weight


def test_members_index_model_lists_and_diagrams_draw():
    from cimba.diagrams import mermaid, process_graph

    model = Lanes()
    model.lanes = [Lane(), Lane(), Lane()]
    model.lanes[2].weight = 5.0
    assert run(model)[model].picked[0, 0] == 5.0
    assert "Node" in mermaid(Node())
    process_graph(Node())
