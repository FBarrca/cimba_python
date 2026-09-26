"""Infer process ↔ entity/input interactions from model method source.

The analysis is static: it reads the source of every process (and event and,
optionally, hook) and resolves attribute chains such as
``self.harbor.facilities.tugs`` against the configured object graph. It
follows children, references, list items, local aliases, helper functions
(including ``numba.njit`` helpers) and the keyword arguments of
``cb.spawn``, so spawned models' processes are connected to the static models
they reference. It never compiles or runs the model.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass
from types import FunctionType, ModuleType
from typing import Any, get_args, get_origin

import cimba.modeling as modeling
from cimba.modeling import (
    Condition, Container, Dataset, Model, PriorityStore, Resource, Store, Sweep,
)
from cimba.schema import Assembly, ClassSchema

from .graph import Edge, Graph, Group, Node

# (entity class, method) -> ("in" | "out", label). "in" means the entity feeds
# the actor (get, acquire, wait); "out" means the actor feeds the entity.
_ENTITY_METHODS: dict[tuple[type, str], tuple[str, str]] = {
    (Container, "put"): ("out", "put"),
    (Container, "get"): ("in", "get"),
    (Store, "put"): ("out", "put"),
    (Store, "get"): ("in", "get"),
    (PriorityStore, "put"): ("out", "put"),
    (PriorityStore, "enqueue"): ("out", "enqueue"),
    (PriorityStore, "cancel"): ("out", "cancel"),
    (PriorityStore, "get"): ("in", "get"),
    (Resource, "acquire"): ("in", "acquire"),
    (Resource, "preempt"): ("in", "preempt"),
    (Resource, "release"): ("out", "release"),
    (Condition, "wait_until"): ("in", "wait"),
    (Condition, "signal"): ("out", "signal"),
    (Dataset, "record"): ("out", "record"),
}
_INPUT_METHODS = {"next", "now", "at"}
_VERBS = {name: getattr(modeling, name) for name in (
    "spawn", "schedule", "release", "hold", "now", "suspend", "end_trial",
    "this_process", "log")}


@dataclass(frozen=True, eq=False)
class _Spawned:
    """All dynamic instances of one model class, spawned during trials."""

    cls: type


@dataclass(frozen=True)
class _Ref:
    kind: str          # model, entity, input, state, collection, event
    owner: Any         # a static Model instance or a _Spawned
    name: str = ""


class _Analysis:
    def __init__(self, root: Model, *, state: bool, hooks: bool):
        self.assembly = Assembly.of(root, strict=False)
        self.state = state
        self.hooks = hooks
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.groups: dict[Any, Group] = {}
        self.spawned: dict[type, _Spawned] = {}
        self.bindings: dict[tuple[_Spawned, str], set[_Ref]] = {}

    # ---------------------------------------------------------------- owners
    def schema(self, owner) -> ClassSchema:
        if isinstance(owner, _Spawned):
            return ClassSchema.of(owner.cls)
        return self.assembly.by_object[owner].schema

    def owner_label(self, owner) -> str:
        if isinstance(owner, _Spawned):
            return owner.cls.__name__
        return self.assembly.by_object[owner].label

    def spawned_of(self, cls: type) -> _Spawned:
        if cls not in self.spawned:
            self.spawned[cls] = _Spawned(cls)
        return self.spawned[cls]

    def group(self, owner) -> str:
        key = self.owner_label(owner)
        if owner not in self.groups:
            if isinstance(owner, _Spawned):
                self.groups[owner] = Group(key, f"{owner.cls.__name__} (spawned)", "spawned")
            else:
                self.groups[owner] = Group(key, f"{key}: {type(owner).__name__}")
        return key

    # ----------------------------------------------------------------- nodes
    def node(self, ref: _Ref, *, actor_kind: str | None = None) -> str:
        owner_key = self.group(ref.owner)
        if ref.kind == "model":
            key = f"{owner_key}"
            label = f"new {ref.owner.cls.__name__}" if isinstance(ref.owner, _Spawned) else owner_key
            kind = "model"
        else:
            key = f"{owner_key}.{ref.name}"
            label, kind = ref.name, actor_kind or ref.kind
            if ref.kind == "entity":
                entity = self.schema(ref.owner)
                field = next(f for f in entity.fields if f.name == ref.name)
                label = f"{ref.name} · {_entity_name(field.value_type)}"
            elif ref.kind == "input" and not isinstance(ref.owner, _Spawned):
                label = f"{ref.name} · {_source_name(self.assembly.by_object[ref.owner].values[ref.name])}"
        if key not in self.nodes:
            self.nodes[key] = Node(key, label, kind, owner_key)
        return key

    def edge(self, source: str, target: str, label: str | None, style: str = "solid"):
        edge = Edge(source, target, label, style)
        if edge not in self.edges:
            self.edges.append(edge)

    # --------------------------------------------------------------- actors
    def actors(self, owner):
        schema = self.schema(owner)
        for name, copies, _priority in schema.processes:
            label = name if copies == 1 else f"{name} ×{copies}"
            yield name, label, "process"
        for name in schema.events:
            yield name, name, "event"
        if self.hooks:
            for name in schema.starts:
                yield name, f"{name} (on_start)", "hook"
            for name in schema.ends:
                yield name, f"{name} (on_end)", "hook"

    def analyze_owner(self, owner):
        for name, label, kind in self.actors(owner):
            key = f"{self.group(owner)}.{name}"
            if key not in self.nodes:
                self.nodes[key] = Node(key, label, kind, self.group(owner))
            method = getattr(self.schema(owner).cls, name)
            _Visitor(self, key, owner).run(method, {_first_param(method): {_Ref("model", owner)}})

    def run(self) -> Graph:
        for instance in self.assembly.instances:
            self.group(instance.model)
        for instance in self.assembly.instances:
            for field in instance.schema.fields:
                if field.kind in {"entity", "input", "series"}:
                    self.node(_Ref("input" if field.kind != "entity" else "entity",
                                   instance.model, field.name))
            self.analyze_owner(instance.model)
        done: dict[_Spawned, int] = {}
        for _ in range(20):
            pending = [s for s in list(self.spawned.values())
                       if done.get(s) != self._binding_size(s)]
            if not pending:
                break
            for spawned in pending:
                done[spawned] = self._binding_size(spawned)
                self.analyze_owner(spawned)
        groups = list(self.groups.values())
        order = {group.key: index for index, group in enumerate(groups)}
        nodes = sorted(self.nodes.values(), key=lambda node: order.get(node.group or "", len(order)))
        return Graph(tuple(nodes), tuple(self.edges), tuple(groups))

    def _binding_size(self, spawned: _Spawned) -> int:
        return sum(len(refs) for (owner, _), refs in self.bindings.items()
                   if owner is spawned)


class _Visitor(ast.NodeVisitor):
    def __init__(self, analysis: _Analysis, actor: str, owner, active=None):
        self.analysis = analysis
        self.actor = actor
        self.owner = owner
        self.aliases: dict[str, set[_Ref]] = {}
        self.globals: dict[str, Any] = {}
        self.active = active if active is not None else set()

    def run(self, function, bound: dict[str, set[_Ref]]):
        function = getattr(function, "py_func", function)
        if not isinstance(function, FunctionType) or function in self.active:
            return
        tree = _function_ast(function)
        if tree is None:
            return
        self.active.add(function)
        saved = self.aliases, self.globals
        self.aliases = {name: set(refs) for name, refs in bound.items()}
        self.globals = function.__globals__
        try:
            for statement in tree.body:
                self.visit(statement)
        finally:
            self.aliases, self.globals = saved
            self.active.discard(function)

    # ------------------------------------------------------------ resolving
    def resolve(self, node: ast.AST) -> set[_Ref]:
        analysis = self.analysis
        if isinstance(node, ast.Name):
            return set(self.aliases.get(node.id, ()))
        if isinstance(node, ast.Attribute):
            found = set()
            for ref in self.resolve(node.value):
                if ref.kind == "model":
                    found |= self.field(ref.owner, node.attr)
            return found
        if isinstance(node, ast.Subscript):
            found = set()
            index = node.slice
            literal = index.value if (isinstance(index, ast.Constant)
                                      and type(index.value) is int) else None
            for ref in self.resolve(node.value):
                if ref.kind == "collection":
                    items = analysis.assembly.by_object[ref.owner].values[ref.name]
                    chosen = ([items[literal]] if literal is not None
                              and -len(items) <= literal < len(items) else items)
                    found |= {_Ref("model", item) for item in chosen if item is not None}
            return found
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "get":
                found = set()
                for ref in self.resolve(func.value):
                    item = self.store_item(ref)
                    if item is not None:
                        found.add(_Ref("model", analysis.spawned_of(item)))
                return found
            if self.verb(func) == "spawn" and node.args:
                cls = self.model_class(node.args[0])
                if cls is not None:
                    return {_Ref("model", analysis.spawned_of(cls))}
            return set()
        if isinstance(node, ast.IfExp):
            return self.resolve(node.body) | self.resolve(node.orelse)
        if isinstance(node, ast.BoolOp):
            return set().union(*(self.resolve(value) for value in node.values))
        return set()

    def field(self, owner, name: str) -> set[_Ref]:
        analysis = self.analysis
        schema = analysis.schema(owner)
        if name in schema.events:
            return {_Ref("event", owner, name)}
        if name in schema.predicates:
            return {_Ref("predicate", owner, name)}
        field = next((f for f in schema.fields if f.name == name), None)
        if field is None:
            return set()
        if isinstance(owner, _Spawned):
            if field.kind in {"ref", "input", "series"}:
                bound = analysis.bindings.get((owner, name), set())
                if field.kind == "ref":
                    return set(bound)
                return set(bound) or {_Ref("input", owner, name)}
            if field.kind in {"param", "state", "output", "constant"}:
                return {_Ref("state", owner, name)}
            return set()
        value = analysis.assembly.by_object[owner].values[name]
        if field.kind == "ref" and isinstance(value, Sweep):
            return {_Ref("model", choice) for choice in value.values if choice is not None}
        if field.kind in {"child", "ref"}:
            return {_Ref("model", value)} if value is not None else set()
        if field.kind == "collection":
            return {_Ref("collection", owner, name)}
        if field.kind == "entity":
            return {_Ref("entity", owner, name)}
        if field.kind in {"input", "series"}:
            return {_Ref("input", owner, name)}
        return {_Ref("state", owner, name)}

    def store_item(self, ref: _Ref):
        if ref.kind != "entity":
            return None
        schema = self.analysis.schema(ref.owner)
        field = next(f for f in schema.fields if f.name == ref.name)
        if get_origin(field.value_type) in (Store, PriorityStore):
            (item,) = get_args(field.value_type)
            if isinstance(item, type) and issubclass(item, Model):
                return item
        return None

    def entity_class(self, ref: _Ref):
        schema = self.analysis.schema(ref.owner)
        field = next(f for f in schema.fields if f.name == ref.name)
        return get_origin(field.value_type) or field.value_type

    def verb(self, func: ast.AST) -> str | None:
        value = None
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            module = self.globals.get(func.value.id)
            if isinstance(module, ModuleType) and getattr(module, func.attr, None) is _VERBS.get(func.attr):
                return func.attr
        elif isinstance(func, ast.Name):
            value = self.globals.get(func.id)
            for name, verb in _VERBS.items():
                if value is verb:
                    return name
        return None

    def model_class(self, node: ast.AST):
        value = None
        if isinstance(node, ast.Name):
            value = self.globals.get(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            value = getattr(self.globals.get(node.value.id), node.attr, None)
        return value if isinstance(value, type) and issubclass(value, Model) else None

    # ------------------------------------------------------------ statements
    def bind(self, target: ast.AST, refs: set[_Ref]):
        if isinstance(target, ast.Name) and refs:
            self.aliases.setdefault(target.id, set()).update(refs)

    def write(self, target: ast.AST):
        if not self.analysis.state:
            return
        if isinstance(target, ast.Subscript):
            target = target.value
        if isinstance(target, ast.Attribute):
            for ref in self.resolve(target):
                if ref.kind == "state":
                    self.analysis.edge(self.actor, self.analysis.node(ref), "write")

    def visit_Assign(self, node: ast.Assign):
        self.visit(node.value)
        refs = self.resolve(node.value)
        for target in node.targets:
            if (isinstance(target, (ast.Tuple, ast.List)) and isinstance(node.value, (ast.Tuple, ast.List))
                    and len(target.elts) == len(node.value.elts)):
                for element, value in zip(target.elts, node.value.elts):
                    self.bind(element, self.resolve(value))
            else:
                self.bind(target, refs)
            self.write(target)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if node.value is not None:
            self.visit(node.value)
            self.bind(node.target, self.resolve(node.value))
            self.write(node.target)

    def visit_AugAssign(self, node: ast.AugAssign):
        self.visit(node.value)
        self.write(node.target)

    def visit_For(self, node: ast.For):
        self.bind(node.target, self.resolve(ast.Subscript(node.iter, ast.Name("_"))))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        analysis = self.analysis
        func = node.func
        verb = self.verb(func)
        if verb == "spawn" and node.args:
            cls = self.model_class(node.args[0])
            if cls is None:
                return
            spawned = analysis.spawned_of(cls)
            analysis.edge(self.actor, analysis.node(_Ref("model", spawned)), "spawn")
            schema = ClassSchema.of(cls)
            kinds = {f.name: f.kind for f in schema.fields}
            for keyword in node.keywords:
                if keyword.arg and kinds.get(keyword.arg) in {"ref", "input", "series"}:
                    refs = {r for r in self.resolve(keyword.value)
                            if r.kind in {"model", "input"}}
                    analysis.bindings.setdefault((spawned, keyword.arg), set()).update(refs)
            return
        if verb == "schedule" and node.args:
            for ref in self.resolve(node.args[0]):
                if ref.kind == "event":
                    key = f"{analysis.group(ref.owner)}.{ref.name}"
                    if key not in analysis.nodes:
                        analysis.nodes[key] = Node(key, ref.name, "event",
                                                   analysis.group(ref.owner))
                    analysis.edge(self.actor, key, "schedule")
            return
        if verb is not None:
            return
        if isinstance(func, ast.Attribute):
            for ref in self.resolve(func.value):
                if ref.kind == "model" and func.attr in analysis.schema(ref.owner).functions:
                    # Follow the call into the implementation of this model's
                    # actual class, so polymorphic policies draw what they do.
                    cls = (ref.owner.cls if isinstance(ref.owner, _Spawned)
                           else type(ref.owner))
                    method = getattr(cls, func.attr)
                    parameters = method.__code__.co_varnames[1:method.__code__.co_argcount]
                    bound = {name: self.resolve(argument)
                             for name, argument in zip(parameters, node.args)}
                    bound = {k: v for k, v in bound.items() if v}
                    bound[_first_param(method)] = {ref}
                    _Visitor(analysis, self.actor, ref.owner, self.active).run(method, bound)
                    continue
                if ref.kind == "input" and func.attr in _INPUT_METHODS:
                    analysis.edge(analysis.node(ref), self.actor, func.attr)
                elif ref.kind == "entity":
                    entity = self.entity_class(ref)
                    spec = None
                    for cls in getattr(entity, "__mro__", (entity,)):
                        spec = _ENTITY_METHODS.get((cls, func.attr))
                        if spec:
                            break
                    if spec:
                        direction, label = spec
                        key = analysis.node(ref)
                        if direction == "in":
                            analysis.edge(key, self.actor, label)
                        else:
                            analysis.edge(self.actor, key, label)
            return
        if isinstance(func, ast.Name):
            helper = self.globals.get(func.id)
            helper = getattr(helper, "py_func", helper)
            if not isinstance(helper, FunctionType):
                return
            parameters = helper.__code__.co_varnames[:helper.__code__.co_argcount]
            bound = {name: self.resolve(argument)
                     for name, argument in zip(parameters, node.args)}
            bound.update({k.arg: self.resolve(k.value) for k in node.keywords if k.arg})
            if any(bound.values()):
                _Visitor(analysis, self.actor, self.owner, self.active).run(
                    helper, {k: v for k, v in bound.items() if v})


def _function_ast(function) -> ast.FunctionDef | None:
    try:
        source = textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError):
        return None
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            return node
    return None


def _first_param(function) -> str:
    function = getattr(function, "py_func", function)
    return function.__code__.co_varnames[0] if function.__code__.co_argcount else "self"


def _entity_name(value_type) -> str:
    origin = get_origin(value_type)
    if origin is None:
        return getattr(value_type, "__name__", str(value_type))
    arguments = ", ".join(getattr(a, "__name__", str(a)) for a in get_args(value_type))
    return f"{origin.__name__}[{arguments}]"


def _source_name(value) -> str:
    if isinstance(value, Sweep):
        return f"sweep of {len(value.values)}"
    describe = getattr(value, "describe", None)
    if describe is None:
        return "no source"
    method = describe().get("method", "source")
    return method.removeprefix("dist.")


def process_graph(model: Model, *, state: bool = False, hooks: bool = False) -> Graph:
    """Infer how processes interact with entities, inputs and spawned models.

    Nodes are grouped by model instance (and by spawned model class). Edges
    point in the direction things flow: a process *puts* into a store, a store
    feeds the process that *gets*, an input feeds the process that reads it,
    a process *spawns* a model. Pass ``state=True`` to add writes to
    ``Param``/``State``/``Output`` fields and ``hooks=True`` to include
    ``on_start``/``on_end`` hooks as actors.
    """
    return _Analysis(model, state=state, hooks=hooks).run()


__all__ = ["process_graph"]
