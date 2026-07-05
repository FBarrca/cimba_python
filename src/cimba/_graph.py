"""Process graph helpers for ``cimba.sim`` models."""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import FunctionType
from typing import Any


@dataclass(frozen=True)
class ProcessDAGNode:
    """A process or model-field node in a model-level dependency graph."""

    name: str
    kind: str = "process"
    copies: int = 1
    priority: int = 0
    spawnable: bool = False
    struct: str | None = None
    indexed: bool = False

    @property
    def key(self) -> str:
        """Stable graph key used by edges and topological order."""
        return f"{self.kind}:{self.name}"


@dataclass(frozen=True)
class ProcessDAGEdge:
    """A directed relationship between two graph nodes."""

    source: str
    target: str
    label: str | None = None


@dataclass(frozen=True)
class ProcessDAGBlock:
    """A presentation group for existing process graph nodes."""

    name: str
    members: tuple[str, ...]
    kind: str = "component"

    def __init__(
        self,
        name: str,
        members: Iterable[str],
        kind: str = "component",
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "members", tuple(members))
        object.__setattr__(self, "kind", kind)

    @property
    def key(self) -> str:
        """Stable block key used by renderers."""
        return f"{self.kind}:{self.name}"


@dataclass(frozen=True)
class ProcessDAG:
    """A model-field-aware graph of registered model processes."""

    nodes: tuple[ProcessDAGNode, ...]
    edges: tuple[ProcessDAGEdge, ...]
    blocks: tuple[ProcessDAGBlock, ...] = ()

    def __init__(
        self,
        nodes: tuple[ProcessDAGNode, ...] | list[ProcessDAGNode],
        edges: tuple[ProcessDAGEdge, ...] | list[ProcessDAGEdge],
        blocks: tuple[ProcessDAGBlock, ...] | list[ProcessDAGBlock] = (),
    ) -> None:
        object.__setattr__(self, "nodes", tuple(nodes))
        object.__setattr__(self, "edges", tuple(edges))
        object.__setattr__(self, "blocks", tuple(blocks))

    def topological_order(self) -> tuple[str, ...]:
        """Return node keys in topological order.

        Raises
        ------
        ValueError
            If an edge references an unknown node or the graph contains a
            cycle.
        """
        node_keys = [node.key for node in self.nodes]
        known = set(node_keys)
        adjacency: dict[str, list[str]] = {key: [] for key in node_keys}
        indegree: dict[str, int] = {key: 0 for key in node_keys}

        unknown = sorted({
            endpoint
            for edge in self.edges
            for endpoint in (edge.source, edge.target)
            if endpoint not in known
        })
        if unknown:
            raise ValueError(
                "process graph edge references unknown node(s): "
                + ", ".join(unknown)
            )

        for edge in self.edges:
            if edge.source == edge.target:
                raise ValueError(
                    f"process graph contains self-edge '{edge.source}'"
                )
            adjacency[edge.source].append(edge.target)
            indegree[edge.target] += 1

        ready = deque(key for key in node_keys if indegree[key] == 0)
        ordered: list[str] = []
        while ready:
            key = ready.popleft()
            ordered.append(key)
            for target in adjacency[key]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)

        if len(ordered) != len(node_keys):
            raise ValueError("process graph contains a cycle")
        return tuple(ordered)

    def to_mermaid(self, direction: str = "TD") -> str:
        """Render the graph as Mermaid flowchart text."""
        lines = [f"flowchart {direction}"]
        nodes_by_key = {node.key: node for node in self.nodes}
        rendered: set[str] = set()
        for block in self.blocks:
            members = [
                key for key in block.members
                if key in nodes_by_key and key not in rendered
            ]
            if not members:
                continue
            block_id = _mermaid_id(block.key)
            label = _escape_mermaid(block.name)
            lines.append(f"    subgraph {block_id}[\"{label}\"]")
            for key in members:
                lines.append(_mermaid_node_line(nodes_by_key[key], "        "))
                rendered.add(key)
            lines.append("    end")
        for node in self.nodes:
            if node.key not in rendered:
                lines.append(_mermaid_node_line(node, "    "))
        for edge in self.edges:
            source = _mermaid_id(edge.source)
            target = _mermaid_id(edge.target)
            if edge.label is None:
                lines.append(f"    {source} --> {target}")
            else:
                label = _escape_mermaid(edge.label)
                lines.append(f"    {source} -->|{label}| {target}")
        return "\n".join(lines)

    def to_dot(self, rankdir: str = "TB") -> str:
        """Render the graph as Graphviz DOT text."""
        lines = ["digraph ProcessDAG {", f"    rankdir={rankdir};"]
        nodes_by_key = {node.key: node for node in self.nodes}
        rendered: set[str] = set()
        for block in self.blocks:
            members = [
                key for key in block.members
                if key in nodes_by_key and key not in rendered
            ]
            if not members:
                continue
            lines.append(f"    subgraph {_dot_cluster_id(block.key)} {{")
            lines.append(f"        label={_dot_quote(block.name)};")
            for key in members:
                lines.append(_dot_node_line(nodes_by_key[key], "        "))
                rendered.add(key)
            lines.append("    }")
        for node in self.nodes:
            if node.key not in rendered:
                lines.append(_dot_node_line(node, "    "))
        for edge in self.edges:
            line = f"    {_dot_quote(edge.source)} -> {_dot_quote(edge.target)}"
            if edge.label is not None:
                line += f" [label={_dot_quote(edge.label)}]"
            lines.append(line + ";")
        lines.append("}")
        return "\n".join(lines)


@dataclass(frozen=True)
class _Ref:
    kind: str
    name: str

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.name}"


_EVENT_PRODUCER_VERBS = {"schedule", "schedule_at"}
_EVENT_CONSUMER_VERBS = {"wait_event"}
_EVENT_CONTROL_VERBS = {
    "event_cancel",
    "event_reschedule",
    "event_reprioritize",
    "event_scheduled",
    "event_time",
    "event_priority",
}
_DIRECT_PROCESS_VERBS = {
    "interrupt",
    "resume",
    "stop",
    "wait_process",
    "set_priority",
    "timer_set",
    "timer_add",
    "timer_cancel",
    "timers_clear",
}
_STATE_KINDS = {"state", "fstate"}

#: (declared field kind, ``env.<field>.<method>(...)`` method name) -> the
#: (edge category, edge label) it infers, mirroring the verb names the old
#: ``sim.put()``/``sim.store_put()``/``sim.pq_put()``/... free functions
#: used, so existing graphs keep the same edge labels under the
#: object-oriented sugar. Used directly against a process's *unlowered*
#: source -- e.g. a plain helper function's own body, inlined via
#: ``_handle_helper_call`` below, which never goes through
#: ``Model.process()``'s entity-method lowering.
_ENTITY_METHOD_VERBS: dict[tuple[str, str], tuple[str, str]] = {
    ("queue", "put"): ("produce", "put"),
    ("queue", "get"): ("consume", "get"),
    ("store", "put"): ("produce", "store_put"),
    ("store", "get"): ("consume", "store_get"),
    ("store", "take"): ("consume", "store_take"),
    ("pqueues", "put"): ("produce", "pq_put"),
    ("pqueues", "get"): ("consume", "pq_get"),
    ("pqueues", "take"): ("consume", "pq_take"),
    ("condition", "signal"): ("produce", "signal"),
    ("condition", "wait_for"): ("consume", "wait_for"),
    ("resource", "acquire"): ("shared", "uses"),
    ("resource", "release"): ("shared", "uses"),
    ("resource", "preempt"): ("shared", "uses"),
    ("resource", "available"): ("shared", "uses"),
    ("resource", "in_use"): ("shared", "uses"),
    ("resource", "held"): ("shared", "uses"),
    ("pool", "acquire"): ("shared", "uses"),
    ("pool", "release"): ("shared", "uses"),
    ("pool", "preempt"): ("shared", "uses"),
    ("pool", "available"): ("shared", "uses"),
    ("pool", "in_use"): ("shared", "uses"),
    ("pool", "held"): ("shared", "uses"),
}

#: declared field kind -> the label ``store/methods.py`` prefixes its
#: lowered helper names with (``_cimba_entity_<label>_<method>``).
_ENTITY_HELPER_LABELS = {
    "queue": "queue",
    "resource": "resource",
    "pool": "pool",
    "store": "store",
    "pqueues": "pq",
    "condition": "condition",
}

#: (field kind, method name) -> the helper-name method segment
#: ``store/methods.py`` actually uses when it differs from the method name
#: itself (only ``Condition.wait_for``, whose helper is ``condition_wait``).
_ENTITY_HELPER_METHOD_OVERRIDES = {("condition", "wait_for"): "wait"}

#: By the time a *registered* process/predicate/event/collect function
#: reaches DAG inference, the ``env.<entity>.method(...)`` sugar in its own
#: body has already been lowered (in ``Model.process()``/
#: ``_lower_component_process()``) into calls to these internal helpers
#: (see ``store/methods.py``), which structurally mirror the old
#: ``sim.put(entity, ...)``-style free functions: a plain call with the
#: entity handle as the first argument. Helper name suffix (after
#: ``_cimba_entity_``) -> (edge category, edge label).
_ENTITY_HELPER_PREFIX = "_cimba_entity_"
_ENTITY_HELPER_VERBS: dict[str, tuple[str, str]] = {
    f"{_ENTITY_HELPER_LABELS[kind]}_"
    f"{_ENTITY_HELPER_METHOD_OVERRIDES.get((kind, method), method)}": spec
    for (kind, method), spec in _ENTITY_METHOD_VERBS.items()
}


def infer_process_dag(
    processes: Iterable[Any],
    *,
    entity_kinds: Mapping[str, str],
    process_fields: Iterable[str],
    spawnable_fields: Iterable[str],
    spawnable_field_processes: Mapping[str, Iterable[str]] | None = None,
    spawnable_index_processes: Mapping[
        tuple[str, int], Iterable[str]] | None = None,
    process_field_processes: Mapping[str, Iterable[str]] | None = None,
    process_index_processes: Mapping[
        tuple[str, int], Iterable[str]] | None = None,
    event_callbacks: Iterable[tuple[str, Callable[..., Any]]] = (),
    blocks: Iterable[ProcessDAGBlock] = (),
) -> ProcessDAG:
    """Infer a model-field-aware process graph from registered process bodies."""
    process_list = tuple(processes)
    process_names = {p.name for p in process_list}
    process_field_names = set(process_fields)
    spawnable_field_names = set(spawnable_fields)
    spawnable_processes = {
        field: set(names)
        for field, names in (spawnable_field_processes or {}).items()
    }
    spawnable_indexed_processes = {
        key: set(names)
        for key, names in (spawnable_index_processes or {}).items()
    }
    process_field_refs = {
        field: set(names)
        for field, names in (process_field_processes or {}).items()
    }
    process_index_refs = {
        key: set(names)
        for key, names in (process_index_processes or {}).items()
    }

    nodes = [
        ProcessDAGNode(
            name=p.name,
            copies=p.copies,
            priority=p.priority,
            spawnable=p.spawnable,
            struct=(p.struct.__name__ if p.struct is not None else None),
            indexed=p.indexed,
        )
        for p in process_list
    ]
    edges: list[ProcessDAGEdge] = []
    resource_nodes: dict[str, ProcessDAGNode] = {}

    context = _InferenceContext(
        entity_kinds=dict(entity_kinds),
        process_names=process_names,
        process_fields=process_field_names,
        spawnable_fields=spawnable_field_names,
        spawnable_field_processes=spawnable_processes,
        spawnable_index_processes=spawnable_indexed_processes,
        process_field_processes=process_field_refs,
        process_index_processes=process_index_refs,
    )

    for process in process_list:
        analyzer = _ProcessAnalyzer(
            context=context,
            actor=_Ref("process", process.name),
            fn_globals=process.fn.__globals__,
            env_names={process.fn.__code__.co_varnames[0]},
        )
        analyzer.analyze_function(process.fn)
        for edge in analyzer.edges:
            if edge not in edges:
                edges.append(edge)
            for key in (edge.source, edge.target):
                ref = _ref_from_key(key)
                if ref.kind != "process" and key not in resource_nodes:
                    resource_nodes[key] = ProcessDAGNode(ref.name, ref.kind)

    for event_name, event_fn in event_callbacks:
        if event_fn.__code__.co_argcount < 1:
            continue
        analyzer = _ProcessAnalyzer(
            context=context,
            actor=_Ref("event", event_name),
            fn_globals=event_fn.__globals__,
            env_names={event_fn.__code__.co_varnames[0]},
        )
        analyzer.analyze_function(event_fn)
        for edge in analyzer.edges:
            if edge not in edges:
                edges.append(edge)
            for key in (edge.source, edge.target):
                ref = _ref_from_key(key)
                if ref.kind != "process" and key not in resource_nodes:
                    resource_nodes[key] = ProcessDAGNode(ref.name, ref.kind)

    nodes.extend(resource_nodes.values())
    return ProcessDAG(nodes, edges, list(blocks))


@dataclass
class _InferenceContext:
    entity_kinds: dict[str, str]
    process_names: set[str]
    process_fields: set[str]
    spawnable_fields: set[str]
    spawnable_field_processes: dict[str, set[str]]
    spawnable_index_processes: dict[tuple[str, int], set[str]]
    process_field_processes: dict[str, set[str]]
    process_index_processes: dict[tuple[str, int], set[str]]


class _ProcessAnalyzer(ast.NodeVisitor):
    def __init__(
        self,
        *,
        context: _InferenceContext,
        actor: _Ref,
        fn_globals: dict[str, Any],
        env_names: set[str],
    ) -> None:
        self.context = context
        self.actor = actor
        self.fn_globals = fn_globals
        self.env_names = env_names
        self.aliases: dict[str, set[_Ref]] = {}
        self.edges: list[ProcessDAGEdge] = []
        self._active_helpers: set[Callable[..., Any]] = set()

    def analyze_function(self, fn: Callable[..., Any]) -> None:
        tree = _function_ast(fn)
        if tree is None:
            return
        self.visit(tree)

    def visit_Assign(self, node: ast.Assign) -> None:
        refs = self._refs(node.value)
        if refs:
            for target in node.targets:
                self._bind_refs(target, refs)
        for target in node.targets:
            self._add_state_writes(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            refs = self._refs(node.value)
            if refs:
                self._bind_refs(node.target, refs)
        self._add_state_writes(node.target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        for ref in self._state_refs(node.target):
            self._add_edge(ref, self.actor, "read")
            self._add_edge(self.actor, ref, "write")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Load) and self._is_env_expr(node.value):
            for ref in self._env_field_refs(node.attr):
                if ref.kind in _STATE_KINDS:
                    self._add_edge(ref, self.actor, "read")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        verb = _sim_verb(node.func)
        if verb is not None:
            self._handle_sim_call(verb, node)
        elif self._handle_entity_helper_call(node):
            pass
        elif self._handle_entity_method_call(node):
            pass
        else:
            self._handle_helper_call(node)
        self.generic_visit(node)

    def _handle_sim_call(self, verb: str, node: ast.Call) -> None:
        if not node.args:
            return

        if verb == "spawn":
            for ref in self._refs(node.args[0]):
                if ref.kind == "process":
                    self._add_edge(self.actor, ref, "spawn")
            return

        if verb in _DIRECT_PROCESS_VERBS:
            for ref in self._refs(node.args[0]):
                if ref.kind == "process":
                    self._add_edge(self.actor, ref, verb)
            return

        if verb in _EVENT_PRODUCER_VERBS:
            for ref in self._refs(node.args[0]):
                if ref.kind == "event":
                    self._add_edge(self.actor, ref, verb)
            return

        if verb in _EVENT_CONSUMER_VERBS:
            for ref in self._refs(node.args[0]):
                if ref.kind == "event":
                    self._add_edge(ref, self.actor, verb)
            return

        if verb in _EVENT_CONTROL_VERBS:
            for ref in self._refs(node.args[0]):
                if ref.kind == "event":
                    self._add_edge(self.actor, ref, verb)
            return

    def _handle_entity_helper_call(self, node: ast.Call) -> bool:
        """Recognize a lowered ``env.<entity>.method(...)`` call (a
        ``_cimba_entity_<label>_<method>(entity, ...)`` helper invocation,
        as it appears in the already-lowered source of a *registered*
        process/predicate/event/collect function) and infer the same edges
        the matching legacy ``sim.put()``/``sim.acquire()``/... free
        function used to. Returns whether the call was recognized (so
        callers don't also try treating it as a helper-function call)."""
        func = node.func
        if not isinstance(func, ast.Name) or not node.args:
            return False
        if not func.id.startswith(_ENTITY_HELPER_PREFIX):
            return False
        spec = _ENTITY_HELPER_VERBS.get(func.id[len(_ENTITY_HELPER_PREFIX):])
        if spec is None:
            return True
        category, label = spec
        for ref in self._refs(node.args[0]):
            if category == "produce":
                self._add_edge(self.actor, ref, label)
            elif category == "consume":
                self._add_edge(ref, self.actor, label)
            else:
                self._add_edge(self.actor, ref, label)
        return True

    def _handle_entity_method_call(self, node: ast.Call) -> bool:
        """Recognize an unlowered ``env.<entity>.method(...)`` call (and
        the indexed ``env.<entity>[i].method(...)`` form), as it appears in
        a plain helper function's own source -- helper functions are
        inlined by AST for this analysis (see ``_handle_helper_call``
        below) but never go through ``Model.process()``'s entity-method
        lowering, since they aren't registered process/predicate/event/
        collect callbacks. Returns whether the call was recognized."""
        func = node.func
        if not isinstance(func, ast.Attribute):
            return False
        method = func.attr
        refs = self._refs(func.value)
        if not refs:
            return False
        handled = False
        for ref in refs:
            spec = _ENTITY_METHOD_VERBS.get((ref.kind, method))
            if spec is None:
                continue
            handled = True
            category, label = spec
            if category == "produce":
                self._add_edge(self.actor, ref, label)
            elif category == "consume":
                self._add_edge(ref, self.actor, label)
            else:
                self._add_edge(self.actor, ref, label)
        return handled

    def _handle_helper_call(self, node: ast.Call) -> None:
        fn = self._helper_function(node)
        if fn is None or fn in self._active_helpers:
            return
        params = fn.__code__.co_varnames[:fn.__code__.co_argcount]
        if not params or not node.args:
            return
        if not self._is_env_expr(node.args[0]):
            return

        self._active_helpers.add(fn)
        old_env_names = self.env_names
        self.env_names = {params[0]}
        try:
            self.analyze_function(fn)
        finally:
            self.env_names = old_env_names
            self._active_helpers.remove(fn)

    def _helper_function(self, node: ast.Call) -> FunctionType | None:
        if not isinstance(node.func, ast.Name):
            return None
        obj = self.fn_globals.get(node.func.id)
        if isinstance(obj, FunctionType):
            return obj
        py_func = getattr(obj, "py_func", None)
        return py_func if isinstance(py_func, FunctionType) else None

    def _refs(self, node: ast.AST) -> set[_Ref]:
        if isinstance(node, ast.Name):
            return set(self.aliases.get(node.id, ()))
        if isinstance(node, ast.Attribute):
            if self._is_env_expr(node.value):
                return self._env_field_refs(node.attr)
            return set()
        if isinstance(node, ast.Subscript):
            if (isinstance(node.value, ast.Attribute)
                    and self._is_env_expr(node.value.value)):
                index = self._literal_int_index(node.slice)
                refs = self._env_field_refs(node.value.attr, index)
                if refs:
                    return refs
            return self._refs(node.value)
        if isinstance(node, ast.Call):
            verb = _sim_verb(node.func)
            if verb in _EVENT_PRODUCER_VERBS and node.args:
                return {
                    ref
                    for ref in self._refs(node.args[0])
                    if ref.kind == "event"
                }
            fn = self._helper_function(node)
            if fn is not None and node.args and self._is_env_expr(node.args[0]):
                return self._helper_return_refs(fn)
        return set()

    def _helper_return_refs(self, fn: FunctionType) -> set[_Ref]:
        params = fn.__code__.co_varnames[:fn.__code__.co_argcount]
        if not params:
            return set()
        tree = _function_ast(fn)
        if tree is None:
            return set()
        old_env_names = self.env_names
        old_aliases = self.aliases
        self.env_names = {params[0]}
        self.aliases = {}
        refs: set[_Ref] = set()
        try:
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    assign_refs = self._refs(node.value)
                    for target in node.targets:
                        self._bind_refs(target, assign_refs)
                elif isinstance(node, ast.Return) and node.value is not None:
                    refs.update(self._refs(node.value))
        finally:
            self.env_names = old_env_names
            self.aliases = old_aliases
        return refs

    def _literal_int_index(self, node: ast.AST) -> int | None:
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        return None

    def _env_field_refs(self, field: str, index: int | None = None) -> set[_Ref]:
        if field in self.context.entity_kinds:
            return {_Ref(self.context.entity_kinds[field], field)}
        if index is not None:
            processes = self.context.process_index_processes.get(
                (field, index))
            if processes:
                return {
                    _Ref("process", name)
                    for name in processes
                    if name in self.context.process_names
                }
            processes = self.context.spawnable_index_processes.get(
                (field, index))
            if processes:
                return {
                    _Ref("process", name)
                    for name in processes
                    if name in self.context.process_names
                }
        processes = self.context.process_field_processes.get(field)
        if processes:
            return {
                _Ref("process", name)
                for name in processes
                if name in self.context.process_names
            }
        processes = self.context.spawnable_field_processes.get(field)
        if processes:
            return {
                _Ref("process", name)
                for name in processes
                if name in self.context.process_names
            }
        if field in self.context.spawnable_fields and field in self.context.process_names:
            return {_Ref("process", field)}
        if field in self.context.process_fields and field in self.context.process_names:
            return {_Ref("process", field)}
        return set()

    def _state_refs(self, node: ast.AST) -> set[_Ref]:
        return {
            ref
            for ref in self._target_refs(node)
            if ref.kind in _STATE_KINDS
        }

    def _target_refs(self, node: ast.AST) -> set[_Ref]:
        if isinstance(node, ast.Attribute) and self._is_env_expr(node.value):
            return self._env_field_refs(node.attr)
        if isinstance(node, ast.Subscript):
            return self._target_refs(node.value)
        if isinstance(node, (ast.Tuple, ast.List)):
            refs: set[_Ref] = set()
            for elt in node.elts:
                refs.update(self._target_refs(elt))
            return refs
        return set()

    def _add_state_writes(self, node: ast.AST) -> None:
        for ref in self._state_refs(node):
            self._add_edge(self.actor, ref, "write")

    def _is_env_expr(self, node: ast.AST) -> bool:
        return isinstance(node, ast.Name) and node.id in self.env_names

    def _bind_refs(self, target: ast.AST, refs: set[_Ref]) -> None:
        if not refs:
            return
        if isinstance(target, ast.Name):
            self.aliases.setdefault(target.id, set()).update(refs)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._bind_refs(elt, refs)

    def _add_edge(self, source: _Ref, target: _Ref,
                  label: str | None = None) -> None:
        edge = ProcessDAGEdge(source.key, target.key, label)
        if edge not in self.edges:
            self.edges.append(edge)


def _function_ast(fn: Callable[..., Any]) -> ast.FunctionDef | None:
    source = getattr(fn, "__cimba_source__", None)
    if source is None:
        try:
            source = inspect.getsource(fn)
        except (OSError, TypeError):
            return None
    tree = ast.parse(textwrap.dedent(source))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    return None


def _sim_verb(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == "sim":
            return node.attr
    return None


def _ref_from_key(key: str) -> _Ref:
    kind, _, name = key.partition(":")
    return _Ref(kind, name)


def _escape_mermaid(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("|", "&#124;")
        .replace("\n", "<br/>")
    )


def _mermaid_id(key: str) -> str:
    return "n_" + re.sub(r"[^0-9A-Za-z_]", "_", key)


def _mermaid_node_line(node: ProcessDAGNode, indent: str) -> str:
    node_id = _mermaid_id(node.key)
    label = _escape_mermaid(node.name)
    if node.kind == "process":
        return f"{indent}{node_id}[\"{label}\"]"
    return f"{indent}{node_id}[(\"{label}\")]"


def _dot_quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _dot_cluster_id(key: str) -> str:
    return "cluster_" + re.sub(r"[^0-9A-Za-z_]", "_", key)


def _dot_node_line(node: ProcessDAGNode, indent: str) -> str:
    shape = "box" if node.kind == "process" else "ellipse"
    return (
        f"{indent}{_dot_quote(node.key)} "
        f"[label={_dot_quote(node.name)}, shape={shape}];"
    )


__all__ = [
    "ProcessDAG",
    "ProcessDAGBlock",
    "ProcessDAGEdge",
    "ProcessDAGNode",
    "infer_process_dag",
]
