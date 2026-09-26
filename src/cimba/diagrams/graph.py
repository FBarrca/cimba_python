"""Renderer-neutral graphs with Mermaid and Graphviz DOT output."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    """A diagram node. ``key`` is unique; ``group`` names its enclosing group."""

    key: str
    label: str
    kind: str
    group: str | None = None


@dataclass(frozen=True)
class Edge:
    """A directed edge; ``style`` is ``"solid"`` or ``"dotted"``."""

    source: str
    target: str
    label: str | None = None
    style: str = "solid"


@dataclass(frozen=True)
class Group:
    """A box around nodes: a model instance or a spawned model class."""

    key: str
    label: str
    kind: str = "model"


_MERMAID_SHAPES = {
    "instance": ('["', '"]'),
    "process": ('(["', '"])'),
    "hook": ('(["', '"])'),
    "event": ('{{"', '"}}'),
    "input": ('[/"', '"/]'),
    "state": ('[("', '")]'),
    "model": ('[["', '"]]'),
}
_DOT_SHAPES = {
    "instance": "box",
    "process": "box, style=rounded",
    "hook": "box, style=\"rounded,dashed\"",
    "event": "hexagon",
    "input": "parallelogram",
    "state": "cylinder",
    "model": "component",
}


def _mermaid_text(text: str) -> str:
    return text.replace('"', "&quot;")


def _dot_text(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


@dataclass(frozen=True)
class Graph:
    """An immutable diagram: nodes, edges and groups."""

    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    groups: tuple[Group, ...] = ()

    def node(self, key: str) -> Node:
        """Return the node with ``key``."""
        for node in self.nodes:
            if node.key == key:
                return node
        raise KeyError(key)

    def topological_order(self) -> tuple[str, ...]:
        """Node keys in dependency order; ``ValueError`` if the graph has a cycle.

        Simulation models often contain intentional cycles (a process signals
        a condition it also waits on), so a cycle is information, not an error
        in the model.
        """
        keys = [node.key for node in self.nodes]
        indegree = dict.fromkeys(keys, 0)
        following: dict[str, list[str]] = {key: [] for key in keys}
        for edge in self.edges:
            following[edge.source].append(edge.target)
            indegree[edge.target] += 1
        ready = deque(key for key in keys if indegree[key] == 0)
        order = []
        while ready:
            key = ready.popleft()
            order.append(key)
            for target in following[key]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
        if len(order) != len(keys):
            raise ValueError("the graph contains a cycle: "
                             + ", ".join(k for k in keys if indegree[k]))
        return tuple(order)

    def to_mermaid(self, direction: str = "TD") -> str:
        """Render as Mermaid ``flowchart`` text."""
        ids = {node.key: f"n{index}" for index, node in enumerate(self.nodes)}
        lines = [f"flowchart {direction}"]

        def node_line(node: Node, indent: str) -> str:
            left, right = _MERMAID_SHAPES.get(node.kind, ('["', '"]'))
            return f"{indent}{ids[node.key]}{left}{_mermaid_text(node.label)}{right}"

        for index, group in enumerate(self.groups):
            members = [node for node in self.nodes if node.group == group.key]
            if not members:
                continue
            lines.append(f'  subgraph g{index}["{_mermaid_text(group.label)}"]')
            lines.extend(node_line(node, "    ") for node in members)
            lines.append("  end")
        grouped = {group.key for group in self.groups}
        lines.extend(node_line(node, "  ") for node in self.nodes
                     if node.group not in grouped)
        for edge in self.edges:
            arrow = "-.->" if edge.style == "dotted" else "-->"
            label = f"|{_mermaid_text(edge.label)}|" if edge.label else ""
            lines.append(f"  {ids[edge.source]} {arrow}{label} {ids[edge.target]}")
        return "\n".join(lines)

    def to_dot(self, rankdir: str = "TB") -> str:
        """Render as Graphviz DOT text."""
        lines = ["digraph cimba {", f"  rankdir={rankdir};",
                 '  node [fontname="Helvetica", fontsize=11];',
                 '  edge [fontname="Helvetica", fontsize=9];']

        def node_line(node: Node, indent: str) -> str:
            shape = _DOT_SHAPES.get(node.kind, "box")
            return (f"{indent}{_dot_text(node.key)} "
                    f"[label={_dot_text(node.label)}, shape={shape}];")

        for index, group in enumerate(self.groups):
            members = [node for node in self.nodes if node.group == group.key]
            if not members:
                continue
            lines.append(f"  subgraph cluster_{index} {{")
            lines.append(f"    label={_dot_text(group.label)};")
            if group.kind == "spawned":
                lines.append("    style=dashed;")
            lines.extend(node_line(node, "    ") for node in members)
            lines.append("  }")
        grouped = {group.key for group in self.groups}
        lines.extend(node_line(node, "  ") for node in self.nodes
                     if node.group not in grouped)
        for edge in self.edges:
            attributes = []
            if edge.label:
                attributes.append(f"label={_dot_text(edge.label)}")
            if edge.style == "dotted":
                attributes.append("style=dashed")
            suffix = f" [{', '.join(attributes)}]" if attributes else ""
            lines.append(f"  {_dot_text(edge.source)} -> {_dot_text(edge.target)}{suffix};")
        lines.append("}")
        return "\n".join(lines)


__all__ = ["Edge", "Graph", "Group", "Node"]
