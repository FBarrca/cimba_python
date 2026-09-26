"""Diagrams of configured models: structure and process interactions.

``structure(model)`` draws the object tree: which model owns which, and which
references which. ``process_graph(model)`` infers how processes interact with
entities, inputs and spawned models from their source. Both return a
:class:`Graph` that renders to Mermaid (``to_mermaid``) or Graphviz DOT
(``to_dot``). Nothing is compiled or run.
"""

from __future__ import annotations

from typing import get_origin

from cimba.modeling import Model, Ref
from cimba.schema import Assembly

from .graph import Edge, Graph, Group, Node
from .processes import process_graph


def structure(model: Model) -> Graph:
    """The configured object tree: solid edges own, dotted edges reference."""
    assembly = Assembly.of(model, strict=False)
    nodes = tuple(Node(instance.label, f"{instance.label}: {instance.schema.cls.__name__}",
                       "instance") for instance in assembly.instances)
    label = {instance.model: instance.label for instance in assembly.instances}
    edges = []
    for instance in assembly.instances:
        for field in instance.schema.fields:
            value = instance.values[field.name]
            if field.kind in {"child", "ref"} and value is not None:
                style = "solid" if field.kind == "child" else "dotted"
                edges.append(Edge(instance.label, label[value], field.name, style))
            elif field.kind == "collection":
                style = "dotted" if get_origin(field.value_type) is Ref else "solid"
                for index, member in enumerate(value or ()):
                    if member is not None:
                        edges.append(Edge(instance.label, label[member],
                                          f"{field.name}[{index}]", style))
    return Graph(nodes, tuple(edges))


def mermaid(model: Model) -> str:
    """Mermaid flowchart text of :func:`structure` (kept for convenience)."""
    return structure(model).to_mermaid()


__all__ = ["Edge", "Graph", "Group", "Node", "mermaid", "process_graph", "structure"]
