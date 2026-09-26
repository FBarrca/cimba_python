"""Mermaid diagrams of a configured model's static object graph."""

from __future__ import annotations

from typing import get_origin

from cimba.modeling import Model, Ref
from cimba.schema import Assembly


def mermaid(model: Model) -> str:
    assembly = Assembly.of(model)
    identifiers = {instance.model: f"n{index}"
                   for index, instance in enumerate(assembly.instances)}
    lines = ["flowchart TD"]
    for instance in assembly.instances:
        label = instance.label.replace('"', "&quot;")
        lines.append(f'  {identifiers[instance.model]}["{label}: '
                     f'{instance.schema.cls.__name__}"]')
    for instance in assembly.instances:
        source = identifiers[instance.model]
        for field in instance.schema.fields:
            value = instance.values[field.name]
            if field.kind in {"child", "ref"} and value is not None:
                style = "-->" if field.kind == "child" else "-.->"
                lines.append(f"  {source} {style}|{field.name}| {identifiers[value]}")
            elif field.kind == "collection":
                style = "-.->" if get_origin(field.value_type) is Ref else "-->"
                for index, member in enumerate(value or ()):
                    if member is not None:
                        lines.append(f"  {source} {style}|{field.name}[{index}]| "
                                     f"{identifiers[member]}")
    return "\n".join(lines)


__all__ = ["mermaid"]
