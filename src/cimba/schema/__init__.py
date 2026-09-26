"""Pure logical descriptions of model classes and configured assemblies."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from types import UnionType
from types import MappingProxyType
from typing import Any, Union, get_args, get_origin, get_type_hints

from cimba.inputs.sources import Source
from cimba.modeling import (
    Condition, Container, Dataset, Entity, Input, Model, Output, Param,
    PriorityStore, Ref, Resource, Series, State, Store, Sweep,
)


class ModelDefinitionError(ValueError):
    pass


@dataclass(frozen=True)
class Field:
    name: str
    kind: str
    value_type: Any
    default: Any
    step: float | None = None
    origin: float = 0.0
    optional: bool = False


@dataclass(frozen=True)
class ClassSchema:
    cls: type[Model]
    fields: tuple[Field, ...]
    processes: tuple[tuple[str, int, int], ...]
    starts: tuple[str, ...]
    ends: tuple[str, ...]
    predicates: tuple[str, ...]
    events: tuple[str, ...]

    @classmethod
    @lru_cache(maxsize=512)
    def of(cls, model_class: type[Model]) -> "ClassSchema":
        if not isinstance(model_class, type) or not issubclass(model_class, Model):
            raise ModelDefinitionError("schema requires a Model class")
        fields: dict[str, Field] = {}
        for base in reversed(model_class.__mro__):
            if not issubclass(base, Model) or base is Model:
                continue
            try:
                annotations = get_type_hints(base, include_extras=True)
            except (NameError, AttributeError) as exc:
                raise ModelDefinitionError(
                    f"cannot resolve annotations of {base.__name__}: {exc}") from exc
            for name, annotation in annotations.items():
                optional = False
                if get_origin(annotation) in (UnionType, Union):
                    members = [member for member in get_args(annotation)
                               if member is not type(None)]
                    if len(members) == 1:
                        optional = len(members) != len(get_args(annotation))
                        annotation = members[0]
                origin = get_origin(annotation) or annotation
                args = get_args(annotation)
                default = getattr(model_class, name, None)
                if origin is Param:
                    kind = "param"
                elif origin is State:
                    kind = "state"
                elif origin is Output:
                    kind = "output"
                elif origin is Input:
                    kind = "input"
                elif origin is Series:
                    kind = "series"
                elif origin is Ref:
                    kind = "ref"
                elif origin in (Container, Store, PriorityStore, Resource,
                                Condition, Dataset) or (
                    isinstance(origin, type) and issubclass(origin, Entity)
                ):
                    kind = "entity"
                elif origin is list:
                    kind = "collection"
                elif isinstance(origin, type) and issubclass(origin, Model):
                    kind = "child"
                else:
                    kind = "constant"
                step = default.step if type(default) is Series else None
                initial_origin = default.origin if type(default) is Series else 0.0
                value_type = annotation if kind == "entity" else (
                    args[0] if args else annotation)
                fields[name] = Field(name, kind, value_type,
                                     default, step, initial_origin, optional)
        roles: dict[str, list[tuple[str, Any]]] = {
            "process": [], "on_start": [], "on_end": [],
            "predicate": [], "event": [],
        }
        for base in reversed(model_class.__mro__):
            for name, method in getattr(base, "__dict__", {}).items():
                role = getattr(method, "cimba_role", None)
                if role in roles:
                    roles[role] = [(n, f) for n, f in roles[role] if n != name]
                    roles[role].append((name, method))
        return cls(
            model_class, tuple(fields.values()),
            tuple((name, method.cimba_copies, method.cimba_priority)
                  for name, method in roles["process"]),
            tuple(name for name, _ in roles["on_start"]),
            tuple(name for name, _ in roles["on_end"]),
            tuple(name for name, _ in roles["predicate"]),
            tuple(name for name, _ in roles["event"]),
        )


@dataclass(frozen=True, eq=False)
class Instance:
    model: Model
    label: str
    schema: ClassSchema
    values: MappingProxyType


@dataclass(frozen=True)
class Assembly:
    root: Model
    instances: tuple[Instance, ...]
    by_object: MappingProxyType

    @classmethod
    def of(cls, root: Model) -> "Assembly":
        if not isinstance(root, Model):
            raise ModelDefinitionError("experiment root must be a Model")
        instances: list[Instance] = []
        visited: dict[Model, Instance] = {}

        def visit(model: Model, label: str) -> None:
            if model in visited:
                raise ModelDefinitionError(f"{label}: model instance reused in tree")
            schema = ClassSchema.of(type(model))
            values: dict[str, Any] = {}
            children: list[tuple[Model, str]] = []
            for field in schema.fields:
                value = model.__dict__.get(field.name, field.default)
                if field.kind == "entity" and value is None:
                    entity_type = get_origin(field.value_type) or field.value_type
                    value = entity_type() if isinstance(entity_type, type) else Entity()
                    object.__setattr__(model, field.name, value)
                elif field.kind == "entity" and value is field.default:
                    value = deepcopy(value)
                    object.__setattr__(model, field.name, value)
                if field.kind == "entity":
                    entity_type = get_origin(field.value_type) or field.value_type
                    if not isinstance(value, entity_type):
                        raise ModelDefinitionError(
                            f"{label}.{field.name}: expected {entity_type.__name__}")
                if field.kind in {"input", "series"}:
                    if type(value) is Series:
                        raise ModelDefinitionError(f"{label}.{field.name}: input source missing")
                    if not isinstance(value, (Source, Sweep)):
                        raise ModelDefinitionError(f"{label}.{field.name}: input source missing")
                    choices = value.values if isinstance(value, Sweep) else (value,)
                    if not all(isinstance(x, Source) for x in choices):
                        raise ModelDefinitionError(f"{label}.{field.name}: invalid source")
                    if field.kind == "series" and field.step is None:
                        raise ModelDefinitionError(f"{label}.{field.name}: Series step missing")
                elif field.kind == "param" and value is None:
                    raise ModelDefinitionError(f"{label}.{field.name}: parameter missing")
                elif field.kind == "state" and value is None:
                    raise ModelDefinitionError(f"{label}.{field.name}: initial state missing")
                elif field.kind == "child":
                    if not isinstance(value, field.value_type):
                        raise ModelDefinitionError(
                            f"{label}.{field.name}: expected "
                            f"{field.value_type.__name__} child")
                    assert isinstance(value, Model)
                    children.append((value, f"{label}.{field.name}"))
                elif field.kind == "collection":
                    if not isinstance(value, list):
                        raise ModelDefinitionError(f"{label}.{field.name}: expected list")
                    is_refs = get_origin(field.value_type) is Ref
                    expected = (get_args(field.value_type)[0] if is_refs
                                else field.value_type)
                    for i, item in enumerate(value):
                        if item is None and is_refs:
                            continue
                        if not isinstance(item, expected):
                            raise ModelDefinitionError(
                                f"{label}.{field.name}[{i}]: expected "
                                f"{expected.__name__}")
                    if not is_refs:
                        for i, item in enumerate(value):
                            if isinstance(item, Model):
                                children.append((item, f"{label}.{field.name}[{i}]"))
                    value = tuple(value)
                values[field.name] = value
            instance = Instance(model, label, schema, MappingProxyType(values))
            visited[model] = instance
            instances.append(instance)
            for child, child_label in children:
                visit(child, child_label)

        visit(root, type(root).__name__.lower())
        for instance in instances:
            for field in instance.schema.fields:
                if field.kind == "ref":
                    target = instance.values[field.name]
                    if target is None and not field.optional:
                        raise ModelDefinitionError(
                            f"{instance.label}.{field.name}: reference missing")
                    if target is not None and not isinstance(target, field.value_type):
                        raise ModelDefinitionError(
                            f"{instance.label}.{field.name}: expected "
                            f"{field.value_type.__name__} reference")
                    if target is not None and target not in visited:
                        raise ModelDefinitionError(
                            f"{instance.label}.{field.name}: dangling reference")
                elif field.kind == "collection" and get_origin(field.value_type) is Ref:
                    for position, target in enumerate(instance.values[field.name]):
                        if target is not None and target not in visited:
                            raise ModelDefinitionError(
                                f"{instance.label}.{field.name}[{position}]: dangling reference")
        return cls(root, tuple(instances), MappingProxyType(visited))
