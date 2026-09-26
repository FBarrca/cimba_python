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


_SCALARS = (float, int, bool)


@dataclass(frozen=True)
class FunctionSignature:
    """Annotated signature of a ``@function``: parameters after ``self``."""

    parameters: tuple[tuple[str, Any], ...]
    defaults: tuple[Any, ...]          # for the trailing parameters
    returns: Any                       # float, int, bool, a Model class or None

    @property
    def types(self) -> tuple[Any, ...]:
        return tuple(kind for _, kind in self.parameters)


def _function_signature(cls, name: str, method) -> FunctionSignature:
    import inspect
    where = f"{cls.__name__}.{name}"
    try:
        hints = get_type_hints(method)
    except (NameError, AttributeError) as exc:
        raise ModelDefinitionError(f"{where}: cannot resolve annotations: {exc}") from exc
    parameters = list(inspect.signature(method).parameters.values())[1:]
    typed, defaults = [], []
    for parameter in parameters:
        if parameter.kind not in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD):
            raise ModelDefinitionError(f"{where}: only positional parameters are supported")
        kind = hints.get(parameter.name)
        if kind is None:
            raise ModelDefinitionError(
                f"{where}: annotate parameter {parameter.name!r} "
                "(float, int, bool or a model class)")
        if kind not in _SCALARS and not (isinstance(kind, type) and issubclass(kind, Model)):
            raise ModelDefinitionError(
                f"{where}: parameter {parameter.name!r} must be float, int, bool "
                "or a model class")
        if parameter.default is not parameter.empty:
            if kind not in _SCALARS:
                raise ModelDefinitionError(f"{where}: only scalar parameters may have defaults")
            defaults.append(kind(parameter.default))
        elif defaults:
            raise ModelDefinitionError(f"{where}: parameters with defaults must come last")
        typed.append((parameter.name, kind))
    if "return" not in hints:
        raise ModelDefinitionError(
            f"{where}: annotate the return type (float, int, bool, a model class or None)")
    returns = hints["return"]
    if returns is type(None):
        returns = None
    if returns is not None and returns not in _SCALARS and not (
        isinstance(returns, type) and issubclass(returns, Model)
    ):
        raise ModelDefinitionError(
            f"{where}: return type must be float, int, bool, a model class or None")
    return FunctionSignature(tuple(typed), tuple(defaults), returns)


@dataclass(frozen=True)
class ClassSchema:
    cls: type[Model]
    fields: tuple[Field, ...]
    processes: tuple[tuple[str, int, int], ...]
    starts: tuple[str, ...]
    ends: tuple[str, ...]
    predicates: tuple[str, ...]
    events: tuple[str, ...]
    functions: tuple[str, ...] = ()
    # Event, predicate and function names in dispatch-table order. A subclass
    # keeps its bases' slots in the same positions and appends new ones, so a
    # base-typed view finds any subclass's implementation at the same index.
    slots: tuple[str, ...] = ()
    signatures: tuple[tuple[str, FunctionSignature], ...] = ()

    def signature(self, name: str) -> FunctionSignature:
        return dict(self.signatures)[name]

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
        }
        slotted: dict[str, str] = {}      # name -> event | predicate | function
        for base in reversed(model_class.__mro__):
            for name, method in getattr(base, "__dict__", {}).items():
                role = getattr(method, "cimba_role", None)
                if name in slotted and role != slotted[name]:
                    raise ModelDefinitionError(
                        f"{base.__name__}.{name} overrides a @{slotted[name]} "
                        f"and must also be decorated with @{slotted[name]}")
                if role in roles:
                    roles[role] = [(n, f) for n, f in roles[role] if n != name]
                    roles[role].append((name, method))
                elif role in {"event", "predicate", "function"}:
                    slotted.setdefault(name, role)   # overrides keep their slot
        for name in slotted:
            if name in fields:
                raise ModelDefinitionError(
                    f"{model_class.__name__}.{name}: a method and a field share this name")
            if hasattr(Model, name):
                raise ModelDefinitionError(
                    f"{model_class.__name__}.{name}: this name is reserved by cimba.Model")
        signatures = []
        for name, role in slotted.items():
            if role != "function":
                continue
            signature = _function_signature(model_class, name, getattr(model_class, name))
            for base in model_class.__mro__[1:]:
                declared = getattr(base, "__dict__", {}).get(name)
                if getattr(declared, "cimba_role", None) == "function":
                    inherited = _function_signature(base, name, declared)
                    if (inherited.types, inherited.returns) != (signature.types, signature.returns):
                        raise ModelDefinitionError(
                            f"{model_class.__name__}.{name}: an override must keep the "
                            f"signature declared in {base.__name__}")
                    break
            signatures.append((name, signature))
        schema = cls(
            model_class, tuple(fields.values()),
            tuple((name, method.cimba_copies, method.cimba_priority)
                  for name, method in roles["process"]),
            tuple(name for name, _ in roles["on_start"]),
            tuple(name for name, _ in roles["on_end"]),
            tuple(name for name, role in slotted.items() if role == "predicate"),
            tuple(name for name, role in slotted.items() if role == "event"),
            tuple(name for name, role in slotted.items() if role == "function"),
            tuple(slotted),
            tuple(signatures),
        )
        for base in model_class.__mro__[1:]:
            if isinstance(base, type) and issubclass(base, Model) and base is not Model:
                inherited_slots = ClassSchema.of(base).slots
                if schema.slots[:len(inherited_slots)] != inherited_slots:
                    raise ModelDefinitionError(
                        f"{model_class.__name__}: events, predicates and functions "
                        f"inherited from {base.__name__} conflict with another base "
                        "class; use single inheritance for model behavior")
        return schema


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
    def of(cls, root: Model, *, strict: bool = True) -> "Assembly":
        """Snapshot the configured tree rooted at ``root``.

        With ``strict=False`` missing sources, parameters, initial states and
        required references are tolerated (used to draw unfinished models).
        """
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
                if field.kind in {"input", "series"} and not strict and (
                    value is None or type(value) is Series
                ):
                    pass
                elif field.kind in {"input", "series"}:
                    if type(value) is Series:
                        raise ModelDefinitionError(f"{label}.{field.name}: input source missing")
                    if not isinstance(value, (Source, Sweep)):
                        raise ModelDefinitionError(f"{label}.{field.name}: input source missing")
                    choices = value.values if isinstance(value, Sweep) else (value,)
                    if not all(isinstance(x, Source) for x in choices):
                        raise ModelDefinitionError(f"{label}.{field.name}: invalid source")
                    if field.kind == "series" and field.step is None:
                        raise ModelDefinitionError(f"{label}.{field.name}: Series step missing")
                elif field.kind == "param" and value is None and strict:
                    raise ModelDefinitionError(f"{label}.{field.name}: parameter missing")
                elif field.kind == "state" and value is None and strict:
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
                    value = instance.values[field.name]
                    # A swept reference picks one of several models per design
                    # point: a pointer is data, so every choice must already be
                    # part of the tree.
                    targets = value.values if isinstance(value, Sweep) else (value,)
                    for target in targets:
                        if target is None and not field.optional and strict:
                            raise ModelDefinitionError(
                                f"{instance.label}.{field.name}: reference missing")
                        if target is not None and not isinstance(target, field.value_type):
                            raise ModelDefinitionError(
                                f"{instance.label}.{field.name}: expected "
                                f"{field.value_type.__name__} reference")
                        if target is not None and target not in visited:
                            raise ModelDefinitionError(
                                f"{instance.label}.{field.name}: "
                                + ("swept reference to a model outside the tree"
                                   if isinstance(value, Sweep) else "dangling reference"))
                elif field.kind == "collection" and get_origin(field.value_type) is Ref:
                    for position, target in enumerate(instance.values[field.name]):
                        if target is not None and target not in visited:
                            raise ModelDefinitionError(
                                f"{instance.label}.{field.name}[{position}]: dangling reference")
        return cls(root, tuple(instances), MappingProxyType(visited))
