"""Pure logical descriptions of model classes and configured assemblies."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, IntEnum
from functools import lru_cache
from math import isfinite
from numbers import Integral, Real
from types import UnionType
from types import MappingProxyType
from collections.abc import Mapping
from typing import Any, TypeGuard, Union, cast, get_args, get_origin, get_type_hints

from cimba.inputs.sources import EmptySource, Source
from cimba.modeling import (
    Array, Condition, Container, Dataset, Decision, Entity, FractionalContainer, Input, Model, Output,
    Param, PriorityStore, Ref, Resource, Series, Store, Sweep,
)


class ModelDefinitionError(ValueError):
    pass


# Field kinds whose value shapes the model tree. A sweep of one of these is a
# structural axis: each design point runs on a tree holding only its option.
STRUCTURAL = frozenset({"child", "ref", "collection"})


def is_int_enum(kind) -> TypeGuard[type[IntEnum]]:
    """Whether ``kind`` is an ``enum.IntEnum`` class, stored natively as int64."""
    return isinstance(kind, type) and issubclass(kind, IntEnum)


def check_scalar_type(kind, where: str) -> None:
    """Reject enum classes that native records cannot hold as int64 codes."""
    if not (isinstance(kind, type) and issubclass(kind, Enum)):
        return
    if not issubclass(kind, IntEnum):
        raise ModelDefinitionError(
            f"{where}: {kind.__name__} is not an enum.IntEnum; compiled model code "
            f"stores enums as their integer values, so derive {kind.__name__} "
            "from enum.IntEnum")
    members = list(kind.__members__.values())
    if not members:
        raise ModelDefinitionError(f"{where}: {kind.__name__} has no members")
    for member in members:
        if not -(1 << 63) <= int(member) < (1 << 63):
            raise ModelDefinitionError(
                f"{where}: {kind.__name__}.{member.name} is outside int64 range")


def enum_member(kind: type[IntEnum], value, path: str) -> IntEnum:
    """``value`` as a member of ``kind``: a member, or the int value of one."""
    if isinstance(value, kind):
        return value
    if isinstance(value, Enum) or isinstance(value, bool) or not isinstance(value, Integral):
        raise ModelDefinitionError(f"{path}: expected a {kind.__name__} member, got {value!r}")
    try:
        return kind(int(value))
    except ValueError:
        raise ModelDefinitionError(
            f"{path}: {value!r} is not a {kind.__name__} value") from None


def scalar_value(field: Field, value, path: str):
    """Validate before NumPy can silently truncate a scalar binding."""
    kind = field.value_type
    if is_int_enum(kind):
        return enum_member(kind, value, path)
    if kind not in (float, int, bool):
        raise ModelDefinitionError(f"{path}: expected a float, int or bool field")
    if not isinstance(value, Real):
        raise ModelDefinitionError(f"{path}: expected a {kind.__name__} value")
    if kind is int and (not isfinite(value) or value % 1 != 0):
        raise ModelDefinitionError(f"{path}: expected an integral value")
    if kind is int and not -(1 << 63) <= cast(Any, value) < (1 << 63):
        raise ModelDefinitionError(f"{path}: integer value is outside int64 range")
    if kind is bool and value not in (0, 1):
        raise ModelDefinitionError(f"{path}: expected a boolean value")
    # Runtime numeric ABCs are broader than typing's conversion protocols.
    return kind(cast(Any, value))


ARRAY_ELEMENTS = (int, float, bool)


def check_array(field: Field, value, path: str) -> None:
    """Validate an ``Array`` configuration, or each option of a sweep of them."""
    for option in (value.values if isinstance(value, Sweep) else (value,)):
        if not isinstance(option, Array):
            raise ModelDefinitionError(
                f"{path}: expected a cb.Array (cb.Array(length) or cb.Array([values]))")
        for position, element in enumerate(option.values or ()):
            scalar_value(field, element, f"{path}[{position}]")


# Entities a ``cb.Ref`` field (or a ``list[cb.Ref[...]]``) can reference.
ENTITY_TYPES = (Container, FractionalContainer, Store, PriorityStore, Resource, Condition,
                Dataset)


def entity_class(annotation) -> type[Entity] | None:
    """The entity class an annotation names (``Store[int]`` gives ``Store``), else None."""
    origin = get_origin(annotation) or annotation
    return origin if origin in ENTITY_TYPES else None


def entity_name(annotation) -> str:
    """``Resource``, ``Store[int]``: an entity annotation as written."""
    origin = get_origin(annotation)
    if origin is None:
        return getattr(annotation, "__name__", str(annotation))
    arguments = ", ".join(getattr(a, "__name__", str(a)) for a in get_args(annotation))
    return f"{origin.__name__}[{arguments}]"


def reference_target(field: "Field"):
    """What a ``Ref`` field, or each item of a ``list[Ref[...]]``, points at, else None."""
    if field.kind == "ref":
        return field.value_type
    if field.kind == "collection" and get_origin(field.value_type) is Ref:
        return get_args(field.value_type)[0]
    return None


def is_entity_reference(field: "Field") -> bool:
    """Whether a field references entities (``Ref[Resource]``, ``list[Ref[Resource]]``)."""
    target = reference_target(field)
    return target is not None and entity_class(target) is not None


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
    returns: Any                       # float, int, bool, an IntEnum, a Model class or None

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
                "(float, int, bool, an IntEnum or a model class)")
        check_scalar_type(kind, f"{where}: parameter {parameter.name!r}")
        if kind not in _SCALARS and not is_int_enum(kind) and not (
            isinstance(kind, type) and issubclass(kind, Model)
        ):
            raise ModelDefinitionError(
                f"{where}: parameter {parameter.name!r} must be float, int, bool, "
                "an IntEnum or a model class")
        if parameter.default is not parameter.empty:
            if is_int_enum(kind):
                defaults.append(enum_member(kind, parameter.default,
                                            f"{where}: parameter {parameter.name!r}"))
            elif kind not in _SCALARS:
                raise ModelDefinitionError(f"{where}: only scalar parameters may have defaults")
            else:
                defaults.append(kind(parameter.default))
        elif defaults:
            raise ModelDefinitionError(f"{where}: parameters with defaults must come last")
        typed.append((parameter.name, kind))
    if "return" not in hints:
        raise ModelDefinitionError(
            f"{where}: annotate the return type "
            "(float, int, bool, an IntEnum, a model class or None)")
    returns = hints["return"]
    if returns is type(None):
        returns = None
    check_scalar_type(returns, f"{where}: return type")
    if returns is not None and returns not in _SCALARS and not is_int_enum(returns) and not (
        isinstance(returns, type) and issubclass(returns, Model)
    ):
        raise ModelDefinitionError(
            f"{where}: return type must be float, int, bool, an IntEnum, "
            "a model class or None")
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
                elif origin is Output:
                    kind = "output"
                elif origin is Input:
                    kind = "input"
                elif origin is Series:
                    kind = "series"
                elif origin is Ref:
                    kind = "ref"
                elif origin is Array:
                    kind = "array"
                    element = args[0] if args else None
                    if element not in ARRAY_ELEMENTS:
                        raise ModelDefinitionError(
                            f"{model_class.__name__}.{name}: cb.Array elements must be "
                            f"int, float or bool, got {element!r}")
                elif origin in (Container, FractionalContainer, Store, PriorityStore, Resource,
                                Condition, Dataset) or (
                    isinstance(origin, type) and issubclass(origin, Entity)
                ):
                    kind = "entity"
                elif origin is list:
                    kind = "collection"
                elif isinstance(origin, type) and issubclass(origin, Model):
                    kind = "child"
                else:
                    # A plain annotated scalar: a per-trial value.
                    kind = "field"
                if kind in {"param", "field", "output"}:
                    scalar = args[0] if args else annotation
                    where = f"{model_class.__name__}.{name}"
                    check_scalar_type(scalar, where)
                    if is_int_enum(scalar) and default is not None and not isinstance(
                            default, Decision):
                        default = (default.map(lambda value: enum_member(scalar, value, where))
                                   if isinstance(default, Sweep) else
                                   enum_member(scalar, default, where))
                step = default.step if type(default) is Series else None
                initial_origin = default.origin if type(default) is Series else 0.0
                value_type = annotation if kind == "entity" else (
                    args[0] if args else annotation)
                fields[name] = Field(name, kind, value_type,
                                     default, step, initial_origin, optional)
                if kind == "array" and default is not None:
                    check_array(fields[name], default, f"{model_class.__name__}.{name}")
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
    # Each entity of the tree -> (owning model, field name)
    entities: MappingProxyType

    def decisions(self):
        """Decision bindings in tree and field order, preserving object identity."""
        return tuple((instance, field, instance.values[field.name])
                     for instance in self.instances for field in instance.schema.fields
                     if isinstance(instance.values[field.name], Decision))

    @classmethod
    def of(cls, root: Model, *, strict: bool = True,
           picks: Mapping[tuple[Model, str], Model] | None = None) -> "Assembly":
        """Snapshot the configured tree rooted at ``root``.

        A child field set to a sweep of models contributes every option, each
        labelled ``path.field#i``, and a swept ``list[Model]`` every option's
        items, as ``path.field#i[j]`` (an item shared by options joins once,
        under its first label). Swept references (``Ref`` and ``list[Ref]``)
        are checked option by option. ``picks`` fixes the option of a swept
        child, reference or collection for that (model, field): the tree then
        holds only the chosen value, under the plain label, and references are
        resolved against that tree. With ``strict=False`` missing sources,
        parameters, initial states and required references are tolerated (used
        to draw unfinished models).
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
                if isinstance(value, Decision) and field.kind != "param":
                    raise ModelDefinitionError(f"{label}.{field.name}: a decision requires a Param field")
                if (field.kind in {"child", "collection"} and value is not None and
                        field.name not in model.__dict__ and
                        not (field.kind == "collection" and
                             get_origin(field.value_type) is Ref)):
                    # Class defaults are templates: materialize and attach once
                    # per owner, never again when snapshotting a swept variant.
                    value = deepcopy(value)
                    setattr(model, field.name, value)
                if (isinstance(value, Sweep) and field.kind in STRUCTURAL and
                        picks is not None and (model, field.name) in picks):
                    # This tree holds the option a design point selects.
                    value = picks[(model, field.name)]
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
                    if field.kind == "input" and field.optional and value is None:
                        value = EmptySource()
                    if type(value) is Series:
                        raise ModelDefinitionError(f"{label}.{field.name}: input source missing")
                    if not isinstance(value, (Source, Sweep)):
                        raise ModelDefinitionError(f"{label}.{field.name}: input source missing")
                    choices = value.values if isinstance(value, Sweep) else (value,)
                    if not all(isinstance(x, Source) or
                               (field.kind == "input" and field.optional and x is None)
                               for x in choices):
                        raise ModelDefinitionError(f"{label}.{field.name}: invalid source")
                    if field.kind == "series" and field.step is None:
                        raise ModelDefinitionError(f"{label}.{field.name}: Series step missing")
                elif field.kind == "param" and value is None and strict:
                    raise ModelDefinitionError(f"{label}.{field.name}: parameter missing")
                elif field.kind == "field" and value is None and strict:
                    raise ModelDefinitionError(f"{label}.{field.name}: initial value missing")
                elif field.kind == "array" and value is None:
                    if strict:
                        raise ModelDefinitionError(
                            f"{label}.{field.name}: array length missing; assign "
                            "cb.Array(length) or give the field a class default")
                elif field.kind == "array":
                    check_array(field, value, f"{label}.{field.name}")
                elif field.kind == "child" and isinstance(value, Sweep):
                    for i, option in enumerate(value.values):
                        if not isinstance(option, field.value_type):
                            raise ModelDefinitionError(
                                f"{label}.{field.name}: sweep option {i} is not a "
                                f"{field.value_type.__name__}")
                        children.append((option, f"{label}.{field.name}#{i}"))
                elif field.kind == "child" and value is None and field.optional:
                    pass
                elif field.kind == "child":
                    if not isinstance(value, field.value_type):
                        raise ModelDefinitionError(
                            f"{label}.{field.name}: expected "
                            f"{field.value_type.__name__} child")
                    assert isinstance(value, Model)
                    children.append((value, f"{label}.{field.name}"))
                elif field.kind == "collection":
                    is_refs = get_origin(field.value_type) is Ref
                    expected = (get_args(field.value_type)[0] if is_refs
                                else field.value_type)
                    axis = value.axis if isinstance(value, Sweep) else None
                    swept = axis is not None
                    lists: tuple[Any, ...] = (value.values if isinstance(value, Sweep)
                                              else (value,))
                    owned: dict[Model, str] = {}
                    for option, items in enumerate(lists):
                        where = f"{label}.{field.name}" + (f"#{option}" if swept else "")
                        # A design point's tree holds the option as a tuple.
                        if not isinstance(items, (list, tuple)):
                            raise ModelDefinitionError(
                                f"{where}: expected list" +
                                (" (sweep lists: cb.sweep([a], [a, b]))" if swept else ""))
                        # Entity references are checked against the tree's entities below
                        for i, item in enumerate(() if is_entity_reference(field) else items):
                            if item is None and is_refs:
                                continue
                            if not isinstance(item, expected):
                                raise ModelDefinitionError(
                                    f"{where}[{i}]: expected {expected.__name__}")
                        if not is_refs:
                            # Options may share items: each joins the tree once,
                            # labelled by its first option.
                            for i, item in enumerate(items):
                                owned.setdefault(item, f"{where}[{i}]")
                    children.extend(owned.items())
                    value = (Sweep(tuple(tuple(items) for items in lists), axis)
                             if axis is not None else tuple(lists[0]))
                values[field.name] = value
            instance = Instance(model, label, schema, MappingProxyType(values))
            visited[model] = instance
            instances.append(instance)
            for child, child_label in children:
                visit(child, child_label)

        visit(root, type(root).__name__.lower())
        owners: dict[Entity, tuple[Instance, Field]] = {}
        for instance in instances:
            for field in instance.schema.fields:
                if field.kind == "entity":
                    owners.setdefault(instance.values[field.name], (instance, field))

        def check_entity(target, annotation, where: str) -> None:
            """A referenced entity: one of this tree's, held by a field of the
            referenced type (a ``Ref[Store[int]]`` needs a ``Store[int]``)."""
            expected = entity_name(annotation)
            entity_type = entity_class(annotation)
            assert entity_type is not None
            if not isinstance(target, entity_type):
                raise ModelDefinitionError(
                    f"{where}: expected a {expected} of this model tree, got {target!r}")
            owner = owners.get(target)
            if owner is None:
                raise ModelDefinitionError(
                    f"{where}: references a {expected} that is not an entity of this "
                    "model tree (reference an entity a model of the tree holds, such "
                    "as role.res)")
            holder, held = owner
            if held.value_type != annotation:
                raise ModelDefinitionError(
                    f"{where}: references {holder.label}.{held.name}, a "
                    f"{entity_name(held.value_type)}, not a {expected}")

        def check_model(target, model_type, where: str) -> None:
            """A referenced model: one of this tree's, of the referenced class."""
            if not isinstance(target, model_type):
                raise ModelDefinitionError(f"{where}: expected {model_type.__name__} reference")
            if target not in visited:
                raise ModelDefinitionError(
                    f"{where}: dangling reference" if picks is None else
                    f"{where}: dangling reference (the referenced model is not part of this "
                    "design point's tree; sweep the reference with the option it points "
                    "into, as in cb.sweeps([a, b], [a.node, b.node]))")

        for instance in instances:
            for field in instance.schema.fields:
                target_type = reference_target(field)
                if target_type is None:
                    continue
                value = instance.values[field.name]
                swept = isinstance(value, Sweep)
                for option, choice in enumerate(value.values if swept else (value,)):
                    where = f"{instance.label}.{field.name}" + (f"#{option}" if swept else "")
                    if field.kind == "ref":
                        if choice is None:
                            if not field.optional and strict:
                                raise ModelDefinitionError(f"{where}: reference missing")
                        elif entity_class(target_type) is not None:
                            check_entity(choice, target_type, where)
                        else:
                            check_model(choice, target_type, where)
                    elif entity_class(target_type) is not None:
                        for position, target in enumerate(choice):
                            check_entity(target, target_type, f"{where}[{position}]")
                    else:
                        for position, target in enumerate(choice):
                            if target is not None:
                                check_model(target, target_type, f"{where}[{position}]")
        return cls(root, tuple(instances), MappingProxyType(visited),
                   MappingProxyType({entity: (holder.model, held.name)
                                     for entity, (holder, held) in owners.items()}))
