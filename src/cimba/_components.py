"""Component declaration, binding metadata, and AST lowering support."""

import ast
import copy
import inspect
import linecache
import textwrap
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar, get_args, get_origin, get_type_hints, overload

import numpy as np

from ._declarations import (_DECL_KINDS, _MISSING, _check_name,
                            _declarations_contain, _field_declarations)

_F = TypeVar("_F", bound=Callable[..., Any])
_COMPONENT_PROCESS_ATTR = "__cimba_component_process__"
_COMPONENT_COLLECT_ATTR = "__cimba_component_collect__"

#: Entity-field kinds whose declaration may be wired to another component
#: instance's same-kind field instead of creating a new entity.
_WIRABLE_FIELD_KINDS = ("queue", "resource", "pool", "store", "condition")

_entity_field_kinds_cache: dict[type, dict[str, str]] = {}


def _entity_field_kinds(cls: type) -> dict[str, str]:
    kinds = _entity_field_kinds_cache.get(cls)
    if kinds is None:
        kinds = {}
        for fname, hint in get_type_hints(cls).items():
            try:
                kind = _DECL_KINDS.get(hint)
            except TypeError:
                kind = None
            if kind in _WIRABLE_FIELD_KINDS:
                kinds[fname] = kind
        _entity_field_kinds_cache[cls] = kinds
    return kinds


class Component:
    """Authoring-time grouping of model fields and process methods.

    Component instances are declared as defaults on a ``Model`` subclass. Their
    declared fields are flattened into the model's trial record, and methods
    decorated with :func:`process` are lowered into ordinary model processes.
    Methods decorated with :func:`collect` run once per instance at the end of
    each trial, before the model-level ``@model.collect`` callback.

    Accessing a declared Queue/Resource/Pool/Store/Condition field on an
    instance yields a wiring reference: passing it as another instance's
    same-kind field value makes both fields name the same entity, e.g.
    ``Station(..., inbox=station_1.outbox)``.
    """

    def __getattr__(self, name: str) -> "_FieldRef":
        if name.startswith("_"):
            raise AttributeError(name)
        kind = _entity_field_kinds(type(self)).get(name)
        if kind is None:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'")
        return _FieldRef(self, name, kind)


@dataclass(frozen=True)
class _FieldRef:
    """Authoring-time reference to a component instance's entity field,
    produced by accessing a declared wirable field on the instance."""

    instance: Component
    field: str
    kind: str


_AMBIGUOUS_WIRING_TARGET: Any = object()
_COLLECTION_WIRING_TARGET: Any = object()


def _component_field_wiring(
    component_name: str,
    templates: Sequence[Component],
    decls: Mapping[str, Any],
) -> dict[str, "_FieldRef"]:
    """Collect declared entity fields overridden with a wiring reference."""
    wiring: dict[str, _FieldRef] = {}
    for kind in _WIRABLE_FIELD_KINDS:
        for fname in decls[kind]:
            refs = [vars(template).get(fname) for template in templates]
            if not any(isinstance(ref, _FieldRef) for ref in refs):
                continue
            if len(templates) > 1:
                raise ValueError(
                    f"component collection '{component_name}' field "
                    f"'{fname}' cannot be wired to another component's "
                    "field; wiring is not supported for collections yet")
            ref = refs[0]
            if ref.kind != kind:
                raise ValueError(
                    f"component '{component_name}' {kind} field '{fname}' "
                    f"cannot be wired to {ref.kind} field '{ref.field}'; "
                    "the field kinds must match")
            wiring[fname] = ref
    return wiring


def _register_wiring_targets(
    registry: dict[int, Any],
    templates: Sequence[Component],
    direct_field_map: dict[str, str],
) -> None:
    single = len(templates) == 1
    for template in templates:
        template_id = id(template)
        if template_id in registry:
            registry[template_id] = _AMBIGUOUS_WIRING_TARGET
        elif single:
            registry[template_id] = direct_field_map
        else:
            registry[template_id] = _COLLECTION_WIRING_TARGET


def _resolve_wiring_target(
    component_name: str,
    fname: str,
    ref: "_FieldRef",
    registry: Mapping[int, Any],
) -> str:
    target = registry.get(id(ref.instance))
    if target is None:
        raise ValueError(
            f"component '{component_name}' field '{fname}' is wired to a "
            f"{type(ref.instance).__name__} instance that is not declared "
            "on the model before it; declare the wiring target first")
    if target is _COLLECTION_WIRING_TARGET:
        raise ValueError(
            f"component '{component_name}' field '{fname}' is wired to a "
            "component collection item, which is not supported yet")
    if target is _AMBIGUOUS_WIRING_TARGET:
        raise ValueError(
            f"component '{component_name}' field '{fname}' is wired to a "
            "component instance that is the default of more than one "
            "model field; the wiring target is ambiguous")
    return target[ref.field]


def _is_component_class(obj: Any) -> bool:
    return (isinstance(obj, type) and issubclass(obj, Component)
            and obj is not Component)


@dataclass(frozen=True)
class _ComponentProcessSpec:
    copies: int | str = 1
    priority: int = 0


@overload
def process(fn: _F) -> _F: ...


@overload
def process(fn: None = None, *, copies: int | str = 1,
            priority: int = 0) -> Callable[[_F], _F]: ...


def process(fn=None, *, copies: int | str = 1, priority: int = 0):
    """Mark a ``Component`` method to be registered as a model process."""
    if isinstance(copies, int):
        if copies < 1:
            raise ValueError("copies must be >= 1")
    elif isinstance(copies, str):
        _check_name(copies, "copies constant")
    else:
        raise TypeError("copies must be an int or the name of an int constant")

    def decorate(f):
        if getattr(f, _COMPONENT_COLLECT_ATTR, False):
            raise ValueError(
                f"'{f.__qualname__}' cannot be both a component process "
                "and a component collect method")
        setattr(f, _COMPONENT_PROCESS_ATTR,
                _ComponentProcessSpec(copies, priority))
        return f

    if fn is None:
        return decorate
    return decorate(fn)


def collect(fn: _F) -> _F:
    """Mark a ``Component`` method as a statistics-collection method.

    The method takes ``(self, env)`` and runs once at the end of each
    trial, before the model-level ``@model.collect`` callback, typically
    assigning the component's declared Output fields."""
    if getattr(fn, _COMPONENT_PROCESS_ATTR, None) is not None:
        raise ValueError(
            f"'{fn.__qualname__}' cannot be both a component process "
            "and a component collect method")
    setattr(fn, _COMPONENT_COLLECT_ATTR, True)
    return fn


@dataclass(frozen=True)
class _ComponentDecl:
    name: str
    cls: type[Component]
    template: Component
    decls: dict[str, Any]
    field_map: dict[str, str]
    local_name: str = ""
    instances: tuple[Component, ...] = ()
    process_names: tuple[str, ...] = ()
    display_name: str = ""
    item_display_name: str = ""
    direct_field_map: dict[str, str] = field(default_factory=dict)
    constants: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    pqueue_counts: dict[str, tuple[int, ...]] = field(default_factory=dict)
    pqueue_offsets: dict[str, tuple[int, ...]] = field(default_factory=dict)
    process_counts: dict[str, tuple[int, ...]] = field(default_factory=dict)
    process_offsets: dict[str, tuple[int, ...]] = field(default_factory=dict)
    aliased_fields: tuple[str, ...] = ()
    children: tuple["_AnyComponentDecl", ...] = ()


@dataclass(frozen=True)
class _ComponentCollectionDecl:
    name: str
    cls: type[Component]
    templates: tuple[Component, ...]
    decls: dict[str, Any]
    field_map: dict[str, str]
    length: int
    constants: dict[str, tuple[Any, ...]]
    pqueue_counts: dict[str, tuple[int, ...]]
    pqueue_offsets: dict[str, tuple[int, ...]]
    process_counts: dict[str, tuple[int, ...]]
    process_offsets: dict[str, tuple[int, ...]]
    local_name: str = ""
    process_names: tuple[str, ...] = ()
    display_name: str = ""
    item_display_name: str = ""
    direct_field_map: dict[str, str] = field(default_factory=dict)
    aliased_fields: tuple[str, ...] = ()
    parent_offsets: tuple[int, ...] = ()
    parent_lengths: tuple[int, ...] = ()
    children: tuple["_AnyComponentDecl", ...] = ()


_AnyComponentDecl = _ComponentDecl | _ComponentCollectionDecl


def _component_declarations(cls: type[Component]) -> dict[str, Any]:
    decls = _field_declarations(cls, allow_symbolic_pqueues=True)
    for kind in ("predicate", "event"):
        if decls[kind]:
            raise ValueError(
                f"component '{cls.__name__}' declares {kind} fields, which "
                "are not supported yet")
    return decls


def _component_collection_class(hint: Any) -> type[Component] | None:
    origin = get_origin(hint)
    args = get_args(hint)
    if origin is list and len(args) == 1 and _is_component_class(args[0]):
        return args[0]
    if (isinstance(hint, list) and len(hint) == 1
            and _is_component_class(hint[0])):
        return hint[0]
    return None


def _component_field_map(name: str, decls: dict[str, Any]) -> dict[str, str]:
    fields: set[str] = set()
    for kind in ("param", "output", "state", "fstate", "resource",
                 "dataset", "condition", "processes", "spawnable", "trace"):
        fields.update(decls[kind])
    for kind in ("queue", "pool", "store", "pqueues"):
        fields.update(decls[kind])
    return {field: f"{name}__{field}" for field in fields}


def _component_constants(
    items: Sequence[Component],
    field_map: Mapping[str, str],
) -> dict[str, tuple[Any, ...]]:
    constants: dict[str, tuple[Any, ...]] = {}
    names = {
        name
        for item in items
        for name in vars(item)
        if not name.startswith("_") and name not in field_map
    }
    for name in names:
        values = tuple(getattr(item, name, _MISSING) for item in items)
        if all(value is not _MISSING and _primitive_constant(value)
               for value in values):
            constants[name] = values
    return constants


def _resolve_component_pqueues(
    component_name: str,
    instance_count: int,
    decls: Mapping[str, Any],
    constants: Mapping[str, tuple[Any, ...]],
) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
    counts_by_field: dict[str, tuple[int, ...]] = {}
    offsets_by_field: dict[str, tuple[int, ...]] = {}
    for field, count_decl in decls["pqueues"].items():
        if isinstance(count_decl, int):
            counts = (count_decl,) * instance_count
        else:
            values = constants.get(count_decl)
            if values is None:
                raise ValueError(
                    f"component '{component_name}' field "
                    f"'{field}' uses PQueues count '{count_decl}', which "
                    "must name an int constant on every item")
            if not all(type(value) is int and value >= 1 for value in values):
                raise ValueError(
                    f"component '{component_name}' field "
                    f"'{field}' uses PQueues count '{count_decl}', which "
                    "must be a positive int on every item")
            counts = values
        offsets: list[int] = []
        total = 0
        for count in counts:
            offsets.append(total)
            total += int(count)
        counts_by_field[field] = tuple(int(count) for count in counts)
        offsets_by_field[field] = tuple(offsets)
    return counts_by_field, offsets_by_field


def _offsets_from_counts(counts: Iterable[int]) -> tuple[tuple[int, ...],
                                                         tuple[int, ...]]:
    counts_tuple = tuple(int(count) for count in counts)
    offsets: list[int] = []
    total = 0
    for count in counts_tuple:
        offsets.append(total)
        total += count
    return counts_tuple, tuple(offsets)


def _resolve_component_processes(
    component_name: str,
    cls: type[Component],
    templates: Sequence[Component],
    decls: Mapping[str, Any],
) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
    methods = {
        name: spec
        for name, _method, spec in _component_process_methods(cls)
    }
    counts_by_field: dict[str, tuple[int, ...]] = {}
    offsets_by_field: dict[str, tuple[int, ...]] = {}
    for field in decls["processes"]:
        spec = methods.get(field)
        if spec is None:
            raise ValueError(
                f"component '{component_name}' Processes field '{field}' "
                "must have a same-named @sim.process method")
        counts = [
            _resolve_component_process_copies(
                component_name, template, field, spec)
            for template in templates
        ]
        counts_tuple, offsets = _offsets_from_counts(counts)
        counts_by_field[field] = counts_tuple
        offsets_by_field[field] = offsets
    return counts_by_field, offsets_by_field


def _rewrite_component_capacity(
    component_name: str,
    field_name: str,
    cap: int | str | None,
    decls: dict[str, Any],
    field_map: dict[str, str],
) -> int | str | None:
    if not isinstance(cap, str):
        return cap
    if cap in decls["param"]:
        return field_map[cap]
    if cap in field_map:
        raise ValueError(
            f"component '{component_name}' field '{field_name}' capacity "
            f"'{cap}' must name a Param field")
    return cap


def _validate_component_instance_declarations(
    component_name: str,
    decls: dict[str, Any],
    *,
    instance_count: int,
    collection: bool,
) -> None:
    for kind in ("predicate", "event"):
        if decls[kind]:
            raise ValueError(
                f"component '{component_name}' declares {kind} fields, which "
                "are not supported yet")


def _flatten_component_declarations(
    target: dict[str, Any],
    component_name: str,
    decls: dict[str, Any],
    field_map: dict[str, str],
    instance_count: int,
    pqueue_counts: Mapping[str, tuple[int, ...]],
    process_counts: Mapping[str, tuple[int, ...]],
    aliased: frozenset[str] = frozenset(),
) -> None:
    for name, flat_name in field_map.items():
        if name in aliased:
            continue
        if _declarations_contain(target, flat_name):
            raise ValueError(f"duplicate field name '{flat_name}'")
    target["param"].extend(field_map[name] for name in decls["param"])
    target["trace"].extend(field_map[name] for name in decls["trace"])
    if instance_count > 1:
        for name in decls["param"]:
            target["field_shapes"][field_map[name]] = (instance_count,)
        for name in decls["trace"]:
            target["field_shapes"][field_map[name]] = (instance_count,)
    for kind in ("output", "state", "fstate", "resource", "dataset",
                 "condition", "spawnable"):
        target[kind].extend(field_map[name] for name in decls[kind]
                            if name not in aliased)
        if instance_count > 1:
            for name in decls[kind]:
                target["field_shapes"][field_map[name]] = (instance_count,)
    for kind in ("queue", "pool", "store"):
        for name, cap in decls[kind].items():
            if name in aliased:
                continue
            target[kind][field_map[name]] = _rewrite_component_capacity(
                component_name, name, cap, decls, field_map)
            if instance_count > 1:
                target["field_shapes"][field_map[name]] = (instance_count,)
    for name, counts in pqueue_counts.items():
        target["pqueues"][field_map[name]] = sum(counts)
    for name, counts in process_counts.items():
        flat_name = field_map[name]
        target["processes"].append(flat_name)
        target["field_shapes"][flat_name] = (sum(counts),)


def _component_child_default(
    component: Component,
    field_name: str,
    child_cls: type[Component],
    component_name: str,
) -> Component:
    child = getattr(component, field_name, _MISSING)
    if child is _MISSING:
        raise ValueError(
            f"component field '{component_name}.{field_name}' needs a "
            f"{child_cls.__name__} instance default")
    if not isinstance(child, child_cls):
        raise TypeError(
            f"component field '{component_name}.{field_name}' default must "
            f"be a {child_cls.__name__} instance")
    return child


def _component_collection_default(
    component: Component,
    field_name: str,
    child_cls: type[Component],
    component_name: str,
) -> tuple[Component, ...]:
    value = getattr(component, field_name, _MISSING)
    if value is _MISSING:
        raise ValueError(
            f"component collection '{component_name}.{field_name}' needs a "
            f"non-empty list or tuple of {child_cls.__name__} instances")
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(
            f"component collection '{component_name}.{field_name}' needs a "
            f"non-empty list or tuple of {child_cls.__name__} instances")
    templates = tuple(value)
    for item in templates:
        if not isinstance(item, child_cls):
            raise TypeError(
                f"component collection '{component_name}.{field_name}' "
                f"items must be {child_cls.__name__} instances")
    return templates


def _component_children_by_name(
    decl: _AnyComponentDecl,
) -> dict[str, _AnyComponentDecl]:
    return {child.local_name: child for child in decl.children}


def _walk_component_declarations(
    decls: Iterable[_AnyComponentDecl],
) -> Iterable[_AnyComponentDecl]:
    for decl in decls:
        yield decl
        yield from _walk_component_declarations(decl.children)


def _build_component_declaration(
    *,
    local_name: str,
    name: str,
    cls: type[Component],
    templates: tuple[Component, ...],
    process_names: tuple[str, ...],
    display_name: str,
    item_display_name: str,
    target: dict[str, Any],
    collection: bool,
    wiring_registry: dict[int, Any],
    parent_offsets: tuple[int, ...] = (),
    parent_lengths: tuple[int, ...] = (),
) -> _AnyComponentDecl:
    decls = _component_declarations(cls)
    direct_field_map = _component_field_map(name, decls)
    instance_count = len(templates)
    _validate_component_instance_declarations(
        name, decls, instance_count=instance_count, collection=collection)
    wiring = _component_field_wiring(name, templates, decls)
    _register_wiring_targets(wiring_registry, templates, direct_field_map)
    for fname, ref in wiring.items():
        direct_field_map[fname] = _resolve_wiring_target(
            name, fname, ref, wiring_registry)
    constants = _component_constants(templates, direct_field_map)
    pqueue_counts, pqueue_offsets = _resolve_component_pqueues(
        name, instance_count, decls, constants)
    process_counts, process_offsets = _resolve_component_processes(
        name, cls, templates, decls)
    _flatten_component_declarations(
        target, name, decls, direct_field_map, instance_count, pqueue_counts,
        process_counts, aliased=frozenset(wiring))

    children: list[_AnyComponentDecl] = []
    for fname, hint in get_type_hints(cls).items():
        if _is_component_class(hint):
            child_templates = tuple(
                _component_child_default(template, fname, hint, name)
                for template in templates
            )
            child_process_names = tuple(
                f"{process_name}__{fname}" for process_name in process_names
            )
            children.append(
                _build_component_declaration(
                    local_name=fname,
                    name=f"{name}__{fname}",
                    cls=hint,
                    templates=child_templates,
                    process_names=child_process_names,
                    display_name=f"{item_display_name}.{fname}",
                    item_display_name=f"{item_display_name}.{fname}",
                    target=target,
                    collection=False,
                    wiring_registry=wiring_registry,
                )
            )
            continue

        collection_cls = _component_collection_class(hint)
        if collection_cls is None:
            continue
        child_templates_list: list[Component] = []
        child_process_names: list[str] = []
        offsets: list[int] = []
        lengths: list[int] = []
        for parent_index, template in enumerate(templates):
            items = _component_collection_default(
                template, fname, collection_cls, name)
            offsets.append(len(child_templates_list))
            lengths.append(len(items))
            child_templates_list.extend(items)
            parent_process_name = process_names[parent_index]
            child_process_names.extend(
                f"{parent_process_name}__{fname}__{index}"
                for index in range(len(items))
            )
        child_display = f"{item_display_name}.{fname}"
        children.append(
            _build_component_declaration(
                local_name=fname,
                name=f"{name}__{fname}",
                cls=collection_cls,
                templates=tuple(child_templates_list),
                process_names=tuple(child_process_names),
                display_name=child_display,
                item_display_name=f"{child_display}[]",
                target=target,
                collection=True,
                wiring_registry=wiring_registry,
                parent_offsets=tuple(offsets),
                parent_lengths=tuple(lengths),
            )
        )

    field_map = dict(direct_field_map)
    for child in children:
        for field, flat_name in child.field_map.items():
            field_map[f"{child.local_name}__{field}"] = flat_name

    if collection:
        return _ComponentCollectionDecl(
            name=name,
            cls=cls,
            templates=templates,
            decls=decls,
            field_map=field_map,
            length=instance_count,
            constants=constants,
            pqueue_counts=pqueue_counts,
            pqueue_offsets=pqueue_offsets,
            process_counts=process_counts,
            process_offsets=process_offsets,
            local_name=local_name,
            process_names=process_names,
            display_name=display_name,
            item_display_name=item_display_name,
            direct_field_map=direct_field_map,
            aliased_fields=tuple(wiring),
            parent_offsets=parent_offsets,
            parent_lengths=parent_lengths,
            children=tuple(children),
        )
    return _ComponentDecl(
        name=name,
        cls=cls,
        template=templates[0],
        decls=decls,
        field_map=field_map,
        local_name=local_name,
        instances=templates,
        process_names=process_names,
        display_name=display_name,
        item_display_name=item_display_name,
        direct_field_map=direct_field_map,
        aliased_fields=tuple(wiring),
        constants=constants,
        pqueue_counts=pqueue_counts,
        pqueue_offsets=pqueue_offsets,
        process_counts=process_counts,
        process_offsets=process_offsets,
        children=tuple(children),
    )


def _class_declarations(cls: type) -> dict[str, Any]:
    """Collect env field declarations from a Model subclass's annotations,
    in declaration order (base classes first)."""
    decls = _field_declarations(cls)
    wiring_registry: dict[int, Any] = {}
    for fname, hint in get_type_hints(cls).items():
        if _is_component_class(hint):
            template = getattr(cls, fname, _MISSING)
            if template is _MISSING:
                raise ValueError(
                    f"component field '{fname}' needs a {hint.__name__} "
                    "instance default")
            if not isinstance(template, hint):
                raise TypeError(
                    f"component field '{fname}' default must be a "
                    f"{hint.__name__} instance")
            component_decl = _build_component_declaration(
                local_name=fname,
                name=fname,
                cls=hint,
                templates=(template,),
                process_names=(fname,),
                display_name=fname,
                item_display_name=fname,
                target=decls,
                collection=False,
                wiring_registry=wiring_registry,
            )
            decls["components"].append(component_decl)
            continue

        collection_cls = _component_collection_class(hint)
        if collection_cls is None:
            continue
        default = getattr(cls, fname, _MISSING)
        if default is _MISSING:
            raise ValueError(
                f"component collection '{fname}' needs a non-empty "
                f"list or tuple of {collection_cls.__name__} instances")
        if (not isinstance(default, (list, tuple)) or not default):
            raise ValueError(
                f"component collection '{fname}' needs a non-empty "
                f"list or tuple of {collection_cls.__name__} instances")
        templates = tuple(default)
        for item in templates:
            if not isinstance(item, collection_cls):
                raise TypeError(
                    f"component collection '{fname}' items must be "
                    f"{collection_cls.__name__} instances")
        component_decl = _build_component_declaration(
            local_name=fname,
            name=fname,
            cls=collection_cls,
            templates=templates,
            process_names=tuple(f"{fname}__{index}"
                                for index in range(len(templates))),
            display_name=fname,
            item_display_name=f"{fname}[]",
            target=decls,
            collection=True,
            wiring_registry=wiring_registry,
            parent_offsets=(0,),
            parent_lengths=(len(templates),),
        )
        decls["component_collections"].append(component_decl)
    return decls


def _component_process_methods(
    cls: type[Component],
) -> list[tuple[str, Callable[..., Any], _ComponentProcessSpec]]:
    methods: dict[str, tuple[Callable[..., Any], _ComponentProcessSpec]] = {}
    for base in reversed(cls.__mro__):
        if base in (object, Component):
            continue
        for name, value in vars(base).items():
            spec = getattr(value, _COMPONENT_PROCESS_ATTR, None)
            if spec is None:
                methods.pop(name, None)
                continue
            if not callable(value):
                raise TypeError(
                    f"component process '{cls.__name__}.{name}' is not "
                    "callable")
            methods[name] = (value, spec)
    return [(name, fn, spec) for name, (fn, spec) in methods.items()]


def _component_collect_methods(
    cls: type[Component],
) -> list[tuple[str, Callable[..., Any]]]:
    methods: dict[str, Callable[..., Any]] = {}
    for base in reversed(cls.__mro__):
        if base in (object, Component):
            continue
        for name, value in vars(base).items():
            if not getattr(value, _COMPONENT_COLLECT_ATTR, False):
                methods.pop(name, None)
                continue
            if not callable(value):
                raise TypeError(
                    f"component collect '{cls.__name__}.{name}' is not "
                    "callable")
            methods[name] = value
    return list(methods.items())


def _primitive_constant(value: Any) -> bool:
    return type(value) in (bool, int, float)


def _resolve_component_process_copies(
    component_name: str,
    component: Component,
    method_name: str,
    spec: _ComponentProcessSpec,
) -> int:
    copies = spec.copies
    if isinstance(copies, int):
        return copies
    value = getattr(component, copies, _MISSING)
    if type(value) is not int or value < 1:
        raise ValueError(
            f"component process '{component_name}.{method_name}' copies "
            f"constant '{copies}' must be a positive int")
    return value


def _collection_const_symbol(collection: str, name: str) -> str:
    return f"_CIMBA_CONST_{collection}__{name}"


def _collection_pqueue_offsets_symbol(collection: str, field: str) -> str:
    return f"_CIMBA_PQOFF_{collection}__{field}"


def _collection_process_offsets_symbol(collection: str, field: str) -> str:
    return f"_CIMBA_PROCOFF_{collection}__{field}"


def _component_offsets_symbol(component: str) -> str:
    return f"_CIMBA_OFF_{component}"


def _env_attr(env_name: str, field: str, ctx: ast.expr_context) -> ast.Attribute:
    return ast.Attribute(
        value=ast.Name(id=env_name, ctx=ast.Load()),
        attr=field,
        ctx=ctx,
    )


def _subscript(
    value: ast.expr,
    index: ast.expr,
    ctx: ast.expr_context,
) -> ast.Subscript:
    value.ctx = ast.Load()
    return ast.Subscript(value=value, slice=index, ctx=ctx)


def _add(left: ast.expr, right: ast.expr) -> ast.expr:
    if (isinstance(left, ast.Constant) and type(left.value) is int
            and isinstance(right, ast.Constant) and type(right.value) is int):
        return ast.Constant(left.value + right.value)
    return ast.BinOp(left=left, op=ast.Add(), right=right)


def _component_instance_count(decl: _AnyComponentDecl) -> int:
    if isinstance(decl, _ComponentCollectionDecl):
        return len(decl.templates)
    return len(decl.instances)


def _component_spawnable_field_names(
    roots: Iterable[_AnyComponentDecl],
) -> set[str]:
    fields: set[str] = set()
    for decl in _walk_component_declarations(roots):
        fields.update(
            decl.direct_field_map[name]
            for name in decl.decls["spawnable"]
        )
    return fields


def _spawnable_slot_label(field: str, index: int | None) -> str:
    return field if index is None else f"{field}[{index}]"


@dataclass(frozen=True)
class _ComponentAccess:
    decl: _AnyComponentDecl
    index: ast.expr | None
    text: str


@dataclass(frozen=True)
class _ComponentFieldAccess:
    decl: _AnyComponentDecl
    index: ast.expr | None
    field: str
    text: str


class _ComponentPathLowerer(ast.NodeTransformer):
    def __init__(self, *, env_name: str):
        self.env_name = env_name

    def _root_namespace_ref(self, node: ast.AST) -> _ComponentAccess | None:
        return None

    def _root_collection_ref(self, node: ast.AST) -> _ComponentAccess | None:
        return None

    def _namespace_ref(self, node: ast.AST) -> _ComponentAccess | None:
        root = self._root_namespace_ref(node)
        if root is not None:
            return root

        if isinstance(node, ast.Subscript):
            collection = self._collection_ref(node.value)
            if collection is None:
                return None
            index = self._collection_item_index(
                collection.decl, collection.index, node.slice)
            return _ComponentAccess(
                collection.decl, index, f"{collection.text}[...]")

        if isinstance(node, ast.Attribute):
            parent = self._namespace_ref(node.value)
            if parent is None:
                return None
            child = _component_children_by_name(parent.decl).get(node.attr)
            if child is None or isinstance(child, _ComponentCollectionDecl):
                return None
            index = (parent.index
                     if _component_instance_count(child) > 1 else None)
            return _ComponentAccess(
                child, index, f"{parent.text}.{node.attr}")

        return None

    def _collection_ref(self, node: ast.AST) -> _ComponentAccess | None:
        root = self._root_collection_ref(node)
        if root is not None:
            return root

        if isinstance(node, ast.Attribute):
            parent = self._namespace_ref(node.value)
            if parent is None:
                return None
            child = _component_children_by_name(parent.decl).get(node.attr)
            if not isinstance(child, _ComponentCollectionDecl):
                return None
            return _ComponentAccess(
                child, parent.index, f"{parent.text}.{node.attr}")

        return None

    def _field_ref(self, node: ast.AST) -> _ComponentFieldAccess | None:
        if not isinstance(node, ast.Attribute):
            return None
        namespace = self._namespace_ref(node.value)
        if namespace is None:
            return None
        field_name = node.attr
        if field_name in namespace.decl.direct_field_map:
            return _ComponentFieldAccess(
                namespace.decl,
                namespace.index,
                field_name,
                f"{namespace.text}.{field_name}",
            )
        if field_name in namespace.decl.constants:
            return _ComponentFieldAccess(
                namespace.decl,
                namespace.index,
                field_name,
                f"{namespace.text}.{field_name}",
            )
        if field_name in _component_children_by_name(namespace.decl):
            return None
        self._raise_unknown_field(namespace, field_name)

    def _collection_item_index(
        self,
        decl: _ComponentCollectionDecl,
        parent_index: ast.expr | None,
        item_index: ast.expr,
    ) -> ast.expr:
        index = self.visit(copy.deepcopy(item_index))
        if not isinstance(index, ast.expr):
            raise TypeError("component collection index did not lower "
                            "to an expression")
        if len(decl.parent_offsets) <= 1:
            offset_value = decl.parent_offsets[0] if decl.parent_offsets else 0
            if offset_value == 0:
                return index
            return _add(ast.Constant(offset_value), index)
        if parent_index is None:
            raise TypeError("nested component collection has no parent index")
        if (isinstance(parent_index, ast.Constant)
                and type(parent_index.value) is int):
            offset_value = decl.parent_offsets[parent_index.value]
            if offset_value == 0:
                return index
            return _add(ast.Constant(offset_value), index)
        offset = _subscript(
            ast.Name(id=_component_offsets_symbol(decl.name), ctx=ast.Load()),
            parent_index,
            ast.Load(),
        )
        return _add(offset, index)

    def _field_target(
        self,
        access: _ComponentFieldAccess,
        ctx: ast.expr_context,
    ) -> ast.expr:
        flat_name = access.decl.direct_field_map[access.field]
        target = _env_attr(self.env_name, flat_name, ctx)
        if _component_instance_count(access.decl) <= 1:
            return target
        if access.index is None:
            raise TypeError("component field has no instance index")
        return _subscript(target, access.index, ctx)

    def _constant_expr(self, access: _ComponentFieldAccess) -> ast.expr:
        values = access.decl.constants[access.field]
        if len(values) == 1:
            return ast.Constant(values[0])
        if (isinstance(access.index, ast.Constant)
                and type(access.index.value) is int):
            return ast.Constant(values[access.index.value])
        if access.index is None:
            raise TypeError("component constant has no instance index")
        return _subscript(
            ast.Name(id=_collection_const_symbol(access.decl.name,
                                                 access.field),
                     ctx=ast.Load()),
            access.index,
            ast.Load(),
        )

    def _pqueue_offset_expr(self, access: _ComponentFieldAccess) -> ast.expr:
        offsets = access.decl.pqueue_offsets[access.field]
        if len(offsets) == 1:
            return ast.Constant(offsets[0])
        if (isinstance(access.index, ast.Constant)
                and type(access.index.value) is int):
            return ast.Constant(offsets[access.index.value])
        if access.index is None:
            raise TypeError("component PQueues field has no instance index")
        return _subscript(
            ast.Name(id=_collection_pqueue_offsets_symbol(access.decl.name,
                                                          access.field),
                     ctx=ast.Load()),
            access.index,
            ast.Load(),
        )

    def _process_offset_expr(self, access: _ComponentFieldAccess) -> ast.expr:
        offsets = access.decl.process_offsets[access.field]
        if len(offsets) == 1:
            return ast.Constant(offsets[0])
        if (isinstance(access.index, ast.Constant)
                and type(access.index.value) is int):
            return ast.Constant(offsets[access.index.value])
        if access.index is None:
            raise TypeError("component Processes field has no instance index")
        return _subscript(
            ast.Name(id=_collection_process_offsets_symbol(access.decl.name,
                                                           access.field),
                     ctx=ast.Load()),
            access.index,
            ast.Load(),
        )

    def _raise_unknown_field(
        self,
        namespace: _ComponentAccess,
        field_name: str,
    ) -> None:
        kind = ("component collection field"
                if isinstance(namespace.decl, _ComponentCollectionDecl)
                else "component field")
        raise ValueError(
            f"{self._callback_label()} references unknown {kind} "
            f"{namespace.text}.{field_name}")

    def _callback_label(self) -> str:
        raise NotImplementedError

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if (isinstance(node.func, ast.Name) and node.func.id == "getattr"
                and node.args):
            target = (self._namespace_ref(node.args[0])
                      or self._collection_ref(node.args[0]))
            if target is not None:
                raise ValueError(
                    f"{self._callback_label()} uses dynamic "
                    f"getattr({target.text}, ...), which is not supported")
        if isinstance(node.func, ast.Attribute):
            target = (self._namespace_ref(node.func.value)
                      or self._collection_ref(node.func.value))
            if target is not None:
                raise ValueError(
                    f"{self._callback_label()} cannot call "
                    f"{target.text}.{node.func.attr}() inside compiled code")
        return self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        access = self._field_ref(node.value)
        if access is not None and access.field in access.decl.decls["pqueues"]:
            queue_index = self.visit(copy.deepcopy(node.slice))
            if not isinstance(queue_index, ast.expr):
                raise TypeError("component PQueues index did not lower "
                                "to an expression")
            offset = self._pqueue_offset_expr(access)
            if isinstance(offset, ast.Constant) and offset.value == 0:
                index = queue_index
            else:
                index = _add(offset, queue_index)
            flat = _env_attr(
                self.env_name,
                access.decl.direct_field_map[access.field],
                ast.Load(),
            )
            return ast.copy_location(_subscript(flat, index, node.ctx), node)

        if access is not None and access.field in access.decl.decls["processes"]:
            process_index = self.visit(copy.deepcopy(node.slice))
            if not isinstance(process_index, ast.expr):
                raise TypeError("component Processes index did not lower "
                                "to an expression")
            offset = self._process_offset_expr(access)
            if isinstance(offset, ast.Constant) and offset.value == 0:
                index = process_index
            else:
                index = _add(offset, process_index)
            flat = _env_attr(
                self.env_name,
                access.decl.direct_field_map[access.field],
                ast.Load(),
            )
            return ast.copy_location(_subscript(flat, index, node.ctx), node)

        collection = self._collection_ref(node.value)
        if collection is not None:
            raise ValueError(
                f"{self._callback_label()} uses {collection.text}[...] "
                "directly; access one of its fields")
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        nested = self._field_ref(node.value)
        if nested is not None:
            raise ValueError(
                f"{self._callback_label()} cannot access attributes below "
                f"component field {nested.text}")

        access = self._field_ref(node)
        if access is not None:
            if (access.field in access.decl.decls["pqueues"]
                    or access.field in access.decl.decls["processes"]):
                raise ValueError(
                    f"{self._callback_label()} must index {access.text} "
                    "before using it")
            if access.field in access.decl.constants:
                if not isinstance(node.ctx, ast.Load):
                    raise ValueError(
                        f"{self._callback_label()} cannot assign to "
                        f"constant {access.text}")
                return ast.copy_location(self._constant_expr(access), node)
            return ast.copy_location(self._field_target(access, node.ctx),
                                     node)

        namespace = self._namespace_ref(node)
        if namespace is not None:
            raise ValueError(
                f"{self._callback_label()} cannot use {namespace.text} "
                "directly; access one of its fields")
        collection = self._collection_ref(node)
        if collection is not None:
            raise ValueError(
                f"{self._callback_label()} cannot use {collection.text} "
                "directly; index it and access one of its fields")
        return self.generic_visit(node)


class _ComponentMethodLowerer(_ComponentPathLowerer):
    def __init__(
        self,
        *,
        component_name: str,
        receiver_name: str,
        env_name: str,
        component_decl: _AnyComponentDecl,
        instance_index: int,
        kind: str = "process",
    ):
        super().__init__(env_name=env_name)
        self.component_name = component_name
        self.receiver_name = receiver_name
        self.component_decl = component_decl
        self.instance_index = ast.Constant(instance_index)
        self.kind = kind

    def _root_namespace_ref(self, node: ast.AST) -> _ComponentAccess | None:
        if isinstance(node, ast.Name) and node.id == self.receiver_name:
            index = (self.instance_index
                     if _component_instance_count(self.component_decl) > 1
                     else None)
            return _ComponentAccess(self.component_decl, index,
                                    self.receiver_name)
        return None

    def _callback_label(self) -> str:
        return f"component '{self.component_name}' {self.kind}"

    def _raise_unknown_field(
        self,
        namespace: _ComponentAccess,
        field_name: str,
    ) -> None:
        raise ValueError(
            f"component '{self.component_name}' {self.kind} references "
            f"unsupported {namespace.text}.{field_name}")

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == self.receiver_name:
            raise ValueError(
                f"component '{self.component_name}' {self.kind} cannot use "
                "self directly inside compiled code")
        return node


def _closure_namespace(fn: Callable[..., Any]) -> dict[str, Any]:
    namespace = dict(fn.__globals__)
    if fn.__closure__ is not None:
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
            namespace[name] = cell.cell_contents
    return namespace


def _component_method_source(fn: Callable[..., Any],
                             kind: str = "process") -> ast.FunctionDef:
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError) as exc:
        raise ValueError(
            f"component {kind} '{fn.__qualname__}' needs inspectable source"
        ) from exc
    tree = ast.parse(textwrap.dedent(source))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node
    raise ValueError(f"component {kind} '{fn.__qualname__}' source does not "
                     "contain a function definition")


def _lower_component_process(
    component_name: str,
    component: Component,
    component_decl: _AnyComponentDecl,
    instance_index: int,
    method_name: str,
    method: Callable[..., Any],
    is_struct_class: Callable[[Any], bool],
) -> Callable[..., Any]:
    node = copy.deepcopy(_component_method_source(method))
    args = node.args
    if (args.posonlyargs or args.vararg or args.kwonlyargs or args.kwarg
            or args.defaults or args.kw_defaults):
        raise ValueError(
            f"component process '{component_name}.{method_name}' must take "
            "(self, env), (self, env, idx), and optionally a final "
            "sim.Struct view parameter, without defaults")

    params = args.args
    hints = get_type_hints(method)
    own = hints.get(params[-1].arg) if len(params) > 2 else None
    injected_struct = own if is_struct_class(own) else None
    injected = injected_struct is not None
    for arg in params[2:len(params) - 1 if injected else len(params)]:
        hint = hints.get(arg.arg)
        if is_struct_class(hint):
            raise ValueError(
                f"component process '{component_name}.{method_name}': the "
                f"{hint.__name__} view must be the last parameter")

    base_arg_count = len(params) - (1 if injected else 0)
    if base_arg_count not in (2, 3):
        raise ValueError(
            f"component process '{component_name}.{method_name}' must take "
            "(self, env), (self, env, idx), and optionally a final "
            "view parameter annotated with a sim.Struct subclass")

    receiver_name = args.args[0].arg
    env_name = args.args[1].arg
    process_name = f"{component_name}__{method_name}"
    node.name = process_name
    node.decorator_list = []
    node.returns = None
    node.type_comment = None
    args.args = args.args[1:]
    for index, arg in enumerate(args.args):
        if injected and index == len(args.args) - 1:
            arg.annotation = ast.Name(id="_CIMBA_STRUCT_VIEW",
                                      ctx=ast.Load())
        else:
            arg.annotation = None
        arg.type_comment = None

    lowerer = _ComponentMethodLowerer(
        component_name=component_name,
        receiver_name=receiver_name,
        env_name=env_name,
        component_decl=component_decl,
        instance_index=instance_index,
    )
    lowered = lowerer.visit(node)
    extra = ({"_CIMBA_STRUCT_VIEW": injected_struct}
             if injected_struct is not None else None)
    return _exec_lowered_method(
        lowered, kind="process", component_name=component_name,
        method_name=method_name, fn_name=process_name, method=method,
        component_decl=component_decl, extra_namespace=extra)


def _exec_lowered_method(
    lowered: ast.AST,
    *,
    kind: str,
    component_name: str,
    method_name: str,
    fn_name: str,
    method: Callable[..., Any],
    component_decl: _AnyComponentDecl,
    extra_namespace: Mapping[str, Any] | None = None,
) -> Callable[..., Any]:
    if not isinstance(lowered, ast.FunctionDef):
        raise TypeError(f"component {kind} lowering produced a non-function")
    module = ast.Module(body=[lowered], type_ignores=[])
    ast.fix_missing_locations(module)
    source = ast.unparse(module) + "\n"

    filename = f"<cimba component '{component_name}.{method_name}'>"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace = _closure_namespace(method)
    if extra_namespace:
        namespace.update(extra_namespace)
    namespace.update(_component_collection_namespace((component_decl,)))
    exec(compile(source, filename, "exec"), namespace)
    generated = namespace[fn_name]
    generated.__module__ = method.__module__
    generated.__qualname__ = fn_name
    generated.__cimba_source__ = source
    return generated


def _lower_component_collect(
    component_name: str,
    component_decl: _AnyComponentDecl,
    instance_index: int,
    method_name: str,
    method: Callable[..., Any],
) -> Callable[..., Any]:
    node = copy.deepcopy(_component_method_source(method, "collect"))
    args = node.args
    if (args.posonlyargs or args.vararg or args.kwonlyargs or args.kwarg
            or args.defaults or args.kw_defaults or len(args.args) != 2):
        raise ValueError(
            f"component collect '{component_name}.{method_name}' must take "
            "(self, env) without defaults")
    receiver_name = args.args[0].arg
    env_name = args.args[1].arg
    fn_name = f"{component_name}__{method_name}"
    node.name = fn_name
    node.decorator_list = []
    node.returns = None
    node.type_comment = None
    args.args = args.args[1:]
    for arg in args.args:
        arg.annotation = None
        arg.type_comment = None

    lowerer = _ComponentMethodLowerer(
        component_name=component_name,
        receiver_name=receiver_name,
        env_name=env_name,
        component_decl=component_decl,
        instance_index=instance_index,
        kind="collect",
    )
    return _exec_lowered_method(
        lowerer.visit(node), kind="collect", component_name=component_name,
        method_name=method_name, fn_name=fn_name, method=method,
        component_decl=component_decl)


class _ModelComponentRefLowerer(_ComponentPathLowerer):
    def __init__(self, *, model_name: str, fn_name: str, env_name: str,
                 component_roots: Mapping[str, _AnyComponentDecl]):
        super().__init__(env_name=env_name)
        self.model_name = model_name
        self.fn_name = fn_name
        self.component_roots = component_roots
        self.changed = False

    def _root_namespace_ref(self, node: ast.AST) -> _ComponentAccess | None:
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == self.env_name):
            decl = self.component_roots.get(node.attr)
            if decl is not None and not isinstance(decl,
                                                   _ComponentCollectionDecl):
                return _ComponentAccess(decl, None,
                                        f"{self.env_name}.{node.attr}")
        return None

    def _root_collection_ref(self, node: ast.AST) -> _ComponentAccess | None:
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == self.env_name):
            decl = self.component_roots.get(node.attr)
            if isinstance(decl, _ComponentCollectionDecl):
                return _ComponentAccess(decl, None,
                                        f"{self.env_name}.{node.attr}")
        return None

    def _callback_label(self) -> str:
        return f"model '{self.model_name}' callback '{self.fn_name}'"

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        lowered = super().visit_Subscript(node)
        if lowered is not node:
            self.changed = True
        return lowered

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        lowered = super().visit_Attribute(node)
        if lowered is not node:
            self.changed = True
        return lowered


def _component_collection_namespace(
    components: Iterable[_AnyComponentDecl],
) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    for decl in _walk_component_declarations(components):
        for name, values in decl.constants.items():
            if len(values) > 1:
                namespace[_collection_const_symbol(decl.name, name)] = \
                    np.asarray(values)
        for field, offsets in decl.pqueue_offsets.items():
            if len(offsets) > 1:
                namespace[
                    _collection_pqueue_offsets_symbol(decl.name, field)
                ] = np.asarray(offsets, dtype=np.int64)
        for field, offsets in decl.process_offsets.items():
            if len(offsets) > 1:
                namespace[
                    _collection_process_offsets_symbol(decl.name, field)
                ] = np.asarray(offsets, dtype=np.int64)
        if (isinstance(decl, _ComponentCollectionDecl)
                and len(decl.parent_offsets) > 1):
            namespace[_component_offsets_symbol(decl.name)] = np.asarray(
                decl.parent_offsets, dtype=np.int64)
    return namespace


def _function_source(fn: Callable[..., Any]) -> str:
    source = getattr(fn, "__cimba_source__", None)
    if source is None:
        source = inspect.getsource(fn)
    return textwrap.dedent(source)


def _function_def_from_source(fn: Callable[..., Any]) -> ast.FunctionDef:
    tree = ast.parse(_function_source(fn))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node
    raise ValueError(f"callback '{fn.__qualname__}' source does not contain "
                     "a function definition")


def _lower_model_component_refs(
    fn: Callable[..., Any],
    *,
    model_name: str,
    component_roots: Mapping[str, _AnyComponentDecl],
) -> Callable[..., Any]:
    if not component_roots:
        return fn
    component_names = set(component_roots)
    if not any(name in fn.__code__.co_names for name in component_names):
        return fn
    try:
        node = copy.deepcopy(_function_def_from_source(fn))
    except (OSError, TypeError) as exc:
        raise ValueError(
            f"model '{model_name}' callback '{fn.__qualname__}' needs "
            "inspectable source to use Component namespaces"
        ) from exc
    if not node.args.args:
        return fn

    env_name = node.args.args[0].arg
    lowerer = _ModelComponentRefLowerer(
        model_name=model_name,
        fn_name=fn.__name__,
        env_name=env_name,
        component_roots=component_roots,
    )
    lowered = lowerer.visit(node)
    if not isinstance(lowered, ast.FunctionDef):
        raise TypeError("model callback lowering produced a non-function")
    if not lowerer.changed:
        return fn

    lowered.decorator_list = []
    lowered.returns = None
    lowered.type_comment = None
    for arg in lowered.args.args:
        arg.annotation = None
        arg.type_comment = None
    module = ast.Module(body=[lowered], type_ignores=[])
    ast.fix_missing_locations(module)
    source = ast.unparse(module) + "\n"

    filename = f"<cimba model callback '{model_name}.{fn.__name__}'>"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace = _closure_namespace(fn)
    namespace.update(_component_collection_namespace(component_roots.values()))
    exec(compile(source, filename, "exec"), namespace)
    generated = namespace[fn.__name__]
    generated.__module__ = fn.__module__
    generated.__qualname__ = fn.__qualname__
    generated.__cimba_source__ = source
    return generated


_PROCESS_DAG_FIELD_KINDS = (
    "queue",
    "resource",
    "pool",
    "store",
    "condition",
    "state",
    "fstate",
    "pqueues",
    "event",
)


def _process_dag_component_process_members(
    decl: _AnyComponentDecl,
    process_names: set[str],
) -> list[str]:
    members: list[str] = []
    for component_name in decl.process_names:
        for method_name, _method, _spec in _component_process_methods(
                decl.cls):
            process_name = f"{component_name}__{method_name}"
            if process_name in process_names:
                members.append(f"process:{process_name}")
    return members


def _process_dag_component_field_members(
    decls: Mapping[str, Any],
    field_map: Mapping[str, str],
    entity_kinds: Mapping[str, str],
    aliased: tuple[str, ...] = (),
) -> list[str]:
    members: list[str] = []
    for kind in _PROCESS_DAG_FIELD_KINDS:
        fields = decls[kind]
        for field in fields:
            if field in aliased:
                # Wired fields name an entity that belongs to (and is
                # displayed in) the target component's block.
                continue
            flat_name = field_map[field]
            graph_kind = entity_kinds.get(flat_name)
            if graph_kind is not None:
                members.append(f"{graph_kind}:{flat_name}")
    return members
