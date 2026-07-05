"""Component declarations, model flattening, and AST lowering.

``sim.Component`` groups related fields and process methods so a model
can be assembled from reusable parts::

    class Station(sim.Component):
        queue: sim.Queue
        served: sim.State

        @sim.process
        def server(self, env):
            ...

    class Line(sim.Model):
        inlet: Station = Station()
        stations: list[Station] = [Station(), Station()]

The trial record compiled by ``Model`` is flat, so everything a
component declares is lowered before compilation:

* fields flatten to prefixed model fields (``inlet__queue``); the items
  of a collection share one shaped field (``stations__queue`` with one
  element per item), and nesting keeps prefixing
  (``zones__gates__queue``);
* ``@sim.process`` / ``@sim.collect`` methods are rewritten into plain
  functions over the flattened env -- ``self.queue`` becomes
  ``env.inlet__queue``. A collection's method compiles once (not once per
  item): ``self.queue`` lowers to ``env.stations__queue[__cimba_inst]``,
  where the instance index is recovered at runtime from the copy index
  (see ``_shared_instance_setup``);
* model callbacks that use component paths
  (``env.zones[i].gates[j].queue``) are rewritten the same way, with
  generated numpy tables backing dynamic item indices, per-item
  constants, and Ref/Refs dereferences.

The module is organized in five parts, in order: the authoring API
(``Component``, the ``@sim.process``/``@sim.collect`` method markers,
and the wiring/Ref metadata captured from instance defaults);
declaration metadata (``_ComponentDecl``, one per component tree node);
declaration building (``_class_declarations`` and ``_DeclBuilder``);
the AST lowerers; and the codegen helpers that compile the lowered
functions.
"""

import ast
import copy
import inspect
import linecache
import textwrap
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar, get_args, get_origin, get_type_hints, overload

import numpy as np
from numba import njit

from ._dataset.methods import (
    DATASET_METHOD_NAMES,
    dataset_lowering_namespace,
    lower_dataset_method_call,
    lower_env_dataset_method_calls,
)
from ._declarations import (_DECL_KINDS, _MISSING, _check_name,
                            _Declarations, _field_declarations, _FieldDecl)
from ._timeseries.methods import (
    HISTORY_GETTER_NAMES,
    lower_env_history_method_calls,
    lower_history_getter_call,
    lower_timeseries_method_call,
    timeseries_lowering_namespace,
)
from .random._lowering import (
    lower_random_calls_in_node,
    random_lowering_namespace,
)
from .store.methods import (
    ENTITY_METHOD_NAMES,
    entity_lowering_namespace,
    lower_entity_method_call,
    lower_env_entity_method_calls,
)

_F = TypeVar("_F", bound=Callable[..., Any])


# --- Authoring API ----------------------------------------------------------
#
# What model authors touch directly: the Component base class, the
# method markers, and the values captured from instance defaults.

_COMPONENT_PROCESS_ATTR = "__cimba_component_process__"
_COMPONENT_COLLECT_ATTR = "__cimba_component_collect__"

_wirable_fields_cache: dict[type, dict[str, str]] = {}

#: declared field kind -> ``_FieldKind.binding``, for the scalar entity
#: kinds whose ``self.<field>.history.method()`` sugar is lowered directly
#: inside component methods. Priority queues are indexed (``self.pq[i]``)
#: before ``.history`` can apply, so they are lowered by the later
#: env-based pass (``lower_env_history_method_calls``) instead.
_COMPONENT_HISTORY_BINDINGS = {
    "queue": "buffer",
    "resource": "resource",
    "pool": "resourcepool",
    "store": "objectqueue",
}

#: scalar entity field kinds whose ``self.<field>.method(...)`` sugar
#: (put/get/acquire/release/...) is lowered directly inside component
#: methods, same restriction as ``_COMPONENT_HISTORY_BINDINGS``: PQueues
#: elements are indexed (``self.pq[i]``) before a method applies, so they
#: are lowered by the later env-based pass (``lower_env_entity_method_calls``)
#: instead.
_COMPONENT_ENTITY_KINDS = frozenset(
    {"queue", "resource", "pool", "store", "condition"})


def _wirable_fields(cls: type) -> dict[str, str]:
    """Declared field name -> kind name, for the wirable entity kinds."""
    kinds = _wirable_fields_cache.get(cls)
    if kinds is None:
        kinds = {}
        for fname, hint in get_type_hints(cls).items():
            try:
                kind = _DECL_KINDS.get(hint)
            except TypeError:
                kind = None
            if kind is not None and kind.wirable:
                kinds[fname] = kind.name
        _wirable_fields_cache[cls] = kinds
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
        kind = _wirable_fields(type(self)).get(name)
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


def _is_component_class(obj: Any) -> bool:
    return (isinstance(obj, type) and issubclass(obj, Component)
            and obj is not Component)


def _collection_item_class(hint: Any) -> type[Component] | None:
    """The item class of a ``list[SomeComponent]`` annotation (also the
    ``[SomeComponent]`` literal shorthand), or None."""
    origin = get_origin(hint)
    args = get_args(hint)
    if origin is list and len(args) == 1 and _is_component_class(args[0]):
        return args[0]
    if (isinstance(hint, list) and len(hint) == 1
            and _is_component_class(hint[0])):
        return hint[0]
    return None


def _component_fields(cls: type) -> Iterator[tuple[str, type[Component], bool]]:
    """Yield ``(field_name, item_class, is_collection)`` for each
    component-typed annotation on a Model or Component class, in
    declaration order."""
    for fname, hint in get_type_hints(cls).items():
        if _is_component_class(hint):
            yield fname, hint, False
        else:
            item_cls = _collection_item_class(hint)
            if item_cls is not None:
                yield fname, item_cls, True


@dataclass(frozen=True)
class _ComponentProcessSpec:
    """The arguments of a ``@sim.process`` method marker."""

    copies: int | str = 1
    priority: int = 0

    def resolve_copies(self, component: Component, label: str) -> int:
        """The copy count for one instance: the literal int, or the value
        of the named per-instance int constant."""
        if isinstance(self.copies, int):
            return self.copies
        value = getattr(component, self.copies, _MISSING)
        if type(value) is not int or value < 1:
            raise ValueError(
                f"component process '{label}' copies "
                f"constant '{self.copies}' must be a positive int")
        return value


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


def _marked_methods(cls: type[Component], marker_attr: str,
                    kind: str) -> dict[str, tuple[Callable[..., Any], Any]]:
    """Methods carrying a marker attribute, walking the MRO base-first so
    an unmarked override drops the inherited registration."""
    methods: dict[str, tuple[Callable[..., Any], Any]] = {}
    for base in reversed(cls.__mro__):
        if base in (object, Component):
            continue
        for name, value in vars(base).items():
            marker = getattr(value, marker_attr, None)
            if marker is None or marker is False:
                methods.pop(name, None)
                continue
            if not callable(value):
                raise TypeError(
                    f"component {kind} '{cls.__name__}.{name}' is not "
                    "callable")
            methods[name] = (value, marker)
    return methods


def _component_process_methods(
    cls: type[Component],
) -> list[tuple[str, Callable[..., Any], _ComponentProcessSpec]]:
    return [(name, fn, spec) for name, (fn, spec)
            in _marked_methods(cls, _COMPONENT_PROCESS_ATTR, "process").items()]


def _component_collect_methods(
    cls: type[Component],
) -> list[tuple[str, Callable[..., Any]]]:
    return [(name, fn) for name, (fn, _marker)
            in _marked_methods(cls, _COMPONENT_COLLECT_ATTR, "collect").items()]


# --- Declaration metadata ---------------------------------------------------
#
# One _ComponentDecl per node of the model's component tree, built by
# _DeclBuilder below and consumed by Model and the lowerers.

@dataclass
class _ComponentRefDecl:
    """A Ref/Refs field: raw per-instance targets captured at declaration
    time, resolved to (target decl, item index) pairs once all component
    declarations exist (so forward references are allowed)."""

    name: str
    table: bool
    #: per template instance: the referenced Component or None (Ref only)
    raw: tuple["Component | None", ...] = field(default=(), repr=False)
    #: per template instance: tuple of referenced Components (Refs only)
    raw_tables: tuple[tuple["Component", ...], ...] = field(
        default=(), repr=False)
    #: per instance: (target decl, target item index | None), or None
    targets: tuple[Any, ...] = field(default=(), repr=False)
    #: single decl every table entry resolves into (Refs only)
    table_decl: Any = field(default=None, repr=False)
    #: flattened per-entry target item indices (Refs only)
    table_indices: tuple[int, ...] = ()
    table_lengths: tuple[int, ...] = ()
    table_offsets: tuple[int, ...] = ()


@dataclass
class _ComponentDecl:
    """One node of a model's component tree.

    A *collection* decl covers every item of a ``list[Component]`` field;
    a scalar decl covers one instance per parent instance -- so a scalar
    component nested under a collection still has several instances. All
    per-instance metadata (constants, counts, offsets) is indexed by the
    flattened instance position.

    ``direct_field_map`` and ``aliased_fields`` are finalized after the
    whole tree is built, when entity wiring is resolved (see
    ``_resolve_component_wiring``).
    """

    #: flattened field prefix, e.g. ``zones__gates``
    name: str
    cls: type[Component]
    collection: bool
    #: bound template instances, one per (parent instance x item)
    instances: tuple[Component, ...]
    #: this class's own field declarations
    decls: _Declarations
    #: field name on the Model subclass that declared this node's root
    local_name: str
    #: per-instance prefix for lowered process/collect function names
    process_names: tuple[str, ...]
    #: authoring-time path, e.g. ``zones[].gates``
    display_name: str
    #: authoring-time path of one item, e.g. ``zones[].gates[]``
    item_display_name: str
    #: own field -> flattened model field (wired fields -> target's field)
    direct_field_map: dict[str, str]
    #: per-instance primitive attribute values captured from the defaults
    constants: dict[str, tuple[Any, ...]]
    #: per-instance PQueues element counts / start offsets, by field
    pqueue_counts: dict[str, tuple[int, ...]]
    pqueue_offsets: dict[str, tuple[int, ...]]
    #: per-instance Processes copy counts / handle offsets, by field
    process_counts: dict[str, tuple[int, ...]]
    process_offsets: dict[str, tuple[int, ...]]
    #: Ref/Refs fields, by name
    component_refs: dict[str, _ComponentRefDecl]
    #: wirable fields overridden with a reference, before resolution
    wiring_raw: dict[str, "_FieldRef"] = field(default_factory=dict)
    #: fields wired to another component's entity (no own model field)
    aliased_fields: tuple[str, ...] = ()
    children: tuple["_ComponentDecl", ...] = ()
    #: for collections: first item index / item count per parent instance
    parent_offsets: tuple[int, ...] = ()
    parent_lengths: tuple[int, ...] = ()

    @property
    def count(self) -> int:
        return len(self.instances)

    def walk(self) -> Iterator["_ComponentDecl"]:
        """This node and all of its descendants, depth-first."""
        yield self
        for child in self.children:
            yield from child.walk()

    def child(self, local_name: str) -> "_ComponentDecl | None":
        for child in self.children:
            if child.local_name == local_name:
                return child
        return None

    def dag_members(self, process_names: set[str],
                    entity_kinds: Mapping[str, str]) -> tuple[str, ...]:
        """Process-graph member ids for this node's block: the lowered
        processes of every instance, then the flattened entities the node
        owns (wired fields belong to, and are displayed in, the wiring
        target's block)."""
        members: list[str] = []
        for method_name, _method, _spec in _component_process_methods(
                self.cls):
            # A method compiled once for every instance registers under
            # the decl name; instance-specialized methods (spawnables,
            # the per-instance fallback) under the per-instance names.
            candidates = (f"{self.name}__{method_name}",
                          *(f"{prefix}__{method_name}"
                            for prefix in self.process_names))
            members.extend(f"process:{candidate}"
                           for candidate in candidates
                           if candidate in process_names)
        for field_decl in self.decls.fields.values():
            if (not field_decl.kind.dag_entity
                    or field_decl.name in self.aliased_fields):
                continue
            flat_name = self.direct_field_map[field_decl.name]
            graph_kind = entity_kinds.get(flat_name)
            if graph_kind is not None:
                members.append(f"{graph_kind}:{flat_name}")
        return tuple(dict.fromkeys(members))


# --- Declaration building ---------------------------------------------------
#
# _class_declarations() walks a Model subclass's annotations and hands
# each component field to _DeclBuilder, which builds the decl tree and
# flattens every declared field into the model-level declarations dict.

def _component_declarations(cls: type[Component]) -> _Declarations:
    decls = _field_declarations(cls, allow_symbolic_pqueues=True,
                                allow_refs=True)
    for field_decl in decls.fields.values():
        if not field_decl.kind.on_component:
            raise ValueError(
                f"component '{cls.__name__}' declares {field_decl.kind.name} "
                "fields, which are not supported yet")
    return decls


def _component_field_map(name: str, decls: _Declarations) -> dict[str, str]:
    return {fname: f"{name}__{fname}" for fname in decls.fields}


def _primitive_constant(value: Any) -> bool:
    return type(value) in (bool, int, float)


def _component_constants(
    items: Sequence[Component],
    field_map: Mapping[str, str],
    exclude: frozenset[str] = frozenset(),
) -> dict[str, tuple[Any, ...]]:
    """Per-instance primitive attribute values (usually set in __init__):
    names present on every item with a bool/int/float value throughout."""
    constants: dict[str, tuple[Any, ...]] = {}
    names = {
        name
        for item in items
        for name in vars(item)
        if (not name.startswith("_") and name not in field_map
            and name not in exclude)
    }
    for name in names:
        values = tuple(getattr(item, name, _MISSING) for item in items)
        if all(value is not _MISSING and _primitive_constant(value)
               for value in values):
            constants[name] = values
    return constants


def _validate_component_consts(
    component_name: str,
    templates: Sequence[Component],
    consts: Mapping[str, type],
) -> dict[str, tuple[Any, ...]]:
    """Per-instance values of the declared ``sim.Const`` fields, checked
    to be present and of the annotated type on every instance."""
    values: dict[str, tuple[Any, ...]] = {}
    for fname, ctype in consts.items():
        instance_values = []
        for template in templates:
            value = getattr(template, fname, _MISSING)
            if value is _MISSING:
                raise ValueError(
                    f"component '{component_name}' constant '{fname}' must "
                    "be set on every item")
            if type(value) is not ctype:
                raise ValueError(
                    f"component '{component_name}' constant '{fname}' must "
                    f"be {ctype.__name__}")
            instance_values.append(value)
        values[fname] = tuple(instance_values)
    return values


def _offsets_from_counts(counts: Iterable[int]) -> tuple[tuple[int, ...],
                                                         tuple[int, ...]]:
    counts_tuple = tuple(int(count) for count in counts)
    offsets: list[int] = []
    total = 0
    for count in counts_tuple:
        offsets.append(total)
        total += count
    return counts_tuple, tuple(offsets)


def _resolve_component_pqueues(
    component_name: str,
    instance_count: int,
    decls: _Declarations,
    constants: Mapping[str, tuple[Any, ...]],
) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
    """Per-instance element counts and start offsets of each PQueues
    field; symbolic counts name a per-instance int constant."""
    counts_by_field: dict[str, tuple[int, ...]] = {}
    offsets_by_field: dict[str, tuple[int, ...]] = {}
    for field_decl in decls.by_kind("pqueues"):
        fname = field_decl.name
        count_decl = field_decl.count
        if isinstance(count_decl, int):
            counts: tuple[Any, ...] = (count_decl,) * instance_count
        else:
            values = constants.get(count_decl)
            if values is None:
                raise ValueError(
                    f"component '{component_name}' field "
                    f"'{fname}' uses PQueues count '{count_decl}', which "
                    "must name an int constant on every item")
            if not all(type(value) is int and value >= 1 for value in values):
                raise ValueError(
                    f"component '{component_name}' field "
                    f"'{fname}' uses PQueues count '{count_decl}', which "
                    "must be a positive int on every item")
            counts = values
        counts_by_field[fname], offsets_by_field[fname] = \
            _offsets_from_counts(counts)
    return counts_by_field, offsets_by_field


def _resolve_component_processes(
    component_name: str,
    cls: type[Component],
    templates: Sequence[Component],
    decls: _Declarations,
) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
    """Per-instance copy counts and handle offsets of each Processes
    field, bound to the same-named @sim.process method."""
    methods = {
        name: spec
        for name, _method, spec in _component_process_methods(cls)
    }
    counts_by_field: dict[str, tuple[int, ...]] = {}
    offsets_by_field: dict[str, tuple[int, ...]] = {}
    for fname in decls.names("processes"):
        spec = methods.get(fname)
        if spec is None:
            raise ValueError(
                f"component '{component_name}' Processes field '{fname}' "
                "must have a same-named @sim.process method")
        counts = [
            spec.resolve_copies(template, f"{component_name}.{fname}")
            for template in templates
        ]
        counts_by_field[fname], offsets_by_field[fname] = \
            _offsets_from_counts(counts)
    return counts_by_field, offsets_by_field


def _rewrite_component_capacity(
    component_name: str,
    field_name: str,
    cap: int | str | None,
    decls: _Declarations,
    field_map: Mapping[str, str],
) -> int | str | None:
    """Rewrite a symbolic Queue/Pool/Store capacity to the flattened name
    of the component's own Param it references; model-level param names
    pass through untouched."""
    if not isinstance(cap, str):
        return cap
    if decls.kind_of(cap) == "param":
        return field_map[cap]
    if cap in field_map:
        raise ValueError(
            f"component '{component_name}' field '{field_name}' capacity "
            f"'{cap}' must name a Param field")
    return cap


def _component_ref_values(
    component_name: str,
    templates: Sequence[Component],
    decls: _Declarations,
) -> dict[str, _ComponentRefDecl]:
    """Capture raw Ref/Refs targets from the template instances."""
    refs: dict[str, _ComponentRefDecl] = {}
    for fname, target_cls in decls.refs.items():
        # A string target (e.g. Ref["Station"] for self-references) is
        # only checked against the Component base; identity resolution
        # does not need the class.
        check_cls = target_cls if isinstance(target_cls, type) else Component
        values = []
        for template in templates:
            value = vars(template).get(fname)
            if value is not None and not isinstance(value, check_cls):
                raise TypeError(
                    f"component '{component_name}' ref '{fname}' value must "
                    f"be a {check_cls.__name__} instance or None")
            values.append(value)
        refs[fname] = _ComponentRefDecl(fname, False, raw=tuple(values))
    for fname, target_cls in decls.ref_tables.items():
        check_cls = target_cls if isinstance(target_cls, type) else Component
        tables = []
        for template in templates:
            value = vars(template).get(fname)
            if value is None:
                value = ()
            if (not isinstance(value, (list, tuple))
                    or not all(isinstance(item, check_cls)
                               for item in value)):
                raise TypeError(
                    f"component '{component_name}' refs table '{fname}' "
                    "value must be a list or tuple of "
                    f"{check_cls.__name__} instances")
            tables.append(tuple(value))
        refs[fname] = _ComponentRefDecl(fname, True, raw_tables=tuple(tables))
    return refs


#: Ref target registry value for an instance that is the default of more
#: than one model field, so it cannot be an unambiguous reference target.
_AMBIGUOUS_REF_TARGET: Any = object()


class _DeclBuilder:
    """Builds the component declaration tree of one Model subclass.

    The tree is built first, capturing each node's declared fields and
    raw wiring references; entity wiring and Ref/Refs targets are resolved
    afterwards (see ``_class_declarations``), once every instance's decl
    exists, so references may point forward. Only then are the fields
    flattened into ``target`` -- the model-level declarations -- because
    wiring decides which fields alias another entity and declare nothing
    of their own.
    """

    def __init__(self, target: _Declarations):
        self.target = target

    # -- instance defaults -------------------------------------------------

    @staticmethod
    def _instance_default(owner: Any, attr: str, label: str,
                          child_cls: type[Component]) -> Component:
        value = getattr(owner, attr, _MISSING)
        if value is _MISSING:
            raise ValueError(
                f"component field '{label}' needs a "
                f"{child_cls.__name__} instance default")
        if not isinstance(value, child_cls):
            raise TypeError(
                f"component field '{label}' default must "
                f"be a {child_cls.__name__} instance")
        return value

    @staticmethod
    def _collection_default(owner: Any, attr: str, label: str,
                            item_cls: type[Component]
                            ) -> tuple[Component, ...]:
        value = getattr(owner, attr, _MISSING)
        if (value is _MISSING or not isinstance(value, (list, tuple))
                or not value):
            raise ValueError(
                f"component collection '{label}' needs a "
                f"non-empty list or tuple of {item_cls.__name__} instances")
        templates = tuple(value)
        for item in templates:
            if not isinstance(item, item_cls):
                raise TypeError(
                    f"component collection '{label}' "
                    f"items must be {item_cls.__name__} instances")
        return templates

    # -- tree construction ---------------------------------------------------

    def build_model(self, cls: type) -> None:
        """Build the root component decls of a Model subclass and append
        them to the target declarations."""
        for fname, item_cls, is_collection in _component_fields(cls):
            decl = self._build_field((cls,), (None,), "", "", fname,
                                     item_cls, is_collection)
            target = (self.target.component_collections if is_collection
                      else self.target.components)
            target.append(decl)

    def _build_field(
        self,
        owners: Sequence[Any],
        prefixes: Sequence[str | None],
        parent_name: str,
        parent_display: str,
        fname: str,
        cls: type[Component],
        collection: bool,
    ) -> _ComponentDecl:
        """Build one component field's decl, gathering its instances from
        each owner (the model class for a root, the parent's instances for
        a child) and deriving the flattened name, per-instance process-name
        prefixes, and display paths from the parent context."""
        label = f"{parent_name}.{fname}" if parent_name else fname
        name = f"{parent_name}__{fname}" if parent_name else fname
        templates: list[Component] = []
        process_names: list[str] = []
        offsets: list[int] = []
        lengths: list[int] = []
        for owner, prefix in zip(owners, prefixes):
            base = fname if prefix is None else f"{prefix}__{fname}"
            if collection:
                items = self._collection_default(owner, fname, label, cls)
                offsets.append(len(templates))
                lengths.append(len(items))
                templates.extend(items)
                process_names.extend(f"{base}__{i}" for i in range(len(items)))
            else:
                templates.append(
                    self._instance_default(owner, fname, label, cls))
                process_names.append(base)

        display = f"{parent_display}.{fname}" if parent_name else fname
        item_display = f"{display}[]" if collection else display
        return self._build(
            local_name=fname, name=name, cls=cls,
            templates=tuple(templates), process_names=tuple(process_names),
            display_name=display, item_display_name=item_display,
            collection=collection,
            parent_offsets=tuple(offsets) if collection else (),
            parent_lengths=tuple(lengths) if collection else ())

    def _build(
        self,
        *,
        local_name: str,
        name: str,
        cls: type[Component],
        templates: tuple[Component, ...],
        process_names: tuple[str, ...],
        display_name: str,
        item_display_name: str,
        collection: bool,
        parent_offsets: tuple[int, ...] = (),
        parent_lengths: tuple[int, ...] = (),
    ) -> _ComponentDecl:
        decls = _component_declarations(cls)
        direct_field_map = _component_field_map(name, decls)
        wiring_raw = self._field_wiring(name, templates, decls)
        component_refs = _component_ref_values(name, templates, decls)
        const_values = _validate_component_consts(name, templates,
                                                  decls.consts)
        constants = {
            **const_values,
            **_component_constants(
                templates, direct_field_map,
                exclude=frozenset(component_refs) | set(decls.consts)),
        }
        pqueue_counts, pqueue_offsets = _resolve_component_pqueues(
            name, len(templates), decls, constants)
        process_counts, process_offsets = _resolve_component_processes(
            name, cls, templates, decls)
        children = tuple(
            self._build_field(templates, process_names, name,
                              item_display_name, child_name, child_cls,
                              child_collection)
            for child_name, child_cls, child_collection
            in _component_fields(cls)
        )
        return _ComponentDecl(
            name=name,
            cls=cls,
            collection=collection,
            instances=templates,
            decls=decls,
            local_name=local_name,
            process_names=process_names,
            display_name=display_name,
            item_display_name=item_display_name,
            direct_field_map=direct_field_map,
            constants=constants,
            pqueue_counts=pqueue_counts,
            pqueue_offsets=pqueue_offsets,
            process_counts=process_counts,
            process_offsets=process_offsets,
            component_refs=component_refs,
            wiring_raw=wiring_raw,
            children=children,
            parent_offsets=parent_offsets,
            parent_lengths=parent_lengths,
        )

    def flatten(self, decl: _ComponentDecl) -> None:
        """Append one built (and wiring-resolved) node's declarations to
        the model-level target under their flattened names; multi-instance
        decls declare shaped fields with one element per instance. Wired
        (aliased) fields name the target's entity and declare nothing of
        their own."""
        shape = (decl.count,) if decl.count > 1 else None
        aliased = set(decl.aliased_fields)
        for field_decl in decl.decls.fields.values():
            fname = field_decl.name
            flat_name = decl.direct_field_map[fname]
            kind = field_decl.kind
            if kind.name == "pqueues":
                total = sum(decl.pqueue_counts[fname])
                self.target.add(_FieldDecl(flat_name, kind, count=total))
            elif kind.name == "processes":
                total = sum(decl.process_counts[fname])
                self.target.add(_FieldDecl(flat_name, kind, shape=(total,)))
            elif fname not in aliased:
                capacity = _rewrite_component_capacity(
                    decl.name, fname, field_decl.capacity, decl.decls,
                    decl.direct_field_map)
                self.target.add(_FieldDecl(flat_name, kind,
                                           capacity=capacity, shape=shape))

    # -- entity wiring -------------------------------------------------------

    @staticmethod
    def _field_wiring(
        name: str,
        templates: tuple[Component, ...],
        decls: _Declarations,
    ) -> dict[str, _FieldRef]:
        """Declared entity fields overridden with a wiring reference,
        validated for matching kinds; the target is resolved later."""
        wiring: dict[str, _FieldRef] = {}
        for field_decl in decls.fields.values():
            if not field_decl.kind.wirable:
                continue
            fname = field_decl.name
            kind = field_decl.kind.name
            refs = [vars(template).get(fname) for template in templates]
            if not any(isinstance(ref, _FieldRef) for ref in refs):
                continue
            if len(templates) > 1:
                raise ValueError(
                    f"component collection '{name}' field "
                    f"'{fname}' cannot be wired to another component's "
                    "field; wiring is not supported for collections yet")
            ref = refs[0]
            if ref.kind != kind:
                raise ValueError(
                    f"component '{name}' {kind} field '{fname}' "
                    f"cannot be wired to {ref.kind} field '{ref.field}'; "
                    "the field kinds must match")
            wiring[fname] = ref
        return wiring


def _class_declarations(cls: type) -> _Declarations:
    """Collect env field declarations from a Model subclass's annotations,
    in declaration order (base classes first). The component trees are
    built, their wiring and references resolved, and every field flattened
    into the returned declarations."""
    decls = _field_declarations(cls)
    builder = _DeclBuilder(decls)
    builder.build_model(cls)
    roots = (*decls.components, *decls.component_collections)
    _resolve_component_wiring(roots)
    # Flatten parent-before-child, in declaration order, so the trial
    # record field order is stable and duplicate names are caught in order.
    for root in roots:
        for decl in root.walk():
            builder.flatten(decl)
    _resolve_component_refs(roots)
    return decls


# -- Entity wiring resolution: runs after the whole tree is built, so a
# field may be wired to a target declared later, and chains of wirings
# resolve through to the entity that actually backs them.

def _resolve_component_wiring(roots: Sequence[_ComponentDecl]) -> None:
    """Resolve each wired field to the flattened name of the entity it
    ultimately names, following chains and rejecting cycles."""
    identity = _instance_identity(roots)
    for root in roots:
        for decl in root.walk():
            aliased: list[str] = []
            for fname, ref in decl.wiring_raw.items():
                decl.direct_field_map[fname] = _resolve_wiring_chain(
                    decl, fname, ref, identity, ())
                aliased.append(fname)
            decl.aliased_fields = tuple(aliased)


def _instance_identity(
    roots: Sequence[_ComponentDecl],
) -> dict[int, Any]:
    """Map each template instance to (its decl, item index or None),
    marking instances shared by more than one field as ambiguous."""
    identity: dict[int, Any] = {}
    for root in roots:
        for decl in root.walk():
            for index, template in enumerate(decl.instances):
                key = id(template)
                if key in identity:
                    identity[key] = _AMBIGUOUS_REF_TARGET
                else:
                    identity[key] = (decl, index if decl.count > 1 else None)
    return identity


def _resolve_wiring_chain(
    decl: _ComponentDecl,
    fname: str,
    ref: _FieldRef,
    identity: Mapping[int, Any],
    visiting: tuple[tuple[int, str], ...],
) -> str:
    target = identity.get(id(ref.instance))
    if target is None:
        raise ValueError(
            f"component '{decl.name}' field '{fname}' is wired to a "
            f"{type(ref.instance).__name__} instance that is not declared "
            "on the model")
    if target is _AMBIGUOUS_REF_TARGET:
        raise ValueError(
            f"component '{decl.name}' field '{fname}' is wired to a "
            "component instance that is the default of more than one "
            "model field; the wiring target is ambiguous")
    target_decl, _index = target
    if target_decl.count > 1:
        raise ValueError(
            f"component '{decl.name}' field '{fname}' is wired to a "
            "component collection item, which is not supported yet")
    next_ref = target_decl.wiring_raw.get(ref.field)
    if next_ref is not None:
        node = (id(target_decl), ref.field)
        if node in visiting:
            raise ValueError(
                f"component '{decl.name}' field '{fname}' is part of a "
                "wiring cycle")
        return _resolve_wiring_chain(target_decl, ref.field, next_ref,
                                     identity, visiting + (node,))
    return target_decl.direct_field_map[ref.field]


# -- Ref/Refs resolution: runs after the whole tree is built, so forward
# references between components work.

def _resolve_component_refs(roots: Sequence[_ComponentDecl]) -> None:
    """Resolve raw Ref/Refs targets to (decl, item index) pairs."""
    identity = _instance_identity(roots)
    for root in roots:
        for decl in root.walk():
            for ref in decl.component_refs.values():
                _resolve_component_ref_decl(decl, ref, identity)


def _resolve_component_ref_target(
    decl: _ComponentDecl,
    ref: _ComponentRefDecl,
    instance: Component,
    identity: Mapping[int, Any],
) -> tuple[_ComponentDecl, int | None]:
    target = identity.get(id(instance))
    if target is None:
        raise ValueError(
            f"component '{decl.name}' ref '{ref.name}' references a "
            f"{type(instance).__name__} instance that is not declared on "
            "the model")
    if target is _AMBIGUOUS_REF_TARGET:
        raise ValueError(
            f"component '{decl.name}' ref '{ref.name}' references a "
            "component instance that is the default of more than one model "
            "field; the target is ambiguous")
    return target


def _resolve_component_ref_decl(
    decl: _ComponentDecl,
    ref: _ComponentRefDecl,
    identity: Mapping[int, Any],
) -> None:
    if not ref.table:
        ref.targets = tuple(
            None if instance is None
            else _resolve_component_ref_target(decl, ref, instance, identity)
            for instance in ref.raw
        )
        return
    resolved = [
        [_resolve_component_ref_target(decl, ref, instance, identity)
         for instance in table]
        for table in ref.raw_tables
    ]
    entries = [target for table in resolved for target in table]
    if entries:
        first = entries[0][0]
        if (any(target[0] is not first for target in entries)
                or first.count <= 1):
            raise ValueError(
                f"component '{decl.name}' refs table '{ref.name}' entries "
                "must all be items of a single component collection")
        ref.table_decl = first
    ref.table_indices = tuple(
        target[1] for table in resolved for target in table)
    ref.table_lengths, ref.table_offsets = _offsets_from_counts(
        len(table) for table in resolved)


# --- Symbols shared with lowered code ----------------------------------------
#
# Lowered functions look up per-instance values that cannot be resolved
# to constants (dynamic item indices) in module-level numpy arrays
# published under these names by _lowering_namespace().

def _const_symbol(component: str, name: str) -> str:
    return f"_CIMBA_CONST_{component}__{name}"


def _pqueue_offsets_symbol(component: str, field_name: str) -> str:
    return f"_CIMBA_PQOFF_{component}__{field_name}"


def _process_offsets_symbol(component: str, field_name: str) -> str:
    return f"_CIMBA_PROCOFF_{component}__{field_name}"


def _collection_offsets_symbol(component: str) -> str:
    return f"_CIMBA_OFF_{component}"


def _ref_index_symbol(component: str, name: str) -> str:
    return f"_CIMBA_REFIDX_{component}__{name}"


def _ref_table_symbol(component: str, name: str) -> str:
    return f"_CIMBA_REFTAB_{component}__{name}"


def _ref_offsets_symbol(component: str, name: str) -> str:
    return f"_CIMBA_REFOFF_{component}__{name}"


def _lowering_namespace(
    components: Iterable[_ComponentDecl],
) -> dict[str, Any]:
    """The numpy lookup tables a lowered function may reference, for the
    given decls, their descendants, and every decl reachable through
    Ref/Refs fields (whose symbols must be present too)."""
    namespace: dict[str, Any] = {}
    seen: set[int] = set()
    stack = list(components)
    while stack:
        root = stack.pop()
        for decl in root.walk():
            if id(decl) in seen:
                continue
            seen.add(id(decl))
            for name, values in decl.constants.items():
                if len(values) > 1:
                    namespace[_const_symbol(decl.name, name)] = \
                        np.asarray(values)
            for fname, offsets in decl.pqueue_offsets.items():
                if len(offsets) > 1:
                    namespace[_pqueue_offsets_symbol(decl.name, fname)] = \
                        np.asarray(offsets, dtype=np.int64)
            for fname, offsets in decl.process_offsets.items():
                if len(offsets) > 1:
                    namespace[_process_offsets_symbol(decl.name, fname)] = \
                        np.asarray(offsets, dtype=np.int64)
            if decl.collection and len(decl.parent_offsets) > 1:
                namespace[_collection_offsets_symbol(decl.name)] = np.asarray(
                    decl.parent_offsets, dtype=np.int64)
            for name, ref in decl.component_refs.items():
                if ref.table:
                    if ref.table_decl is not None:
                        namespace[_ref_table_symbol(decl.name, name)] = \
                            np.asarray(ref.table_indices, dtype=np.int64)
                        stack.append(ref.table_decl)
                    if len(ref.table_offsets) > 1:
                        namespace[_ref_offsets_symbol(decl.name, name)] = \
                            np.asarray(ref.table_offsets, dtype=np.int64)
                    continue
                targets = [t for t in ref.targets if t is not None]
                stack.extend(target_decl for target_decl, _index in targets)
                if len(ref.targets) > 1 and len(targets) == len(ref.targets):
                    first = targets[0][0]
                    if (all(t[0] is first for t in targets)
                            and first.count > 1):
                        namespace[_ref_index_symbol(decl.name, name)] = \
                            np.asarray([t[1] for t in targets],
                                       dtype=np.int64)
    return namespace


# --- AST lowering -------------------------------------------------------------
#
# _ComponentPathLowerer resolves component paths in an expression tree:
# a *namespace* (one component instance), a *collection* (must be
# indexed), or a Refs *table* (must be indexed), ending in a field or
# constant access that lowers to the flattened env field. Subclasses
# define the path roots: `self` inside component methods, `env.<name>`
# inside model callbacks.

def _env_attr(env_name: str, field_name: str,
              ctx: ast.expr_context) -> ast.Attribute:
    return ast.Attribute(
        value=ast.Name(id=env_name, ctx=ast.Load()),
        attr=field_name,
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


@dataclass(frozen=True)
class _ComponentAccess:
    """A resolved component-instance path: the decl plus the instance
    index expression (None when the decl has a single instance)."""

    decl: _ComponentDecl
    index: ast.expr | None
    text: str


@dataclass(frozen=True)
class _ComponentFieldAccess:
    """A resolved path to a declared field or captured constant."""

    decl: _ComponentDecl
    index: ast.expr | None
    field: str
    text: str


@dataclass(frozen=True)
class _RefTableAccess:
    """A resolved path to a Refs table, before indexing."""

    parent: _ComponentAccess
    name: str
    ref: _ComponentRefDecl
    text: str


class _ComponentPathLowerer(ast.NodeTransformer):
    #: When set (method lowering with a runtime instance index), indexing
    #: a Refs table with an unknown instance requires uniform per-instance
    #: lengths, so constant indices stay bounds-checkable.
    strict_ref_tables = False

    def __init__(self, *, env_name: str):
        self.env_name = env_name

    # -- path roots, defined by the subclasses -------------------------------

    def _root_namespace_ref(self, node: ast.AST) -> _ComponentAccess | None:
        return None

    def _root_collection_ref(self, node: ast.AST) -> _ComponentAccess | None:
        return None

    def _callback_label(self) -> str:
        raise NotImplementedError

    # -- path resolution -------------------------------------------------------

    def _namespace_ref(self, node: ast.AST) -> _ComponentAccess | None:
        root = self._root_namespace_ref(node)
        if root is not None:
            return root

        if isinstance(node, ast.Subscript):
            collection = self._collection_ref(node.value)
            if collection is not None:
                index = self._collection_item_index(
                    collection.decl, collection.index, node.slice)
                return _ComponentAccess(
                    collection.decl, index, f"{collection.text}[...]")
            table = self._ref_table_ref(node.value)
            if table is not None:
                return self._ref_table_item(table, node.slice)
            return None

        if isinstance(node, ast.Attribute):
            parent = self._namespace_ref(node.value)
            if parent is None:
                return None
            child = parent.decl.child(node.attr)
            if child is not None:
                if child.collection:
                    return None
                index = parent.index if child.count > 1 else None
                return _ComponentAccess(
                    child, index, f"{parent.text}.{node.attr}")
            ref = parent.decl.component_refs.get(node.attr)
            if ref is not None and not ref.table:
                return self._ref_namespace(parent, node.attr, ref)
            return None

        return None

    def _collection_ref(self, node: ast.AST) -> _ComponentAccess | None:
        root = self._root_collection_ref(node)
        if root is not None:
            return root

        if isinstance(node, ast.Attribute):
            parent = self._namespace_ref(node.value)
            if parent is None:
                return None
            child = parent.decl.child(node.attr)
            if child is None or not child.collection:
                return None
            return _ComponentAccess(
                child, parent.index, f"{parent.text}.{node.attr}")

        return None

    def _ref_table_ref(self, node: ast.AST) -> _RefTableAccess | None:
        if not isinstance(node, ast.Attribute):
            return None
        parent = self._namespace_ref(node.value)
        if parent is None:
            return None
        ref = parent.decl.component_refs.get(node.attr)
        if ref is None or not ref.table:
            return None
        return _RefTableAccess(parent, node.attr, ref,
                               f"{parent.text}.{node.attr}")

    def _ref_namespace(
        self,
        parent: _ComponentAccess,
        name: str,
        ref: _ComponentRefDecl,
    ) -> _ComponentAccess:
        """Dereference a Ref field: a static target when the instance is
        known, else an index lookup through the REFIDX table."""
        text = f"{parent.text}.{name}"
        index = parent.index
        if index is None or (isinstance(index, ast.Constant)
                             and type(index.value) is int):
            position = 0 if index is None else index.value
            target = ref.targets[position]
            if target is None:
                raise ValueError(
                    f"{self._callback_label()} dereferences {text}, which "
                    "has no target for this instance")
            target_decl, target_index = target
            target_expr = (None if target_index is None
                           else ast.Constant(target_index))
            return _ComponentAccess(target_decl, target_expr, text)
        if any(target is None for target in ref.targets):
            raise ValueError(
                f"{self._callback_label()} dereferences {text} with a "
                "dynamic instance index, but some instances have no target")
        first = ref.targets[0][0]
        if any(target[0] is not first for target in ref.targets):
            raise ValueError(
                f"{self._callback_label()} dereferences {text} with a "
                "dynamic instance index, which requires every instance to "
                "reference the same component declaration")
        if first.count <= 1:
            return _ComponentAccess(first, None, text)
        lookup = _subscript(
            ast.Name(id=_ref_index_symbol(parent.decl.name, name),
                     ctx=ast.Load()),
            index, ast.Load())
        return _ComponentAccess(first, lookup, text)

    def _ref_table_item(
        self,
        table: _RefTableAccess,
        item_slice: ast.expr,
    ) -> _ComponentAccess:
        """Index a Refs table: a static target when both the instance and
        the entry are known, else a lookup through the REFTAB table."""
        item_index = self.visit(copy.deepcopy(item_slice))
        if not isinstance(item_index, ast.expr):
            raise TypeError("component refs table index did not lower to "
                            "an expression")
        ref = table.ref
        text = f"{table.text}[...]"
        parent_index = table.parent.index
        parent_pos: int | None
        if parent_index is None:
            parent_pos = 0
        elif (isinstance(parent_index, ast.Constant)
                and type(parent_index.value) is int):
            parent_pos = parent_index.value
        else:
            parent_pos = None

        if (parent_pos is not None and isinstance(item_index, ast.Constant)
                and type(item_index.value) is int):
            length = ref.table_lengths[parent_pos]
            position = item_index.value
            if not 0 <= position < length:
                raise ValueError(
                    f"{self._callback_label()} index {position} is out of "
                    f"range for {table.text} (length {length})")
            target_index = ref.table_indices[
                ref.table_offsets[parent_pos] + position]
            return _ComponentAccess(ref.table_decl,
                                    ast.Constant(target_index), text)

        if ref.table_decl is None:
            raise ValueError(
                f"{self._callback_label()} indexes {table.text}, which has "
                "no entries")
        if parent_pos is None and self.strict_ref_tables:
            lengths = ref.table_lengths
            if len(set(lengths)) > 1:
                raise ValueError(
                    f"{self._callback_label()} indexes {table.text}, whose "
                    "per-instance lengths differ")
            if (isinstance(item_index, ast.Constant)
                    and type(item_index.value) is int
                    and not 0 <= item_index.value < lengths[0]):
                raise ValueError(
                    f"{self._callback_label()} index {item_index.value} is "
                    f"out of range for {table.text} (length {lengths[0]})")
        if parent_pos is not None:
            offset: ast.expr = ast.Constant(ref.table_offsets[parent_pos])
        else:
            offset = _subscript(
                ast.Name(id=_ref_offsets_symbol(
                    table.parent.decl.name, table.name), ctx=ast.Load()),
                parent_index, ast.Load())
        lookup = _subscript(
            ast.Name(id=_ref_table_symbol(
                table.parent.decl.name, table.name), ctx=ast.Load()),
            _add(offset, item_index), ast.Load())
        return _ComponentAccess(ref.table_decl, lookup, text)

    def _field_ref(self, node: ast.AST) -> _ComponentFieldAccess | None:
        if not isinstance(node, ast.Attribute):
            return None
        namespace = self._namespace_ref(node.value)
        if namespace is None:
            return None
        field_name = node.attr
        if (field_name in namespace.decl.direct_field_map
                or field_name in namespace.decl.constants):
            return _ComponentFieldAccess(
                namespace.decl,
                namespace.index,
                field_name,
                f"{namespace.text}.{field_name}",
            )
        if namespace.decl.child(field_name) is not None:
            return None
        if field_name in namespace.decl.component_refs:
            return None
        self._raise_unknown_field(namespace, field_name)

    def _collection_item_index(
        self,
        decl: _ComponentDecl,
        parent_index: ast.expr | None,
        item_index: ast.expr,
    ) -> ast.expr:
        """The flattened instance index of a collection item: the item
        index plus the parent instance's start offset."""
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
            ast.Name(id=_collection_offsets_symbol(decl.name),
                     ctx=ast.Load()),
            parent_index,
            ast.Load(),
        )
        return _add(offset, index)

    # -- lowered expressions ---------------------------------------------------

    def _instance_table_expr(
        self,
        values: Sequence[Any],
        index: ast.expr | None,
        symbol: str,
        what: str,
    ) -> ast.expr:
        """A per-instance value: a constant when the instance is known,
        else an element of the numpy array published under `symbol`."""
        if len(values) == 1:
            return ast.Constant(values[0])
        if (isinstance(index, ast.Constant) and type(index.value) is int):
            return ast.Constant(values[index.value])
        if index is None:
            raise TypeError(f"component {what} has no instance index")
        return _subscript(ast.Name(id=symbol, ctx=ast.Load()), index,
                          ast.Load())

    def _field_target(
        self,
        access: _ComponentFieldAccess,
        ctx: ast.expr_context,
    ) -> ast.expr:
        flat_name = access.decl.direct_field_map[access.field]
        target = _env_attr(self.env_name, flat_name, ctx)
        if access.decl.count <= 1:
            return target
        if access.index is None:
            raise TypeError("component field has no instance index")
        return _subscript(target, access.index, ctx)

    def _constant_expr(self, access: _ComponentFieldAccess) -> ast.expr:
        return self._instance_table_expr(
            access.decl.constants[access.field], access.index,
            _const_symbol(access.decl.name, access.field), "constant")

    def _lower_indexed_field(
        self,
        access: _ComponentFieldAccess,
        node: ast.Subscript,
    ) -> ast.Subscript | None:
        """Lower ``<pqueues/processes field>[i]`` to an element of the
        flattened shared array, at the instance's offset plus ``i``."""
        decl = access.decl
        kind = decl.decls.kind_of(access.field)
        if kind == "pqueues":
            what = "PQueues"
            offsets = decl.pqueue_offsets[access.field]
            symbol = _pqueue_offsets_symbol(decl.name, access.field)
        elif kind == "processes":
            what = "Processes"
            offsets = decl.process_offsets[access.field]
            symbol = _process_offsets_symbol(decl.name, access.field)
        else:
            return None
        item = self.visit(copy.deepcopy(node.slice))
        if not isinstance(item, ast.expr):
            raise TypeError(f"component {what} index did not lower "
                            "to an expression")
        offset = self._instance_table_expr(offsets, access.index, symbol,
                                           f"{what} field")
        if isinstance(offset, ast.Constant) and offset.value == 0:
            index = item
        else:
            index = _add(offset, item)
        flat = _env_attr(self.env_name,
                         decl.direct_field_map[access.field], ast.Load())
        return ast.copy_location(_subscript(flat, index, node.ctx), node)

    def _raise_unknown_field(
        self,
        namespace: _ComponentAccess,
        field_name: str,
    ) -> None:
        kind = ("component collection field"
                if namespace.decl.collection else "component field")
        raise ValueError(
            f"{self._callback_label()} references unknown {kind} "
            f"{namespace.text}.{field_name}")

    # -- node visitors -----------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if (isinstance(node.func, ast.Name) and node.func.id == "getattr"
                and node.args):
            target = (self._namespace_ref(node.args[0])
                      or self._collection_ref(node.args[0])
                      or self._ref_table_ref(node.args[0]))
            if target is not None:
                raise ValueError(
                    f"{self._callback_label()} uses dynamic "
                    f"getattr({target.text}, ...), which is not supported")
        if isinstance(node.func, ast.Attribute):
            # self.<field>.history().method(...)
            history_call = node.func.value
            if (isinstance(history_call, ast.Call)
                    and isinstance(history_call.func, ast.Attribute)
                    and history_call.func.attr == "history"
                    and not history_call.args and not history_call.keywords):
                access = self._field_ref(history_call.func.value)
                if access is not None:
                    binding = _COMPONENT_HISTORY_BINDINGS.get(
                        access.decl.decls.kind_of(access.field))
                    if binding is not None:
                        return lower_timeseries_method_call(
                            node,
                            self._field_target(access, ast.Load()),
                            binding=binding,
                            visit=self.visit,
                            label=self._callback_label(),
                        )
        # bare self.<field>.history()
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "history"
                and not node.args and not node.keywords):
            access = self._field_ref(node.func.value)
            if access is not None:
                binding = _COMPONENT_HISTORY_BINDINGS.get(
                    access.decl.decls.kind_of(access.field))
                if binding is not None:
                    return lower_history_getter_call(
                        node,
                        self._field_target(access, ast.Load()),
                        binding=binding,
                        label=self._callback_label(),
                    )
        if isinstance(node.func, ast.Attribute):
            access = self._field_ref(node.func.value)
            if access is not None:
                field_kind = access.decl.decls.kind_of(access.field)
                if field_kind == "dataset":
                    return lower_dataset_method_call(
                        node,
                        self._field_target(access, ast.Load()),
                        visit=self.visit,
                        label=self._callback_label(),
                    )
                if field_kind in _COMPONENT_ENTITY_KINDS:
                    return lower_entity_method_call(
                        node,
                        self._field_target(access, ast.Load()),
                        kind=field_kind,
                        visit=self.visit,
                        label=self._callback_label(),
                        env_expr=ast.Name(id=self.env_name, ctx=ast.Load()),
                    )
                raise ValueError(
                    f"{self._callback_label()} cannot call "
                    f"{access.text}.{node.func.attr}() inside compiled code")
            target = (self._namespace_ref(node.func.value)
                      or self._collection_ref(node.func.value)
                      or self._ref_table_ref(node.func.value))
            if target is not None:
                raise ValueError(
                    f"{self._callback_label()} cannot call "
                    f"{target.text}.{node.func.attr}() inside compiled code")
        return self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        access = self._field_ref(node.value)
        if access is not None:
            lowered = self._lower_indexed_field(access, node)
            if lowered is not None:
                return lowered
        collection = self._collection_ref(node.value)
        if collection is not None:
            raise ValueError(
                f"{self._callback_label()} uses {collection.text}[...] "
                "directly; access one of its fields")
        table = self._ref_table_ref(node.value)
        if table is not None:
            raise ValueError(
                f"{self._callback_label()} uses {table.text}[...] "
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
            if access.decl.decls.kind_of(access.field) in ("pqueues",
                                                           "processes"):
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
        table = self._ref_table_ref(node)
        if table is not None:
            raise ValueError(
                f"{self._callback_label()} must index {table.text} before "
                "using it")
        return self.generic_visit(node)


class _ComponentMethodLowerer(_ComponentPathLowerer):
    """Lowers a component method body: `self` is the path root, resolved
    to the given instance-index expression -- a constant when the method
    is specialized to one instance, or the runtime ``__cimba_inst``
    value when one compiled function covers every instance."""

    def __init__(
        self,
        *,
        component_name: str,
        receiver_name: str,
        env_name: str,
        component_decl: _ComponentDecl,
        instance_index: ast.expr,
        kind: str = "process",
    ):
        super().__init__(env_name=env_name)
        self.component_name = component_name
        self.receiver_name = receiver_name
        self.component_decl = component_decl
        self.instance_index = instance_index
        self.strict_ref_tables = not isinstance(instance_index, ast.Constant)
        self.kind = kind

    def _root_namespace_ref(self, node: ast.AST) -> _ComponentAccess | None:
        if isinstance(node, ast.Name) and node.id == self.receiver_name:
            index = (copy.deepcopy(self.instance_index)
                     if self.component_decl.count > 1 else None)
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


class _ModelComponentRefLowerer(_ComponentPathLowerer):
    """Lowers a model callback body: `env.<component field>` is the path
    root; `changed` records whether anything was rewritten."""

    def __init__(self, *, model_name: str, fn_name: str, env_name: str,
                 component_roots: Mapping[str, _ComponentDecl]):
        super().__init__(env_name=env_name)
        self.model_name = model_name
        self.fn_name = fn_name
        self.component_roots = component_roots
        self.changed = False

    def _root_ref(self, node: ast.AST) -> _ComponentDecl | None:
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == self.env_name):
            return self.component_roots.get(node.attr)
        return None

    def _root_namespace_ref(self, node: ast.AST) -> _ComponentAccess | None:
        decl = self._root_ref(node)
        if decl is not None and not decl.collection:
            return _ComponentAccess(decl, None,
                                    f"{self.env_name}.{node.attr}")
        return None

    def _root_collection_ref(self, node: ast.AST) -> _ComponentAccess | None:
        decl = self._root_ref(node)
        if decl is not None and decl.collection:
            return _ComponentAccess(decl, None,
                                    f"{self.env_name}.{node.attr}")
        return None

    def _callback_label(self) -> str:
        return f"model '{self.model_name}' callback '{self.fn_name}'"

    def visit_Call(self, node: ast.Call) -> ast.AST:
        lowered = super().visit_Call(node)
        if lowered is not node:
            self.changed = True
        return lowered

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


# --- Codegen ------------------------------------------------------------------
#
# The lowered ASTs are unparsed, exec'd, and returned as plain functions
# whose source (kept in __cimba_source__ and linecache) reflects the
# rewrite -- Numba and the process-DAG inference both re-read it.

def _closure_namespace(fn: Callable[..., Any]) -> dict[str, Any]:
    namespace = dict(fn.__globals__)
    if fn.__closure__ is not None:
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
            namespace[name] = cell.cell_contents
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


def _component_method_source(fn: Callable[..., Any],
                             kind: str) -> ast.FunctionDef:
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


def _compile_lowered(
    node: ast.FunctionDef,
    *,
    filename: str,
    fn_name: str,
    qualname: str,
    namespace: dict[str, Any],
    like: Callable[..., Any],
) -> Callable[..., Any]:
    """Exec a lowered FunctionDef and return the generated function; the
    source goes into linecache so tracebacks and inspect.getsource()
    resolve against the rewritten code."""
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    source = ast.unparse(module) + "\n"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    exec(compile(source, filename, "exec"), namespace)
    generated = namespace[fn_name]
    generated.__module__ = like.__module__
    generated.__qualname__ = qualname
    generated.__cimba_source__ = source
    return generated


def _lower_component_method(
    node: ast.FunctionDef,
    *,
    kind: str,
    component_name: str,
    component_decl: _ComponentDecl,
    instance_index: ast.expr,
    method_name: str,
    method: Callable[..., Any],
    struct_view: type | None = None,
    prologue: Sequence[ast.stmt] = (),
    extra_namespace: Mapping[str, Any] | None = None,
    model_dataset_fields: Iterable[str] = (),
    model_history_fields: Mapping[str, str] = {},
    model_entity_fields: Mapping[str, str] = {},
) -> Callable[..., Any]:
    """Shared tail of process/collect lowering: drop `self`, rewrite the
    body against the flattened env, and compile the result."""
    args = node.args
    receiver_name = args.args[0].arg
    env_name = args.args[1].arg
    fn_name = f"{component_name}__{method_name}"
    node.name = fn_name
    node.decorator_list = []
    node.returns = None
    node.type_comment = None
    args.args = args.args[1:]
    for index, arg in enumerate(args.args):
        if struct_view is not None and index == len(args.args) - 1:
            # Keep an annotation on the view parameter so Model.process()
            # detects it on the lowered function; the exec namespace maps
            # _CIMBA_STRUCT_VIEW to the struct class.
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
        kind=kind,
    )
    lowered = lowerer.visit(node)
    if not isinstance(lowered, ast.FunctionDef):
        raise TypeError(f"component {kind} lowering produced a non-function")
    if model_dataset_fields:
        lowered, _ = lower_env_dataset_method_calls(
            lowered,
            env_name=env_name,
            dataset_fields=model_dataset_fields,
            label=f"component {kind} '{component_name}.{method_name}'",
        )
    if model_history_fields:
        lowered, _ = lower_env_history_method_calls(
            lowered,
            env_name=env_name,
            history_fields=model_history_fields,
            label=f"component {kind} '{component_name}.{method_name}'",
        )
    if model_entity_fields:
        lowered, _ = lower_env_entity_method_calls(
            lowered,
            env_name=env_name,
            entity_fields=model_entity_fields,
            label=f"component {kind} '{component_name}.{method_name}'",
        )
    lowered.body[:0] = list(prologue)

    namespace = _closure_namespace(method)
    if model_entity_fields:
        _rewire_entity_method_helpers(
            namespace, set(method.__code__.co_names),
            model_name=component_name, entity_fields=model_entity_fields,
            cache={})
    lowered, random_changed = lower_random_calls_in_node(
        lowered,
        namespace=namespace,
        label=f"component {kind} '{component_name}.{method_name}'",
    )
    if struct_view is not None:
        namespace["_CIMBA_STRUCT_VIEW"] = struct_view
    if extra_namespace:
        namespace.update(extra_namespace)
    namespace.update(dataset_lowering_namespace())
    namespace.update(timeseries_lowering_namespace())
    namespace.update(entity_lowering_namespace())
    if random_changed:
        namespace.update(random_lowering_namespace())
    namespace.update(_lowering_namespace((component_decl,)))
    return _compile_lowered(
        lowered,
        filename=f"<cimba component '{component_name}.{method_name}'>",
        fn_name=fn_name,
        qualname=fn_name,
        namespace=namespace,
        like=method,
    )


def _shared_instance_setup(
    node: ast.FunctionDef,
    base: str,
    counts: tuple[int, ...],
    base_arg_count: int,
) -> tuple[ast.expr, list[ast.stmt], dict[str, Any]]:
    """Prepare one compiled body to serve every instance of a collection.

    A collection's process is started once per copy with the *global* copy
    index (0 .. sum(counts) - 1). Three indices are in play, and this
    derives the latter two from the first:

    * ``__cimba_idx`` -- the global copy index, inserted here as the body's
      new second parameter;
    * ``__cimba_inst`` -- which collection item the copy belongs to, used
      as the runtime instance index wherever the body reads a per-instance
      field or constant (returned as the index expression);
    * the user's own copy-index parameter, if the method declares one --
      the copy's position *within* its item.

    Uniform copy counts reduce to arithmetic; ragged counts use small
    generated lookup tables, keyed by ``base`` so methods never collide.
    Returns ``(instance_index_expr, prologue_statements, lookup_tables)``.
    """
    params = node.args.args
    user_idx = params[2].arg if base_arg_count == 3 else None
    if user_idx is not None:
        params[2] = ast.arg(arg="__cimba_idx")
    else:
        params.insert(2, ast.arg(arg="__cimba_idx"))

    inst_symbol = f"_CIMBA_PROCINST_{base}"
    copybase_symbol = f"_CIMBA_COPYBASE_{base}"
    uniform = len(set(counts)) == 1
    per_instance = counts[0]
    lines: list[str] = []
    tables: dict[str, Any] = {}

    # __cimba_inst: the collection item this global copy belongs to.
    if uniform and per_instance == 1:
        lines.append("__cimba_inst = __cimba_idx")
    elif uniform:
        lines.append(f"__cimba_inst = __cimba_idx // {per_instance}")
    else:
        tables[inst_symbol] = np.repeat(
            np.arange(len(counts), dtype=np.int64),
            np.asarray(counts, dtype=np.int64))
        lines.append(f"__cimba_inst = {inst_symbol}[__cimba_idx]")

    # the user's copy index: this copy's position within its own item.
    if user_idx is not None:
        if uniform and per_instance == 1:
            lines.append(f"{user_idx} = 0")
        elif uniform:
            lines.append(f"{user_idx} = __cimba_idx % {per_instance}")
        else:
            tables[copybase_symbol] = np.asarray(
                _offsets_from_counts(counts)[1], dtype=np.int64)
            lines.append(
                f"{user_idx} = __cimba_idx - {copybase_symbol}[__cimba_inst]")

    return (ast.Name(id="__cimba_inst", ctx=ast.Load()),
            ast.parse("\n".join(lines)).body, tables)


def _component_process_signature(
    node: ast.FunctionDef,
    component_name: str,
    method_name: str,
    method: Callable[..., Any],
    is_struct_class: Callable[[Any], bool],
) -> tuple[type | None, int]:
    """Validate a component process method's ``(self, env[, idx][, view])``
    signature and return ``(struct_view_class_or_None, base_arg_count)``
    where the base count excludes the optional view parameter."""
    args = node.args
    signature = (f"component process '{component_name}.{method_name}' must "
                 "take (self, env), (self, env, idx), and optionally a "
                 "final sim.Struct view parameter, without defaults")
    if (args.posonlyargs or args.vararg or args.kwonlyargs or args.kwarg
            or args.defaults or args.kw_defaults):
        raise ValueError(signature)

    params = args.args
    hints = get_type_hints(method)
    own = hints.get(params[-1].arg) if len(params) > 2 else None
    struct_view = own if is_struct_class(own) else None
    injected = struct_view is not None
    for arg in params[2:len(params) - 1 if injected else len(params)]:
        if is_struct_class(hints.get(arg.arg)):
            raise ValueError(
                f"component process '{component_name}.{method_name}': the "
                f"{hints[arg.arg].__name__} view must be the last parameter")

    base_arg_count = len(params) - (1 if injected else 0)
    if base_arg_count not in (2, 3):
        raise ValueError(signature)
    return struct_view, base_arg_count


def _lower_component_process(
    component_name: str,
    component_decl: _ComponentDecl,
    method_name: str,
    method: Callable[..., Any],
    is_struct_class: Callable[[Any], bool],
    *,
    instance_index: int | None = None,
    copies_per_instance: tuple[int, ...] | None = None,
    model_dataset_fields: Iterable[str] = (),
    model_history_fields: Mapping[str, str] = {},
    model_entity_fields: Mapping[str, str] = {},
) -> Callable[..., Any]:
    """Lower a component process method into a flat process function.

    With ``instance_index``, the function is specialized to one instance
    (spawnable methods, and the fallback for methods whose per-instance
    Ref targets cannot share one body). With ``copies_per_instance``, one
    function covers every instance: it takes the global copy index as a
    runtime argument, recovers the instance and the user's local copy
    index from it, and reads per-instance values through the published
    lookup tables."""
    node = copy.deepcopy(_component_method_source(method, "process"))
    struct_view, base_arg_count = _component_process_signature(
        node, component_name, method_name, method, is_struct_class)

    if copies_per_instance is None:
        index_expr: ast.expr = ast.Constant(instance_index)
        prologue: list[ast.stmt] = []
        tables: dict[str, Any] = {}
    else:
        index_expr, prologue, tables = _shared_instance_setup(
            node, f"{component_name}__{method_name}",
            tuple(copies_per_instance), base_arg_count)

    return _lower_component_method(
        node, kind="process", component_name=component_name,
        component_decl=component_decl, instance_index=index_expr,
        method_name=method_name, method=method, struct_view=struct_view,
        prologue=prologue, extra_namespace=tables,
        model_dataset_fields=model_dataset_fields,
        model_history_fields=model_history_fields,
        model_entity_fields=model_entity_fields)


def _lower_component_collect(
    component_name: str,
    component_decl: _ComponentDecl,
    method_name: str,
    method: Callable[..., Any],
    *,
    instance_index: int | None = None,
    per_class: bool = False,
    model_dataset_fields: Iterable[str] = (),
    model_history_fields: Mapping[str, str] = {},
    model_entity_fields: Mapping[str, str] = {},
) -> Callable[..., Any]:
    """Lower a component collect method; with ``per_class``, one function
    covers every instance and takes the instance index as its second
    argument."""
    node = copy.deepcopy(_component_method_source(method, "collect"))
    args = node.args
    if (args.posonlyargs or args.vararg or args.kwonlyargs or args.kwarg
            or args.defaults or args.kw_defaults or len(args.args) != 2):
        raise ValueError(
            f"component collect '{component_name}.{method_name}' must take "
            "(self, env) without defaults")
    if per_class:
        args.args.append(ast.arg(arg="__cimba_inst"))
        index_expr: ast.expr = ast.Name(id="__cimba_inst", ctx=ast.Load())
    else:
        index_expr = ast.Constant(instance_index)
    return _lower_component_method(
        node, kind="collect", component_name=component_name,
        component_decl=component_decl, instance_index=index_expr,
        method_name=method_name, method=method,
        model_dataset_fields=model_dataset_fields,
        model_history_fields=model_history_fields,
        model_entity_fields=model_entity_fields)


def _lower_model_component_refs(
    fn: Callable[..., Any],
    *,
    model_name: str,
    component_roots: Mapping[str, _ComponentDecl],
) -> Callable[..., Any]:
    """Rewrite a model callback's component paths against the flattened
    env; returns the callback unchanged when it uses none."""
    if not component_roots:
        return fn
    if not any(name in fn.__code__.co_names for name in component_roots):
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

    namespace = _closure_namespace(fn)
    namespace.update(dataset_lowering_namespace())
    namespace.update(timeseries_lowering_namespace())
    namespace.update(entity_lowering_namespace())
    namespace.update(_lowering_namespace(component_roots.values()))
    return _compile_lowered(
        lowered,
        filename=f"<cimba model callback '{model_name}.{fn.__name__}'>",
        fn_name=fn.__name__,
        qualname=fn.__qualname__,
        namespace=namespace,
        like=fn,
    )


def _lower_dataset_methods(
    fn: Callable[..., Any],
    *,
    model_name: str,
    dataset_fields: Iterable[str],
) -> Callable[..., Any]:
    """Rewrite ``env.<dataset>.method(...)`` to native dataset helper calls."""
    fields = set(dataset_fields)
    if not fields:
        return fn
    names = set(fn.__code__.co_names)
    if not (names.intersection(fields)
            and names.intersection(DATASET_METHOD_NAMES)):
        return fn
    try:
        node = copy.deepcopy(_function_def_from_source(fn))
    except (OSError, TypeError) as exc:
        raise ValueError(
            f"model '{model_name}' callback '{fn.__qualname__}' needs "
            "inspectable source to use Dataset methods"
        ) from exc
    if not node.args.args:
        return fn

    env_name = node.args.args[0].arg
    lowered, changed = lower_env_dataset_method_calls(
        node,
        env_name=env_name,
        dataset_fields=fields,
        label=f"model '{model_name}' callback '{fn.__name__}'",
    )
    if not changed:
        return fn

    lowered.decorator_list = []
    lowered.returns = None
    lowered.type_comment = None
    for arg in lowered.args.args:
        arg.annotation = None
        arg.type_comment = None

    namespace = _closure_namespace(fn)
    namespace.update(dataset_lowering_namespace())
    return _compile_lowered(
        lowered,
        filename=f"<cimba model callback '{model_name}.{fn.__name__}'>",
        fn_name=fn.__name__,
        qualname=fn.__qualname__,
        namespace=namespace,
        like=fn,
    )


def _lower_history_methods(
    fn: Callable[..., Any],
    *,
    model_name: str,
    history_fields: Mapping[str, str],
) -> Callable[..., Any]:
    """Rewrite ``env.<entity>.history.method(...)`` (and the indexed
    ``env.<entity>[i].history.method(...)`` form) to native timeseries
    helper calls."""
    if not history_fields:
        return fn
    names = set(fn.__code__.co_names)
    if "history" not in names or not names.intersection(history_fields):
        return fn
    try:
        node = copy.deepcopy(_function_def_from_source(fn))
    except (OSError, TypeError) as exc:
        raise ValueError(
            f"model '{model_name}' callback '{fn.__qualname__}' needs "
            "inspectable source to use timeseries history methods"
        ) from exc
    if not node.args.args:
        return fn

    env_name = node.args.args[0].arg
    lowered, changed = lower_env_history_method_calls(
        node,
        env_name=env_name,
        history_fields=history_fields,
        label=f"model '{model_name}' callback '{fn.__name__}'",
    )
    if not changed:
        return fn

    lowered.decorator_list = []
    lowered.returns = None
    lowered.type_comment = None
    for arg in lowered.args.args:
        arg.annotation = None
        arg.type_comment = None

    namespace = _closure_namespace(fn)
    namespace.update(timeseries_lowering_namespace())
    return _compile_lowered(
        lowered,
        filename=f"<cimba model callback '{model_name}.{fn.__name__}'>",
        fn_name=fn.__name__,
        qualname=fn.__qualname__,
        namespace=namespace,
        like=fn,
    )


def _lower_entity_method_helper(
    helper: Any,
    *,
    model_name: str,
    entity_fields: Mapping[str, str],
    cache: dict[int, Any],
) -> Any:
    """If ``helper`` is a Numba dispatcher (an ``@njit``-decorated plain
    helper function, the kind processes commonly factor shared logic
    into) whose own body uses ``env.<entity>.method(...)`` sugar --
    directly, or transitively through helpers *it* calls -- rewrite and
    recompile it, returning the new dispatcher. Otherwise (not a
    dispatcher, or nothing to rewrite) returns ``helper`` unchanged.

    Entity-method lowering is normally per-registered-callback (see
    ``_lower_entity_methods`` below), because that's the only place the
    AST rewrite naturally has an ``env`` name to anchor on. A plain helper
    called with ``env`` as its own first argument has exactly the same
    shape, so it gets the same treatment here -- memoized by
    ``id(py_func)`` in ``cache`` so a helper shared by several processes
    is only rewritten once."""
    py_func = getattr(helper, "py_func", None)
    if py_func is None:
        return helper
    key = id(py_func)
    if key in cache:
        return cache[key]
    # Guard recursive/mutually-recursive helpers: while we're rewriting
    # this one, references to it (including from itself) resolve to the
    # original -- an accepted limitation for the exotic self-recursive case.
    cache[key] = helper
    names = set(py_func.__code__.co_names)
    direct = bool(names.intersection(entity_fields)
                 and names.intersection(ENTITY_METHOD_NAMES))
    namespace = _closure_namespace(py_func)
    helpers_changed = _rewire_entity_method_helpers(
        namespace, names, model_name=model_name, entity_fields=entity_fields,
        cache=cache)
    if not direct and not helpers_changed:
        return helper
    try:
        node = copy.deepcopy(_function_def_from_source(py_func))
    except (OSError, TypeError):
        return helper
    if not node.args.args:
        return helper

    env_name = node.args.args[0].arg
    lowered, changed = lower_env_entity_method_calls(
        node,
        env_name=env_name,
        entity_fields=entity_fields,
        label=f"model '{model_name}' helper '{py_func.__qualname__}'",
    )
    if not changed and not helpers_changed:
        return helper

    lowered.decorator_list = []
    lowered.returns = None
    lowered.type_comment = None
    for arg in lowered.args.args:
        arg.annotation = None
        arg.type_comment = None

    namespace.update(entity_lowering_namespace())
    plain = _compile_lowered(
        lowered,
        filename=f"<cimba model '{model_name}' helper "
                f"'{py_func.__qualname__}'>",
        fn_name=py_func.__name__,
        qualname=py_func.__qualname__,
        namespace=namespace,
        like=py_func,
    )
    result = njit(plain)
    cache[key] = result
    return result


def _rewire_entity_method_helpers(
    namespace: dict[str, Any],
    names: Iterable[str],
    *,
    model_name: str,
    entity_fields: Mapping[str, str],
    cache: dict[int, Any],
) -> bool:
    """Rewrite, in place, every referenced global name in ``namespace``
    that is a helper dispatcher needing entity-method lowering. Returns
    whether anything changed."""
    changed = False
    for name in names:
        obj = namespace.get(name)
        if obj is None:
            continue
        rewritten = _lower_entity_method_helper(
            obj, model_name=model_name, entity_fields=entity_fields,
            cache=cache)
        if rewritten is not obj:
            namespace[name] = rewritten
            changed = True
    return changed


def _lower_entity_methods(
    fn: Callable[..., Any],
    *,
    model_name: str,
    entity_fields: Mapping[str, str],
) -> Callable[..., Any]:
    """Rewrite ``env.<entity>.method(...)`` (and the indexed
    ``env.<entity>[i].method(...)`` form) to native helper calls, e.g.
    ``env.queue.put(1)`` or ``env.server.acquire()`` -- in ``fn``'s own
    body, and in any ``@njit`` helper function it (transitively) calls."""
    if not entity_fields:
        return fn
    names = set(fn.__code__.co_names)
    direct = bool(names.intersection(entity_fields)
                 and names.intersection(ENTITY_METHOD_NAMES))
    namespace = _closure_namespace(fn)
    cache: dict[int, Any] = {}
    helpers_changed = _rewire_entity_method_helpers(
        namespace, names, model_name=model_name, entity_fields=entity_fields,
        cache=cache)
    if not direct and not helpers_changed:
        return fn
    try:
        node = copy.deepcopy(_function_def_from_source(fn))
    except (OSError, TypeError) as exc:
        raise ValueError(
            f"model '{model_name}' callback '{fn.__qualname__}' needs "
            "inspectable source to use entity methods"
        ) from exc
    if not node.args.args:
        return fn

    env_name = node.args.args[0].arg
    lowered, changed = lower_env_entity_method_calls(
        node,
        env_name=env_name,
        entity_fields=entity_fields,
        label=f"model '{model_name}' callback '{fn.__name__}'",
    )
    if not changed and not helpers_changed:
        return fn

    lowered.decorator_list = []
    lowered.returns = None
    lowered.type_comment = None
    for arg in lowered.args.args:
        arg.annotation = None
        arg.type_comment = None

    namespace.update(entity_lowering_namespace())
    return _compile_lowered(
        lowered,
        filename=f"<cimba model callback '{model_name}.{fn.__name__}'>",
        fn_name=fn.__name__,
        qualname=fn.__qualname__,
        namespace=namespace,
        like=fn,
    )
