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
  (``self.zones[i].gates[j].queue``) are rewritten the same way, with
  generated numpy tables backing dynamic item indices, per-item
  constants, and Ref/Refs dereferences.
* read-only ``@sim.function`` methods become explicitly typed Numba helpers.
  Calls keep their component syntax in user code, while lowering passes the
  ordinary arguments followed by the scalar component values the helper reads.

The module is organized in five parts, in order: the nested-owner API
(``Component`` and the wiring/Ref metadata captured from instance defaults;
callback markers and shared MRO normalization live in ``_callbacks``);
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
import weakref
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, get_args, get_origin, get_type_hints

import numpy as np
from numba import njit, types

from ._callbacks import (
    _DeclarationOwner,
    _ProcessSpec,
    _callback_arg_count,
    _process_signature,
)
from ._dataset.methods import (
    dataset_lowering_namespace,
    lower_env_dataset_method_calls,
)
from ._declarations import (_DECL_KINDS, _FIELD_KINDS, _MISSING,
                            _STANDARD_FIELDS, _ConstHint, _Declarations,
                            _RefHint, _FieldDecl, _param_default,
                            class_type_hints)
from ._timeseries.methods import (
    lower_env_history_method_calls,
    timeseries_lowering_namespace,
)
from .random._lowering import (
    lower_random_calls_in_node,
    random_lowering_namespace,
)
from .store.methods import (
    ENTITY_METHOD_NAMES,
    entity_lowering_namespace,
    lower_env_entity_method_calls,
)

# --- Authoring API ----------------------------------------------------------
#
# The Component base class and values captured from instance defaults.

_wirable_fields_cache: dict[type, dict[str, str]] = {}


def _wirable_fields(cls: type) -> dict[str, str]:
    """Declared field name -> kind name, for the wirable entity kinds."""
    kinds = _wirable_fields_cache.get(cls)
    if kinds is None:
        kinds = {}
        for fname, hint in class_type_hints(cls).items():
            try:
                kind = _DECL_KINDS.get(hint)
            except TypeError:
                kind = None
            if kind is not None and kind.wirable:
                kinds[fname] = kind.name
        _wirable_fields_cache[cls] = kinds
    return kinds


class Component(_DeclarationOwner):
    """Authoring-time grouping of model fields and process methods.

    Component instances are declared as defaults on a ``Model`` subclass. Their
    declared fields are flattened into the model's trial record, and methods
    decorated with :func:`process` are lowered into ordinary model processes.
    Methods decorated with :func:`collect` run once per instance at the end of
    each trial, before the model-level ``@sim.collect`` callback.
    Methods decorated with :func:`predicate` and :func:`event` publish
    component-local callback handles through explicit or hidden fields.
    Read-only methods decorated with :func:`function` are lowered into
    explicitly typed synchronous Numba helpers callable from compiled model
    and component callbacks.

    Accessing a declared Queue/Resource/Pool/Store/Condition field on an
    instance yields a wiring reference: passing it as another instance's
    same-kind field value makes both fields name the same entity, e.g.
    ``Station(..., inbox=station_1.outbox)``.
    """

    def __init__(self, **values: Any) -> None:
        """Configure declared ``Param`` and ``Const`` fields.

        Component subclasses with their own constructor arguments should
        forward declaration values explicitly with
        ``super().__init__(**kwargs)``. Runtime fields such as states, queues,
        and nested components remain constructor-owned by the subclass and
        are rejected here when passed to the base constructor. Ref/Refs values
        are assigned directly and validated when the model declaration tree
        is built.
        """
        configurable, runtime = _component_constructor_fields(type(self))
        converted: dict[str, Any] = {}
        cls_name = type(self).__name__
        for name, value in values.items():
            declaration = configurable.get(name)
            if declaration is None:
                if name in runtime:
                    raise TypeError(
                        f"{cls_name} field '{name}' is a runtime declaration "
                        "and cannot be configured by Component.__init__; "
                        "only Param, Const, Ref, and Refs fields are "
                        "configurable")
                raise TypeError(
                    f"{cls_name}.__init__() got an unexpected keyword "
                    f"argument '{name}'")

            kind, expected = declaration
            if kind in ("ref", "refs"):
                converted[name] = value
                continue
            if kind == "param":
                try:
                    converted[name] = _param_default(
                        value, f"component '{cls_name}' Param '{name}'")
                except (TypeError, ValueError, OverflowError) as exc:
                    raise TypeError(
                        f"component '{cls_name}' Param '{name}' must be a "
                        "real scalar") from exc
                continue

            expected_name = getattr(expected, "__name__", repr(expected))
            try:
                coerced = expected(value)
            except Exception as exc:
                raise TypeError(
                    f"component '{cls_name}' Const '{name}' could not be "
                    f"converted to {expected_name}") from exc
            if type(coerced) is not expected:
                raise TypeError(
                    f"component '{cls_name}' Const '{name}' must convert to "
                    f"exactly {expected_name}, got "
                    f"{type(coerced).__name__}")
            converted[name] = coerced

        self.__dict__.update(converted)

    def __getattr__(self, name: str) -> "_FieldRef":
        if name.startswith("_"):
            raise AttributeError(name)
        kind = _wirable_fields(type(self)).get(name)
        if kind is None:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'")
        return _FieldRef(self, name, kind)


def _component_constructor_fields(
    cls: type[Component],
) -> tuple[dict[str, tuple[str, Any]], set[str]]:
    """Return configurable and runtime declarations for ``cls``.

    ``class_type_hints`` supplies the same inherited, resolved annotation view
    used by model declaration building. Param, Const, Ref, and Refs
    declarations are accepted by the base constructor; other Cimba
    declarations and nested components are tracked separately so they get a
    useful rejection rather than being reported as unknown keywords.
    """
    configurable: dict[str, tuple[str, Any]] = {}
    runtime: set[str] = set()
    for name, hint in class_type_hints(cls).items():
        if isinstance(hint, _ConstHint):
            configurable[name] = ("const", hint.type)
            continue
        if isinstance(hint, _RefHint):
            configurable[name] = ("refs" if hint.table else "ref",
                                   hint.target)
            continue
        try:
            kind = _DECL_KINDS.get(hint)
        except TypeError:
            kind = None
        if kind is not None:
            if kind.name == "param":
                configurable[name] = ("param", float)
            else:
                runtime.add(name)
            continue
        if (isinstance(hint, _RefHint) or _is_component_class(hint)
                or _collection_item_class(hint) is not None):
            runtime.add(name)
    return configurable, runtime


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
    for fname, hint in class_type_hints(cls).items():
        if _is_component_class(hint):
            yield fname, hint, False
        else:
            item_cls = _collection_item_class(hint)
            if item_cls is not None:
                yield fname, item_cls, True


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
class _OwnerDecl:
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
    #: compatibility constraint from the annotation. ``cls`` remains the
    #: concrete class for homogeneous declarations for backwards-compatible
    #: private introspection; polymorphic declarations use the annotated base.
    cls: type[Any]
    declared_cls: type[Any]
    #: exact class selected by each template instance.
    instance_classes: tuple[type[Any], ...]
    collection: bool
    #: bound template instances, one per (parent instance x item)
    instances: tuple[Any, ...]
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
    #: optional per-instance Param defaults, by local field name
    param_defaults: dict[str, tuple[float, ...]]
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
    children: tuple["_OwnerDecl", ...] = ()
    #: for collections: first item index / item count per parent instance
    parent_offsets: tuple[int, ...] = ()
    parent_lengths: tuple[int, ...] = ()
    #: scalar children are packed over the parents that declare them:
    #: parent instance index -> child instance index, or -1 when absent.
    parent_slots: tuple[int, ...] = ()
    #: field/constant owners and logical-instance -> packed-slot mappings.
    field_owners: dict[str, tuple[int, ...]] = field(default_factory=dict)
    field_slots: dict[str, tuple[int, ...]] = field(default_factory=dict)
    constant_owners: dict[str, tuple[int, ...]] = field(default_factory=dict)
    constant_slots: dict[str, tuple[int, ...]] = field(default_factory=dict)
    owner_root: bool = False

    @property
    def count(self) -> int:
        return len(self.instances)

    @property
    def polymorphic(self) -> bool:
        """Whether this path or one of its descendants needs per-instance
        specialization."""
        return len(self.specialization_groups()) > 1

    def class_at(self, index: int = 0) -> type[Any]:
        return self.instance_classes[index]

    def specialization_key(self, index: int) -> tuple[Any, ...]:
        """Recursive concrete layout of one logical instance."""
        children: list[Any] = []
        for child in self.children:
            if child.collection:
                start = child.parent_offsets[index]
                length = child.parent_lengths[index]
                children.append((
                    child.local_name,
                    tuple(child.specialization_key(item)
                          for item in range(start, start + length)),
                ))
            else:
                slot = child.parent_slots[index] if child.parent_slots else index
                children.append(
                    (
                        child.local_name,
                        None if slot < 0 else child.specialization_key(slot),
                    )
                )
        return (self.instance_classes[index], tuple(children))

    def specialization_groups(self) -> tuple[tuple[int, ...], ...]:
        """Logical instance indexes grouped by identical recursive layout."""
        groups: list[list[int]] = []
        positions: dict[tuple[Any, ...], int] = {}
        for index in range(self.count):
            key = self.specialization_key(index)
            position = positions.get(key)
            if position is None:
                positions[key] = len(groups)
                groups.append([index])
            else:
                groups[position].append(index)
        return tuple(tuple(group) for group in groups)

    def specialization_slots(self) -> tuple[int, ...]:
        slots = [0] * self.count
        for variant, group in enumerate(self.specialization_groups()):
            for index in group:
                slots[index] = variant
        return tuple(slots)

    def walk(self) -> Iterator["_OwnerDecl"]:
        """This node and all of its descendants, depth-first."""
        yield self
        for child in self.children:
            yield from child.walk()

    def child(self, local_name: str) -> "_OwnerDecl | None":
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
        methods: dict[str, tuple[Callable[..., Any],
                                 _ProcessSpec]] = {}
        for cls in dict.fromkeys(self.instance_classes):
            for callback in cls._callbacks().processes:
                methods.setdefault(callback.name, (callback.fn, callback.spec))
        for method_name in methods:
            # A method compiled once for every instance registers under
            # the decl name; instance-specialized methods (spawnables,
            # the per-instance fallback) under the per-instance names.
            candidates = (
                f"{self.name}__{method_name}",
                *(f"{prefix}__{method_name}" for prefix in self.process_names),
            )
            members.extend(
                f"process:{candidate}"
                for candidate in candidates
                if candidate in process_names
            )
        for field_decl in self.decls.fields.values():
            if not field_decl.kind.dag_entity or field_decl.name in self.aliased_fields:
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
    decls = cls._field_declarations(allow_symbolic_pqueues=True, allow_refs=True)
    cls._bind_callbacks(decls, owner="component")
    for field_decl in decls.fields.values():
        if not field_decl.kind.on_component:
            raise ValueError(
                f"component '{cls.__name__}' declares {field_decl.kind.name} "
                "fields, which are not supported yet")
    return decls


def _merge_component_declarations(
    component_name: str,
    classes: Sequence[type[Component]],
) -> tuple[_Declarations, tuple[_Declarations, ...]]:
    """Merge the declarations of concrete classes sharing one logical path.

    Per-instance declarations are retained for ownership checks. The merged
    declarations drive the flattened schema; structurally incompatible fields
    that would otherwise receive the same flattened name are rejected.
    """
    per_instance = tuple(_component_declarations(cls) for cls in classes)
    merged = _Declarations()
    field_sources: dict[str, type[Component]] = {}
    for cls, decls in zip(classes, per_instance):
        for name, candidate in decls.fields.items():
            existing = merged.fields.get(name)
            if existing is None:
                merged.add(candidate)
                field_sources[name] = cls
                continue
            compatible = (
                existing.kind.name == candidate.kind.name
                and existing.capacity == candidate.capacity
                and existing.count == candidate.count
            )
            if not compatible:
                other = field_sources[name]
                raise TypeError(
                    f"component '{component_name}' concrete classes "
                    f"{other.__name__} and {cls.__name__} declare "
                    f"incompatible field '{name}'")
        for name, target in decls.refs.items():
            existing = merged.refs.get(name, _MISSING)
            if existing is not _MISSING and existing != target:
                raise TypeError(
                    f"component '{component_name}' concrete classes declare "
                    f"incompatible Ref field '{name}'")
            merged.refs[name] = target
        for name, target in decls.ref_tables.items():
            existing = merged.ref_tables.get(name, _MISSING)
            if existing is not _MISSING and existing != target:
                raise TypeError(
                    f"component '{component_name}' concrete classes declare "
                    f"incompatible Refs field '{name}'")
            merged.ref_tables[name] = target
        for name, ctype in decls.consts.items():
            existing = merged.consts.get(name, _MISSING)
            if existing is not _MISSING and existing != ctype:
                raise TypeError(
                    f"component '{component_name}' concrete classes declare "
                    f"incompatible Const field '{name}'")
            merged.consts[name] = ctype
    return merged, per_instance


def _owner_slots(
    count: int,
    owners: Sequence[int],
) -> tuple[int, ...]:
    slots = [-1] * count
    for slot, owner in enumerate(owners):
        slots[owner] = slot
    return tuple(slots)


def _component_field_map(name: str, decls: _Declarations) -> dict[str, str]:
    return {fname: f"{name}__{fname}" for fname in decls.fields}


def _primitive_constant(value: Any) -> bool:
    return type(value) in (bool, int, float)


def _polymorphic_component_constants(
    items: Sequence[Component],
    field_map: Mapping[str, str],
    exclude: frozenset[str],
) -> tuple[dict[str, tuple[Any, ...]], dict[str, tuple[int, ...]]]:
    """Capture primitive attributes over only the instances that own them."""
    constants: dict[str, tuple[Any, ...]] = {}
    owners_by_name: dict[str, tuple[int, ...]] = {}
    names = {
        name
        for item in items
        for name in vars(item)
        if (not name.startswith("_") and name not in field_map
            and name not in exclude)
    }
    for name in names:
        owned = tuple(
            index for index, item in enumerate(items)
            if (value := getattr(item, name, _MISSING)) is not _MISSING
            and _primitive_constant(value)
        )
        if not owned:
            continue
        values = tuple(getattr(items[index], name) for index in owned)
        constants[name] = values
        owners_by_name[name] = owned
    return constants, owners_by_name


def _validate_component_consts(
    component_name: str,
    templates: Sequence[Component],
    consts: Mapping[str, type],
    owners_by_name: Mapping[str, tuple[int, ...]] | None = None,
) -> tuple[dict[str, tuple[Any, ...]], dict[str, tuple[int, ...]]]:
    """Per-instance values of the declared ``sim.Const`` fields, checked
    to be present and of the annotated type on every instance."""
    values: dict[str, tuple[Any, ...]] = {}
    resolved_owners: dict[str, tuple[int, ...]] = {}
    for fname, ctype in consts.items():
        instance_values = []
        owners = ((tuple(range(len(templates)))
                   if owners_by_name is None else owners_by_name[fname]))
        for index in owners:
            template = templates[index]
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
        resolved_owners[fname] = tuple(owners)
    return values, resolved_owners


def _component_param_defaults(
    component_name: str,
    templates: Sequence[Component],
    decls: _Declarations,
    field_owners: Mapping[str, tuple[int, ...]] | None = None,
) -> dict[str, tuple[float, ...]]:
    """Capture Param defaults from class attributes or component instances.

    A shaped flattened parameter can only be omitted as a whole, so every
    instance covered by a component declaration must either provide the
    default or leave it required.
    """
    defaults: dict[str, tuple[float, ...]] = {}
    for fname in decls.names("param"):
        owners = (tuple(range(len(templates))) if field_owners is None
                  else field_owners[fname])
        values = tuple(
            getattr(templates[index], fname, _MISSING) for index in owners)
        present = tuple(value is not _MISSING for value in values)
        if not any(present):
            continue
        if not all(present):
            raise ValueError(
                f"component '{component_name}' Param '{fname}' must have "
                "a default on every instance or none")
        defaults[fname] = tuple(
            _param_default(
                value, f"component '{component_name}' Param '{fname}'")
            for value in values
        )
    return defaults


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
    field_owners: Mapping[str, tuple[int, ...]] | None = None,
    constant_slots: Mapping[str, tuple[int, ...]] | None = None,
) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
    """Per-instance element counts and start offsets of each PQueues
    field; symbolic counts name a per-instance int constant."""
    counts_by_field: dict[str, tuple[int, ...]] = {}
    offsets_by_field: dict[str, tuple[int, ...]] = {}
    for field_decl in decls.by_kind("pqueues"):
        fname = field_decl.name
        owners = (tuple(range(instance_count)) if field_owners is None
                  else field_owners[fname])
        count_decl = field_decl.count
        if isinstance(count_decl, int):
            owned_counts: tuple[Any, ...] = (count_decl,) * len(owners)
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
            if constant_slots is None:
                owned_counts = values
            else:
                slots = constant_slots[count_decl]
                owned_counts = tuple(values[slots[index]] for index in owners)
        owned_counts_tuple, owned_offsets = _offsets_from_counts(owned_counts)
        counts = [0] * instance_count
        offsets = [0] * instance_count
        for owner, count, offset in zip(
                owners, owned_counts_tuple, owned_offsets):
            counts[owner] = count
            offsets[owner] = offset
        counts_by_field[fname] = tuple(counts)
        offsets_by_field[fname] = tuple(offsets)
    return counts_by_field, offsets_by_field


def _resolve_component_processes(
    component_name: str,
    cls: type[Component],
    templates: Sequence[Component],
    decls: _Declarations,
    instance_classes: Sequence[type[Component]] | None = None,
    field_owners: Mapping[str, tuple[int, ...]] | None = None,
) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
    """Per-instance copy counts and handle offsets of Processes fields."""
    classes = tuple(instance_classes or (cls,) * len(templates))
    counts_by_field: dict[str, tuple[int, ...]] = {}
    offsets_by_field: dict[str, tuple[int, ...]] = {}
    for fname in decls.names("processes"):
        owners = (tuple(range(len(templates))) if field_owners is None
                  else field_owners[fname])
        owned_counts: list[int] = []
        for index in owners:
            specs = [
                callback.spec
                for callback in classes[index]._callbacks().processes
                if callback.spec.field == fname
            ]
            spec = specs[0] if len(specs) == 1 else None
            if spec is None:
                raise ValueError(
                    f"component '{component_name}' Processes field '{fname}' must be bound by exactly one @sim.process(field=...)"
                )
            owned_counts.append(
                spec.resolve_copies(templates[index], f"{component_name}.{fname}")
            )
        resolved_counts, resolved_offsets = _offsets_from_counts(owned_counts)
        counts = [0] * len(templates)
        offsets = [0] * len(templates)
        for owner, count, offset in zip(
                owners, resolved_counts, resolved_offsets):
            counts[owner] = count
            offsets[owner] = offset
        counts_by_field[fname] = tuple(counts)
        offsets_by_field[fname] = tuple(offsets)
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
    """Build an owner tree, then flatten it after wiring is resolved."""

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
                                     (item_cls,), is_collection,
                                     owner_positions=(0,), parent_count=1)
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
        declared_classes: Sequence[type[Component]],
        collection: bool,
        *,
        owner_positions: Sequence[int],
        parent_count: int,
    ) -> _OwnerDecl:
        """Build one component field's decl, gathering its instances from
        each owner (the model class for a root, the parent's instances for
        a child) and deriving the flattened name, per-instance process-name
        prefixes, and display paths from the parent context."""
        label = f"{parent_name}.{fname}" if parent_name else fname
        name = f"{parent_name}__{fname}" if parent_name else fname
        templates: list[Component] = []
        process_names: list[str] = []
        offsets: list[int] = [0] * parent_count if collection else []
        lengths: list[int] = [0] * parent_count if collection else []
        parent_slots = [-1] * parent_count
        for owner, prefix, owner_position, declared_cls in zip(
                owners, prefixes, owner_positions, declared_classes):
            base = fname if prefix is None else f"{prefix}__{fname}"
            if collection:
                items = self._collection_default(
                    owner, fname, label, declared_cls)
                offsets[owner_position] = len(templates)
                lengths[owner_position] = len(items)
                templates.extend(items)
                process_names.extend(f"{base}__{i}" for i in range(len(items)))
            else:
                parent_slots[owner_position] = len(templates)
                templates.append(
                    self._instance_default(owner, fname, label, declared_cls))
                process_names.append(base)

        display = f"{parent_display}.{fname}" if parent_name else fname
        item_display = f"{display}[]" if collection else display
        return self._build(
            local_name=fname, name=name,
            declared_cls=declared_classes[0],
            templates=tuple(templates), process_names=tuple(process_names),
            display_name=display, item_display_name=item_display,
            collection=collection,
            parent_offsets=tuple(offsets) if collection else (),
            parent_lengths=tuple(lengths) if collection else (),
            parent_slots=tuple(parent_slots) if not collection else ())

    def _build(
        self,
        *,
        local_name: str,
        name: str,
        declared_cls: type[Component],
        templates: tuple[Component, ...],
        process_names: tuple[str, ...],
        display_name: str,
        item_display_name: str,
        collection: bool,
        parent_offsets: tuple[int, ...] = (),
        parent_lengths: tuple[int, ...] = (),
        parent_slots: tuple[int, ...] = (),
    ) -> _OwnerDecl:
        instance_classes = tuple(type(template) for template in templates)
        cls = (instance_classes[0] if len(set(instance_classes)) == 1
               else declared_cls)
        decls, per_instance_decls = _merge_component_declarations(
            name, instance_classes)
        field_owners = {
            fname: tuple(
                index for index, own in enumerate(per_instance_decls)
                if fname in own.fields)
            for fname in decls.fields
        }
        field_slots = {
            fname: _owner_slots(len(templates), owners)
            for fname, owners in field_owners.items()
        }
        direct_field_map = _component_field_map(name, decls)
        wiring_raw = self._field_wiring(name, templates, decls)
        component_refs = _component_ref_values(name, templates, decls)
        param_defaults = _component_param_defaults(
            name, templates, decls, field_owners)
        const_owners_declared = {
            fname: tuple(
                index for index, own in enumerate(per_instance_decls)
                if fname in own.consts)
            for fname in decls.consts
        }
        const_values, declared_const_owners = _validate_component_consts(
            name, templates, decls.consts, const_owners_declared)
        implicit_constants, implicit_owners = \
            _polymorphic_component_constants(
                templates, direct_field_map,
                exclude=frozenset(component_refs) | set(decls.consts))
        constants = {
            **const_values,
            **implicit_constants,
        }
        constant_owners = {**declared_const_owners, **implicit_owners}
        constant_slots = {
            fname: _owner_slots(len(templates), owners)
            for fname, owners in constant_owners.items()
        }
        pqueue_counts, pqueue_offsets = _resolve_component_pqueues(
            name, len(templates), decls, constants, field_owners,
            constant_slots)
        process_counts, process_offsets = _resolve_component_processes(
            name, cls, templates, decls, instance_classes, field_owners)
        child_specs: dict[
            str, list[tuple[int, type[Component], bool]]] = {}
        child_order: list[str] = []
        for index, instance_cls in enumerate(instance_classes):
            for child_name, child_cls, child_collection in \
                    _component_fields(instance_cls):
                if child_name not in child_specs:
                    child_specs[child_name] = []
                    child_order.append(child_name)
                child_specs[child_name].append(
                    (index, child_cls, child_collection))
        children_list: list[_OwnerDecl] = []
        for child_name in child_order:
            specs = child_specs[child_name]
            collection_values = {spec[2] for spec in specs}
            if len(collection_values) != 1:
                raise TypeError(
                    f"component '{name}' concrete classes declare "
                    f"'{child_name}' as both a component and a collection")
            positions = tuple(spec[0] for spec in specs)
            children_list.append(self._build_field(
                tuple(templates[index] for index in positions),
                tuple(process_names[index] for index in positions),
                name, item_display_name, child_name,
                tuple(spec[1] for spec in specs),
                specs[0][2], owner_positions=positions,
                parent_count=len(templates)))
        children = tuple(children_list)
        return _OwnerDecl(
            name=name,
            cls=cls,
            declared_cls=declared_cls,
            instance_classes=instance_classes,
            collection=collection,
            instances=templates,
            decls=decls,
            local_name=local_name,
            process_names=process_names,
            display_name=display_name,
            item_display_name=item_display_name,
            direct_field_map=direct_field_map,
            constants=constants,
            param_defaults=param_defaults,
            pqueue_counts=pqueue_counts,
            pqueue_offsets=pqueue_offsets,
            process_counts=process_counts,
            process_offsets=process_offsets,
            component_refs=component_refs,
            wiring_raw=wiring_raw,
            children=children,
            parent_offsets=parent_offsets,
            parent_lengths=parent_lengths,
            parent_slots=parent_slots,
            field_owners=field_owners,
            field_slots=field_slots,
            constant_owners=constant_owners,
            constant_slots=constant_slots,
        )

    def flatten(self, decl: _OwnerDecl) -> None:
        """Append one built (and wiring-resolved) node's declarations to
        the model-level target under their flattened names; multi-instance
        decls declare shaped fields with one element per instance. Wired
        (aliased) fields name the target's entity and declare nothing of
        their own."""
        aliased = set(decl.aliased_fields)
        for field_decl in decl.decls.fields.values():
            fname = field_decl.name
            owner_count = len(decl.field_owners[fname])
            shape = (owner_count,) if owner_count > 1 else None
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
                capacity_slots = None
                if (isinstance(field_decl.capacity, str)
                        and decl.decls.kind_of(field_decl.capacity) == "param"):
                    param_slots = decl.field_slots[field_decl.capacity]
                    slots = tuple(
                        param_slots[owner]
                        for owner in decl.field_owners[fname])
                    if any(slot < 0 for slot in slots):
                        raise ValueError(
                            f"component '{decl.name}' field '{fname}' "
                            f"capacity '{field_decl.capacity}' is not "
                            "declared by every owning concrete type")
                    capacity_slots = slots
                default = field_decl.default
                if kind.name == "param" and fname in decl.param_defaults:
                    values = decl.param_defaults[fname]
                    default = values[0] if decl.count == 1 else values
                self.target.add(_FieldDecl(flat_name, kind,
                                           capacity=capacity,
                                           capacity_slots=capacity_slots,
                                           shape=shape,
                                           default=default))

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
    decls = cls._field_declarations()
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


def _owner_declaration(cls: type, decls: _Declarations) -> _OwnerDecl:
    """Represent the Model root in the same declaration tree as Components."""
    owner_decls = _Declarations()
    owner_decls.fields.update(decls.fields)
    for name, _format in _STANDARD_FIELDS:
        kind = _FIELD_KINDS["state" if name == "seed" else "fstate"]
        owner_decls.fields[name] = _FieldDecl(name, kind)
    fields = tuple(owner_decls.fields)
    pqueue_offsets = {name: (0,) for name in owner_decls.names("pqueues")}
    process_offsets = {name: (0,) for name in owner_decls.names("processes")}
    return _OwnerDecl(
        name="model",
        cls=cls,
        declared_cls=cls,
        instance_classes=(cls,),
        collection=False,
        instances=(None,),
        decls=owner_decls,
        local_name="model",
        process_names=("model",),
        display_name="model",
        item_display_name="model",
        direct_field_map={name: name for name in fields},
        constants={},
        param_defaults={},
        pqueue_counts={},
        pqueue_offsets=pqueue_offsets,
        process_counts={},
        process_offsets=process_offsets,
        component_refs={},
        children=tuple((*decls.components, *decls.component_collections)),
        field_owners={name: (0,) for name in fields},
        field_slots={name: (0,) for name in fields},
        owner_root=True,
    )


# -- Entity wiring resolution: runs after the whole tree is built, so a
# field may be wired to a target declared later, and chains of wirings
# resolve through to the entity that actually backs them.


def _resolve_component_wiring(roots: Sequence[_OwnerDecl]) -> None:
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


def _instance_identity(roots: Sequence[_OwnerDecl]) -> dict[int, Any]:
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
    decl: _OwnerDecl,
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


def _resolve_component_refs(roots: Sequence[_OwnerDecl]) -> None:
    """Resolve raw Ref/Refs targets to (decl, item index) pairs."""
    identity = _instance_identity(roots)
    for root in roots:
        for decl in root.walk():
            for ref in decl.component_refs.values():
                _resolve_component_ref_decl(decl, ref, identity)


def _resolve_component_ref_target(
    decl: _OwnerDecl,
    ref: _ComponentRefDecl,
    instance: Component,
    identity: Mapping[int, Any],
) -> tuple[_OwnerDecl, int | None]:
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
    decl: _OwnerDecl, ref: _ComponentRefDecl, identity: Mapping[int, Any]
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
        if any(target[0] is not first for target in entries):
            raise ValueError(
                f"component '{decl.name}' refs table '{ref.name}' entries "
                "must all be items of a single component collection")
        if not first.collection:
            raise ValueError(
                f"component '{decl.name}' refs table '{ref.name}' entries "
                f"must all be items of a single component collection; "
                f"'{first.name}' is a lone component, not a collection")
        ref.table_decl = first
    ref.table_indices = tuple(
        # A one-item collection stores its fields unindexed, so identity
        # reports no item index for it; the table still addresses item 0.
        0 if target[1] is None else target[1]
        for table in resolved for target in table)
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


def _collection_lengths_symbol(component: str) -> str:
    return f"_CIMBA_LEN_{component}"


def _component_slots_symbol(component: str) -> str:
    return f"_CIMBA_COMPSLOT_{component}"


def _field_slots_symbol(component: str, field_name: str) -> str:
    return f"_CIMBA_FIELDSLOT_{component}__{field_name}"


def _constant_slots_symbol(component: str, field_name: str) -> str:
    return f"_CIMBA_CONSTSLOT_{component}__{field_name}"


def _variant_slots_symbol(component: str) -> str:
    return f"_CIMBA_VARIANT_{component}"


def _ref_index_symbol(component: str, name: str) -> str:
    return f"_CIMBA_REFIDX_{component}__{name}"


def _ref_table_symbol(component: str, name: str) -> str:
    return f"_CIMBA_REFTAB_{component}__{name}"


def _ref_offsets_symbol(component: str, name: str) -> str:
    return f"_CIMBA_REFOFF_{component}__{name}"


def _ref_lengths_symbol(component: str, name: str) -> str:
    return f"_CIMBA_REFLEN_{component}__{name}"


def _lowering_namespace(components: Iterable[_OwnerDecl]) -> dict[str, Any]:
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
                slots = decl.constant_slots.get(name, ())
                if slots and slots != tuple(range(len(slots))):
                    namespace[_constant_slots_symbol(decl.name, name)] = \
                        np.asarray(slots, dtype=np.int64)
            for fname, slots in decl.field_slots.items():
                if slots != tuple(range(len(slots))):
                    namespace[_field_slots_symbol(decl.name, fname)] = \
                        np.asarray(slots, dtype=np.int64)
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
            if decl.collection and len(decl.parent_lengths) > 1:
                namespace[_collection_lengths_symbol(decl.name)] = np.asarray(
                    decl.parent_lengths, dtype=np.int64)
            if decl.parent_slots:
                namespace[_component_slots_symbol(decl.name)] = np.asarray(
                    decl.parent_slots, dtype=np.int64)
            if decl.polymorphic:
                namespace[_variant_slots_symbol(decl.name)] = np.asarray(
                    decl.specialization_slots(), dtype=np.int64)
            for name, ref in decl.component_refs.items():
                if ref.table:
                    if ref.table_decl is not None:
                        namespace[_ref_table_symbol(decl.name, name)] = \
                            np.asarray(ref.table_indices, dtype=np.int64)
                        stack.append(ref.table_decl)
                    if len(ref.table_offsets) > 1:
                        namespace[_ref_offsets_symbol(decl.name, name)] = \
                            np.asarray(ref.table_offsets, dtype=np.int64)
                    if len(ref.table_lengths) > 1:
                        namespace[_ref_lengths_symbol(decl.name, name)] = \
                            np.asarray(ref.table_lengths, dtype=np.int64)
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
# define the path roots: `self` inside component methods and model callbacks.

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
class _OwnerAccess:
    """A resolved component-instance path: the decl plus the instance
    index expression (None when the decl has a single instance)."""

    decl: _OwnerDecl
    index: ast.expr | None
    text: str
    #: logical indexes this access may select when ``index`` is dynamic.
    possible_indices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class _FieldAccess:
    """A resolved path to a declared field or captured constant."""

    decl: _OwnerDecl
    index: ast.expr | None
    field: str
    text: str
    possible_indices: tuple[int, ...] | None = None


@dataclass
class _FunctionDependency:
    """One owner value threaded into a compiled function helper.

    Normally a scalar the caller reads at the call site. When the helper
    indexes a collection with a value it computes itself (a loop target
    or a local), the caller cannot pick the element, so ``array`` threads
    the whole flattened field and the helper subscripts it."""

    access: _FieldAccess
    parameter: str
    direct: bool = True
    array: bool = False


@dataclass
class _FunctionSpec:
    """A Model/Component synchronous function lowered to one helper."""

    decl: _OwnerDecl
    name: str
    method: Callable[..., Any]
    graph_name: str
    symbol: str
    parameter_names: tuple[str, ...]
    argument_types: tuple[Any, ...]
    return_type: Any
    dependencies: tuple[_FunctionDependency, ...]
    helper: Any
    callees: tuple[str, ...]
    receiver_indexed: bool = False
    #: None for the homogeneous shared helper; otherwise the recursive
    #: specialization ordinal and its logical instance indexes.
    variant: int | None = None
    instance_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class _RefTableAccess:
    """A resolved path to a Refs table, before indexing."""

    parent: _OwnerAccess
    name: str
    ref: _ComponentRefDecl
    text: str


class _OwnerPathLowerer(ast.NodeTransformer):
    #: When set (method lowering with a runtime instance index), literal
    #: indices into a Refs table are checked against every possible instance.
    strict_ref_tables = False

    def __init__(
        self, *, env_name: str, functions: Mapping[str, _FunctionSpec] | None = None
    ):
        self.env_name = env_name
        self.functions = functions or {}
        self.called_functions: set[str] = set()
        self._ref_loop_tables: list[tuple[str, str, str | None, str]] = []

    # -- path roots, defined by the subclasses -------------------------------

    def _root_namespace_ref(self, node: ast.AST) -> _OwnerAccess | None:
        return None

    def _root_collection_ref(self, node: ast.AST) -> _OwnerAccess | None:
        return None

    def _callback_label(self) -> str:
        raise NotImplementedError

    @staticmethod
    def _possible_positions(access: _OwnerAccess) -> tuple[int, ...]:
        if isinstance(access.index, ast.Constant) and type(access.index.value) is int:
            return (access.index.value,)
        if access.possible_indices is not None:
            return access.possible_indices
        return tuple(range(access.decl.count))

    @staticmethod
    def _ref_table_key(table: _RefTableAccess) -> tuple[str, str, str | None]:
        index = table.parent.index
        return (
            table.parent.decl.name,
            table.name,
            None if index is None else ast.dump(index),
        )

    def _lower_len_call(self, node: ast.Call) -> ast.expr | None:
        """Lower ``len`` for a resolved component collection or Refs table."""
        if (not isinstance(node.func, ast.Name)
                or node.func.id != "len"
                or len(node.args) != 1
                or node.keywords):
            return None
        collection = self._collection_ref(node.args[0])
        if collection is not None:
            return ast.copy_location(
                self._instance_table_expr(
                    collection.decl.parent_lengths,
                    collection.index,
                    _collection_lengths_symbol(collection.decl.name),
                    "component collection",
                ),
                node,
            )
        table = self._ref_table_ref(node.args[0])
        if table is not None:
            return ast.copy_location(
                self._instance_table_expr(
                    table.ref.table_lengths,
                    table.parent.index,
                    _ref_lengths_symbol(table.parent.decl.name,
                                        table.name),
                    "Refs table",
                ),
                node,
            )
        return None

    # -- path resolution -------------------------------------------------------

    def _present_position(
        self,
        parent: _OwnerAccess,
        values: Sequence[int],
        text: str,
        absent: Callable[[int], bool],
    ) -> int | None:
        """Validate a polymorphic child and return a static parent slot."""
        position = (
            parent.index.value
            if isinstance(parent.index, ast.Constant)
            and type(parent.index.value) is int
            else (0 if parent.index is None else None)
        )
        if position is not None:
            if absent(values[position]):
                raise ValueError(
                    f"{self._callback_label()} accesses {text}, which is not declared by that concrete component type"
                )
        elif any(absent(values[item]) for item in self._possible_positions(parent)):
            raise ValueError(
                f"{self._callback_label()} dynamically accesses {text}, which is not declared by every concrete component type"
            )
        return position

    def _namespace_ref(self, node: ast.AST) -> _OwnerAccess | None:
        root = self._root_namespace_ref(node)
        if root is not None:
            return root

        if isinstance(node, ast.Subscript):
            collection = self._collection_ref(node.value)
            if collection is not None:
                index = self._collection_item_index(
                    collection.decl, collection.index, node.slice
                )
                possible = None
                if not (isinstance(index, ast.Constant)
                        and type(index.value) is int):
                    if collection.index is None:
                        possible = tuple(range(collection.decl.count))
                    else:
                        possible_items: list[int] = []
                        for parent_index in self._possible_positions(
                                collection):
                            start = collection.decl.parent_offsets[parent_index]
                            length = collection.decl.parent_lengths[parent_index]
                            possible_items.extend(
                                range(start, start + length))
                        possible = tuple(possible_items)
                return _OwnerAccess(
                    collection.decl, index, f"{collection.text}[...]", possible
                )
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
                text = f"{parent.text}.{node.attr}"
                if not child.parent_slots:
                    index = parent.index if child.count > 1 else None
                else:
                    position = self._present_position(
                        parent, child.parent_slots, text, lambda slot: slot < 0
                    )
                    index = (
                        ast.Constant(child.parent_slots[position])
                        if position is not None and child.count > 1
                        else (
                            None
                            if position is not None
                            else _subscript(
                                ast.Name(
                                    id=_component_slots_symbol(child.name),
                                    ctx=ast.Load(),
                                ),
                                parent.index,
                                ast.Load(),
                            )
                        )
                    )
                possible = None
                if not (isinstance(index, ast.Constant)
                        and type(index.value) is int):
                    possible = tuple(
                        child.parent_slots[position]
                        for position in self._possible_positions(parent)
                    )
                return _OwnerAccess(child, index, text, possible)
            ref = parent.decl.component_refs.get(node.attr)
            if ref is not None and not ref.table:
                return self._ref_namespace(parent, node.attr, ref)
            return None

        return None

    def _collection_ref(self, node: ast.AST) -> _OwnerAccess | None:
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
            if child.parent_lengths:
                self._present_position(
                    parent,
                    child.parent_lengths,
                    f"{parent.text}.{node.attr}",
                    lambda length: length == 0,
                )
            return _OwnerAccess(
                child,
                parent.index,
                f"{parent.text}.{node.attr}",
                parent.possible_indices,
            )

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
        self, parent: _OwnerAccess, name: str, ref: _ComponentRefDecl
    ) -> _OwnerAccess:
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
                    f"{self._callback_label()} dereferences {text}, which has no target for this instance"
                )
            target_decl, target_index = target
            target_expr = None if target_index is None else ast.Constant(target_index)
            return _OwnerAccess(target_decl, target_expr, text)
        parent_possible = self._possible_positions(parent)
        if any(ref.targets[position] is None
               for position in parent_possible):
            raise ValueError(
                f"{self._callback_label()} dereferences {text} with a dynamic instance index, but some instances have no target"
            )
        targets = [ref.targets[position] for position in parent_possible]
        first = targets[0][0]
        if any(target[0] is not first for target in targets):
            raise ValueError(
                f"{self._callback_label()} dereferences {text} with a "
                "dynamic instance index, which requires every instance to "
                "reference the same component declaration"
            )
        if first.count <= 1:
            return _OwnerAccess(first, None, text, (0,))
        lookup = _subscript(
            ast.Name(id=_ref_index_symbol(parent.decl.name, name), ctx=ast.Load()),
            index,
            ast.Load(),
        )
        return _OwnerAccess(first, lookup, text, tuple(target[1] for target in targets))

    def _ref_table_item(
        self, table: _RefTableAccess, item_slice: ast.expr
    ) -> _OwnerAccess:
        """Index a Refs table: a static target when both the instance and
        the entry are known, else a lookup through the REFTAB table."""
        item_index = self.visit(copy.deepcopy(item_slice))
        if not isinstance(item_index, ast.expr):
            raise TypeError("component refs table index did not lower to an expression")
        ref = table.ref
        text = f"{table.text}[...]"
        parent_index = table.parent.index
        parent_pos: int | None
        if parent_index is None:
            parent_pos = 0
        elif isinstance(parent_index, ast.Constant) and type(parent_index.value) is int:
            parent_pos = parent_index.value
        else:
            parent_pos = None

        if (parent_pos is not None and isinstance(item_index, ast.Constant)
                and type(item_index.value) is int):
            length = ref.table_lengths[parent_pos]
            position = item_index.value
            if not 0 <= position < length:
                raise ValueError(
                    f"{self._callback_label()} index {position} is out of range for {table.text} (length {length})"
                )
            target_index = ref.table_indices[ref.table_offsets[parent_pos] + position]
            return _OwnerAccess(ref.table_decl, ast.Constant(target_index), text)

        if ref.table_decl is None:
            raise ValueError(
                f"{self._callback_label()} indexes {table.text}, which has no entries"
            )
        if parent_pos is None and self.strict_ref_tables:
            lengths = tuple(
                ref.table_lengths[position]
                for position in self._possible_positions(table.parent)
            )
            loop_bound = (
                isinstance(item_index, ast.Name)
                and (*self._ref_table_key(table), item_index.id)
                in self._ref_loop_tables
            )
            if len(set(lengths)) > 1 and not loop_bound:
                raise ValueError(
                    f"{self._callback_label()} indexes {table.text}, whose per-instance lengths differ"
                )
            if (
                isinstance(item_index, ast.Constant)
                and type(item_index.value) is int
                and any(
                    not 0 <= item_index.value < ref.table_lengths[position]
                    for position in self._possible_positions(table.parent)
                )
            ):
                raise ValueError(
                    f"{self._callback_label()} index {item_index.value} is out of range for {table.text} (lengths {lengths})"
                )
        if parent_pos is not None:
            offset: ast.expr = ast.Constant(ref.table_offsets[parent_pos])
        else:
            offset = _subscript(
                ast.Name(
                    id=_ref_offsets_symbol(table.parent.decl.name, table.name),
                    ctx=ast.Load(),
                ),
                parent_index,
                ast.Load(),
            )
        lookup = _subscript(
            ast.Name(
                id=_ref_table_symbol(table.parent.decl.name, table.name), ctx=ast.Load()
            ),
            _add(offset, item_index),
            ast.Load(),
        )
        return _OwnerAccess(
            ref.table_decl, lookup, text, tuple(dict.fromkeys(ref.table_indices))
        )

    def _field_ref(self, node: ast.AST) -> _FieldAccess | None:
        if not isinstance(node, ast.Attribute):
            return None
        namespace = self._namespace_ref(node.value)
        if namespace is None:
            return None
        field_name = node.attr
        if (
            field_name in namespace.decl.direct_field_map
            or field_name in namespace.decl.constants
        ):
            return _FieldAccess(
                namespace.decl,
                namespace.index,
                field_name,
                f"{namespace.text}.{field_name}",
                namespace.possible_indices,
            )
        if namespace.decl.child(field_name) is not None:
            return None
        if field_name in namespace.decl.component_refs:
            return None
        self._raise_unknown_field(namespace, field_name)

    def _collection_item_index(
        self, decl: _OwnerDecl, parent_index: ast.expr | None, item_index: ast.expr
    ) -> ast.expr:
        """The flattened instance index of a collection item: the item
        index plus the parent instance's start offset."""
        index = self.visit(copy.deepcopy(item_index))
        if not isinstance(index, ast.expr):
            raise TypeError("component collection index did not lower to an expression")
        if isinstance(index, ast.Constant) and type(index.value) is int:
            length: int | None = None
            if len(decl.parent_lengths) <= 1:
                length = decl.parent_lengths[0] if decl.parent_lengths else 0
            elif (isinstance(parent_index, ast.Constant)
                  and type(parent_index.value) is int):
                length = decl.parent_lengths[parent_index.value]
            if length is not None and not 0 <= index.value < length:
                raise ValueError(
                    f"{self._callback_label()} collection index {index.value} is out of range (length {length})"
                )
        if len(decl.parent_offsets) <= 1:
            offset_value = decl.parent_offsets[0] if decl.parent_offsets else 0
            if offset_value == 0:
                return index
            return _add(ast.Constant(offset_value), index)
        if parent_index is None:
            raise TypeError("nested component collection has no parent index")
        if isinstance(parent_index, ast.Constant) and type(parent_index.value) is int:
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
        self, values: Sequence[Any], index: ast.expr | None, symbol: str, what: str
    ) -> ast.expr:
        """A per-instance value: a constant when the instance is known,
        else an element of the numpy array published under `symbol`."""
        if len(values) == 1:
            return ast.Constant(values[0])
        if isinstance(index, ast.Constant) and type(index.value) is int:
            return ast.Constant(values[index.value])
        if index is None:
            raise TypeError(f"component {what} has no instance index")
        return _subscript(ast.Name(id=symbol, ctx=ast.Load()), index,
                          ast.Load())

    def _field_target(self, access: _FieldAccess, ctx: ast.expr_context) -> ast.expr:
        flat_name = access.decl.direct_field_map[access.field]
        target = _env_attr(self.env_name, flat_name, ctx)
        owners = access.decl.field_owners[access.field]
        if len(owners) <= 1:
            self._owned_index(access, owners)
            return target
        if access.index is None:
            raise TypeError("component field has no instance index")
        packed_index = self._owned_index(
            access,
            owners,
            access.decl.field_slots[access.field],
            _field_slots_symbol(access.decl.name, access.field),
        )
        assert packed_index is not None
        return _subscript(target, packed_index, ctx)

    def _owned_index(
        self,
        access: _FieldAccess,
        owners: tuple[int, ...],
        slots: tuple[int, ...] | None = None,
        slot_symbol: str | None = None,
    ) -> ast.expr | None:
        """Validate ownership and map a logical component index to storage."""
        index = access.index
        if isinstance(index, ast.Constant) and type(index.value) is int:
            if index.value not in owners:
                raise ValueError(
                    f"{self._callback_label()} accesses {access.text}, which is not declared by that concrete component type"
                )
            return ast.Constant(slots[index.value]) if slots is not None else index
        if index is not None:
            possible = set(access.possible_indices or range(access.decl.count))
            if not possible.issubset(owners):
                raise ValueError(
                    f"{self._callback_label()} dynamically accesses {access.text}, which is not declared by every concrete component type"
                )
        if index is None or slots is None or slots == tuple(range(len(slots))):
            return index
        assert slot_symbol is not None
        return _subscript(ast.Name(id=slot_symbol, ctx=ast.Load()), index, ast.Load())

    def _field_array_target(self, access: _FieldAccess) -> ast.expr:
        """The whole flattened field, for a component function helper that
        indexes it itself. Only reached for fields every instance of the
        collection declares, so logical index == storage slot."""
        flat_name = access.decl.direct_field_map[access.field]
        return _env_attr(self.env_name, flat_name, ast.Load())

    def _constant_expr(self, access: _FieldAccess) -> ast.expr:
        owners = access.decl.constant_owners[access.field]
        slots = access.decl.constant_slots[access.field]
        index = self._owned_index(
            access,
            owners,
            slots if len(owners) > 1 else None,
            _constant_slots_symbol(access.decl.name, access.field),
        )
        return self._instance_table_expr(
            access.decl.constants[access.field],
            index,
            _const_symbol(access.decl.name, access.field),
            "constant",
        )

    def _lower_indexed_field(
        self, access: _FieldAccess, node: ast.Subscript
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
            raise TypeError(f"component {what} index did not lower to an expression")
        offset = self._instance_table_expr(
            offsets, access.index, symbol, f"{what} field"
        )
        if isinstance(offset, ast.Constant) and offset.value == 0:
            index = item
        else:
            index = _add(offset, item)
        flat = _env_attr(self.env_name,
                         decl.direct_field_map[access.field], ast.Load())
        return ast.copy_location(_subscript(flat, index, node.ctx), node)

    def _raise_unknown_field(self, namespace: _OwnerAccess, field_name: str) -> None:
        kind = (
            "component collection field"
            if namespace.decl.collection
            else "component field"
        )
        raise ValueError(
            f"{self._callback_label()} references unknown {kind} {namespace.text}.{field_name}"
        )

    def _function_specs(
        self, namespace: _OwnerAccess, method_name: str
    ) -> tuple[_FunctionSpec, ...]:
        candidates = [
            spec
            for spec in self.functions.values()
            if spec.decl is namespace.decl and spec.name == method_name
        ]
        if not candidates:
            return ()
        if (isinstance(namespace.index, ast.Constant)
                and type(namespace.index.value) is int):
            return tuple(
                spec
                for spec in candidates
                if namespace.index.value in spec.instance_indices
            )
        shared = [spec for spec in candidates if spec.variant is None]
        if shared:
            return (shared[0],)
        possible = set(namespace.possible_indices
                       or range(namespace.decl.count))
        ordered = sorted(
            (
                spec
                for spec in candidates
                if possible.intersection(spec.instance_indices)
            ),
            key=lambda spec: spec.variant,
        )
        covered = {
            index for spec in ordered for index in spec.instance_indices
            if index in possible
        }
        if covered != possible:
            raise ValueError(
                f"{self._callback_label()} dynamically calls "
                f"{namespace.text}.{method_name}(), which is not declared "
                "as @sim.function by every concrete component type"
            )
        return tuple(ordered)

    @staticmethod
    def _substitute_expr(
        expression: ast.expr | None, replacements: Mapping[str, ast.expr]
    ) -> ast.expr | None:
        if expression is None:
            return None

        class Substitute(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.AST:
                replacement = replacements.get(node.id)
                if replacement is None:
                    return node
                return ast.copy_location(copy.deepcopy(replacement), node)

        result = Substitute().visit(copy.deepcopy(expression))
        if not isinstance(result, ast.expr):
            raise TypeError(
                "component function dependency did not lower to an expression"
            )
        return result

    def _lower_one_component_function_call(
        self, node: ast.Call, receiver: _OwnerAccess, spec: _FunctionSpec
    ) -> ast.Call:
        arguments = [self.visit(copy.deepcopy(arg)) for arg in node.args]
        replacements = {
            name: arg
            for name, arg in zip(spec.parameter_names, arguments)
        }
        if receiver.index is not None:
            replacements["__cimba_receiver_index"] = receiver.index

        dependency_args: list[ast.expr] = []
        for dependency in spec.dependencies:
            access = dependency.access
            bound = _FieldAccess(
                access.decl,
                self._substitute_expr(access.index, replacements),
                access.field,
                access.text,
                access.possible_indices,
            )
            if bound.field in bound.decl.constants:
                value = self._constant_expr(bound)
            elif dependency.array:
                value = self._field_array_target(bound)
            else:
                value = self._field_target(bound, ast.Load())
            dependency_args.append(value)
            replacements[dependency.parameter] = value

        self.called_functions.add(spec.graph_name)
        helper_args = list(arguments)
        if spec.receiver_indexed:
            if receiver.index is None:
                raise TypeError(
                    "indexed component function has no receiver index")
            helper_args.append(copy.deepcopy(receiver.index))
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id=spec.symbol, ctx=ast.Load()),
                args=[*helper_args, *dependency_args],
                keywords=[],
            ),
            node,
        )

    def _validate_function_call(self, node: ast.Call, spec: _FunctionSpec) -> None:
        if node.keywords:
            raise ValueError(
                f"{self._callback_label()} call to component function '{spec.graph_name}' must use positional arguments"
            )
        if len(node.args) != len(spec.parameter_names):
            raise ValueError(
                f"{self._callback_label()} call to component function '{spec.graph_name}' takes {len(spec.parameter_names)} argument(s), got {len(node.args)}"
            )

    def _dispatch_function_call(
        self,
        node: ast.Call,
        receiver: _OwnerAccess,
        specs: Sequence[_FunctionSpec],
        lower_one: Callable[[ast.Call, _OwnerAccess, _FunctionSpec], ast.Call],
    ) -> ast.expr:
        self._validate_function_call(node, specs[0])
        if len(specs) == 1:
            return lower_one(node, receiver, specs[0])
        first = specs[0]
        contract = (first.parameter_names, first.argument_types,
                    first.return_type)
        if any((spec.parameter_names, spec.argument_types, spec.return_type)
               != contract for spec in specs[1:]):
            raise TypeError(
                f"{self._callback_label()} dynamically calls {receiver.text}.{node.func.attr}(), whose concrete implementations have incompatible signatures"
            )
        if receiver.index is None:
            raise TypeError("polymorphic component function has no index")
        expression = lower_one(copy.deepcopy(node), receiver, specs[-1])
        variant_expr = _subscript(
            ast.Name(id=_variant_slots_symbol(receiver.decl.name), ctx=ast.Load()),
            copy.deepcopy(receiver.index),
            ast.Load(),
        )
        for spec in reversed(specs[:-1]):
            expression = ast.IfExp(
                test=ast.Compare(
                    left=copy.deepcopy(variant_expr),
                    ops=[ast.Eq()],
                    comparators=[ast.Constant(spec.variant)],
                ),
                body=lower_one(copy.deepcopy(node), receiver, spec),
                orelse=expression,
            )
        return ast.copy_location(expression, node)

    def _lower_component_function_call(
        self, node: ast.Call, receiver: _OwnerAccess, specs: Sequence[_FunctionSpec]
    ) -> ast.expr:
        return self._dispatch_function_call(
            node, receiver, specs, self._lower_one_component_function_call
        )

    # -- node visitors -----------------------------------------------------------

    def visit_For(self, node: ast.For) -> ast.AST:
        """Track the canonical ``range(len(refs))`` loop bound.

        The runtime table offset makes dynamic indexing valid for the
        current owner even when different owners have different table
        lengths.  The marker is intentionally scoped to the loop body so
        an index variable used after the loop does not inherit that proof.
        """
        loop_ref: tuple[str, str, str | None, str] | None = None
        if (isinstance(node.target, ast.Name)
                and isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"
                and len(node.iter.args) == 1
                and not node.iter.keywords
                and isinstance(node.iter.args[0], ast.Call)):
            length_call = node.iter.args[0]
            if (isinstance(length_call.func, ast.Name)
                    and length_call.func.id == "len"
                    and len(length_call.args) == 1
                    and not length_call.keywords):
                table = self._ref_table_ref(length_call.args[0])
                if table is not None:
                    decl_name, table_name, parent_key = \
                        self._ref_table_key(table)
                    loop_ref = (decl_name, table_name, parent_key,
                                node.target.id)

        node.target = self.visit(node.target)
        node.iter = self.visit(node.iter)
        if loop_ref is not None:
            self._ref_loop_tables.append(loop_ref)
        node.body = [self.visit(statement) for statement in node.body]
        if loop_ref is not None:
            self._ref_loop_tables.pop()
        node.orelse = [self.visit(statement) for statement in node.orelse]
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        lowered_len = self._lower_len_call(node)
        if lowered_len is not None:
            return lowered_len
        if isinstance(node.func, ast.Name) and node.func.id == "getattr" and node.args:
            target = (
                self._namespace_ref(node.args[0])
                or self._collection_ref(node.args[0])
                or self._ref_table_ref(node.args[0])
            )
            if target is not None:
                raise ValueError(
                    f"{self._callback_label()} uses dynamic getattr({target.text}, ...), which is not supported"
                )
        if isinstance(node.func, ast.Attribute):
            receiver = self._namespace_ref(node.func.value)
            if receiver is not None:
                specs = self._function_specs(receiver, node.func.attr)
                if specs:
                    return self._lower_component_function_call(node, receiver, specs)
            access = self._field_ref(node.func.value)
            if access is not None and access.decl.owner_root:
                node.args = [self.visit(arg) for arg in node.args]
                node.keywords = [
                    ast.keyword(arg=item.arg, value=self.visit(item.value))
                    for item in node.keywords
                ]
                return node
            if access is not None:
                kind = access.decl.decls.kind_of(access.field)
                if kind == "pqueues" or (
                    access.decl.decls.fields[access.field].kind.binding is None
                    and kind not in ("condition", "event")
                ):
                    raise ValueError(
                        f"{self._callback_label()} cannot call {access.text}.{node.func.attr}() inside compiled code"
                    )
                return ast.copy_location(
                    ast.Call(
                        func=ast.Attribute(
                            value=self._field_target(access, ast.Load()),
                            attr=node.func.attr,
                            ctx=ast.Load(),
                        ),
                        args=[self.visit(arg) for arg in node.args],
                        keywords=[
                            ast.keyword(arg=item.arg, value=self.visit(item.value))
                            for item in node.keywords
                        ],
                    ),
                    node,
                )
            target = (
                self._namespace_ref(node.func.value)
                or self._collection_ref(node.func.value)
                or self._ref_table_ref(node.func.value)
            )
            if target is not None:
                raise ValueError(
                    f"{self._callback_label()} cannot call {target.text}.{node.func.attr}() inside compiled code"
                )
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
                f"{self._callback_label()} uses {collection.text}[...] directly; access one of its fields"
            )
        table = self._ref_table_ref(node.value)
        if table is not None:
            raise ValueError(
                f"{self._callback_label()} uses {table.text}[...] directly; access one of its fields"
            )
        return self.generic_visit(node)

    def _lower_attribute(self, access: _FieldAccess, node: ast.Attribute) -> ast.AST:
        if access.decl.decls.kind_of(access.field) in ("pqueues", "processes"):
            raise ValueError(
                f"{self._callback_label()} must index {access.text} before using it"
            )
        if access.field in access.decl.constants:
            if not isinstance(node.ctx, ast.Load):
                raise ValueError(
                    f"{self._callback_label()} cannot assign to constant {access.text}"
                )
            return ast.copy_location(self._constant_expr(access), node)
        return ast.copy_location(self._field_target(access, node.ctx), node)

    def _direct_path_error(self, kind: str, text: str) -> ValueError:
        suffix = {
            "namespace": "directly; access one of its fields",
            "collection": "directly; index it and access one of its fields",
            "table": "before using it",
        }[kind]
        action = "must index" if kind == "table" else "cannot use"
        return ValueError(f"{self._callback_label()} {action} {text} {suffix}")

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        nested = self._field_ref(node.value)
        if nested is not None:
            raise ValueError(
                f"{self._callback_label()} cannot access attributes below component field {nested.text}"
            )
        access = self._field_ref(node)
        if access is not None:
            return self._lower_attribute(access, node)
        for kind, resolve in (
            ("namespace", self._namespace_ref),
            ("collection", self._collection_ref),
            ("table", self._ref_table_ref),
        ):
            if (path := resolve(node)) is not None:
                raise self._direct_path_error(kind, path.text)
        return self.generic_visit(node)


class _RootedOwnerLowerer(_OwnerPathLowerer):
    """One path lowerer shared by component and model callbacks."""

    def __init__(
        self,
        *,
        env_name: str,
        label: str,
        component_decl: _OwnerDecl | None = None,
        receiver_name: str | None = None,
        instance_index: ast.expr | None = None,
        possible_indices: tuple[int, ...] | None = None,
        owner_decl: _OwnerDecl | None = None,
        component_roots: Mapping[str, _OwnerDecl] | None = None,
        track_changes: bool = False,
        functions: Mapping[str, _FunctionSpec] | None = None,
    ):
        super().__init__(env_name=env_name, functions=functions)
        self.label = label
        self.component_decl = component_decl
        self.receiver_name = receiver_name
        self.instance_index = instance_index
        self.possible_indices = possible_indices
        self.owner_decl = owner_decl
        self.component_roots = component_roots or {}
        self.strict_ref_tables = instance_index is not None and not isinstance(
            instance_index, ast.Constant
        )
        self.changed = False
        self._track_changes = track_changes

    def _root_namespace_ref(self, node: ast.AST) -> _OwnerAccess | None:
        if self.component_decl is not None:
            if isinstance(node, ast.Name) and node.id == self.receiver_name:
                index = (
                    copy.deepcopy(self.instance_index)
                    if self.component_decl.count > 1
                    else None
                )
                return _OwnerAccess(
                    self.component_decl,
                    index,
                    self.receiver_name,
                    self.possible_indices,
                )
            if (
                self.owner_decl is not None
                and isinstance(node, ast.Name)
                and node.id == self.env_name
            ):
                return _OwnerAccess(self.owner_decl, None, self.env_name, (0,))
            return None
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == self.env_name
        ):
            decl = self.component_roots.get(node.attr)
            if decl is not None and not decl.collection:
                return _OwnerAccess(decl, None, f"{self.env_name}.{node.attr}")
        return None

    def _root_collection_ref(self, node: ast.AST) -> _OwnerAccess | None:
        if self.component_decl is not None:
            return None
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == self.env_name
        ):
            decl = self.component_roots.get(node.attr)
            if decl is not None and decl.collection:
                return _OwnerAccess(decl, None, f"{self.env_name}.{node.attr}")
        return None

    def _callback_label(self) -> str:
        return self.label

    def _raise_unknown_field(self, namespace: _OwnerAccess, field_name: str) -> None:
        if self.component_decl is None or (
            self.component_decl.owner_root and not namespace.decl.owner_root
        ):
            return super()._raise_unknown_field(namespace, field_name)
        raise ValueError(
            f"{self.label} references unsupported {namespace.text}.{field_name}"
        )

    def visit(self, node: ast.AST) -> ast.AST:
        lowered = super().visit(node)
        if self._track_changes and lowered is not node:
            self.changed = True
        return lowered

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if (
            self.component_decl is not None
            and not self.component_decl.owner_root
            and node.id == self.receiver_name
        ):
            raise ValueError(
                f"{self.label} cannot use self directly inside compiled code"
            )
        return node


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
        return _function_def_from_source(fn)
    except (OSError, TypeError) as exc:
        raise ValueError(
            f"component {kind} '{fn.__qualname__}' needs inspectable source"
        ) from exc


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
    if hasattr(like, "__cimba_function_calls__"):
        generated.__cimba_function_calls__ = \
            like.__cimba_function_calls__
    return generated


def _strip_function_annotations(node: ast.FunctionDef) -> None:
    node.decorator_list = []
    node.returns = node.type_comment = None
    for arg in node.args.args:
        arg.annotation = arg.type_comment = None


_FUNCTION_SCALAR_TYPES = {
    bool: types.boolean,
    int: types.int64,
    float: types.float64,
}

_FORBIDDEN_FUNCTION_SIM_CALLS = frozenset({
    "hold", "interrupt", "stop", "wait_process", "wait_event", "resume",
    "spawn", "despawn", "suspend", "set_priority", "timer_set",
    "timer_add", "timer_cancel", "timers_clear", "clear_events",
})

_FUNCTION_CACHE: weakref.WeakValueDictionary[tuple[Any, ...], Any] = (
    weakref.WeakValueDictionary()
)


def _function_scalar_type(annotation: Any, label: str) -> Any:
    numba_type = next(
        (candidate for scalar, candidate in _FUNCTION_SCALAR_TYPES.items()
         if annotation is scalar),
        None,
    )
    if numba_type is None:
        name = getattr(annotation, "__name__", repr(annotation))
        raise TypeError(
            f"{label} has unsupported type annotation {name}; expected "
            "bool, int/sim.Handle, or float")
    return numba_type


def _function_signature(
    node: ast.FunctionDef,
    method: Callable[..., Any],
    label: str,
    receiver: str,
    localns: Mapping[str, Any] | None = None,
) -> tuple[tuple[str, ...], tuple[Any, ...], Any]:
    args = node.args
    signature = (
        f"{label} must take {receiver} followed by explicitly annotated "
        "positional scalar arguments, without defaults or variadics, and "
        "declare a scalar return annotation")
    if (args.posonlyargs or args.vararg or args.kwonlyargs or args.kwarg
            or args.defaults or args.kw_defaults or not args.args):
        raise ValueError(signature)
    try:
        hints = get_type_hints(method, localns=localns)
    except Exception as exc:
        raise TypeError(f"{label} annotations could not be resolved") from exc

    parameter_names = tuple(arg.arg for arg in args.args[1:])
    argument_types = []
    for name in parameter_names:
        if name not in hints:
            raise TypeError(f"{label} argument '{name}' needs a type "
                            "annotation")
        argument_types.append(
            _function_scalar_type(hints[name], f"{label} argument '{name}'"))
    if "return" not in hints:
        raise TypeError(f"{label} needs a return type annotation")
    return_type = _function_scalar_type(
        hints["return"], f"{label} return value")

    returns = [item for item in ast.walk(node) if isinstance(item, ast.Return)]
    if not returns or any(item.value is None for item in returns):
        raise ValueError(f"{label} must return a scalar value")
    return parameter_names, tuple(argument_types), return_type


def _rooted_at_name(node: ast.AST, name: str) -> bool:
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return isinstance(node, ast.Name) and node.id == name


def _locally_bound_names(node: ast.AST) -> set[str]:
    """Names a function body binds itself: loop targets, assignments,
    comprehension targets, ``with``/``except`` bindings, walrus.

    Parameters are deliberately excluded -- a component function's
    dependencies are read at the call site, where the arguments are in
    scope but these names are not."""
    bound: set[str] = set()

    def add(target: ast.AST) -> None:
        for sub in ast.walk(target):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                bound.add(sub.id)

    for sub in ast.walk(node):
        if isinstance(sub, (ast.Assign, ast.Delete)):
            for target in sub.targets:
                add(target)
        elif isinstance(sub, (ast.AnnAssign, ast.AugAssign, ast.For,
                              ast.AsyncFor, ast.comprehension)):
            add(sub.target)
        elif isinstance(sub, ast.NamedExpr):
            add(sub.target)
        elif isinstance(sub, (ast.With, ast.AsyncWith)):
            for item in sub.items:
                if item.optional_vars is not None:
                    add(item.optional_vars)
        elif isinstance(sub, ast.ExceptHandler):
            if sub.name is not None:
                bound.add(sub.name)
    return bound


class _FunctionValidator(ast.NodeVisitor):
    """Reject side effects that must never enter a synchronous helper."""

    def __init__(self, *, receiver_name: str, method: Callable[..., Any],
                 label: str):
        self.receiver_name = receiver_name
        self.namespace = _closure_namespace(method)
        self.label = label

    def _check_target(self, node: ast.AST) -> None:
        if _rooted_at_name(node, self.receiver_name):
            raise ValueError(
                f"{self.label} cannot mutate component field {ast.unparse(node)}"
            )
        if isinstance(node, (ast.Tuple, ast.List)):
            for item in node.elts:
                self._check_target(item)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._check_target(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._check_target(node.target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._check_target(node.target)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            self._check_target(target)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        forbidden: str | None = None
        if (isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.attr in _FORBIDDEN_FUNCTION_SIM_CALLS):
            module = self.namespace.get(node.func.value.id)
            if getattr(module, "__name__", None) == "cimba.sim":
                forbidden = node.func.attr
        elif isinstance(node.func, ast.Name):
            obj = self.namespace.get(node.func.id)
            obj_name = getattr(obj, "__name__", node.func.id)
            if obj_name in _FORBIDDEN_FUNCTION_SIM_CALLS:
                module_name = getattr(obj, "__module__", "")
                if (module_name.startswith("cimba.")
                        or node.func.id in _FORBIDDEN_FUNCTION_SIM_CALLS):
                    forbidden = obj_name
        if forbidden is not None:
            raise ValueError(
                f"{self.label} cannot call scheduling/process operation sim.{forbidden}()"
            )
        self.generic_visit(node)


class _FunctionBodyLowerer(_OwnerPathLowerer):
    """Turn a component function body into a scalar-only helper body."""

    def __init__(
        self,
        *,
        builder: "_FunctionBuilder",
        decl: _OwnerDecl,
        method_name: str,
        receiver_name: str,
        parameter_names: tuple[str, ...],
        instance_indices: tuple[int, ...],
        variant: int | None,
    ):
        super().__init__(env_name="__cimba_no_env")
        self.builder = builder
        self.decl = decl
        self.method_name = method_name
        self.receiver_name = receiver_name
        self.parameter_names = parameter_names
        self.instance_indices = instance_indices
        self.variant = variant
        self.dependencies: list[_FunctionDependency] = []
        self._dependency_keys: dict[tuple[Any, ...], int] = {}
        self.callees: list[str] = []
        self.helper_namespace: dict[str, Any] = {}
        #: Names the body binds itself; an instance index built from one of
        #: these cannot be resolved at the call site (see _array_dependency).
        self._local_names: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self._local_names = _locally_bound_names(node)
        return self.generic_visit(node)

    def _root_namespace_ref(self, node: ast.AST) -> _OwnerAccess | None:
        if isinstance(node, ast.Name) and node.id == self.receiver_name:
            if self.variant is not None and len(self.instance_indices) == 1:
                index = (ast.Constant(self.instance_indices[0])
                         if self.decl.count > 1 else None)
            else:
                index = (
                    ast.Name(id="__cimba_receiver_index", ctx=ast.Load())
                    if self.decl.count > 1
                    else None
                )
            possible = (
                self.instance_indices if not isinstance(index, ast.Constant) else None
            )
            return _OwnerAccess(self.decl, index, self.receiver_name, possible)
        return None

    def _callback_label(self) -> str:
        return f"component function '{self.decl.name}.{self.method_name}'"

    def _dependency(
        self, access: _FieldAccess, *, direct: bool, array: bool = False
    ) -> ast.Name:
        key = (
            access.decl.name,
            access.field,
            None if access.index is None else ast.dump(access.index),
            array,
        )
        index = self._dependency_keys.get(key)
        if index is None:
            index = len(self.dependencies)
            self._dependency_keys[key] = index
            self.dependencies.append(
                _FunctionDependency(
                    access=copy.deepcopy(access),
                    parameter=f"__cimba_dep_{index}",
                    direct=direct,
                    array=array,
                )
            )
        elif direct:
            self.dependencies[index].direct = True
        return ast.Name(
            id=self.dependencies[index].parameter, ctx=ast.Load())

    def _index_is_local(self, index: ast.expr | None) -> bool:
        """Whether an instance index depends on a name the body binds, and
        so cannot be evaluated by the caller."""
        if index is None or not self._local_names:
            return False
        return any(isinstance(sub, ast.Name) and sub.id in self._local_names
                   for sub in ast.walk(index))

    def _array_dependency(self, access: _FieldAccess) -> ast.expr:
        """Thread the whole flattened field in and subscript it here, for a
        collection read whose index the body computes for itself."""
        label = self._callback_label()
        detail = (
            f"{label} indexes {access.text} with a value computed inside the function"
        )
        if access.field in access.decl.constants:
            raise ValueError(
                f"{detail}, which is only supported for Param, Output, State, and FloatState fields; index '{access.field}' with a function argument instead"
            )
        owners = access.decl.field_owners[access.field]
        slots = access.decl.field_slots[access.field]
        if len(owners) > 1 and slots != tuple(range(len(slots))):
            raise ValueError(
                f"{detail}, which requires every instance of '{access.decl.name}' to declare '{access.field}'; index it with a function argument instead"
            )
        whole = _FieldAccess(access.decl, None, access.field, access.text, None)
        if len(owners) <= 1:
            # A single owner means the flattened field is a plain scalar;
            # there is no array to index and the index is irrelevant.
            return self._dependency(whole, direct=True)
        parameter = self._dependency(whole, direct=True, array=True)
        return _subscript(parameter, access.index, ast.Load())

    def _validate_scalar_field(self, access: _FieldAccess) -> None:
        if access.field in access.decl.constants:
            if access.field not in access.decl.decls.consts:
                raise ValueError(
                    f"{self._callback_label()} cannot read undeclared constant {access.text}; declare it as sim.Const"
                )
            ctype = access.decl.decls.consts[access.field]
            _function_scalar_type(
                ctype, f"{self._callback_label()} constant '{access.field}'"
            )
            return
        kind = access.decl.decls.kind_of(access.field)
        if kind not in ("param", "output", "state", "fstate"):
            raise ValueError(
                f"{self._callback_label()} cannot read non-scalar component field {access.text} ({kind})"
            )

    def visit_Call(self, node: ast.Call) -> ast.AST:
        lowered_len = self._lower_len_call(node)
        if lowered_len is not None:
            return lowered_len
        if isinstance(node.func, ast.Attribute):
            receiver = self._namespace_ref(node.func.value)
            if receiver is not None:
                candidates = self.builder.specs_for(
                    receiver.decl,
                    receiver.index,
                    node.func.attr,
                    receiver.possible_indices,
                )
                if candidates:
                    return self._dispatch_function_call(
                        node, receiver, candidates, self._callee_call
                    )

            access = self._field_ref(node.func.value)
            if access is not None:
                operation = (
                    "entity or runtime operation"
                    if self.decl.owner_root
                    else "component field operation"
                )
                raise ValueError(
                    f"{self._callback_label()} cannot call {operation} {access.text}.{node.func.attr}()"
                )
            if receiver is not None:
                raise ValueError(
                    f"{self._callback_label()} cannot call unmarked component method {receiver.text}.{node.func.attr}()"
                )
        return self.generic_visit(node)

    def _callee_call(
        self, node: ast.Call, receiver: _OwnerAccess, callee: _FunctionSpec
    ) -> ast.Call:
        arguments = [self.visit(copy.deepcopy(arg)) for arg in node.args]
        replacements = dict(zip(callee.parameter_names, arguments))
        if receiver.index is not None:
            replacements["__cimba_receiver_index"] = receiver.index
        dependencies: list[ast.expr] = []
        for dependency in callee.dependencies:
            access = dependency.access
            bound = _FieldAccess(
                access.decl,
                self._substitute_expr(access.index, replacements),
                access.field,
                access.text,
                access.possible_indices,
            )
            value = self._dependency(bound, direct=False, array=dependency.array)
            dependencies.append(value)
            replacements[dependency.parameter] = value
        self.helper_namespace[callee.symbol] = callee.helper
        if callee.graph_name not in self.callees:
            self.callees.append(callee.graph_name)
        if callee.receiver_indexed:
            if receiver.index is None:
                raise TypeError("indexed component function has no receiver index")
            arguments.append(copy.deepcopy(receiver.index))
        return ast.Call(
            func=ast.Name(id=callee.symbol, ctx=ast.Load()),
            args=[*arguments, *dependencies],
            keywords=[],
        )

    def _lower_attribute(self, access: _FieldAccess, node: ast.Attribute) -> ast.AST:
        if not isinstance(node.ctx, ast.Load):
            raise ValueError(
                f"{self._callback_label()} cannot mutate component field {access.text}"
            )
        self._validate_scalar_field(access)
        value = (
            self._array_dependency(access)
            if self._index_is_local(access.index)
            else self._dependency(access, direct=True)
        )
        return ast.copy_location(value, node)

    def _direct_path_error(self, kind: str, text: str) -> ValueError:
        suffix = {
            "namespace": "directly; access a scalar field or marked function",
            "collection": "directly; index it",
            "table": "",
        }[kind]
        action = "must index" if kind == "table" else "cannot use"
        return ValueError(f"{self._callback_label()} {action} {text} {suffix}".rstrip())

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == self.receiver_name:
            raise ValueError(
                f"{self._callback_label()} cannot use self directly")
        return node


class _FunctionBuilder:
    """Validate and compile all synchronous functions for one model tree."""

    def __init__(self):
        self.specs: dict[str, _FunctionSpec] = {}
        self._building: list[str] = []

    @staticmethod
    def _dependency_type(dependency: _FunctionDependency) -> Any:
        access = dependency.access
        if access.field in access.decl.constants:
            return _function_scalar_type(
                access.decl.decls.consts[access.field],
                f"component function constant '{access.field}'",
            )
        kind = access.decl.decls.kind_of(access.field)
        scalar = types.int64 if kind == "state" else types.float64
        # A shaped env field reaches the helper as a NestedArray, which
        # matches an unspecified-layout array parameter but not "::1".
        return scalar[:] if dependency.array else scalar

    def build(
        self,
        decl: _OwnerDecl,
        method_name: str,
        method: Callable[..., Any],
        variant: int | None,
        instance_indices: tuple[int, ...],
    ) -> _FunctionSpec:
        owner = "model" if decl.owner_root else "component"
        base_name = (
            f"model:{method_name}" if decl.owner_root else f"{decl.name}__{method_name}"
        )
        graph_name = base_name if variant is None else f"{base_name}__variant_{variant}"
        existing = self.specs.get(graph_name)
        if existing is not None:
            return existing
        if graph_name in self._building:
            start = self._building.index(graph_name)
            cycle = [*self._building[start:], graph_name]
            raise ValueError(f"recursive {owner} function call: " + " -> ".join(cycle))

        self._building.append(graph_name)
        try:
            node = copy.deepcopy(_component_method_source(method, "function"))
            display_name = decl.cls.__name__ if decl.owner_root else decl.name
            label = f"{owner} function '{display_name}.{method_name}'"
            parameter_names, argument_types, return_type = _function_signature(
                node,
                method,
                label,
                "self",
                {base.__name__: base for base in decl.cls.__mro__},
            )
            receiver_name = node.args.args[0].arg
            _FunctionValidator(
                receiver_name=receiver_name, method=method, label=label
            ).visit(node)

            # Two names, because callers and the compiler want different
            # things. `symbol` identifies the declaration: callers bind
            # helpers into their namespace under it, and two collections of
            # the same class can lower to helpers with different signatures
            # (only a multi-instance one takes a receiver index), so sharing
            # one name across declarations binds the wrong helper.
            # `canonical` names the generated function itself and so decides
            # source_key: keeping it per class/method lets structurally
            # identical declarations share one compiled helper via the cache.
            symbol = (
                f"_CIMBA_MODEL_FUNCTION_{method_name}_{id(method):x}"
                if decl.owner_root
                else f"_CIMBA_FUNCTION_{graph_name}_{id(method):x}"
            )
            canonical = (
                f"_CIMBA_FUNCTION_{decl.cls.__name__}_{method_name}_{id(method):x}"
                + ("" if variant is None else f"_V{variant}")
            )
            lowerer = _FunctionBodyLowerer(
                builder=self,
                decl=decl,
                method_name=method_name,
                receiver_name=receiver_name,
                parameter_names=parameter_names,
                instance_indices=instance_indices,
                variant=variant,
            )
            lowered = lowerer.visit(node)
            if not isinstance(lowered, ast.FunctionDef):
                raise TypeError(f"{label} lowering produced a non-function")
            lowered.name = canonical
            _strip_function_annotations(lowered)
            lowered.args.args = lowered.args.args[1:]
            receiver_indexed = decl.count > 1 and len(instance_indices) > 1
            if receiver_indexed:
                lowered.args.args.append(
                    ast.arg(arg="__cimba_receiver_index"))
            lowered.args.args.extend(
                ast.arg(arg=dependency.parameter) for dependency in lowerer.dependencies
            )

            namespace = _closure_namespace(method)
            # An array dependency keeps its index expression inside the
            # helper, so the offset/slot tables it may reference have to be
            # in scope here as well as at the call site.
            namespace.update(_lowering_namespace((decl,)))
            namespace.update(lowerer.helper_namespace)
            lowered, random_changed = lower_random_calls_in_node(
                lowered, namespace=namespace, label=label
            )
            if random_changed:
                namespace.update(random_lowering_namespace())
            ast.fix_missing_locations(lowered)
            source_key = ast.unparse(
                ast.Module(body=[lowered], type_ignores=[]))
            dependency_types = tuple(
                self._dependency_type(dependency) for dependency in lowerer.dependencies
            )
            signature = return_type(
                *argument_types,
                *((types.int64,) if receiver_indexed else ()),
                *dependency_types,
            )
            closure_key = tuple(
                (name,
                 value if _primitive_constant(value) else id(value))
                for name, value in (
                    (name, cell.cell_contents)
                    for name, cell in zip(
                        method.__code__.co_freevars, method.__closure__ or ()
                    )
                )
            )
            length_table_key = tuple(
                (name, tuple(int(item) for item in value.tolist()))
                for name, value in namespace.items()
                if (name.startswith("_CIMBA_LEN_")
                    or name.startswith("_CIMBA_REFLEN_"))
                and isinstance(value, np.ndarray)
            )
            cache_key = (
                decl.class_at(instance_indices[0]),
                decl.specialization_key(instance_indices[0]),
                id(method),
                source_key,
                str(signature),
                closure_key,
                length_table_key,
                tuple(id(value)
                      for value in lowerer.helper_namespace.values()),
            )
            helper = _FUNCTION_CACHE.get(cache_key)
            if helper is None:
                plain = _compile_lowered(
                    lowered,
                    filename=f"<cimba {owner} function '{decl.name}.{method_name}'>",
                    fn_name=canonical,
                    qualname=canonical,
                    namespace=namespace,
                    like=method,
                )
                try:
                    helper = njit(signature)(plain)
                    helper.disable_compile()
                except Exception as exc:
                    raise TypeError(
                        f"{label} failed Numba nopython compilation"
                    ) from exc
                helper.__cimba_source__ = plain.__cimba_source__
                _FUNCTION_CACHE[cache_key] = helper

            spec = _FunctionSpec(
                decl=decl,
                name=method_name,
                method=method,
                graph_name=graph_name,
                symbol=symbol,
                parameter_names=parameter_names,
                argument_types=argument_types,
                return_type=return_type,
                dependencies=tuple(lowerer.dependencies),
                helper=helper,
                callees=tuple(lowerer.callees),
                receiver_indexed=receiver_indexed,
                variant=variant,
                instance_indices=instance_indices,
            )
            self.specs[graph_name] = spec
            return spec
        finally:
            self._building.pop()

    def build_all(self, roots: Iterable[_OwnerDecl]) -> dict[str, _FunctionSpec]:
        for root in roots:
            for decl in root.walk():
                groups = decl.specialization_groups()
                for ordinal, instance_indices in enumerate(groups):
                    variant = None if len(groups) == 1 else ordinal
                    cls = decl.class_at(instance_indices[0])
                    for callback in cls._callbacks().functions:
                        self.build(
                            decl, callback.name, callback.fn, variant, instance_indices
                        )
        return self.specs

    def specs_for(
        self,
        decl: _OwnerDecl,
        index: ast.expr | None,
        method_name: str,
        possible_indices: tuple[int, ...] | None = None,
    ) -> tuple[_FunctionSpec, ...]:
        """Resolve/build the concrete helper candidates for an access."""
        if isinstance(index, ast.Constant) and type(index.value) is int:
            instance_index = index.value
            groups = decl.specialization_groups()
            variant = (
                None
                if len(groups) == 1
                else decl.specialization_slots()[instance_index]
            )
            group = groups[0] if variant is None else groups[variant]
            cls = decl.class_at(index.value)
            method = next(
                (
                    callback.fn
                    for callback in cls._callbacks().functions
                    if callback.name == method_name
                ),
                None,
            )
            if method is None:
                return ()
            return (self.build(
                decl, method_name, method, variant, group),)
        if not decl.polymorphic:
            method = next(
                (
                    callback.fn
                    for callback in decl.class_at()._callbacks().functions
                    if callback.name == method_name
                ),
                None,
            )
            if method is None:
                return ()
            group = decl.specialization_groups()[0]
            return (self.build(decl, method_name, method, None, group),)
        candidates: list[_FunctionSpec] = []
        possible = set(possible_indices or range(decl.count))
        groups = decl.specialization_groups()
        for variant, group in enumerate(groups):
            if not possible.intersection(group):
                continue
            cls = decl.class_at(group[0])
            method = next(
                (
                    callback.fn
                    for callback in cls._callbacks().functions
                    if callback.name == method_name
                ),
                None,
            )
            if method is None:
                raise ValueError(
                    f"dynamic component function call to '{decl.name}.{method_name}' requires every concrete component type to declare that @sim.function"
                )
            candidates.append(self.build(decl, method_name, method, variant, group))
        return tuple(candidates)


def _build_functions(roots: Iterable[_OwnerDecl]) -> dict[str, _FunctionSpec]:
    return _FunctionBuilder().build_all(roots)


@dataclass(frozen=True)
class _CallbackLoweringContext:
    datasets: Sequence[str]
    histories: Mapping[str, str]
    entities: Mapping[str, str]
    owner_decl: _OwnerDecl
    functions: Mapping[str, _FunctionSpec]


def _lower_owner_methods(
    node: ast.FunctionDef,
    fn: Callable[..., Any],
    *,
    env_name: str,
    label: str,
    owner_name: str,
    context: _CallbackLoweringContext,
    namespace: dict[str, Any],
    entity_fields: Mapping[str, str] | None = None,
) -> tuple[ast.FunctionDef, bool]:
    """Apply the method-sugar passes shared by both owner callback types."""
    changed = False
    for lower, fields, keyword, helpers in (
        (
            lower_env_dataset_method_calls,
            context.datasets,
            "dataset_fields",
            dataset_lowering_namespace,
        ),
        (
            lower_env_history_method_calls,
            context.histories,
            "history_fields",
            timeseries_lowering_namespace,
        ),
    ):
        node, lowered = lower(node, env_name=env_name, label=label, **{keyword: fields})
        if lowered:
            namespace.update(helpers())
            changed = True

    entities = context.entities if entity_fields is None else entity_fields
    helpers_changed = _rewire_entity_method_helpers(
        namespace,
        set(fn.__code__.co_names),
        model_name=owner_name,
        entity_fields=entities,
        cache={},
    )
    node, lowered = lower_env_entity_method_calls(
        node, env_name=env_name, entity_fields=entities, label=label
    )
    if lowered or helpers_changed:
        namespace.update(entity_lowering_namespace())
        changed = True

    node, lowered = lower_random_calls_in_node(node, namespace=namespace, label=label)
    if lowered:
        namespace.update(random_lowering_namespace())
        changed = True
    return node, changed


def _lower_component_method(
    node: ast.FunctionDef,
    *,
    kind: str,
    component_name: str,
    component_decl: _OwnerDecl,
    instance_index: ast.expr,
    possible_indices: tuple[int, ...] | None = None,
    method_name: str,
    method: Callable[..., Any],
    struct_view: type | None = None,
    prologue: Sequence[ast.stmt] = (),
    extra_namespace: Mapping[str, Any] | None = None,
    context: _CallbackLoweringContext,
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
            # Keep an annotation on the view parameter so process registration
            # detects it on the lowered function; the exec namespace maps
            # _CIMBA_STRUCT_VIEW to the struct class.
            arg.annotation = ast.Name(id="_CIMBA_STRUCT_VIEW",
                                      ctx=ast.Load())
        else:
            arg.annotation = None
        arg.type_comment = None

    lowerer = _RootedOwnerLowerer(
        label=f"component '{component_name}' {kind}",
        receiver_name=receiver_name,
        env_name=env_name,
        component_decl=component_decl,
        owner_decl=context.owner_decl,
        instance_index=instance_index,
        possible_indices=possible_indices,
        functions=context.functions,
    )
    lowered = lowerer.visit(node)
    if not isinstance(lowered, ast.FunctionDef):
        raise TypeError(f"component {kind} lowering produced a non-function")
    label = f"component {kind} '{component_name}.{method_name}'"
    namespace = _closure_namespace(method)
    lowered, _ = _lower_owner_methods(
        lowered,
        method,
        env_name=env_name,
        label=label,
        owner_name=component_name,
        context=context,
        namespace=namespace,
    )
    lowered.body[:0] = list(prologue)

    if struct_view is not None:
        namespace["_CIMBA_STRUCT_VIEW"] = struct_view
    if extra_namespace:
        namespace.update(extra_namespace)
    namespace.update(
        _model_lowering_namespace(
            {component_decl.name: component_decl}, context.functions
        )
    )
    generated = _compile_lowered(
        lowered,
        filename=f"<cimba component '{component_name}.{method_name}'>",
        fn_name=fn_name,
        qualname=fn_name,
        namespace=namespace,
        like=method,
    )
    generated.__cimba_function_calls__ = tuple(
        sorted(lowerer.called_functions))
    return generated


def _shared_instance_setup(
    node: ast.FunctionDef,
    base: str,
    counts: tuple[int, ...],
    base_arg_count: int,
    instance_indices: tuple[int, ...] | None = None,
) -> tuple[ast.expr, list[ast.stmt], dict[str, Any]]:
    """Map global process-copy indexes to collection items/local copies."""
    params = node.args.args
    user_idx = params[2].arg if base_arg_count == 3 else None
    if user_idx is not None:
        params[2] = ast.arg(arg="__cimba_idx")
    else:
        params.insert(2, ast.arg(arg="__cimba_idx"))

    inst_symbol = f"_CIMBA_PROCINST_{base}"
    copybase_symbol = f"_CIMBA_COPYBASE_{base}"
    group_symbol = f"_CIMBA_PROCGROUP_{base}"
    mapped = (instance_indices is not None
              and instance_indices != tuple(range(len(instance_indices))))
    local_inst = "__cimba_local_inst" if mapped else "__cimba_inst"
    uniform = len(set(counts)) == 1
    per_instance = counts[0]
    lines: list[str] = []
    tables: dict[str, Any] = {}

    # __cimba_inst: the collection item this global copy belongs to.
    if uniform and per_instance == 1:
        lines.append(f"{local_inst} = __cimba_idx")
    elif uniform:
        lines.append(f"{local_inst} = __cimba_idx // {per_instance}")
    else:
        tables[inst_symbol] = np.repeat(
            np.arange(len(counts), dtype=np.int64),
            np.asarray(counts, dtype=np.int64))
        lines.append(f"{local_inst} = {inst_symbol}[__cimba_idx]")

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
                f"{user_idx} = __cimba_idx - "
                f"{copybase_symbol}[{local_inst}]")

    if mapped:
        tables[group_symbol] = np.asarray(
            instance_indices, dtype=np.int64)
        lines.append(f"__cimba_inst = {group_symbol}[{local_inst}]")

    return (ast.Name(id="__cimba_inst", ctx=ast.Load()),
            ast.parse("\n".join(lines)).body, tables)


def _component_process_signature(
    component_name: str,
    method_name: str,
    method: Callable[..., Any],
    is_struct_class: Callable[[Any], bool],
) -> tuple[type | None, int]:
    """Validate a component process method's ``(self, env[, idx][, view])``
    signature and return ``(struct_view_class_or_None, base_arg_count)``."""
    signature = (f"component process '{component_name}.{method_name}' must "
                 "take (self, env), (self, env, idx), and optionally a "
                 "final sim.Struct view parameter, without defaults")
    return _process_signature(
        method, 2, is_struct_class,
        f"component process '{component_name}.{method_name}'", signature)


def _lower_component_process(
    component_name: str,
    component_decl: _OwnerDecl,
    method_name: str,
    method: Callable[..., Any],
    is_struct_class: Callable[[Any], bool],
    *,
    instance_index: int | None = None,
    copies_per_instance: tuple[int, ...] | None = None,
    instance_indices: tuple[int, ...] | None = None,
    context: _CallbackLoweringContext,
) -> Callable[..., Any]:
    """Lower one specialized or collection-shared component process."""
    node = copy.deepcopy(_component_method_source(method, "process"))
    struct_view, base_arg_count = _component_process_signature(
        component_name, method_name, method, is_struct_class)

    if copies_per_instance is None:
        index_expr: ast.expr = ast.Constant(instance_index)
        prologue: list[ast.stmt] = []
        tables: dict[str, Any] = {}
    else:
        index_expr, prologue, tables = _shared_instance_setup(
            node, f"{component_name}__{method_name}",
            tuple(copies_per_instance), base_arg_count, instance_indices)

    return _lower_component_method(
        node, kind="process", component_name=component_name,
        component_decl=component_decl, instance_index=index_expr,
        possible_indices=(
            instance_indices if copies_per_instance is not None
            else ((instance_index,) if instance_index is not None else None)),
        method_name=method_name, method=method, struct_view=struct_view,
        prologue=prologue, extra_namespace=tables,
        context=context)


def _lower_component_collect(
    component_name: str,
    component_decl: _OwnerDecl,
    method_name: str,
    method: Callable[..., Any],
    *,
    instance_index: int | None = None,
    per_class: bool = False,
    instance_indices: tuple[int, ...] | None = None,
    context: _CallbackLoweringContext,
) -> Callable[..., Any]:
    """Lower a component collect method; with ``per_class``, one function
    covers every instance and takes the instance index as its second
    argument."""
    node = copy.deepcopy(_component_method_source(method, "collect"))
    args = node.args
    signature = (f"component collect '{component_name}.{method_name}' must "
                 "take (self, env) without defaults")
    _callback_arg_count(method, (2,), signature)
    if per_class:
        args.args.append(ast.arg(arg="__cimba_group_inst"))
        indices = (instance_indices
                   if instance_indices is not None
                   else tuple(range(component_decl.count)))
        if indices == tuple(range(len(indices))):
            index_expr = ast.Name(
                id="__cimba_group_inst", ctx=ast.Load())
            prologue: list[ast.stmt] = []
            tables: dict[str, Any] = {}
        else:
            symbol = f"_CIMBA_COLLECTGROUP_{component_name}__{method_name}"
            index_expr = ast.Name(id="__cimba_inst", ctx=ast.Load())
            prologue = ast.parse(
                f"__cimba_inst = {symbol}[__cimba_group_inst]").body
            tables = {
                symbol: np.asarray(indices, dtype=np.int64),
            }
    else:
        index_expr = ast.Constant(instance_index)
        prologue = []
        tables = {}
    return _lower_component_method(
        node, kind="collect", component_name=component_name,
        component_decl=component_decl, instance_index=index_expr,
        possible_indices=(
            instance_indices if per_class
            else ((instance_index,) if instance_index is not None else None)),
        method_name=method_name, method=method,
        prologue=prologue, extra_namespace=tables,
        context=context)


def _lower_component_signal(
    component_name: str,
    component_decl: _OwnerDecl,
    method_name: str,
    method: Callable[..., Any],
    *,
    kind: str,
    instance_index: int,
    context: _CallbackLoweringContext,
) -> Callable[..., Any]:
    """Lower one instance of a component predicate or event callback."""
    node = copy.deepcopy(_component_method_source(method, kind))
    arity = (2,) if kind == "predicate" else (2, 3)
    suffix = " or (self, env, data)" if kind == "event" else ""
    signature = f"component {kind} '{component_name}.{method_name}' must take (self, env){suffix} without defaults"
    _callback_arg_count(method, arity, signature)
    if kind == "predicate" and get_type_hints(method).get("return") is not bool:
        raise ValueError(
            f"component predicate '{component_name}.{method_name}' must return bool"
        )
    generated = _lower_component_method(
        node,
        kind=kind,
        component_name=component_name,
        component_decl=component_decl,
        instance_index=ast.Constant(instance_index),
        possible_indices=(instance_index,),
        method_name=method_name,
        method=method,
        context=context,
    )
    if kind == "predicate":
        generated.__annotations__["return"] = bool
    return generated


def _lower_model_component_refs_in_node(
    node: ast.FunctionDef,
    *,
    model_name: str,
    owner_decl: _OwnerDecl,
    functions: Mapping[str, _FunctionSpec] | None = None,
) -> tuple[ast.FunctionDef, bool, tuple[str, ...]]:
    if not node.args.args:
        return node, False, ()

    env_name = node.args.args[0].arg
    lowerer = _RootedOwnerLowerer(
        label=f"model '{model_name}' callback '{node.name}'",
        receiver_name=env_name,
        env_name=env_name,
        component_decl=owner_decl,
        owner_decl=owner_decl,
        instance_index=ast.Constant(0),
        possible_indices=(0,),
        track_changes=True,
        functions=functions,
    )
    lowered = lowerer.visit(node)
    if not isinstance(lowered, ast.FunctionDef):
        raise TypeError("model callback lowering produced a non-function")
    return lowered, lowerer.changed, tuple(sorted(lowerer.called_functions))


def _model_lowering_namespace(
    component_roots: Mapping[str, _OwnerDecl],
    functions: Mapping[str, _FunctionSpec] | None,
) -> dict[str, Any]:
    namespace = dataset_lowering_namespace()
    namespace.update(timeseries_lowering_namespace())
    namespace.update(entity_lowering_namespace())
    if functions:
        namespace.update({spec.symbol: spec.helper for spec in functions.values()})
    namespace.update(_lowering_namespace(component_roots.values()))
    return namespace


def _compile_model_callback_lowering(
    fn: Callable[..., Any],
    lowered: ast.FunctionDef,
    model_name: str,
    lowering_namespace: Mapping[str, Any],
    namespace: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    _strip_function_annotations(lowered)
    if namespace is None:
        namespace = _closure_namespace(fn)
    namespace.update(lowering_namespace)
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
    """Recursively lower entity sugar inside a referenced Numba helper."""
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

    _strip_function_annotations(lowered)

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
