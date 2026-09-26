"""Components and the flattened declaration tree of a model.

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

The trial record compiled by ``Model`` is flat. This module builds one
``_OwnerDecl`` per node of the component tree and flattens every field a
component declares into prefixed model fields (``inlet__queue``); the items
of a collection share one shaped field (``stations__queue`` with one element
per item), and nesting keeps prefixing (``zones__gates__queue``). Entity
wiring (``Station(inbox=other.outbox)``) and Ref/Refs targets are resolved
once the whole tree exists. Rewriting callbacks against the flattened
record lives in ``_lowering``.

The module is organized in three parts, in order: the authoring API
(``Component`` and the wiring/Ref metadata captured from instance defaults;
callback markers live in ``_callbacks``); declaration metadata
(``_OwnerDecl``); and declaration building (``_class_declarations``,
``_DeclBuilder``, and wiring/Ref resolution).
"""

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, get_args, get_origin

from ._callbacks import (
    _ProcessSpec,
    _bind_callback_fields,
    _callback_set,
)
from ._declarations import (_FIELD_KINDS, _MISSING,
                            _STANDARD_FIELDS, _ConstHint, _Declarations,
                            _RefHint, _FieldDecl, _field_declarations,
                            _declared_kind, _param_default,
                            class_type_hints)

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
            kind = _declared_kind(hint)
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
        kind = _declared_kind(hint)
        if kind is not None:
            if kind.name == "param":
                configurable[name] = ("param", float)
            else:
                runtime.add(name)
            continue
        if (_is_component_class(hint)
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
    #: the concrete class shared by every instance, or the annotated base
    #: class when the instances are of different (polymorphic) classes
    cls: type[Any]
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
            for callback in _callback_set(cls).processes:
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
    decls = _field_declarations(cls, allow_symbolic_pqueues=True,
                                allow_refs=True)
    _bind_callback_fields(cls, decls, owner="component")
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
    owners_by_name: Mapping[str, tuple[int, ...]],
) -> dict[str, tuple[Any, ...]]:
    """Values of the declared ``sim.Const`` fields on the instances that
    declare them, checked to be present and of the annotated type."""
    values: dict[str, tuple[Any, ...]] = {}
    for fname, ctype in consts.items():
        instance_values = []
        for index in owners_by_name[fname]:
            value = getattr(templates[index], fname, _MISSING)
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


def _component_param_defaults(
    component_name: str,
    templates: Sequence[Component],
    decls: _Declarations,
    field_owners: Mapping[str, tuple[int, ...]],
) -> dict[str, tuple[float, ...]]:
    """Capture Param defaults from class attributes or component instances.

    A shaped flattened parameter can only be omitted as a whole, so every
    instance covered by a component declaration must either provide the
    default or leave it required.
    """
    defaults: dict[str, tuple[float, ...]] = {}
    for fname in decls.names("param"):
        values = tuple(getattr(templates[index], fname, _MISSING)
                       for index in field_owners[fname])
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


def _packed_counts(
    instance_count: int,
    owners: Sequence[int],
    owned_counts: Iterable[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Per-instance element counts and start offsets of a field packed over
    its owning instances; instances that don't own it get zero elements."""
    counts = [0] * instance_count
    offsets = [0] * instance_count
    for owner, count, offset in zip(owners, *_offsets_from_counts(owned_counts)):
        counts[owner] = count
        offsets[owner] = offset
    return tuple(counts), tuple(offsets)


def _resolve_component_pqueues(
    component_name: str,
    instance_count: int,
    decls: _Declarations,
    constants: Mapping[str, tuple[Any, ...]],
    field_owners: Mapping[str, tuple[int, ...]],
    constant_slots: Mapping[str, tuple[int, ...]],
) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
    """Per-instance element counts and start offsets of each PQueues
    field; symbolic counts name a per-instance int constant."""
    counts_by_field: dict[str, tuple[int, ...]] = {}
    offsets_by_field: dict[str, tuple[int, ...]] = {}
    for field_decl in decls.by_kind("pqueues"):
        fname = field_decl.name
        owners = field_owners[fname]
        count_decl = field_decl.count
        if isinstance(count_decl, int):
            owned_counts: tuple[Any, ...] = (count_decl,) * len(owners)
        else:
            # PQueues declarations are rejected without a count.
            assert count_decl is not None
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
            slots = constant_slots[count_decl]
            owned_counts = tuple(values[slots[index]] for index in owners)
        counts_by_field[fname], offsets_by_field[fname] = _packed_counts(
            instance_count, owners, owned_counts)
    return counts_by_field, offsets_by_field


def _resolve_component_processes(
    component_name: str,
    templates: Sequence[Component],
    instance_classes: Sequence[type[Component]],
    decls: _Declarations,
    field_owners: Mapping[str, tuple[int, ...]],
) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
    """Per-instance copy counts and handle offsets of Processes fields."""
    counts_by_field: dict[str, tuple[int, ...]] = {}
    offsets_by_field: dict[str, tuple[int, ...]] = {}
    for fname in decls.names("processes"):
        owners = field_owners[fname]
        owned_counts: list[int] = []
        for index in owners:
            specs = [
                callback.spec
                for callback in _callback_set(instance_classes[index]).processes
                if callback.spec.field == fname
            ]
            if len(specs) != 1:
                raise ValueError(
                    f"component '{component_name}' Processes field '{fname}' must be bound by exactly one @sim.process(field=...)"
                )
            owned_counts.append(specs[0].resolve_copies(
                templates[index], f"{component_name}.{fname}"))
        counts_by_field[fname], offsets_by_field[fname] = _packed_counts(
            len(templates), owners, owned_counts)
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
        parent_offsets: tuple[int, ...],
        parent_lengths: tuple[int, ...],
        parent_slots: tuple[int, ...],
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
        declared_const_owners = {
            fname: tuple(
                index for index, own in enumerate(per_instance_decls)
                if fname in own.consts)
            for fname in decls.consts
        }
        const_values = _validate_component_consts(
            name, templates, decls.consts, declared_const_owners)
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
            name, templates, instance_classes, decls, field_owners)
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
            (ref,) = refs
            assert isinstance(ref, _FieldRef)
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
