"""Model declaration, callback registration, and compilation.

A ``Model`` collects declared entities, parameters, outputs, and process
functions, then compiles everything on first ``experiment()``:

* component trees are flattened and every callback is lowered to a plain
  function over the flat trial record at construction time;
* user callbacks compile once per concrete model, after construction and
  registration, into native callbacks (see ``_runtime``); only the fixed
  lifecycle callbacks are shared between models;
* ``experiment()`` expands its arguments into one trial record per trial
  (see ``_trial_inputs``) and returns an ``Experiment`` that runs them
  across all cores.
"""

import copy
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Self, get_type_hints, overload

import numpy as np
from numpy.typing import ArrayLike

from numba import carray, from_dtype, njit, types
from numba.core.dispatcher import Dispatcher

from . import _bindings as _b
from ._callbacks import (
    _bind_callback_fields,
    _callback_arg_count,
    _callback_set,
    _process_signature,
)
from ._components import (
    Component,
    _class_declarations,
    _OwnerDecl,
    _owner_declaration,
)
from ._functions import _build_functions
from ._lowering import (
    _CallbackLoweringContext,
    _closure_namespace,
    _compile_model_callback,
    _direct_collect_callback,
    _direct_process_callback,
    _function_calls,
    _function_def_from_source,
    _lower_component_collect,
    _lower_component_signal,
    _lower_component_process,
    _lower_owner_methods,
    _lower_model_component_refs_in_node,
    _model_lowering_namespace,
    _set_function_calls,
)
from ._declarations import (
    _MISSING,
    _Capacities,
    _check_name,
    _FIELD_KINDS,
    _FieldDecl,
)
from ._experiment import Experiment, _ExperimentResultT
from ._graph import (ProcessDAG, ProcessDAGBlock, ProcessDAGEdge,
                     ProcessDAGNode, infer_process_dag)
from ._capture import (
    HISTORY_CAPTURE_STORE_FIELD,
    HISTORY_CAPTURE_TRIAL_FIELD,
    HistoryCaptureSpec,
    lower_dataset_capture_calls,
    lower_history_capture_calls,
)
from ._intrinsics import ptr_caster
from ._struct import Struct, _is_struct_class
from ._runtime import (
    _CAPACITY_CONSTANT,
    _CAPACITY_FIELD,
    _COLLECT_DESCRIPTOR_COUNT_FIELD,
    _COLLECT_DESCRIPTORS_FIELD,
    _ENTITY_BUFFER,
    _ENTITY_CONDITION,
    _ENTITY_DATASET,
    _ENTITY_DESCRIPTOR_COUNT_FIELD,
    _ENTITY_DESCRIPTOR_WIDTH,
    _ENTITY_DESCRIPTORS_FIELD,
    _ENTITY_OBJECTQUEUE,
    _ENTITY_PRIORITYQUEUE,
    _ENTITY_RESOURCE,
    _ENTITY_RESOURCEPOOL,
    _HAS_SPAWNED_FIELD,
    _LIFECYCLE_ABI_FIELDS,
    _LIFECYCLE_FIELDS,
    _LIFECYCLE_JOBS,
    _PROC_DATA_OFFSET,
    _PROCESS_CONTEXTS_FIELD,
    _PROCESS_DESCRIPTOR_COUNT_FIELD,
    _PROCESS_DESCRIPTOR_WIDTH,
    _PROCESS_DESCRIPTORS_FIELD,
    _PROCESS_HANDLE_COUNT_FIELD,
    _PROCESS_HANDLES_FIELD,
    _compile_cfuncs,
    _compile_parallel_cfunc,
    _native_names,
)
from ._trial_inputs import (
    _as_param_axis,
    _as_trace_grid,
    _draw_trial_seeds,
    _n_design_points,
)


@dataclass
class _ProcDecl:
    """A lowered class-declared ``@sim.process`` callback."""

    name: str
    fn: Callable[..., Any]
    copies: int
    priority: int
    indexed: bool                  # takes the copy index argument
    struct: type[Struct] | None    # per-process fields, if any
    injected: bool                 # fn receives its own struct view
    spawnable: bool                # created by sim.spawn(), not at setup
    spawn_field: str | None = None # env spawn descriptor lands in
    spawn_index: int | None = None # shaped descriptor element, if any
    process_field: str | None = None # env Processes field handles land in
    process_offset: int = 0        # first handle slot for this process

    @property
    def alloc_size(self) -> int:
        return (self.struct._alloc_size if self.struct is not None
                else _PROC_DATA_OFFSET)


@dataclass(frozen=True)
class _BoundCallbackDecl:
    """A predicate or event callback bound to a record field."""

    name: str
    fn: Callable[..., Any]
    field: str
    index: int | None = None
    takes_data: bool = False

    @property
    def key(self) -> str | tuple[str, int]:
        return self.field if self.index is None else (self.field, self.index)


@dataclass(frozen=True)
class _CFuncJob:
    category: str
    name: str
    key: Any
    signature: Any
    function: Callable[..., Any]
    process: _ProcDecl | None = None


@dataclass(frozen=True)
class _CompiledModel:
    """The native artifacts of one compiled model, kept alive with it."""

    dtype: np.dtype
    #: the fixed trial dispatcher handed to ``cimba_run``
    trial: Any
    #: lifecycle callbacks stored in the leading ``_LIFECYCLE_FIELDS``
    lifecycle: tuple[Any, ...]
    processes: dict[str, Any]
    #: predicate/event callbacks by bound field (or ``(field, index)``)
    predicates: dict[Any, Any]
    events: dict[Any, Any]
    collects: dict[int, Any]
    collect_descriptors: np.ndarray
    process_descriptors: np.ndarray
    process_handle_count: int
    entity_descriptors: np.ndarray
    entity_descriptor_count: int
    #: spawnable process name -> [callback, native name, allocation size]
    spawn_descriptors: dict[str, np.ndarray]
    #: (spawnable field, element index or None, process name)
    spawn_assignments: tuple[tuple[str, int | None, str], ...]


@dataclass(frozen=True)
class ComponentFieldSchema:
    """Public flattened-layout metadata for one component-owned field."""

    path: str
    flattened_name: str
    kind: str
    owners: tuple[int, ...]
    concrete_types: tuple[type[Component], ...]
    logical_count: int
    shape: tuple[int, ...] | None

    @property
    def packed(self) -> bool:
        """Whether the field omits logical component instances."""
        return self.owners != tuple(range(self.logical_count))


def _field_offsets(dtype: np.dtype) -> dict[str, int]:
    """Byte offset of every field in a structured trial-record dtype."""
    return {name: spec[1] for name, spec in (dtype.fields or {}).items()}


def _as_capacity_dict(value: _Capacities) -> dict[str, int | str | None]:
    """Normalize a `stores`/`pools` declaration: a list means unbounded
    capacity; dict values are an int capacity or a param name string."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return {name: None for name in value}


def _spawnable_slot_label(field: str, index: int | None) -> str:
    return field if index is None else f"{field}[{index}]"


class Model(Generic[_ExperimentResultT]):
    """A simulation model. Subclass it and declare the root environment fields as
    annotations (Param, Output, Queue, Resource, Pool, Store, Dataset,
    Condition, State, Predicate) -- model callbacks use their first argument,
    conventionally ``self``, as that root trial environment view. Entity names
    may also be passed as keyword lists for
    quick callback-free untyped models. Class-declared model and component
    callbacks compile once per model instance and are reused by its
    experiments."""

    start_time: float
    warmup_s: float
    duration_s: float
    cooldown_s: float
    seed: int

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "__cimba_precompile__" in cls.__dict__:
            raise TypeError(
                "__cimba_precompile__ was removed; construct the model and "
                "call model.compile(), or let experiment() compile it")

    def __init__(self, name: str | None = None, *,
                 params: Iterable[str] = (),
                 outputs: Iterable[str] = (),
                 queues: _Capacities = None,
                 resources: Iterable[str] = (),
                 pools: _Capacities = None,
                 stores: _Capacities = None,
                 datasets: Iterable[str] = (),
                 conditions: Iterable[str] = (),
                 state: Iterable[str] = ()):
        decls = _class_declarations(type(self))
        self.name = name if name is not None else type(self).__name__
        for kind_name, names in (("param", params), ("output", outputs),
                                 ("resource", resources),
                                 ("dataset", datasets),
                                 ("condition", conditions),
                                 ("state", state)):
            for n in names:
                decls.add(_FieldDecl(n, _FIELD_KINDS[kind_name]))
        for kind_name, capacities in (("queue", queues), ("pool", pools),
                                      ("store", stores)):
            for n, cap in _as_capacity_dict(capacities).items():
                decls.add(_FieldDecl(n, _FIELD_KINDS[kind_name],
                                     capacity=cap))
        _bind_callback_fields(
            type(self),
            decls,
            owner="model",
            protected=frozenset(
                name for name in vars(Model) if not name.startswith("_")
            ),
        )
        self._decls = decls

        # Backwards-compatible views of the declarations, by kind
        self.params = decls.names("param")
        self.param_defaults: dict[str, Any] = {
            field.name: field.default
            for field in decls.by_kind("param")
            if field.default is not _MISSING
        }
        self.outputs = decls.names("output")
        self.queues = decls.values("queue", attribute="capacity")
        self.resources = decls.names("resource")
        self.pools = decls.values("pool", attribute="capacity")
        self.stores = decls.values("store", attribute="capacity")
        self.datasets = decls.names("dataset")
        self.conditions = decls.names("condition")
        self.state = decls.names("state")
        self.float_state: list[str] = decls.names("fstate")
        self.traces: list[str] = decls.names("trace")
        self.pqueues: dict[str, int] = decls.values("pqueues", attribute="count")
        #: entity field name -> native binding prefix, for fields whose
        #: ``.history()`` compiles to a native timeseries lookup
        self._history_fields: dict[str, str] = decls.values(
            "queue", "resource", "pool", "store", "pqueues",
            attribute="kind.binding")
        self._history_captures: dict[str, HistoryCaptureSpec] = {}
        self._dataset_captures: dict[str, HistoryCaptureSpec] = {}
        self._predicate_fields: list[str] = decls.names("predicate")
        self._event_fields: list[str] = decls.names("event")
        self._process_fields: list[str] = decls.names("processes")
        self._spawnable_fields: list[str] = decls.names("spawnable")
        self._component_decls: list[_OwnerDecl] = decls.components
        self._component_collection_decls: list[_OwnerDecl] = decls.component_collections
        self._component_roots: dict[str, _OwnerDecl] = {
            decl.name: decl
            for decl in (*decls.components, *decls.component_collections)}
        self._field_shapes: dict[str, tuple[int, ...]] = {
            f.name: f.shape for f in decls.fields.values()
            if f.shape is not None}
        # Component-owned entity fields retain a collection axis even when
        # the flattened dtype is scalar (a one-item collection).  The
        # authoring path's ``[]`` marker is the stable indication that a
        # field belongs to a component collection.
        self._indexed_history_fields: dict[str, int] = {}
        for root in self._component_roots.values():
            for decl in root.walk():
                if "[]" not in decl.item_display_name:
                    continue
                for field_decl in decl.decls.fields.values():
                    flat_name = decl.direct_field_map[field_decl.name]
                    if flat_name not in self._history_fields:
                        continue
                    count = len(decl.field_owners[field_decl.name])
                    self._indexed_history_fields[flat_name] = max(
                        count,
                        self._indexed_history_fields.get(flat_name, 0),
                    )
        self._component_bindings: dict[str, tuple[Component, ...]] = {}

        seen: set[str] = set()
        for field in decls.fields.values():
            if not field.name.startswith(("_pred_", "_ev_")):
                _check_name(field.name, field.kind.name)
            seen.add(field.name)
        for root in self._component_roots.values():
            label = ("component collection" if root.collection
                     else "component")
            _check_name(root.name, label)
            if root.name in seen:
                raise ValueError(f"duplicate field name '{root.name}'")
            seen.add(root.name)
        for field in decls.by_kind("queue", "pool", "store"):
            cap = field.capacity
            if cap is not None and not isinstance(cap, int) \
                    and decls.kind_of(cap) != "param":
                raise ValueError(f"capacity '{cap}' is neither an int nor "
                                 "a declared param")
        self._seen = seen
        self._processes: list[_ProcDecl] = []
        self._predicates: list[_BoundCallbackDecl] = []
        self._events: list[_BoundCallbackDecl] = []
        self._collect: Callable[..., Any] | None = None
        # (lowered collect, instance count); count > 1 collects take the
        # instance index as their second argument
        self._component_collects: list[tuple[Callable[..., Any], int]] = []
        self._compiled: _CompiledModel | None = None
        self._compile_lock = threading.RLock()
        self._owner_decl = _owner_declaration(type(self), decls)
        self._functions = _build_functions((self._owner_decl,))
        self._lowering_context = _CallbackLoweringContext(
            decls.values("queue", "resource", "pool", "store", "pqueues",
                         "condition", "event", "dataset",
                         attribute="kind.name"),
            self._owner_decl,
            self._functions,
        )
        self._bind_components()
        self._register_component_processes()
        self._register_model_callbacks()

    def _bind_components(self) -> None:
        for decl in self._component_roots.values():
            components = tuple(copy.copy(item) for item in decl.instances)
            self._component_bindings[decl.name] = components
            if decl.collection:
                setattr(self, decl.name, list(components))
            else:
                setattr(self, decl.name, components[0])
            self._bind_component_children(decl, components)

    def compile(self) -> Self:
        """Compile this fully constructed model once and return it.

        experiment() calls this automatically. Reuse the same model for
        parameter sweeps; construct a new model after changing callback code
        or compile-time configuration. Compilation errors propagate here.
        """
        self._compile()
        return self

    def _bind_component_children(
        self, decl: _OwnerDecl, parents: tuple[Component, ...]
    ) -> None:
        for child in decl.children:
            bound: list[Component] = []
            if child.collection:
                for parent_index, parent in enumerate(parents):
                    start = child.parent_offsets[parent_index]
                    length = child.parent_lengths[parent_index]
                    items: list[Component] = []
                    for item_index in range(length):
                        child_index = start + item_index
                        component = copy.copy(child.instances[child_index])
                        bound.append(component)
                        items.append(component)
                    setattr(parent, child.local_name, items)
            else:
                for parent_index, parent in enumerate(parents):
                    child_index = (child.parent_slots[parent_index]
                                   if child.parent_slots else parent_index)
                    if child_index < 0:
                        continue
                    component = copy.copy(child.instances[child_index])
                    bound.append(component)
                    setattr(parent, child.local_name, component)
            bound_tuple = tuple(bound)
            self._component_bindings[child.name] = bound_tuple
            self._bind_component_children(child, bound_tuple)

    def _register_component_processes(self) -> None:
        for root in self._component_roots.values():
            for decl in root.walk():
                self._register_component_decl_processes(decl)

    def _register_model_callbacks(self) -> None:
        callbacks = _callback_set(type(self))
        for kind in ("predicate", "event"):
            for decl in getattr(callbacks, f"{kind}s"):
                self._register_signal(decl.fn, kind, decl.bound_field, None)

        for decl in callbacks.processes:
            spec = decl.spec
            self._register_process(
                decl.fn, copies=spec.copies, priority=spec.priority,
                struct=spec.struct,
                spawn_field=decl.bound_field if spec.spawnable else None,
                process_field=None if spec.spawnable else spec.field,
                owner_cls=type(self))

        for decl in callbacks.collectors:
            self._register_collect(decl.fn)

    @staticmethod
    def _lower_shared_or_per_instance(
        indices: Sequence[int],
        lower_shared: Callable[[], Any],
        lower_instance: Callable[[int], Any],
    ) -> list[tuple[Any, int | None]]:
        """Lower one callback shared by several instances, or one per
        instance when sharing is rejected (a path that only some instances
        can resolve dynamically); index ``None`` marks the shared one."""
        if len(indices) > 1:
            try:
                return [(lower_shared(), None)]
            except ValueError:
                pass
        return [(lower_instance(index), index) for index in indices]

    def _register_component_decl_processes(self, decl: _OwnerDecl) -> None:
        """Lower and register one decl's process and collect methods.
        Spawnable process methods are always per-instance -- the spawn descriptor
        is what identifies the instance at runtime."""
        components = self._component_bindings[decl.name]
        lowering = self._lowering_context

        # Predicate/event native ABIs carry no component-instance context,
        # so collections receive one specialized callback per logical item.
        for index in range(decl.count):
            cls = decl.class_at(index)
            prefix = decl.process_names[index]
            for callback in (*_callback_set(cls).predicates, *_callback_set(cls).events):
                flat_field = decl.direct_field_map[callback.bound_field]
                field_index = (
                    decl.field_slots[callback.bound_field][index]
                    if self._field_shapes.get(flat_field) is not None
                    else None
                )
                lowered = _lower_component_signal(
                    prefix,
                    decl,
                    callback.name,
                    callback.fn,
                    kind=callback.kind,
                    instance_index=index,
                    context=lowering,
                )
                self._register_signal(
                    lowered, callback.kind, flat_field, field_index
                )

        groups = decl.specialization_groups()
        for variant, indices in enumerate(groups):
            first_index = indices[0]
            cls = decl.class_at(first_index)
            group_name = (
                decl.name if len(groups) == 1 else f"{decl.name}__variant_{variant}"
            )
            for callback in _callback_set(cls).processes:
                method_name, method, spec = (callback.name, callback.fn, callback.spec)
                counts = tuple(
                    spec.resolve_copies(
                        components[index], f"{decl.process_names[index]}.{method_name}"
                    )
                    for index in indices
                )
                process_local_field: str | None = spec.field
                # The Processes field when this group's instances own it.
                field_owners = (decl.field_owners.get(process_local_field)
                                if process_local_field is not None else None)
                owned_process_field = (
                    process_local_field
                    if field_owners is not None and first_index in field_owners
                    else None
                )

                def lower_instance(index: int) -> Any:
                    return _lower_component_process(
                        decl.process_names[index],
                        decl,
                        method_name,
                        method,
                        instance_index=index,
                        context=lowering,
                    )

                if spec.spawnable or (owned_process_field is not None
                                      and len(groups) > 1):
                    for position, index in enumerate(indices):
                        spawn_field = (
                            decl.direct_field_map[method_name]
                            if spec.spawnable
                            else None
                        )
                        spawn_index = None
                        if spawn_field is not None:
                            owners = decl.field_owners[method_name]
                            if len(owners) > 1:
                                spawn_index = decl.field_slots[method_name][index]
                        process_field = (
                            decl.direct_field_map[owned_process_field]
                            if owned_process_field is not None
                            else None
                        )
                        process_offset = (
                            decl.process_offsets[owned_process_field][index]
                            if owned_process_field is not None
                            else 0
                        )
                        self._register_process(
                            lower_instance(index), copies=counts[position],
                            priority=spec.priority, struct=spec.struct,
                            spawn_field=spawn_field,
                            spawn_index=spawn_index,
                            process_field=process_field,
                            process_offset=process_offset,
                        )
                    continue

                process_field = (
                    decl.direct_field_map[process_local_field]
                    if process_local_field is not None
                    else None
                )

                def lower_group() -> Any:
                    if len(indices) == 1:
                        return _lower_component_process(
                            group_name,
                            decl,
                            method_name,
                            method,
                            instance_index=first_index,
                            context=lowering,
                        )
                    return _lower_component_process(
                        group_name, decl, method_name, method,
                        copies_per_instance=counts,
                        instance_indices=indices,
                        context=lowering,
                    )

                lowered = self._lower_shared_or_per_instance(
                    indices,
                    lower_group,
                    (
                        lower_instance
                        if len(indices) > 1 or len(groups) == 1
                        else lambda _index: lower_group()
                    ),
                )
                for fn, index in lowered:
                    if index is None:
                        copies, offset = sum(counts), 0
                    else:
                        position = indices.index(index)
                        copies = counts[position]
                        offset = (
                            decl.process_offsets[process_local_field][index]
                            if process_local_field is not None
                            else 0
                        )
                    self._register_process(
                        fn, copies=copies, priority=spec.priority,
                        struct=spec.struct,
                        process_field=process_field,
                        process_offset=offset,
                    )

            for callback in _callback_set(cls).collectors:
                method_name, method = callback.name, callback.fn
                lowered = self._lower_shared_or_per_instance(
                    indices,
                    lambda: _lower_component_collect(
                        group_name,
                        decl,
                        method_name,
                        method,
                        instance_indices=indices,
                        context=lowering,
                    ),
                    lambda index: _lower_component_collect(
                        decl.process_names[index], decl, method_name, method,
                        instance_index=index,
                        context=lowering,
                    ),
                )
                for fn, index in lowered:
                    self._component_collects.append(
                        (fn, len(indices) if index is None else 1)
                    )

    @overload
    def component_schema(self) -> tuple[ComponentFieldSchema, ...]: ...

    @overload
    def component_schema(self, path: str) -> ComponentFieldSchema: ...

    def component_schema(
        self,
        path: str | None = None,
    ) -> ComponentFieldSchema | tuple[ComponentFieldSchema, ...]:
        """Describe component-owned flattened fields and packed ownership.

        ``path`` accepts the authoring path (with or without ``[]`` markers)
        or the flattened field name. With no path, returns every component
        field in declaration order.
        """
        schemas: list[ComponentFieldSchema] = []
        for root in self._component_roots.values():
            for decl in root.walk():
                for field_decl in decl.decls.fields.values():
                    name = field_decl.name
                    owners = decl.field_owners[name]
                    flat_name = decl.direct_field_map[name]
                    schemas.append(ComponentFieldSchema(
                        path=f"{decl.item_display_name}.{name}",
                        flattened_name=flat_name,
                        kind=field_decl.kind.name,
                        owners=owners,
                        concrete_types=tuple(
                            decl.instance_classes[index]
                            for index in owners),
                        logical_count=decl.count,
                        shape=self._field_shapes.get(flat_name),
                    ))
        if path is None:
            return tuple(schemas)
        normalized = path.replace("[]", "")
        matches = [
            schema for schema in schemas
            if (path == schema.flattened_name
                or normalized == schema.path.replace("[]", ""))
        ]
        if not matches:
            raise KeyError(f"unknown component field: {path}")
        if len(matches) > 1:
            raise KeyError(f"ambiguous component field: {path}")
        return matches[0]

    def _next_capture_slot(self) -> int:
        return sum(spec.slot_count for spec in (
            *self._history_captures.values(),
            *self._dataset_captures.values(),
        ))

    def _register_history_capture(
        self,
        name: str,
        binding: str,
        indexed_count: int | None = None,
    ) -> int:
        spec = self._history_captures.get(name)
        if spec is not None:
            if ((spec.shape is None) != (indexed_count is None)
                    or (spec.shape is not None
                        and spec.shape != (indexed_count,))):
                raise ValueError(
                    f"history field '{name}' is captured with inconsistent "
                    "indexing")
            return spec.slot
        if indexed_count is not None and indexed_count < 1:
            raise ValueError(
                f"indexed history field '{name}' has no collection items")
        slot = self._next_capture_slot()
        self._history_captures[name] = HistoryCaptureSpec(
            name=name,
            binding=binding,
            slot=slot,
            columns=3,
            shape=(indexed_count,) if indexed_count is not None else None,
        )
        return slot

    def _register_dataset_capture(self, name: str, binding: str) -> int:
        spec = self._dataset_captures.get(name)
        if spec is not None:
            return spec.slot
        slot = self._next_capture_slot()
        self._dataset_captures[name] = HistoryCaptureSpec(
            name=name,
            binding=binding,
            slot=slot,
            columns=1,
            shape=None,
        )
        return slot

    def _lower_model_callback(
        self,
        fn: Callable[..., Any],
        *,
        collect: bool = False,
    ) -> Callable[..., Any]:
        """Lower one root callback through a single AST and compilation."""
        context = self._lowering_context
        try:
            node = copy.deepcopy(_function_def_from_source(fn))
        except (OSError, TypeError) as exc:
            raise ValueError(
                f"model '{self.name}' callback '{fn.__qualname__}' needs "
                "inspectable source"
            ) from exc
        if not node.args.args:
            return fn

        namespace = _closure_namespace(fn)
        node, changed, called_functions = _lower_model_component_refs_in_node(
            node,
            model_name=self.name,
            owner_decl=context.owner_decl,
            functions=context.functions,
        )
        if changed:
            namespace.update(
                _model_lowering_namespace(self._component_roots, context.functions)
            )

        if collect:
            node, lowered = lower_history_capture_calls(
                node, model_name=self.name,
                history_fields=self._history_fields,
                indexed_history_fields=self._indexed_history_fields,
                register=self._register_history_capture,
                namespace=namespace)
            changed |= lowered
            node, lowered = lower_dataset_capture_calls(
                node, model_name=self.name,
                dataset_fields=set(self.datasets),
                register=self._register_dataset_capture,
                namespace=namespace)
            changed |= lowered

        node, lowered = _lower_owner_methods(
            node,
            fn,
            env_name=node.args.args[0].arg,
            label=f"model '{self.name}' callback '{fn.__name__}'",
            owner_name=self.name,
            context=context,
            namespace=namespace,
        )
        changed |= lowered
        if not changed:
            return fn

        generated = _compile_model_callback(fn, node, self.name, namespace)
        _set_function_calls(
            generated, (*_function_calls(fn), *called_functions))
        return generated

    def _process_dag_blocks(
        self,
        entity_kinds: Mapping[str, str],
        function_members: Mapping[str, Iterable[str]],
    ) -> tuple[ProcessDAGBlock, ...]:
        process_names = {process.name for process in self._processes}
        return tuple(
            ProcessDAGBlock(
                decl.display_name or decl.name,
                tuple(dict.fromkeys((
                    *decl.dag_members(process_names, entity_kinds),
                    *function_members.get(decl.name, ()),
                ))),
                kind=("component_collection" if decl.collection
                      else "component"),
            )
            for root in self._component_roots.values()
            for decl in root.walk()
        )

    # --- Internal callback registration ----------------------------------
    def _register_process(
        self,
        fn: Callable[..., Any],
        *,
        copies: int,
        priority: int,
        struct: Any,
        spawn_field: str | None = None,
        spawn_index: int | None = None,
        process_field: str | None = None,
        process_offset: int = 0,
        owner_cls: type | None = None,
    ) -> None:
        """Register one (lowered) ``@sim.process`` callback.

        ``fn`` takes ``(self)`` or ``(self, idx)`` plus an optional final
        ``sim.Struct`` view parameter. A spawnable process publishes its
        spawn descriptor in ``spawn_field`` (element ``spawn_index`` of a
        shaped field) and is started by ``sim.spawn()`` instead of at trial
        setup; otherwise ``process_field`` optionally receives the handles
        of its ``copies`` starting at ``process_offset``."""
        if struct is not None and not _is_struct_class(struct):
            raise ValueError("struct= expects a sim.Struct subclass")
        name = fn.__name__
        spawnable = spawn_field is not None

        # A process named like the field it publishes shares that already
        # registered name; any other process claims its own name.
        if name not in (process_field, spawn_field):
            self._register_name(name, "process")

        localns = ({base.__name__: base for base in owner_cls.__mro__}
                   if owner_cls is not None else None)
        signature = (
            "process functions take (self), (self, idx), and optionally "
            "a final view parameter annotated with a sim.Struct subclass")
        own, base_arg_count = _process_signature(
            fn, 1, f"process '{name}'", signature,
            localns)
        injected = own is not None
        if injected:
            if struct is not None and struct is not own:
                raise ValueError(f"process '{name}': struct= and the view "
                                 "annotation disagree")
            struct = own
        indexed = base_arg_count == 2
        if spawnable:
            if copies != 1:
                raise ValueError(f"spawnable process '{name}' cannot take "
                                 "copies; sim.spawn() creates them")
            if indexed:
                raise ValueError(f"spawnable process '{name}' takes (self) "
                                 "or (self, view), not a copy index")
        self._processes.append(_ProcDecl(
            name, self._lower_model_callback(fn), copies, priority, indexed,
            struct, injected, spawnable, spawn_field, spawn_index,
            process_field, process_offset))

    def process_dag(self, *, validate: bool = True) -> ProcessDAG:
        """Infer a resource-aware graph from class-declared processes.

        ``validate`` is accepted for API stability. Inferred graphs may contain
        legitimate resource cycles, so acyclicity is checked only when callers
        explicitly ask for :meth:`ProcessDAG.topological_order`.
        """
        entity_kinds = {f.name: f.kind.name
                        for f in self._decls.fields.values()
                        if f.kind.dag_entity}
        # Registered events without a declared field publish their address
        # in a hidden _ev_<name> field.
        entity_kinds.update({callback.field: "event"
                             for callback in self._events})
        spawnable_field_processes: dict[str, set[str]] = {}
        spawnable_index_processes: dict[tuple[str, int], set[str]] = {}
        process_field_processes: dict[str, set[str]] = {}
        process_index_processes: dict[tuple[str, int], set[str]] = {}
        for process in self._processes:
            if process.spawn_field is not None:
                spawnable_field_processes.setdefault(
                    process.spawn_field, set()).add(process.name)
                if process.spawn_index is not None:
                    spawnable_index_processes.setdefault(
                        (process.spawn_field, process.spawn_index),
                        set()).add(process.name)
            elif process.process_field is not None:
                process_field_processes.setdefault(
                    process.process_field, set()).add(process.name)
                for slot in range(process.process_offset,
                                  process.process_offset + process.copies):
                    process_index_processes.setdefault(
                        (process.process_field, slot),
                        set()).add(process.name)

        function_nodes: dict[str, ProcessDAGNode] = {}
        function_edges: list[ProcessDAGEdge] = []
        function_members: dict[str, list[str]] = {}

        def add_function_member(decl_name: str, key: str) -> None:
            members = function_members.setdefault(decl_name, [])
            if key not in members:
                members.append(key)

        for spec in self._functions.values():
            function_node = ProcessDAGNode(spec.graph_name, "function")
            function_nodes[function_node.key] = function_node
            add_function_member(spec.decl.name, function_node.key)
            for access in spec.reads:
                if access.field in access.decl.constants:
                    continue
                field_kind = access.decl.decls.kind_of(access.field)
                if field_kind not in ("param", "output", "state", "fstate"):
                    continue
                flat_name = access.decl.direct_field_map[access.field]
                field_node = ProcessDAGNode(flat_name, field_kind)
                function_nodes[field_node.key] = field_node
                add_function_member(access.decl.name, field_node.key)
                edge = ProcessDAGEdge(
                    field_node.key, function_node.key, "read")
                if edge not in function_edges:
                    function_edges.append(edge)
            for callee in spec.callees:
                edge = ProcessDAGEdge(function_node.key, f"function:{callee}", "call")
                if edge not in function_edges:
                    function_edges.append(edge)

        for process in self._processes:
            for called in _function_calls(process.fn):
                edge = ProcessDAGEdge(
                    f"process:{process.name}",
                    f"function:{called}",
                    "call",
                )
                if edge not in function_edges:
                    function_edges.append(edge)
        return infer_process_dag(
            self._processes,
            entity_kinds=entity_kinds,
            process_fields=self._process_fields,
            spawnable_fields=self._spawnable_fields,
            spawnable_field_processes=spawnable_field_processes,
            spawnable_index_processes=spawnable_index_processes,
            process_field_processes=process_field_processes,
            process_index_processes=process_index_processes,
            event_callbacks=((callback.field, callback.fn)
                             for callback in self._events),
            blocks=self._process_dag_blocks(
                entity_kinds, function_members),
            extra_nodes=function_nodes.values(),
            extra_edges=function_edges,
        )

    def _register_signal(
        self, fn: Callable[..., Any], kind: str, field: str, index: int | None
    ) -> None:
        """Register a predicate or event callback published in ``field``
        (element ``index`` of a shaped field)."""
        name = fn.__name__
        takes_data = False
        if kind == "predicate":
            _callback_arg_count(fn, (1,), "predicate functions take (self)")
            localns = {base.__name__: base for base in type(self).__mro__}
            if get_type_hints(fn, localns=localns).get("return") is not bool:
                raise ValueError("predicate functions must return bool")
        else:
            takes_data = (
                _callback_arg_count(
                    fn, (1, 2), "event functions take (self) or (self, data)"
                )
                == 2
            )
        self._register_name(name, kind)
        registered = self._predicates if kind == "predicate" else self._events
        registered.append(_BoundCallbackDecl(
            name, self._lower_model_callback(fn), field, index, takes_data))

    def _register_collect(self, fn: Callable[..., Any]) -> None:
        """Register the model ``@sim.collect`` callback ``def fn(self)``; it
        runs after every component collector, so it can aggregate them."""
        _callback_arg_count(
            fn, (1,), "model collect functions take (self)")
        self._collect = self._lower_model_callback(fn, collect=True)

    @property
    def _collects(self) -> list[tuple[Callable[..., Any], int]]:
        """All end-of-trial collect functions in execution order --
        component-owned collects first, the model-level one last -- each
        with the number of instances it is called for (multi-instance
        collects take the instance index as their second argument)."""
        fns = list(self._component_collects)
        if self._collect is not None:
            fns.append((self._collect, 1))
        return fns

    def _register_name(self, name: str, kind: str) -> None:
        _check_name(name, kind)
        if name in self._seen:
            raise ValueError(f"duplicate name '{name}'")
        self._seen.add(name)

    def _runtime_process_descriptors(
        self,
        dtype: np.dtype,
        callbacks: Mapping[str, Any],
        native_names: Mapping[str, str],
    ) -> tuple[np.ndarray, int]:
        """Build the fixed-width process table consumed by lifecycle ABI."""
        rows: list[list[int]] = []
        handle_start = 0
        offsets = _field_offsets(dtype)
        for process in self._processes:
            if process.spawnable:
                continue
            destination = -1
            if process.process_field is not None:
                destination = (
                    offsets[process.process_field]
                    + 8 * process.process_offset
                )
            rows.append([
                callbacks[process.name].address,
                _b.cstring(native_names[process.name]),
                process.alloc_size,
                process.priority,
                process.copies,
                int(process.indexed),
                handle_start,
                destination,
            ])
            handle_start += process.copies
        descriptors = np.asarray(rows, dtype=np.int64)
        if not rows:
            descriptors = np.zeros(
                (1, _PROCESS_DESCRIPTOR_WIDTH), dtype=np.int64)
        return descriptors, handle_start

    def _runtime_entity_descriptors(self, dtype: np.dtype) -> np.ndarray:
        """Build entity setup/recording/cleanup descriptors for one model."""
        offsets = _field_offsets(dtype)
        kind_codes = {
            "queue": _ENTITY_BUFFER,
            "resource": _ENTITY_RESOURCE,
            "pool": _ENTITY_RESOURCEPOOL,
            "store": _ENTITY_OBJECTQUEUE,
            "dataset": _ENTITY_DATASET,
            "condition": _ENTITY_CONDITION,
        }
        logical_names = [
            native_name
            for field in self._decls.by_kind(
                "queue", "resource", "pool", "store", "condition")
            for _key, native_name in self._field_name_keys(field.name)
        ]
        logical_names.extend(
            f"{field}_{index}"
            for field, count in self.pqueues.items()
            for index in range(count)
        )
        native_names = _native_names(logical_names)
        rows: list[list[int]] = []
        for field in self._decls.by_kind(
                "queue", "resource", "pool", "store", "dataset",
                "condition"):
            names = self._field_name_keys(field.name)
            for index, (_key, native_name) in enumerate(names):
                capacity_mode = _CAPACITY_CONSTANT
                capacity = -1
                if isinstance(field.capacity, int):
                    capacity = field.capacity
                elif isinstance(field.capacity, str):
                    capacity_mode = _CAPACITY_FIELD
                    slot = index
                    if field.capacity_slots is not None:
                        slot = field.capacity_slots[index]
                    capacity = offsets[field.capacity] + 8 * slot
                rows.append([
                    kind_codes[field.kind.name],
                    (0 if field.kind.name == "dataset"
                     else _b.cstring(native_names[native_name])),
                    capacity_mode,
                    capacity,
                    offsets[field.name] + 8 * index,
                ])
        for field, count in self.pqueues.items():
            offset = offsets[field]
            for index in range(count):
                rows.append([
                    _ENTITY_PRIORITYQUEUE,
                    _b.cstring(native_names[f"{field}_{index}"]),
                    _CAPACITY_CONSTANT,
                    -1,
                    offset + 8 * index,
                ])
        descriptors = np.asarray(rows, dtype=np.int64)
        if not rows:
            descriptors = np.zeros(
                (1, _ENTITY_DESCRIPTOR_WIDTH), dtype=np.int64)
        return descriptors

    def _field_spec(self, name: str, fmt: str) -> tuple[Any, ...]:
        shape = self._field_shapes.get(name)
        if shape is None:
            return (name, fmt)
        return (name, fmt, shape)

    def _param_axes(self, param_values: Mapping[str, Any]) -> list[np.ndarray]:
        return [
            _as_param_axis(param_values[p], self._field_shapes.get(p), p)
            for p in self.params
        ]

    def _resolve_param_values(
        self,
        param_values: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Fill omitted Params from declared defaults and validate names."""
        resolved = dict(param_values)
        schemas = {
            schema.flattened_name: schema
            for schema in self.component_schema()
            if schema.kind == "param"
        }
        for name, value in tuple(resolved.items()):
            schema = schemas.get(name)
            if schema is not None and isinstance(value, Mapping):
                resolved[name] = self._resolve_indexed_component_param(
                    name, value, schema)
        for name, default in self.param_defaults.items():
            resolved.setdefault(name, default)
        missing = set(self.params) - set(resolved)
        if missing:
            raise ValueError(f"missing parameter values: {sorted(missing)}")
        unknown = set(param_values) - set(self.params) - set(self.traces)
        if unknown:
            raise ValueError(f"unknown parameters: {sorted(unknown)}")
        return resolved

    def _resolve_indexed_component_param(
        self,
        name: str,
        value: Mapping[Any, Any],
        schema: ComponentFieldSchema,
    ) -> Any:
        """Convert ``{logical component index: value}`` to packed order."""
        invalid = [key for key in value if type(key) is not int]
        if invalid:
            raise TypeError(
                f"component parameter '{name}' owner indexes must be ints")
        owner_set = set(schema.owners)
        extra = set(value) - owner_set
        if extra:
            raise ValueError(
                f"component parameter '{name}' has non-owning indexes: "
                f"{sorted(extra)}")

        defaults = self.param_defaults.get(name, _MISSING)
        if defaults is _MISSING:
            default_by_owner: dict[int, Any] = {}
        else:
            packed_defaults = (
                (defaults,) if schema.shape is None else tuple(defaults))
            default_by_owner = dict(zip(schema.owners, packed_defaults))
        missing = owner_set - set(value) - set(default_by_owner)
        if missing:
            raise ValueError(
                f"component parameter '{name}' is missing owner indexes: "
                f"{sorted(missing)}")
        ordered = [
            value[owner] if owner in value else default_by_owner[owner]
            for owner in schema.owners
        ]
        if schema.shape is None:
            return ordered[0]
        arrays = [np.asarray(item, dtype=np.float64) for item in ordered]
        if all(item.ndim == 0 for item in arrays):
            return np.asarray(ordered, dtype=np.float64)
        if (all(item.ndim == 1 for item in arrays)
                and len({item.shape for item in arrays}) == 1):
            return np.column_stack(arrays)
        raise ValueError(
            f"component parameter '{name}' indexed values must be all "
            "scalars or equal-length 1-D sweep arrays")

    def _trace_field_spec(self, name: str) -> tuple[Any, ...]:
        shape = self._field_shapes.get(name)
        if shape is None:
            return (name, "<i8", (2,))
        return (name, "<i8", (*shape, 2))

    def _field_name_keys(self, name: str) -> list[tuple[str, str]]:
        shape = self._field_shapes.get(name)
        if shape is None:
            return [(f"NAME_{name}", name)]
        return [(f"NAME_{name}_{i}", f"{name}_{i}")
                for i in range(shape[0])]

    @property
    def dtype(self) -> np.dtype:
        # (name, format) or (name, format, shape) numpy field specs
        fields: list[Any] = list(_LIFECYCLE_ABI_FIELDS)
        for f in self._decls.by_kind("param", "output", "queue", "resource",
                                     "pool", "store", "dataset", "condition",
                                     "state", "fstate"):
            fields.append((f.name, f.kind.fmt) if f.shape is None
                          else (f.name, f.kind.fmt, f.shape))
        fields += [self._trace_field_spec(t) for t in self.traces]
        fields += [(f.name, "<i8", (f.count,))
                   for f in self._decls.by_kind("pqueues")]
        fields += [self._field_spec(p, "<i8")
                   for p in self._predicate_fields]
        fields += [self._field_spec(e, "<i8") for e in self._event_fields]
        fields += [self._field_spec(s, "<i8") for s in self._spawnable_fields]
        if self._history_captures or self._dataset_captures:
            fields += [
                (HISTORY_CAPTURE_TRIAL_FIELD, "<u8"),
                (HISTORY_CAPTURE_STORE_FIELD, "<i8"),
            ]
        process_fields_added: set[str] = set()
        for p in self._processes:
            if p.spawnable:
                continue
            if p.process_field is not None:
                if p.process_field not in process_fields_added:
                    shape = self._field_shapes.get(p.process_field)
                    if shape is None:
                        fields += [(p.process_field, "<i8", (p.copies,))]
                    else:
                        fields += [self._field_spec(p.process_field, "<i8")]
                    process_fields_added.add(p.process_field)
        return np.dtype(fields)

    # --- Compilation --------------------------------------------------------
    def _compile_callbacks(
        self,
        rec: Any,
        extra_jobs: Sequence[tuple[Any, Callable[..., Any]]],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[Any]]:
        """Compile lowered class callbacks to Cimba's native callback ABIs."""
        trial_ptr = types.CPointer(rec)
        proc_sig = types.intp(types.intp, trial_ptr)
        proc_sig_ix = types.intp(types.intp, types.CPointer(types.int64))
        pred_sig = types.boolean(types.intp, types.intp, trial_ptr)
        # cmb_event_func: subject is the env pointer, object the data word
        ev_sig = types.void(trial_ptr, types.intp)
        rec_from_addr = ptr_caster(rec)

        def make_pred(inner):
            def pred(cnd, prc, ctxp):
                return bool(inner(carray(ctxp, 1)[0]))
            return pred

        def make_event(inner, takes_data):
            def ev(subject, data):
                if takes_data:
                    inner(carray(subject, 1)[0], data)
                else:
                    inner(carray(subject, 1)[0])
            return ev

        def make_proc_indexed(p):
            """Build the indexed-process adapter; workers compile its body."""
            inner = njit(p.fn)
            assert isinstance(inner, Dispatcher)
            struct = p.struct if p.injected else None

            def proc(me, ctxp):
                pair = carray(ctxp, 2)
                env = carray(rec_from_addr(pair[0]), 1)[0]
                if struct is None:
                    inner(env, pair[1])
                else:
                    inner(env, pair[1], struct(me))
                return 0

            return proc

        def raise_process_compile_error(p, exc):
            calls = _function_calls(p.fn)
            if calls:
                raise TypeError(
                    f"process '{p.name}' has an invalid call to "
                    "component function(s): " + ", ".join(calls)
                ) from exc
            raise exc

        proc_cfuncs: dict[str, Any] = {}
        pred_cfuncs: dict[Any, Any] = {}
        event_cfuncs: dict[Any, Any] = {}
        extra_callbacks = [None] * len(extra_jobs)

        jobs: list[_CFuncJob] = []
        for p in self._processes:
            try:
                signature, function = (
                    (proc_sig_ix, make_proc_indexed(p))
                    if p.indexed
                    else (proc_sig, _direct_process_callback(
                        p.fn, p.name, p.struct if p.injected else None))
                )
                jobs.append(
                    _CFuncJob("process", p.name, p.name, signature, function, p)
                )
            except Exception as exc:
                raise_process_compile_error(p, exc)
        signal_groups = (
            (
                "predicate",
                self._predicates,
                pred_sig,
                lambda item: make_pred(njit(item.fn)),
            ),
            (
                "event",
                self._events,
                ev_sig,
                lambda item: make_event(njit(item.fn), item.takes_data),
            ),
        )
        for category, declarations, signature, adapt in signal_groups:
            for callback in declarations:
                jobs.append(
                    _CFuncJob(
                        category,
                        callback.name,
                        callback.key,
                        signature,
                        adapt(callback),
                    )
                )
        for index, (signature, function) in enumerate(extra_jobs):
            jobs.append(_CFuncJob("extra", str(index), index, signature, function))

        try:
            callbacks = _compile_cfuncs(
                [(job.signature, job.function) for job in jobs],
            )
        except Exception:
            # Recompile in the parent so invalid user code retains its full
            # process-specific diagnostic instead of a worker traceback.
            callbacks = []
            for job in jobs:
                try:
                    callbacks.append(
                        _compile_parallel_cfunc(job.signature, job.function)
                    )
                except Exception as exc:
                    if job.process is not None:
                        raise_process_compile_error(job.process, exc)
                    if job.category != "extra":
                        raise TypeError(
                            f"callback '{job.name}' failed to compile"
                        ) from exc
                    raise
        targets = {
            "process": proc_cfuncs,
            "predicate": pred_cfuncs,
            "event": event_cfuncs,
        }
        for job, callback in zip(jobs, callbacks):
            if job.category == "extra":
                extra_callbacks[job.key] = callback
            else:
                targets[job.category][job.key] = callback
        return proc_cfuncs, pred_cfuncs, event_cfuncs, extra_callbacks

    def _compile(self) -> _CompiledModel:
        with self._compile_lock:
            return self._compile_model()

    def _check_bindings(self) -> None:
        """Reject declared callback fields that no callback fills."""
        if not self._processes:
            raise ValueError("model has no processes")
        for kind, fields, callbacks in (
            ("Predicate", self._predicate_fields, self._predicates),
            ("Event", self._event_fields, self._events),
        ):
            bound = {callback.key for callback in callbacks}
            unbound = [
                field
                for field in fields
                if not (
                    {field}
                    if (shape := self._field_shapes.get(field)) is None
                    else {(field, index) for index in range(shape[0])}
                ).issubset(bound)
            ]
            if unbound:
                raise ValueError(
                    f"{kind} field(s) {unbound} declared but no @{kind.lower()} of that name registered"
                )
        process_field_slots: dict[str, dict[int, int]] = {}
        for process in self._processes:
            if process.spawnable or process.process_field is None:
                continue
            slots = process_field_slots.setdefault(process.process_field, {})
            for slot in range(process.process_offset,
                              process.process_offset + process.copies):
                slots[slot] = slots.get(slot, 0) + 1
        unbound_process_fields: list[str] = []
        bad_process_fields: list[str] = []
        for field in self._process_fields:
            slots = process_field_slots.get(field)
            if slots is None:
                unbound_process_fields.append(field)
                continue
            shape = self._field_shapes.get(field)
            expected = len(slots) if shape is None else shape[0]
            if (set(slots) != set(range(expected))
                    or any(count != 1 for count in slots.values())):
                bad_process_fields.append(field)
        if unbound_process_fields:
            raise ValueError(
                f"Processes field(s) {unbound_process_fields} declared but "
                "no @process of that name registered")
        if bad_process_fields:
            raise ValueError(
                f"Processes field(s) {bad_process_fields} have incomplete or "
                "overlapping process handle bindings")
        bound_spawn_slots = {
            (p.spawn_field, p.spawn_index)
            for p in self._processes
            if p.spawnable
        }
        expected_spawn_slots: list[tuple[str, int | None]] = []
        for field in self._spawnable_fields:
            shape = self._field_shapes.get(field)
            if shape is None:
                expected_spawn_slots.append((field, None))
            else:
                expected_spawn_slots.extend(
                    (field, index) for index in range(shape[0]))
        unbound = [
            _spawnable_slot_label(field, index)
            for field, index in expected_spawn_slots
            if (field, index) not in bound_spawn_slots
        ]
        if unbound:
            raise ValueError(f"Spawnable field(s) {unbound} declared but "
                             "no @process of that name registered")

    def _compile_model(self) -> _CompiledModel:
        if self._compiled is not None:
            return self._compiled
        self._check_bindings()

        dtype = self.dtype
        rec = from_dtype(dtype)
        trial_ptr = types.CPointer(rec)
        # Callees precede callers in this insertion-ordered mapping. Explicit
        # signatures retain the public scalar annotation contract.
        for spec in self._functions.values():
            try:
                spec.helper.compile(spec.return_type(
                    rec, types.int64, *spec.argument_types))
                spec.helper.disable_compile()
            except Exception as exc:
                raise TypeError(
                    f"function '{spec.graph_name}' failed Numba nopython compilation"
                ) from exc

        direct_collects = [
            (index, _direct_collect_callback(fn, index, count))
            for index, (fn, count) in enumerate(self._collects)
        ]
        native_process_names = _native_names(
            process.name for process in self._processes)

        proc_cfuncs, pred_cfuncs, event_cfuncs, lifecycle = self._compile_callbacks(
            rec,
            [
                *_LIFECYCLE_JOBS,
                *((types.void(trial_ptr), function)
                  for _index, function in direct_collects),
            ],
        )
        (recording_event, trial_initialize, trial_entities, trial_processes,
         trial_teardown, trial, trial_stop, trial_process_cleanup,
         trial_collect, *collect_callbacks) = lifecycle
        compiled_collects = {
            index: callback
            for (index, _function), callback in zip(
                direct_collects, collect_callbacks)
        }
        spawn_descs = {
            p.name: np.array([proc_cfuncs[p.name].address,
                              _b.cstring(native_process_names[p.name]),
                              p.alloc_size],
                             dtype=np.int64)
            for p in self._processes if p.spawnable
        }
        spawn_assignments = tuple(
            (p.spawn_field, p.spawn_index, p.name)
            for p in self._processes
            if p.spawnable and p.spawn_field is not None
        )
        process_descriptors, process_handle_count = \
            self._runtime_process_descriptors(
                dtype, proc_cfuncs, native_process_names)
        entity_descriptors = self._runtime_entity_descriptors(dtype)
        entity_descriptor_count = (
            sum(
                1 if field.shape is None else field.shape[0]
                for field in self._decls.by_kind(
                    "queue", "resource", "pool", "store", "dataset",
                    "condition")
            )
            + sum(self.pqueues.values())
        )
        collect_descriptors = np.asarray(
            [compiled_collects[index].address
             for index in range(len(compiled_collects))],
            dtype=np.int64,
        )
        if collect_descriptors.size == 0:
            collect_descriptors = np.zeros(1, dtype=np.int64)

        self._compiled = _CompiledModel(
            dtype=dtype,
            trial=trial,
            lifecycle=(
                recording_event,
                trial_initialize,
                trial_entities,
                trial_processes,
                trial_teardown,
                trial_stop,
                trial_process_cleanup,
                trial_collect,
            ),
            processes=proc_cfuncs,
            predicates=pred_cfuncs,
            events=event_cfuncs,
            collects=compiled_collects,
            collect_descriptors=collect_descriptors,
            process_descriptors=process_descriptors,
            process_handle_count=process_handle_count,
            entity_descriptors=entity_descriptors,
            entity_descriptor_count=entity_descriptor_count,
            spawn_descriptors=spawn_descs,
            spawn_assignments=spawn_assignments,
        )
        return self._compiled

    # --- Experiments ----------------------------------------------------------
    def trial_seeds(self, *,
                    seed: int,
                    replications: int = 1,
                    **param_values: Any) -> np.ndarray:
        """The per-trial seeds that experiment() with this seed, these
        swept parameter values, and this replication count will assign,
        in trial order (design-point-major, replications innermost).

        Use this to generate trace data outside experiment() -- e.g. in
        parallel when a generator is expensive -- while staying
        reproducible from the experiment seed: feed
        ``trace_rng(seeds[i], field_name)`` to the generator and pass
        the finished rows to experiment() with the same seed. Trace
        fields passed here are ignored, so the experiment() keyword
        arguments can be reused as-is."""
        param_values = self._resolve_param_values(param_values)
        if replications < 1:
            raise ValueError("replications must be >= 1")
        n_points = _n_design_points(self._param_axes(param_values))
        return _draw_trial_seeds(seed, n_points * replications)

    def experiment(self,
                   *,
                   replications: int = 1,
                   duration: float | None = 1.0e6,
                   warmup: float = 1.0e3,
                   cooldown: float = 0.0,
                   start_time: float = 0.0,
                   seed: int | None = None,
                   **param_values: "ArrayLike | Callable[..., ArrayLike]",
                   ) -> "Experiment[_ExperimentResultT]":
        """Build an experiment: the cross product of the swept parameter
        values (scalars are held fixed), replicated with distinct seeds.
        Omitted Params use their declaration defaults; Params without a
        default remain required.

        ``duration=None, warmup=0, cooldown=0`` runs until the event queue is
        empty. This mode has no automatic entity-history recording or dataset
        reset window; explicit model sampling and @collect callbacks still run.
        Suspended processes are cleaned up when the queue empties. A model
        which keeps scheduling events needs a finite duration instead.

        Trace fields take their replay data here as well: a 1-D array
        shared by every trial, a 2-D array whose row i replays in trial i
        (trial order is design-point-major with replications innermost),
        or a sequence of 1-D arrays for ragged per-trial traces.

        A trace field also accepts a callable ``f(rng)`` or
        ``f(rng, trial_index)`` returning a 1-D array; it is invoked once
        per trial with ``trace_rng(trial_seed, field_name)``, a numpy
        Generator derived from that trial's own seed, so the experiment
        ``seed`` reproduces the generated traces too. A callable's
        ``trace_rng_name`` attribute overrides the field name in that
        derivation (see ``trace_rng``)."""
        if duration is None and (warmup != 0.0 or cooldown != 0.0):
            raise ValueError("duration=None requires warmup=0 and cooldown=0")
        if duration is not None and not np.isfinite(duration):
            raise ValueError("duration must be finite, or None to run until idle")
        compiled = self._compile()

        param_values = self._resolve_param_values(param_values)
        missing_traces = set(self.traces) - set(param_values)
        if missing_traces:
            raise ValueError(f"missing trace values: "
                             f"{sorted(missing_traces)}")
        if replications < 1:
            raise ValueError("replications must be >= 1")

        axes = self._param_axes(param_values)
        n_points = _n_design_points(axes)
        n_trials = n_points * replications

        trials = np.zeros(n_trials, dtype=compiled.dtype)
        trials["start_time"] = start_time
        trials["warmup_s"] = warmup
        trials["duration_s"] = np.inf if duration is None else duration
        trials["cooldown_s"] = cooldown
        for field, callback in zip(_LIFECYCLE_FIELDS, compiled.lifecycle):
            trials[field] = callback.address
        process_handle_count = compiled.process_handle_count
        process_width = max(1, process_handle_count)
        runtime_handles = np.zeros(
            (n_trials, process_width), dtype=np.int64)
        runtime_contexts = np.zeros(
            (n_trials, process_width, 2), dtype=np.int64)
        trial_indexes = np.arange(n_trials, dtype=np.int64)
        trials[_PROCESS_DESCRIPTORS_FIELD] = \
            compiled.process_descriptors.ctypes.data
        trials[_PROCESS_DESCRIPTOR_COUNT_FIELD] = sum(
            not process.spawnable for process in self._processes)
        trials[_PROCESS_HANDLES_FIELD] = (
            runtime_handles.ctypes.data
            + trial_indexes * runtime_handles.strides[0]
        )
        trials[_PROCESS_HANDLE_COUNT_FIELD] = process_handle_count
        trials[_PROCESS_CONTEXTS_FIELD] = (
            runtime_contexts.ctypes.data
            + trial_indexes * runtime_contexts.strides[0]
        )
        trials[_HAS_SPAWNED_FIELD] = int(
            any(process.spawnable for process in self._processes))
        trials[_COLLECT_DESCRIPTORS_FIELD] = \
            compiled.collect_descriptors.ctypes.data
        trials[_COLLECT_DESCRIPTOR_COUNT_FIELD] = \
            len(compiled.collects)
        trials[_ENTITY_DESCRIPTORS_FIELD] = \
            compiled.entity_descriptors.ctypes.data
        trials[_ENTITY_DESCRIPTOR_COUNT_FIELD] = \
            compiled.entity_descriptor_count
        if self._history_captures or self._dataset_captures:
            trials[HISTORY_CAPTURE_TRIAL_FIELD] = np.arange(
                n_trials, dtype=np.uint64)
            trials[HISTORY_CAPTURE_STORE_FIELD] = 0
        # Index each axis directly rather than meshgrid'ing all of them: the
        # design points are a mixed-radix count over the axis sizes, which is
        # exactly meshgrid(indexing="ij") ravel order. Singleton axes cost a
        # broadcast instead of a mesh dimension, so numpy's 32-dimension
        # ceiling no longer applies to the declared parameter count.
        points = np.arange(n_points, dtype=np.int64)
        stride = 1
        for p, axis in zip(reversed(self.params), reversed(axes)):
            size = axis.shape[0]
            if size == 1:
                selected = np.repeat(axis, n_points, axis=0)
            else:
                selected = axis[(points // stride) % size]
            trials[p] = np.repeat(selected, replications, axis=0)
            stride *= size
        for o in self.outputs:
            trials[o] = np.nan
        for field, pred in compiled.predicates.items():
            if isinstance(field, tuple):
                trials[field[0]][:, field[1]] = pred.address
            else:
                trials[field] = pred.address
        for field, ev in compiled.events.items():
            if isinstance(field, tuple):
                trials[field[0]][:, field[1]] = ev.address
            else:
                trials[field] = ev.address
        for field, index, process_name in compiled.spawn_assignments:
            desc = compiled.spawn_descriptors[process_name]
            if index is None:
                trials[field] = desc.ctypes.data
            else:
                trials[field][:, index] = desc.ctypes.data

        trials["seed"] = _draw_trial_seeds(seed, n_trials)

        trace_rows: list[np.ndarray] = []
        for tname in self.traces:
            value = param_values[tname]
            shape = self._field_shapes.get(tname)
            slots = 1 if shape is None else shape[0]
            rows = _as_trace_grid(value, trials["seed"], n_trials, slots, tname)
            flat_rows = [row for trial_rows in rows for row in trial_rows]
            trace_rows.extend(flat_rows)
            trials[tname] = np.asarray(
                [(row.ctypes.data, row.size) for row in flat_rows], dtype=np.int64
            ).reshape(trials[tname].shape)

        swept = tuple(p for p, axis in zip(self.params, axes)
                      if axis.shape[0] > 1)
        return Experiment(self, trials, compiled.trial.address,
                          keepalive=[*trace_rows, runtime_handles,
                                     runtime_contexts],
                          replications=replications,
                          swept=swept,
                          history_captures=tuple(
                              self._history_captures.values()),
                          dataset_captures=tuple(
                              self._dataset_captures.values()))


