"""Model declaration, compilation, and experiment execution.

A ``Model`` collects declared entities, parameters, outputs, and process
functions, then compiles everything on first ``experiment()``:

* each process body is njit-compiled and wrapped in a ``cfunc`` whose
  address cimba can start as a stackful fiber;
* a trial function is generated as Python source (see ``_trial_source``),
  compiled, and turned into a ``cfunc``: it creates the declared entities,
  schedules the recording window, starts the processes, runs the event
  queue, collects statistics, and tears everything down;
* an ``Experiment`` is a structured numpy array with one record per trial
  (the ``env`` seen by process bodies) handed to ``cimba_run_experiment``,
  which runs trials in parallel across all cores.
"""

import ast
import copy
import inspect
import keyword
import linecache
import textwrap
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import (TYPE_CHECKING, Any, TypedDict, TypeVar, get_args,
                    get_origin, get_type_hints, overload)

import numpy as np
from numpy.typing import ArrayLike, NDArray

from numba import carray, cfunc, from_dtype, njit, types
from numba.extending import overload as _nb_overload

from . import _bindings as _b
from ._cimba import ffi, lib
from ._graph import (ProcessDAG, ProcessDAGBlock, ProcessDAGEdge,
                     ProcessDAGNode, infer_process_dag)
from ._intrinsics import addressof, ptr_caster

#: Opaque native entity handle (process, queue, resource, ...) as stored
#: in env fields.
Handle = int

#: The per-trial record passed to process bodies: a numpy structured
#: scalar whose fields are accessed as attributes inside nopython code.
#: Annotate env with your Model subclass to get typed fields; Env is the
#: untyped fallback.
Env = Any

_F = TypeVar("_F", bound=Callable[..., Any])
_MISSING = object()
_COMPONENT_PROCESS_ATTR = "__cimba_component_process__"

#: A `pools`/`stores` declaration: list of names (unbounded capacity) or
#: name -> capacity mapping (an int, a param name, or None for unbounded).
_Capacities = Mapping[str, int | str | None] | Iterable[str] | None

# --- Env field declarations ---------------------------------------------------
# A model is declared as a Model subclass whose annotated fields describe
# the trial record:
#
#     class RepairShop(sim.Model):
#         mtbf: sim.Param
#         avg_broken: sim.Output
#         broken: sim.Queue
#         repairman: sim.Resource
#
# The subclass doubles as the static type of `env` in process bodies, so
# the checker knows each field. For the checker the markers ARE the value
# types the fields carry (Param -> float, Queue -> Handle, ...); at
# runtime they are distinct sentinel classes that Model.__init__ collects
# from the annotations.

if TYPE_CHECKING:
    Param = float        #: swept input, set by experiment()
    Output = float       #: result, written by the model
    State = int          #: mutable per-trial counter
    FloatState = float   #: mutable per-trial real-valued state
    Queue = Handle       #: cmb_buffer; default declares capacity
    Resource = Handle    #: cmb_resource, single holder
    Pool = Handle        #: cmb_resourcepool; default declares capacity
    Store = Handle       #: cmb_objectqueue; default declares capacity
    Dataset = Handle     #: cmb_dataset tally statistics
    Condition = Handle   #: cmb_condition variable
    Predicate = int      #: address of the matching @model.predicate
    Event = int          #: address of the matching @model.event callback
    #: handles of the same-named @model.process's copies, indexable
    Processes = Sequence[Handle]
    #: indexable array of priority queues; default declares the count
    PQueues = Sequence[Handle]
    #: the same-named @model.process, created at runtime by sim.spawn()
    Spawnable = int

    class Trace:
        """Per-trial replay array, fed to experiment(); inside model code
        ``Trace(env.<field>)`` returns the trial's trace as a float64
        NumPy view."""

        def __new__(cls, field: "Trace") -> "NDArray[np.float64]": ...

    class _Capacity:
        cap: int | str
        def __init__(self, cap: int | str) -> None: ...

    def capacity(cap: int | str) -> Any: ...

    def count(n: int | str) -> Any: ...

    _DECL_KINDS: dict[Any, str] = {}
else:
    class _Decl:
        """Marker for env field declarations in Model subclasses."""

    class Param(_Decl): ...
    class Output(_Decl): ...
    class State(_Decl): ...
    class FloatState(_Decl): ...
    class Queue(_Decl): ...
    class Resource(_Decl): ...
    class Pool(_Decl): ...
    class Store(_Decl): ...
    class Dataset(_Decl): ...
    class Condition(_Decl): ...
    class Predicate(_Decl): ...
    class Event(_Decl): ...
    class Processes(_Decl): ...
    class PQueues(_Decl): ...
    class Spawnable(_Decl): ...

    class Trace(_Decl):
        def __new__(cls, *args, **kwargs):
            raise TypeError("Trace(field) views are only available "
                            "inside compiled model code")

    class _Capacity:
        def __init__(self, cap):
            self.cap = cap

    def capacity(cap):
        """Declare a Pool/Store capacity: an int or the name of a param."""
        return _Capacity(cap)

    def count(n):
        """Declare the number of elements in a PQueues field."""
        return n

    _DECL_KINDS = {Param: "param", Output: "output", State: "state",
                   FloatState: "fstate",
                   Queue: "queue", Resource: "resource", Pool: "pool",
                   Store: "store", Dataset: "dataset",
                   Condition: "condition", Predicate: "predicate",
                   Event: "event",
                   Processes: "processes", PQueues: "pqueues",
                   Spawnable: "spawnable", Trace: "trace"}

    _trace_data = ptr_caster(types.float64)

    @_nb_overload(Trace)
    def _trace_view(field):
        # env.<field> is the [data_ptr, length] int64 pair in the record
        if not isinstance(field, types.Array) or field.dtype != types.int64:
            return None

        def view(field):
            return carray(_trace_data(field[0]), field[1])

        return view


class Component:
    """Authoring-time grouping of model fields and process methods.

    Component instances are declared as defaults on a ``Model`` subclass. Their
    declared fields are flattened into the model's trial record, and methods
    decorated with :func:`process` are lowered into ordinary model processes.
    """


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
        setattr(f, _COMPONENT_PROCESS_ATTR,
                _ComponentProcessSpec(copies, priority))
        return f

    if fn is None:
        return decorate
    return decorate(fn)


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
    local_name: str = ""
    process_names: tuple[str, ...] = ()
    display_name: str = ""
    item_display_name: str = ""
    direct_field_map: dict[str, str] = field(default_factory=dict)
    parent_offsets: tuple[int, ...] = ()
    parent_lengths: tuple[int, ...] = ()
    children: tuple["_AnyComponentDecl", ...] = ()


_AnyComponentDecl = _ComponentDecl | _ComponentCollectionDecl


def _empty_declarations() -> dict[str, Any]:
    decls: dict[str, Any] = {"param": [], "output": [], "state": [],
                             "fstate": [], "resource": [],
                             "dataset": [], "condition": [],
                             "predicate": [], "event": [], "processes": [],
                             "spawnable": [], "trace": [],
                             "queue": {}, "pool": {}, "store": {},
                             "pqueues": {}, "components": [],
                             "component_collections": [],
                             "field_shapes": {}}
    return decls


def _field_declarations(
    cls: type,
    *,
    allow_symbolic_pqueues: bool = False,
) -> dict[str, Any]:
    """Collect direct env field declarations from a Model/Component class."""
    decls = _empty_declarations()
    for fname, hint in get_type_hints(cls).items():
        try:
            kind = _DECL_KINDS.get(hint)
        except TypeError:
            kind = None
        if kind is None:
            continue
        default = getattr(cls, fname, None)
        if kind in ("queue", "pool", "store"):
            if isinstance(default, _Capacity):
                default = default.cap
            decls[kind][fname] = default
        elif kind == "pqueues":
            if isinstance(default, int) and default >= 1:
                decls[kind][fname] = default
            elif allow_symbolic_pqueues and isinstance(default, str):
                _check_name(default, "PQueues count constant")
                decls[kind][fname] = default
            else:
                raise ValueError(
                    f"field '{fname}': a PQueues declaration needs a "
                    "positive count default, e.g. "
                    "'qs: sim.PQueues = sim.count(4)'")
        else:
            if default is not None:
                raise ValueError(
                    f"field '{fname}': only Queue/Pool/Store declarations "
                    "may carry a capacity default")
            decls[kind].append(fname)
    return decls


def _component_declarations(cls: type[Component]) -> dict[str, Any]:
    decls = _field_declarations(cls, allow_symbolic_pqueues=True)
    for kind in ("predicate", "event", "processes", "spawnable"):
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
                 "dataset", "condition", "trace"):
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
    if collection or instance_count > 1:
        for kind in ("param", "trace"):
            if decls[kind]:
                raise ValueError(
                    f"component collection '{component_name}' declares "
                    f"{kind} fields, which are not supported yet")
    for kind in ("predicate", "event", "processes", "spawnable"):
        if decls[kind]:
            raise ValueError(
                f"component '{component_name}' declares {kind} fields, which "
                "are not supported yet")
    if collection or instance_count > 1:
        for kind in ("queue", "pool", "store"):
            for field, cap in decls[kind].items():
                if isinstance(cap, str):
                    raise ValueError(
                        f"component collection '{component_name}' field "
                        f"'{field}' uses symbolic capacity '{cap}', which is "
                        "not supported yet")


def _declarations_contain(decls: dict[str, Any], name: str) -> bool:
    for kind in ("param", "output", "state", "fstate", "resource",
                 "dataset", "condition", "predicate", "event", "processes",
                 "spawnable", "trace"):
        if name in decls[kind]:
            return True
    for kind in ("queue", "pool", "store", "pqueues"):
        if name in decls[kind]:
            return True
    return False


def _flatten_component_declarations(
    target: dict[str, Any],
    component_name: str,
    decls: dict[str, Any],
    field_map: dict[str, str],
    instance_count: int,
    pqueue_counts: Mapping[str, tuple[int, ...]],
) -> None:
    for flat_name in field_map.values():
        if _declarations_contain(target, flat_name):
            raise ValueError(f"duplicate field name '{flat_name}'")
    for kind in ("param", "trace"):
        target[kind].extend(field_map[name] for name in decls[kind])
    for kind in ("output", "state", "fstate", "resource", "dataset",
                 "condition"):
        target[kind].extend(field_map[name] for name in decls[kind])
        if instance_count > 1:
            for name in decls[kind]:
                target["field_shapes"][field_map[name]] = (instance_count,)
    for kind in ("queue", "pool", "store"):
        for name, cap in decls[kind].items():
            target[kind][field_map[name]] = _rewrite_component_capacity(
                component_name, name, cap, decls, field_map)
            if instance_count > 1:
                target["field_shapes"][field_map[name]] = (instance_count,)
    for name, counts in pqueue_counts.items():
        target["pqueues"][field_map[name]] = sum(counts)


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
    parent_offsets: tuple[int, ...] = (),
    parent_lengths: tuple[int, ...] = (),
) -> _AnyComponentDecl:
    decls = _component_declarations(cls)
    direct_field_map = _component_field_map(name, decls)
    instance_count = len(templates)
    _validate_component_instance_declarations(
        name, decls, instance_count=instance_count, collection=collection)
    constants = _component_constants(templates, direct_field_map)
    pqueue_counts, pqueue_offsets = _resolve_component_pqueues(
        name, instance_count, decls, constants)
    _flatten_component_declarations(
        target, name, decls, direct_field_map, instance_count, pqueue_counts)

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
            local_name=local_name,
            process_names=process_names,
            display_name=display_name,
            item_display_name=item_display_name,
            direct_field_map=direct_field_map,
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
        constants=constants,
        pqueue_counts=pqueue_counts,
        pqueue_offsets=pqueue_offsets,
        children=tuple(children),
    )


def _class_declarations(cls: type) -> dict[str, Any]:
    """Collect env field declarations from a Model subclass's annotations,
    in declaration order (base classes first)."""
    decls = _field_declarations(cls)
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
            parent_offsets=(0,),
            parent_lengths=(len(templates),),
        )
        decls["component_collections"].append(component_decl)
    return decls

_STANDARD_FIELDS = [
    ("start_time", "<f8"),
    ("warmup_s", "<f8"),
    ("duration_s", "<f8"),
    ("cooldown_s", "<f8"),
    ("seed", "<u8"),
]
_RESERVED = {name for name, _ in _STANDARD_FIELDS}

# All ExternalFunction bindings, by name, for the generated trial source.
_EXTERN_FUNCS = {name: obj for name, obj in vars(_b).items()
                 if isinstance(obj, types.ExternalFunction)}

_UNBOUNDED = np.uint64(0xFFFFFFFFFFFFFFFF)

# Offset of derived-struct fields inside an extended process allocation:
# the cmb_process header, rounded up to the 8-byte record alignment.
_PROC_DATA_OFFSET = (int(lib.cpy_process_sizeof()) + 7) & ~7


class Struct:
    """Per-process data fields, declared like a dataclass: subclass it
    and annotate the fields (``float`` or ``int``). A process function
    asks for its own view by annotating a final parameter with the
    subclass::

        class Visitor(sim.Struct):
            patience: float
            rides: int

        @model.process(copies=4)
        def visitor(env, vip: Visitor):
            vip.patience = sim.triangular(0.5, 1.0, 1.5)

    Each process copy then carries its own fields, zeroed at creation, in
    the same native allocation as the process (this is the Python form of
    the C tutorial's ``struct visitor { struct cmb_process core; ... }``).
    Subclassing a Struct subclass inherits its fields.

    Other processes reach the same fields through the process handle:
    inside model code, ``Visitor(handle)`` returns a read/write view --
    so a handle pulled from a queue is all a server needs to update a
    visitor's statistics. ``@model.process(struct=Visitor)`` attaches the
    fields without the view parameter.
    """

    if TYPE_CHECKING:
        def __init__(self, process: Handle) -> None: ...

    def __new__(cls, *args: Any, **kwargs: Any) -> "Struct":
        raise TypeError(f"{cls.__name__}(handle) views are only available "
                        "inside compiled model code")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        fields = []
        for fname, hint in get_type_hints(cls).items():
            if fname.startswith("_"):
                continue
            if hint is float:
                fields.append((fname, "<f8"))
            elif hint is int:
                fields.append((fname, "<i8"))
            else:
                raise TypeError(f"struct field '{fname}': only float and "
                                "int fields are supported")
        if not fields:
            raise ValueError(f"struct '{cls.__name__}' declares no fields")
        cls._dtype = np.dtype(fields)
        cls._alloc_size = _PROC_DATA_OFFSET + cls._dtype.itemsize

        cast = ptr_caster(from_dtype(cls._dtype))
        offset = _PROC_DATA_OFFSET

        def view(process):
            return carray(cast(process + offset), 1)[0]

        @_nb_overload(cls)
        def struct_view(process):
            if not isinstance(process, types.Integer):
                return None
            return view


def _is_struct_class(obj: Any) -> bool:
    return (isinstance(obj, type) and issubclass(obj, Struct)
            and obj is not Struct)


@dataclass
class _ProcDecl:
    """A registered @model.process function."""

    name: str
    fn: Callable[..., Any]
    copies: int
    priority: int
    indexed: bool                  # takes the copy index argument
    struct: type[Struct] | None    # per-process fields, if any
    injected: bool                 # fn receives its own struct view
    spawnable: bool                # created by sim.spawn(), not at setup

    @property
    def alloc_size(self) -> int:
        return (self.struct._alloc_size if self.struct is not None
                else _PROC_DATA_OFFSET)


class _Compiled(TypedDict):
    """Artifacts of Model._compile(), kept alive for the model's lifetime.
    The callables are Numba cfunc/dispatcher objects (untyped upstream)."""

    trial: Any
    events: tuple[Any, Any, Any]
    procs: dict[str, Any]
    preds: dict[str, Any]
    user_events: dict[str, Any]
    #: per-Spawnable-field descriptor arrays sim.spawn() reads:
    #: [cfunc address, name cstring, allocation size]
    spawns: dict[str, np.ndarray]
    collect: Any
    dtype: np.dtype


def _check_name(name: str, kind: str) -> None:
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"{kind} name '{name}' is not a valid identifier")
    if name in _RESERVED or name.startswith("_"):
        raise ValueError(f"{kind} name '{name}' is reserved")


def _as_capacity_dict(value: _Capacities) -> dict[str, int | str | None]:
    """Normalize a `stores`/`pools` declaration: a list means unbounded
    capacity; dict values are an int capacity or a param name string."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return {name: None for name in value}


def _as_trace_rows(value: Any, n_trials: int, name: str) -> list[np.ndarray]:
    """Normalize a Trace value into one contiguous float64 row per trial:
    a 1-D array is shared by every trial, a 2-D array maps row i to trial
    i, and a sequence of 1-D arrays gives ragged per-trial traces."""
    if not isinstance(value, np.ndarray):
        try:
            value = np.asarray(value, dtype=np.float64)
        except (ValueError, TypeError):
            # Ragged: a sequence of per-trial 1-D arrays
            rows = [np.ascontiguousarray(row, dtype=np.float64)
                    for row in value]
            if len(rows) != n_trials:
                raise ValueError(
                    f"trace '{name}': expected {n_trials} per-trial "
                    f"arrays (one per trial), got {len(rows)}") from None
            for row in rows:
                if row.ndim != 1:
                    raise ValueError(f"trace '{name}': per-trial arrays "
                                     "must be 1-D") from None
            return rows
    arr = np.ascontiguousarray(value, dtype=np.float64)
    if arr.ndim == 1:
        return [arr] * n_trials
    if arr.ndim == 2:
        if arr.shape[0] != n_trials:
            raise ValueError(
                f"trace '{name}': expected {n_trials} rows (one per "
                f"trial, design-point-major with replications innermost), "
                f"got {arr.shape[0]}")
        return list(arr)
    raise ValueError(f"trace '{name}': expected a 1-D array (shared), a "
                     "2-D array (row per trial), or a sequence of 1-D "
                     "arrays")


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


def _add(left: ast.expr, right: ast.expr) -> ast.BinOp:
    return ast.BinOp(left=left, op=ast.Add(), right=right)


def _component_instance_count(decl: _AnyComponentDecl) -> int:
    if isinstance(decl, _ComponentCollectionDecl):
        return len(decl.templates)
    return len(decl.instances)


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
            if access.field in access.decl.decls["pqueues"]:
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
    ):
        super().__init__(env_name=env_name)
        self.component_name = component_name
        self.receiver_name = receiver_name
        self.component_decl = component_decl
        self.instance_index = ast.Constant(instance_index)

    def _root_namespace_ref(self, node: ast.AST) -> _ComponentAccess | None:
        if isinstance(node, ast.Name) and node.id == self.receiver_name:
            index = (self.instance_index
                     if _component_instance_count(self.component_decl) > 1
                     else None)
            return _ComponentAccess(self.component_decl, index,
                                    self.receiver_name)
        return None

    def _callback_label(self) -> str:
        return f"component '{self.component_name}' process"

    def _raise_unknown_field(
        self,
        namespace: _ComponentAccess,
        field_name: str,
    ) -> None:
        raise ValueError(
            f"component '{self.component_name}' process references "
            f"unsupported {namespace.text}.{field_name}")

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == self.receiver_name:
            raise ValueError(
                f"component '{self.component_name}' process cannot use "
                "self directly inside compiled code")
        return node


def _closure_namespace(fn: Callable[..., Any]) -> dict[str, Any]:
    namespace = dict(fn.__globals__)
    if fn.__closure__ is not None:
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
            namespace[name] = cell.cell_contents
    return namespace


def _component_method_source(fn: Callable[..., Any]) -> ast.FunctionDef:
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError) as exc:
        raise ValueError(
            f"component process '{fn.__qualname__}' needs inspectable source"
        ) from exc
    tree = ast.parse(textwrap.dedent(source))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node
    raise ValueError(f"component process '{fn.__qualname__}' source does not "
                     "contain a function definition")


def _lower_component_process(
    component_name: str,
    component: Component,
    component_decl: _AnyComponentDecl,
    instance_index: int,
    method_name: str,
    method: Callable[..., Any],
) -> Callable[..., Any]:
    node = copy.deepcopy(_component_method_source(method))
    args = node.args
    if (args.posonlyargs or args.vararg or args.kwonlyargs or args.kwarg
            or args.defaults or args.kw_defaults):
        raise ValueError(
            f"component process '{component_name}.{method_name}' must take "
            "(self, env) or (self, env, idx) without defaults")
    if len(args.args) not in (2, 3):
        raise ValueError(
            f"component process '{component_name}.{method_name}' must take "
            "(self, env) or (self, env, idx)")

    receiver_name = args.args[0].arg
    env_name = args.args[1].arg
    process_name = f"{component_name}__{method_name}"
    node.name = process_name
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
    )
    lowered = lowerer.visit(node)
    if not isinstance(lowered, ast.FunctionDef):
        raise TypeError("component process lowering produced a non-function")
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
    namespace.update(_component_collection_namespace((component_decl,)))
    exec(compile(source, filename, "exec"), namespace)
    generated = namespace[process_name]
    generated.__module__ = method.__module__
    generated.__qualname__ = process_name
    generated.__cimba_source__ = source
    return generated


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
) -> list[str]:
    members: list[str] = []
    for kind in _PROCESS_DAG_FIELD_KINDS:
        fields = decls[kind]
        for field in fields:
            flat_name = field_map[field]
            graph_kind = entity_kinds.get(flat_name)
            if graph_kind is not None:
                members.append(f"{graph_kind}:{flat_name}")
    return members


def _dedupe_process_dag_members(members: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(members))


class Model:
    """A simulation model. Subclass it and declare the env fields as
    annotations (Param, Output, Queue, Resource, Pool, Store, Dataset,
    Condition, State, Predicate) -- the subclass then types `env` in
    process bodies. Entity names may also be passed as keyword lists for
    quick untyped models. Process functions are registered with
    @model.process; compilation happens once, on first experiment()."""

    # Standard trial-record fields, readable as env attributes in process
    # bodies (plain annotations, not declaration markers).
    start_time: float
    warmup_s: float
    duration_s: float
    cooldown_s: float
    seed: int

    _source: str

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
        self.params = decls["param"] + list(params)
        self.outputs = decls["output"] + list(outputs)
        self.queues = decls["queue"] | _as_capacity_dict(queues)
        self.resources = decls["resource"] + list(resources)
        self.pools = decls["pool"] | _as_capacity_dict(pools)
        self.stores = decls["store"] | _as_capacity_dict(stores)
        self.datasets = decls["dataset"] + list(datasets)
        self.conditions = decls["condition"] + list(conditions)
        self.state = decls["state"] + list(state)
        self.float_state: list[str] = decls["fstate"]
        self.traces: list[str] = decls["trace"]
        self.pqueues: dict[str, int] = decls["pqueues"]
        self._predicate_fields: list[str] = decls["predicate"]
        self._event_fields: list[str] = decls["event"]
        self._process_fields: list[str] = decls["processes"]
        self._spawnable_fields: list[str] = decls["spawnable"]
        self._component_decls: list[_ComponentDecl] = decls["components"]
        self._component_collection_decls: list[_ComponentCollectionDecl] = \
            decls["component_collections"]
        self._field_shapes: dict[str, tuple[int, ...]] = \
            decls["field_shapes"]
        self._components: dict[str, Component] = {}
        self._component_collections: dict[str, tuple[Component, ...]] = {}
        self._component_bindings: dict[str, tuple[Component, ...]] = {}

        seen: set[str] = set()
        for kind, names in (("param", self.params),
                            ("output", self.outputs),
                            ("queue", self.queues),
                            ("resource", self.resources),
                            ("pool", self.pools),
                            ("store", self.stores),
                            ("dataset", self.datasets),
                            ("condition", self.conditions),
                            ("state", self.state),
                            ("fstate", self.float_state),
                            ("trace", self.traces),
                            ("pqueues", self.pqueues),
                            ("predicate", self._predicate_fields),
                            ("event", self._event_fields),
                            ("processes", self._process_fields),
                            ("spawnable", self._spawnable_fields)):
            for n in names:
                _check_name(n, kind)
                if n in seen:
                    raise ValueError(f"duplicate field name '{n}'")
                seen.add(n)
        for component in self._component_decls:
            _check_name(component.name, "component")
            if component.name in seen:
                raise ValueError(f"duplicate field name '{component.name}'")
            seen.add(component.name)
        for collection in self._component_collection_decls:
            _check_name(collection.name, "component collection")
            if collection.name in seen:
                raise ValueError(f"duplicate field name '{collection.name}'")
            seen.add(collection.name)
        for cap in (list(self.queues.values()) + list(self.pools.values())
                    + list(self.stores.values())):
            if cap is not None and not isinstance(cap, int) \
                    and cap not in self.params:
                raise ValueError(f"capacity '{cap}' is neither an int nor "
                                 "a declared param")
        self._seen = seen
        self._processes: list[_ProcDecl] = []
        # (name, fn, env field holding the compiled address)
        self._predicates: list[tuple[str, Callable[..., Any], str]] = []
        # (name, fn, env field holding the compiled address, takes_data)
        self._events: list[tuple[str, Callable[..., Any], str, bool]] = []
        self._collect: Callable[..., Any] | None = None
        self._compiled: _Compiled | None = None
        self._bind_components()
        self._register_component_processes()

    def _bind_components(self) -> None:
        for decl in self._component_decls:
            component = copy.copy(decl.instances[0])
            self._components[decl.name] = component
            self._component_bindings[decl.name] = (component,)
            setattr(self, decl.name, component)
            self._bind_component_metadata(component, decl.name)
            self._bind_component_children(decl, (component,))
        for decl in self._component_collection_decls:
            components = tuple(copy.copy(template)
                               for template in decl.templates)
            self._component_collections[decl.name] = components
            self._component_bindings[decl.name] = components
            setattr(self, decl.name, list(components))
            for index, component in enumerate(components):
                self._bind_component_metadata(
                    component, f"{decl.name}[{index}]",
                    collection=decl.name, index=index)
            self._bind_component_children(decl, components)

    def _bind_component_metadata(
        self,
        component: Component,
        name: str,
        *,
        collection: str | None = None,
        index: int | None = None,
    ) -> None:
        try:
            component._cimba_model = self
            component._cimba_name = name
            if collection is not None:
                component._cimba_collection = collection
            if index is not None:
                component._cimba_index = index
        except AttributeError:
            pass

    def _bind_component_children(
        self,
        decl: _AnyComponentDecl,
        parents: tuple[Component, ...],
    ) -> None:
        for child in decl.children:
            bound: list[Component] = []
            if isinstance(child, _ComponentCollectionDecl):
                for parent_index, parent in enumerate(parents):
                    start = child.parent_offsets[parent_index]
                    length = child.parent_lengths[parent_index]
                    items: list[Component] = []
                    for item_index in range(length):
                        child_index = start + item_index
                        component = copy.copy(child.templates[child_index])
                        bound.append(component)
                        items.append(component)
                        self._bind_component_metadata(
                            component,
                            child.process_names[child_index],
                            collection=child.name,
                            index=child_index,
                        )
                    setattr(parent, child.local_name, items)
            else:
                for child_index, parent in enumerate(parents):
                    component = copy.copy(child.instances[child_index])
                    bound.append(component)
                    setattr(parent, child.local_name, component)
                    self._bind_component_metadata(
                        component, child.process_names[child_index])
            bound_tuple = tuple(bound)
            self._component_bindings[child.name] = bound_tuple
            self._bind_component_children(child, bound_tuple)

    def _register_component_processes(self) -> None:
        roots: list[_AnyComponentDecl] = []
        roots.extend(self._component_decls)
        roots.extend(self._component_collection_decls)
        for decl in _walk_component_declarations(roots):
            components = self._component_bindings[decl.name]
            for index, component in enumerate(components):
                component_name = decl.process_names[index]
                for method_name, method, spec in _component_process_methods(
                        decl.cls):
                    copies = _resolve_component_process_copies(
                        component_name, component, method_name, spec)
                    lowered = _lower_component_process(
                        component_name, component, decl, index, method_name,
                        method)
                    self.process(lowered, copies=copies,
                                 priority=spec.priority)

    @property
    def _component_field_maps(self) -> dict[str, dict[str, str]]:
        return {decl.name: decl.field_map for decl in self._component_decls}

    @property
    def _component_collection_maps(
        self,
    ) -> dict[str, _ComponentCollectionDecl]:
        return {decl.name: decl for decl in self._component_collection_decls}

    @property
    def _component_roots(self) -> dict[str, _AnyComponentDecl]:
        roots: dict[str, _AnyComponentDecl] = {}
        roots.update({decl.name: decl for decl in self._component_decls})
        roots.update({decl.name: decl
                      for decl in self._component_collection_decls})
        return roots

    def _lower_component_refs(self, fn: _F) -> _F:
        return _lower_model_component_refs(
            fn, model_name=self.name,
            component_roots=self._component_roots,
        )

    def _process_dag_blocks(
        self,
        entity_kinds: Mapping[str, str],
    ) -> tuple[ProcessDAGBlock, ...]:
        process_names = {process.name for process in self._processes}
        blocks: list[ProcessDAGBlock] = []

        for decl in self._component_decls:
            members = _process_dag_component_process_members(
                decl, process_names)
            members.extend(_process_dag_component_field_members(
                decl.decls, decl.direct_field_map, entity_kinds))
            blocks.append(
                ProcessDAGBlock(
                    decl.display_name or decl.name,
                    _dedupe_process_dag_members(members),
                )
            )
            for child in _walk_component_declarations(decl.children):
                members = _process_dag_component_process_members(
                    child, process_names)
                members.extend(_process_dag_component_field_members(
                    child.decls, child.direct_field_map, entity_kinds))
                blocks.append(
                    ProcessDAGBlock(
                        child.display_name or child.name,
                        _dedupe_process_dag_members(members),
                        kind=("component_collection"
                              if isinstance(child, _ComponentCollectionDecl)
                              else "component"),
                    )
                )

        for decl in self._component_collection_decls:
            members = _process_dag_component_process_members(
                decl, process_names)
            members.extend(_process_dag_component_field_members(
                decl.decls, decl.direct_field_map, entity_kinds))
            blocks.append(
                ProcessDAGBlock(
                    decl.display_name or decl.name,
                    _dedupe_process_dag_members(members),
                    kind="component_collection",
                )
            )
            for child in _walk_component_declarations(decl.children):
                members = _process_dag_component_process_members(
                    child, process_names)
                members.extend(_process_dag_component_field_members(
                    child.decls, child.direct_field_map, entity_kinds))
                blocks.append(
                    ProcessDAGBlock(
                        child.display_name or child.name,
                        _dedupe_process_dag_members(members),
                        kind=("component_collection"
                              if isinstance(child, _ComponentCollectionDecl)
                              else "component"),
                    )
                )

        return tuple(blocks)

    # --- Declaration decorators ------------------------------------------
    @overload
    def process(self, fn: _F) -> _F: ...

    @overload
    def process(self, fn: None = None, *, copies: int = 1,
                priority: int = 0,
                struct: "type[Struct] | None" = None
                ) -> Callable[[_F], _F]: ...

    def process(self, fn=None, *, copies: int = 1, priority: int = 0,
                struct=None):
        """Register a process function `def fn(env)` or `def fn(env, idx)`
        (the latter receives its copy index). A final parameter annotated
        with a sim.Struct subclass receives the process's own field view:
        `def fn(env, vip: Visitor)` or `def fn(env, idx, vip: Visitor)`.
        copies=n starts n identical processes; priority sets the cimba
        process priority; struct= attaches the per-process fields without
        the view parameter. A process named in a sim.Spawnable field is
        not started at setup -- sim.spawn(env.<name>, env) creates its
        copies at runtime."""
        if fn is None:
            return lambda f: self.process(f, copies=copies,
                                          priority=priority, struct=struct)
        if copies < 1:
            raise ValueError("copies must be >= 1")
        if struct is not None and not _is_struct_class(struct):
            raise ValueError("struct= expects a sim.Struct subclass")
        name = fn.__name__
        spawnable = name in self._spawnable_fields
        if name in self._process_fields or spawnable:
            # The declared field publishes the handles (Processes) or the
            # spawn reference (Spawnable)
            if self._compiled is not None:
                raise RuntimeError("model is already compiled")
            if any(p.name == name for p in self._processes):
                raise ValueError(f"process '{name}' already registered")
        else:
            self._register_name(name, "process")

        nargs = fn.__code__.co_argcount
        params = fn.__code__.co_varnames[:nargs]
        hints = get_type_hints(fn)
        own = hints.get(params[-1]) if nargs > 1 else None
        injected = _is_struct_class(own)
        for p in params[1:len(params) - 1 if injected else None]:
            if _is_struct_class(hints.get(p)):
                raise ValueError(f"process '{name}': the {hints[p].__name__}"
                                 " view must be the last parameter")
        if injected:
            if struct is not None and struct is not own:
                raise ValueError(f"process '{name}': struct= and the view "
                                 "annotation disagree")
            struct = own
        indexed = nargs - injected == 2
        if nargs - injected not in (1, 2):
            raise ValueError(
                "process functions take (env), (env, idx), and optionally "
                "a final view parameter annotated with a sim.Struct "
                "subclass")
        if spawnable:
            if copies != 1:
                raise ValueError(f"spawnable process '{name}' cannot take "
                                 "copies; sim.spawn() creates them")
            if indexed:
                raise ValueError(f"spawnable process '{name}' takes (env) "
                                 "or (env, view), not a copy index")
        fn = self._lower_component_refs(fn)
        self._processes.append(_ProcDecl(name, fn, copies, priority,
                                         indexed, struct, injected,
                                         spawnable))
        return fn

    def process_dag(self, *, validate: bool = True) -> ProcessDAG:
        """Infer a resource-aware process graph from registered processes.

        ``validate`` is accepted for API stability. Inferred graphs may contain
        legitimate resource cycles, so acyclicity is checked only when callers
        explicitly ask for :meth:`ProcessDAG.topological_order`.
        """
        entity_kinds: dict[str, str] = {}
        entity_kinds.update({name: "queue" for name in self.queues})
        entity_kinds.update({name: "resource" for name in self.resources})
        entity_kinds.update({name: "pool" for name in self.pools})
        entity_kinds.update({name: "store" for name in self.stores})
        entity_kinds.update({name: "condition" for name in self.conditions})
        entity_kinds.update({name: "state" for name in self.state})
        entity_kinds.update({name: "fstate" for name in self.float_state})
        entity_kinds.update({name: "pqueues" for name in self.pqueues})
        event_fields = set(self._event_fields)
        event_fields.update(field for _n, _fn, field, _d in self._events)
        entity_kinds.update({name: "event" for name in event_fields})
        return infer_process_dag(
            self._processes,
            entity_kinds=entity_kinds,
            process_fields=self._process_fields,
            spawnable_fields=self._spawnable_fields,
            event_callbacks=((field, fn) for _n, fn, field, _d in self._events),
            blocks=self._process_dag_blocks(entity_kinds),
        )

    def predicate(self, fn: _F) -> _F:
        """Register a condition predicate `def fn(env) -> bool`. Its
        compiled address is published in the declared Predicate field of
        the same name, for use with sim.wait_for(env.<cond>, env.<name>,
        env). (Without a declared field, it is published as the hidden
        field `_pred_<name>`.)"""
        name = fn.__name__
        if name in self._predicate_fields:
            if self._compiled is not None:
                raise RuntimeError("model is already compiled")
            if any(f == name for _n, _fn, f in self._predicates):
                raise ValueError(f"predicate '{name}' already registered")
            field = name
        else:
            self._register_name(name, "predicate")
            field = f"_pred_{name}"
        fn = self._lower_component_refs(fn)
        self._predicates.append((name, fn, field))
        return fn

    def event(self, fn: _F) -> _F:
        """Register a low-level event callback `def fn(env)` or
        `def fn(env, data)` (the latter receives the int64 data word given
        at scheduling time). Its compiled address is published in the
        declared Event field of the same name, for use with
        sim.schedule(env.<name>, env, delay, ...). (Without a declared
        field, it is published as the hidden field `_ev_<name>`.)"""
        name = fn.__name__
        nargs = fn.__code__.co_argcount
        if nargs not in (1, 2):
            raise ValueError("event functions take (env) or (env, data)")
        if name in self._event_fields:
            if self._compiled is not None:
                raise RuntimeError("model is already compiled")
            if any(f == name for _n, _fn, f, _d in self._events):
                raise ValueError(f"event '{name}' already registered")
            field = name
        else:
            self._register_name(name, "event")
            field = f"_ev_{name}"
        fn = self._lower_component_refs(fn)
        self._events.append((name, fn, field, nargs == 2))
        return fn

    def collect(self, fn: _F) -> _F:
        """Register the statistics-collection function, run once at the
        end of each trial."""
        if self._collect is not None:
            raise ValueError("collect() already registered")
        if self._compiled is not None:
            raise RuntimeError("model is already compiled")
        fn = self._lower_component_refs(fn)
        self._collect = fn
        return fn

    def _register_name(self, name: str, kind: str) -> None:
        _check_name(name, kind)
        if name in self._seen:
            raise ValueError(f"duplicate name '{name}'")
        if self._compiled is not None:
            raise RuntimeError("model is already compiled")
        self._seen.add(name)

    # --- Trial record layout ----------------------------------------------
    @property
    def _entities(self) -> list[str]:
        return (list(self.queues) + self.resources + list(self.pools)
                + list(self.stores) + self.datasets + self.conditions)

    def _handle_expr(self, pname: str, i: int) -> str:
        """Env expression for a process handle: an element of the declared
        Processes field, or the hidden per-copy scalar."""
        if pname in self._process_fields:
            return f"env['{pname}'][{i}]"
        return f"env['_p_{pname}_{i}']"

    @property
    def _process_handles(self) -> list[str]:
        return [self._handle_expr(p.name, i)
                for p in self._processes if not p.spawnable
                for i in range(p.copies)]

    def _field_spec(self, name: str, fmt: str) -> tuple[Any, ...]:
        shape = self._field_shapes.get(name)
        if shape is None:
            return (name, fmt)
        return (name, fmt, shape)

    def _field_refs(self, name: str) -> list[str]:
        shape = self._field_shapes.get(name)
        if shape is None:
            return [f"env['{name}']"]
        if len(shape) != 1:
            raise ValueError(f"field '{name}' has unsupported shape {shape}")
        return [f"env['{name}'][{i}]" for i in range(shape[0])]

    def _field_name_keys(self, name: str) -> list[tuple[str, str]]:
        shape = self._field_shapes.get(name)
        if shape is None:
            return [(f"NAME_{name}", name)]
        if len(shape) != 1:
            raise ValueError(f"field '{name}' has unsupported shape {shape}")
        return [(f"NAME_{name}_{i}", f"{name}_{i}")
                for i in range(shape[0])]

    @property
    def dtype(self) -> np.dtype:
        # (name, format) or (name, format, shape) numpy field specs
        fields: list[Any] = list(_STANDARD_FIELDS)
        fields += [(p, "<f8") for p in self.params]
        fields += [self._field_spec(o, "<f8") for o in self.outputs]
        fields += [self._field_spec(h, "<i8") for h in self._entities]
        fields += [self._field_spec(s, "<i8") for s in self.state]
        fields += [self._field_spec(s, "<f8") for s in self.float_state]
        fields += [(t, "<i8", (2,)) for t in self.traces]
        fields += [(f, "<i8", (n,)) for f, n in self.pqueues.items()]
        fields += [(p, "<i8") for p in self._predicate_fields]
        fields += [(f, "<i8") for _n, _fn, f in self._predicates
                   if f.startswith("_pred_")]
        fields += [(e, "<i8") for e in self._event_fields]
        fields += [(f, "<i8") for _n, _fn, f, _d in self._events
                   if f.startswith("_ev_")]
        fields += [(s, "<i8") for s in self._spawnable_fields]
        for p in self._processes:
            if p.spawnable:
                continue
            if p.name in self._process_fields:
                fields += [(p.name, "<i8", (p.copies,))]
            else:
                fields += [(f"_p_{p.name}_{i}", "<i8")
                           for i in range(p.copies)]
            if p.indexed:
                for i in range(p.copies):
                    fields += [(f"_cx_{p.name}_{i}_env", "<i8"),
                               (f"_cx_{p.name}_{i}_idx", "<i8")]
        return np.dtype(fields)

    # --- Compilation --------------------------------------------------------
    def _compile_callbacks(
        self, rec: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Any]:
        """njit-compile the registered process/predicate/event/collect
        functions and wrap all but collect in cfuncs cimba can call."""
        trial_ptr = types.CPointer(rec)
        proc_sig = types.intp(types.intp, trial_ptr)
        proc_sig_ix = types.intp(types.intp, types.CPointer(types.int64))
        pred_sig = types.boolean(types.intp, types.intp, trial_ptr)
        # cmb_event_func: subject is the env pointer, object the data word
        ev_sig = types.void(trial_ptr, types.intp)
        rec_from_addr = ptr_caster(rec)

        def make_proc(inner, struct):
            if struct is None:
                @cfunc(proc_sig)
                def proc(me, ctxp):
                    inner(carray(ctxp, 1)[0])
                    return 0
            else:
                @cfunc(proc_sig)
                def proc(me, ctxp):
                    inner(carray(ctxp, 1)[0], struct(me))
                    return 0
            return proc

        def make_proc_indexed(inner, struct):
            if struct is None:
                @cfunc(proc_sig_ix)
                def proc(me, ctxp):
                    pair = carray(ctxp, 2)
                    env = carray(rec_from_addr(pair[0]), 1)[0]
                    inner(env, pair[1])
                    return 0
            else:
                @cfunc(proc_sig_ix)
                def proc(me, ctxp):
                    pair = carray(ctxp, 2)
                    env = carray(rec_from_addr(pair[0]), 1)[0]
                    inner(env, pair[1], struct(me))
                    return 0
            return proc

        def make_pred(inner):
            @cfunc(pred_sig)
            def pred(cnd, prc, ctxp):
                return inner(carray(ctxp, 1)[0])
            return pred

        def make_event(inner, takes_data):
            if takes_data:
                @cfunc(ev_sig)
                def ev(subject, data):
                    inner(carray(subject, 1)[0], data)
            else:
                @cfunc(ev_sig)
                def ev(subject, data):
                    inner(carray(subject, 1)[0])
            return ev

        proc_cfuncs = {}
        for p in self._processes:
            inner = njit(p.fn)
            view = p.struct if p.injected else None
            proc_cfuncs[p.name] = (make_proc_indexed(inner, view)
                                   if p.indexed else make_proc(inner, view))
        # Predicates and events keyed by the env field that publishes
        # their compiled address
        pred_cfuncs = {field: make_pred(njit(fn))
                       for _n, fn, field in self._predicates}
        event_cfuncs = {field: make_event(njit(fn), takes_data)
                        for _n, fn, field, takes_data in self._events}
        collect_inner = njit(self._collect) if self._collect else None
        return proc_cfuncs, pred_cfuncs, event_cfuncs, collect_inner

    def _codegen_namespace(self, proc_cfuncs: dict[str, Any],
                           collect_inner: Any) -> dict[str, Any]:
        """Globals for the generated trial source: the extern bindings,
        interned entity/process name strings, and process cfunc addresses."""
        ns = dict(_EXTERN_FUNCS)
        ns.update(carray=carray, addressof=addressof, np=np,
                  COLLECT=collect_inner, CAP=_UNBOUNDED)
        for e in self._entities:
            for key, name in self._field_name_keys(e):
                ns[key] = _b.cstring(name)
        for f, n in self.pqueues.items():
            for k in range(n):
                ns[f"NAME_{f}_{k}"] = _b.cstring(f"{f}_{k}")
        for pname, proc in proc_cfuncs.items():
            ns[f"NAME_{pname}"] = _b.cstring(pname)
            ns[f"F_{pname}"] = proc.address
        for p in self._processes:
            if p.struct is not None and not p.spawnable:
                ns[f"SZ_{p.name}"] = np.uint64(p.alloc_size)
        return ns

    def _trial_source(self, dtype: np.dtype) -> str:
        """Generate the trial function and the three recording-window event
        callbacks as Python source (njit-compilable against the namespace
        from _codegen_namespace)."""

        def cap_expr(cap):
            if cap is None:
                return "CAP"
            if isinstance(cap, int):
                return f"np.uint64({cap})"
            return f"np.uint64(env['{cap}'])"

        recorded = [("buffer", q) for q in self.queues]
        recorded += [("resource", r) for r in self.resources]
        recorded += [("resourcepool", p) for p in self.pools]
        recorded += [("objectqueue", s) for s in self.stores]
        handles = self._process_handles

        src = ["def _start_rec(subject, obj):",
               "    env = carray(subject, 1)[0]"]
        for kind, name in recorded:
            src += [f"    {kind}_recording_start({ref})"
                    for ref in self._field_refs(name)]
        for f, n in self.pqueues.items():
            for k in range(n):
                src += [
                    f"    priorityqueue_recording_start(env['{f}'][{k}])"
                ]
        # Datasets tally over the measurement window only
        for dataset in self.datasets:
            src += [f"    dataset_reset({ref})"
                    for ref in self._field_refs(dataset)]
        src += ["def _stop_rec(subject, obj):",
                "    env = carray(subject, 1)[0]"]
        for kind, name in recorded:
            src += [f"    {kind}_recording_stop({ref})"
                    for ref in self._field_refs(name)]
        for f, n in self.pqueues.items():
            for k in range(n):
                src += [
                    f"    priorityqueue_recording_stop(env['{f}'][{k}])"
                ]
        # Stop only processes still running; some may have finished on
        # their own (e.g. a source that produced a fixed number of items)
        has_spawns = any(p.spawnable for p in self._processes)
        src = (src
               + ["def _end_sim(subject, obj):",
                  "    env = carray(subject, 1)[0]"]
               + [f"    if process_status({h}) == 1:\n"
                  f"        process_stop({h}, 0)" for h in handles]
               + (["    spawned_stop_all()"] if has_spawns else []))

        src += ["def _trial(vtrl):",
                "    arr = carray(vtrl, 1)",
                "    env = arr[0]",
                "    self_addr = addressof(vtrl)",
                "    logger_apply_flags()",
                "    event_queue_initialize(env['start_time'])",
                "    random_initialize(env['seed'])",
                "    t = env['start_time'] + env['warmup_s']",
                "    event_schedule(EV_START, self_addr, 0, t, 0)",
                "    t = t + env['duration_s']",
                "    event_schedule(EV_STOP, self_addr, 0, t, 0)",
                "    t = t + env['cooldown_s']",
                "    event_schedule(EV_END, self_addr, 0, t, 0)"]
        for q, cap in self.queues.items():
            for i, ref in enumerate(self._field_refs(q)):
                name_key = self._field_name_keys(q)[i][0]
                src += ["    h = buffer_create()",
                        f"    buffer_initialize(h, {name_key}, "
                        f"{cap_expr(cap)})",
                        f"    {ref} = h"]
        for r in self.resources:
            for i, ref in enumerate(self._field_refs(r)):
                name_key = self._field_name_keys(r)[i][0]
                src += ["    h = resource_create()",
                        f"    resource_initialize(h, {name_key})",
                        f"    {ref} = h"]
        for p, cap in self.pools.items():
            for i, ref in enumerate(self._field_refs(p)):
                name_key = self._field_name_keys(p)[i][0]
                src += ["    h = resourcepool_create()",
                        f"    resourcepool_initialize(h, {name_key}, "
                        f"{cap_expr(cap)})",
                        f"    {ref} = h"]
        for s, cap in self.stores.items():
            for i, ref in enumerate(self._field_refs(s)):
                name_key = self._field_name_keys(s)[i][0]
                src += ["    h = objectqueue_create()",
                        f"    objectqueue_initialize(h, {name_key}, "
                        f"{cap_expr(cap)})",
                        f"    {ref} = h"]
        for d in self.datasets:
            for ref in self._field_refs(d):
                src += ["    h = dataset_create()",
                        "    dataset_initialize(h)",
                        f"    {ref} = h"]
        for c in self.conditions:
            for i, ref in enumerate(self._field_refs(c)):
                name_key = self._field_name_keys(c)[i][0]
                src += ["    h = condition_create()",
                        f"    condition_initialize(h, {name_key})",
                        f"    {ref} = h"]
        for f, n in self.pqueues.items():
            for k in range(n):
                src += ["    h = priorityqueue_create()",
                        f"    priorityqueue_initialize(h, NAME_{f}_{k}, "
                        "CAP)",
                        f"    env['{f}'][{k}] = h"]
        offsets = {n: off for n, (_dt, off) in dtype.fields.items()}
        for p in self._processes:
            if p.spawnable:
                continue
            pname = p.name
            create = ("process_create()" if p.struct is None
                      else f"process_create_sized(SZ_{pname})")
            for i in range(p.copies):
                if p.indexed:
                    src += [f"    env['_cx_{pname}_{i}_env'] = self_addr",
                            f"    env['_cx_{pname}_{i}_idx'] = {i}",
                            f"    ctx = self_addr + "
                            f"{offsets[f'_cx_{pname}_{i}_env']}"]
                else:
                    src += ["    ctx = self_addr"]
                src += [f"    p = {create}",
                        f"    process_initialize(p, NAME_{pname}, "
                        f"F_{pname}, ctx, {p.priority})",
                        "    process_start(p)",
                        f"    {self._handle_expr(pname, i)} = p"]
        src += ["    event_queue_execute()"]
        if self._collect is not None:
            src += ["    COLLECT(env)"]
        for h in handles:
            src += [f"    process_terminate({h})",
                    f"    process_destroy({h})"]
        if has_spawns:
            src += ["    spawned_reclaim()"]
        for q in self.queues:
            src += [f"    buffer_destroy({ref})"
                    for ref in self._field_refs(q)]
        for r in self.resources:
            src += [f"    resource_destroy({ref})"
                    for ref in self._field_refs(r)]
        for p in self.pools:
            src += [f"    resourcepool_destroy({ref})"
                    for ref in self._field_refs(p)]
        for s in self.stores:
            src += [f"    objectqueue_destroy({ref})"
                    for ref in self._field_refs(s)]
        for d in self.datasets:
            src += [f"    dataset_destroy({ref})"
                    for ref in self._field_refs(d)]
        for c in self.conditions:
            src += [f"    condition_destroy({ref})"
                    for ref in self._field_refs(c)]
        for f, n in self.pqueues.items():
            for k in range(n):
                src += [f"    priorityqueue_terminate(env['{f}'][{k}])",
                        f"    priorityqueue_destroy(env['{f}'][{k}])"]
        src += ["    event_queue_terminate()",
                "    random_terminate()"]
        return "\n".join(src)

    def _compile(self) -> _Compiled:
        if self._compiled is not None:
            return self._compiled
        if not self._processes:
            raise ValueError("model has no processes")
        bound = {f for _n, _fn, f in self._predicates}
        unbound = [f for f in self._predicate_fields if f not in bound]
        if unbound:
            raise ValueError(f"Predicate field(s) {unbound} declared but "
                             "no @predicate of that name registered")
        bound = {f for _n, _fn, f, _d in self._events}
        unbound = [f for f in self._event_fields if f not in bound]
        if unbound:
            raise ValueError(f"Event field(s) {unbound} declared but "
                             "no @event of that name registered")
        registered = {p.name for p in self._processes}
        unbound = [f for f in self._process_fields if f not in registered]
        if unbound:
            raise ValueError(f"Processes field(s) {unbound} declared but "
                             "no @process of that name registered")
        unbound = [f for f in self._spawnable_fields if f not in registered]
        if unbound:
            raise ValueError(f"Spawnable field(s) {unbound} declared but "
                             "no @process of that name registered")

        dtype = self.dtype
        rec = from_dtype(dtype)
        trial_ptr = types.CPointer(rec)
        evt_sig = types.void(trial_ptr, types.intp)

        proc_cfuncs, pred_cfuncs, event_cfuncs, collect_inner = \
            self._compile_callbacks(rec)
        spawn_descs = {
            p.name: np.array([proc_cfuncs[p.name].address,
                              _b.cstring(p.name), p.alloc_size],
                             dtype=np.int64)
            for p in self._processes if p.spawnable
        }
        ns = self._codegen_namespace(proc_cfuncs, collect_inner)
        self._source = self._trial_source(dtype)
        exec(compile(self._source, f"<cimba model '{self.name}'>", "exec"),
             ns)

        start_rec = cfunc(evt_sig)(ns["_start_rec"])
        stop_rec = cfunc(evt_sig)(ns["_stop_rec"])
        end_sim = cfunc(evt_sig)(ns["_end_sim"])
        ns["EV_START"] = start_rec.address
        ns["EV_STOP"] = stop_rec.address
        ns["EV_END"] = end_sim.address
        trial = cfunc(types.void(trial_ptr))(ns["_trial"])

        # Keep every compiled artifact alive for the model's lifetime
        self._compiled = {
            "trial": trial,
            "events": (start_rec, stop_rec, end_sim),
            "procs": proc_cfuncs,
            "preds": pred_cfuncs,
            "user_events": event_cfuncs,
            "spawns": spawn_descs,
            "collect": collect_inner,
            "dtype": dtype,
        }
        return self._compiled

    # --- Experiments ----------------------------------------------------------
    def experiment(self,
                   *,
                   replications: int = 1,
                   duration: float = 1.0e6,
                   warmup: float = 1.0e3,
                   cooldown: float = 0.0,
                   start_time: float = 0.0,
                   seed: int | None = None,
                   **param_values: ArrayLike) -> "Experiment":
        """Build an experiment: the cross product of the swept parameter
        values (scalars are held fixed), replicated with distinct seeds.

        Trace fields take their replay data here as well: a 1-D array
        shared by every trial, a 2-D array whose row i replays in trial i
        (trial order is design-point-major with replications innermost),
        or a sequence of 1-D arrays for ragged per-trial traces."""
        compiled = self._compile()

        missing = set(self.params) - set(param_values)
        unknown = set(param_values) - set(self.params) - set(self.traces)
        missing_traces = set(self.traces) - set(param_values)
        if missing:
            raise ValueError(f"missing parameter values: {sorted(missing)}")
        if missing_traces:
            raise ValueError(f"missing trace values: "
                             f"{sorted(missing_traces)}")
        if unknown:
            raise ValueError(f"unknown parameters: {sorted(unknown)}")
        if replications < 1:
            raise ValueError("replications must be >= 1")

        axes = [np.atleast_1d(np.asarray(param_values[p], dtype=np.float64))
                for p in self.params]
        mesh = np.meshgrid(*axes, indexing="ij") if axes else []
        n_points = mesh[0].size if mesh else 1
        n_trials = n_points * replications

        trials = np.zeros(n_trials, dtype=compiled["dtype"])
        trials["start_time"] = start_time
        trials["warmup_s"] = warmup
        trials["duration_s"] = duration
        trials["cooldown_s"] = cooldown
        for p, m in zip(self.params, mesh):
            trials[p] = np.repeat(m.ravel(), replications)
        for o in self.outputs:
            trials[o] = np.nan
        for field, pred in compiled["preds"].items():
            trials[field] = pred.address
        for field, ev in compiled["user_events"].items():
            trials[field] = ev.address
        for field, desc in compiled["spawns"].items():
            trials[field] = desc.ctypes.data

        trace_rows: list[np.ndarray] = []
        for tname in self.traces:
            rows = _as_trace_rows(param_values[tname], n_trials, tname)
            trace_rows += rows
            field = trials[tname]
            for i, row in enumerate(rows):
                field[i, 0] = row.ctypes.data
                field[i, 1] = row.size

        rng = np.random.default_rng(
            seed if seed is not None else int(lib.cmb_random_hwseed())
        )
        trials["seed"] = rng.integers(1, np.iinfo(np.uint64).max,
                                      size=n_trials, dtype=np.uint64)
        return Experiment(self, trials, compiled["trial"].address,
                          keepalive=trace_rows)


class Experiment:
    model: Model
    #: One structured record per trial; outputs are filled in by run().
    trials: np.ndarray
    #: Number of failed trials in the last run(), or None before it.
    failures: int | None

    def __init__(self, model: Model, trials: np.ndarray, trial_addr: int,
                 keepalive: Sequence[np.ndarray] = ()):
        self.model = model
        self.trials = trials
        self._trial_addr = trial_addr
        # Trace arrays whose data pointers live in the trial records
        self._keepalive = tuple(keepalive)
        self.failures = None

    def run(self) -> int:
        """Run all trials in parallel, in place. Returns the number of
        failed trials (their outputs stay NaN)."""
        trials = self.trials
        if trials.dtype.names is None:
            raise TypeError("experiment must be a structured array")
        if not trials.flags["C_CONTIGUOUS"]:
            raise ValueError("experiment array must be C-contiguous")
        if trials.ndim != 1 or trials.size == 0:
            raise ValueError("experiment must be a non-empty 1-D array")

        fptr = ffi.cast("void(*)(void *)", self._trial_addr)
        buf = ffi.from_buffer(trials, require_writable=True)
        lib.cimba_run_experiment(buf, trials.size, trials.itemsize, fptr)

        if not self.model.outputs:
            self.failures = 0
        else:
            failed = np.isnan(self.trials[self.model.outputs[0]])
            if failed.ndim > 1:
                failed = failed.reshape(failed.shape[0], -1).any(axis=1)
            self.failures = int(failed.sum())
        return self.failures

    def __getitem__(self, field: str) -> np.ndarray:
        return self.trials[field]

    def __len__(self) -> int:
        return self.trials.size
