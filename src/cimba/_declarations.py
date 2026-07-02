"""Shared declaration marker types for the sim modeling API."""

import keyword
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, get_type_hints

import numpy as np
from numpy.typing import NDArray

from numba import carray, types
from numba.extending import overload as _nb_overload

from ._intrinsics import ptr_caster

#: Opaque native entity handle (process, queue, resource, ...) as stored
#: in env fields.
Handle = int

#: The per-trial record passed to process bodies: a numpy structured
#: scalar whose fields are accessed as attributes inside nopython code.
#: Annotate env with your Model subclass to get typed fields; Env is the
#: untyped fallback.
Env = Any

_MISSING = object()

#: A `pools`/`stores` declaration: list of names (unbounded capacity) or
#: name -> capacity mapping (an int, a param name, or None for unbounded).
_Capacities = Mapping[str, int | str | None] | Iterable[str] | None

_STANDARD_FIELDS = [
    ("start_time", "<f8"),
    ("warmup_s", "<f8"),
    ("duration_s", "<f8"),
    ("cooldown_s", "<f8"),
    ("seed", "<u8"),
]
_RESERVED = {name for name, _ in _STANDARD_FIELDS}


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


def _check_name(name: str, kind: str) -> None:
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"{kind} name '{name}' is not a valid identifier")
    if name in _RESERVED or name.startswith("_"):
        raise ValueError(f"{kind} name '{name}' is reserved")


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
            method_binding = (
                kind in ("processes", "spawnable")
                and getattr(default, "__cimba_component_process__", None)
                is not None
            )
            if default is not None and not method_binding:
                raise ValueError(
                    f"field '{fname}': only Queue/Pool/Store declarations "
                    "may carry a capacity default")
            decls[kind].append(fname)
    return decls


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
