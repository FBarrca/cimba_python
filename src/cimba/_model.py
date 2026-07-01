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

import keyword
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, TypedDict, TypeVar, get_type_hints,
                    overload)

import numpy as np
from numpy.typing import ArrayLike

from numba import carray, cfunc, from_dtype, njit, types
from numba.extending import overload as _nb_overload

from . import _bindings as _b
from ._cimba import ffi, lib
from ._graph import (ProcessDAG, ProcessDAGEdge, ProcessDAGNode,
                     infer_process_dag)
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

    class _Capacity:
        cap: int | str
        def __init__(self, cap: int | str) -> None: ...

    def capacity(cap: int | str) -> Any: ...

    def count(n: int) -> Any: ...

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
                   Spawnable: "spawnable"}


def _class_declarations(cls: type) -> dict[str, Any]:
    """Collect env field declarations from a Model subclass's annotations,
    in declaration order (base classes first)."""
    decls: dict[str, Any] = {"param": [], "output": [], "state": [],
                             "fstate": [], "resource": [],
                             "dataset": [], "condition": [],
                             "predicate": [], "event": [], "processes": [],
                             "spawnable": [],
                             "queue": {}, "pool": {}, "store": {},
                             "pqueues": {}}
    for fname, hint in get_type_hints(cls).items():
        kind = _DECL_KINDS.get(hint)
        if kind is None:
            continue
        default = getattr(cls, fname, None)
        if kind in ("queue", "pool", "store"):
            if isinstance(default, _Capacity):
                default = default.cap
            decls[kind][fname] = default
        elif kind == "pqueues":
            if not isinstance(default, int) or default < 1:
                raise ValueError(
                    f"field '{fname}': a PQueues declaration needs a "
                    "positive count default, e.g. "
                    "'qs: sim.PQueues = sim.count(4)'")
            decls[kind][fname] = default
        else:
            if default is not None:
                raise ValueError(
                    f"field '{fname}': only Queue/Pool/Store declarations "
                    "may carry a capacity default")
            decls[kind].append(fname)
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
        self.pqueues: dict[str, int] = decls["pqueues"]
        self._predicate_fields: list[str] = decls["predicate"]
        self._event_fields: list[str] = decls["event"]
        self._process_fields: list[str] = decls["processes"]
        self._spawnable_fields: list[str] = decls["spawnable"]

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
        self._events.append((name, fn, field, nargs == 2))
        return fn

    def collect(self, fn: _F) -> _F:
        """Register the statistics-collection function, run once at the
        end of each trial."""
        if self._collect is not None:
            raise ValueError("collect() already registered")
        if self._compiled is not None:
            raise RuntimeError("model is already compiled")
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

    @property
    def dtype(self) -> np.dtype:
        # (name, format) or (name, format, shape) numpy field specs
        fields: list[Any] = list(_STANDARD_FIELDS)
        fields += [(p, "<f8") for p in self.params]
        fields += [(o, "<f8") for o in self.outputs]
        fields += [(h, "<i8") for h in self._entities]
        fields += [(s, "<i8") for s in self.state]
        fields += [(s, "<f8") for s in self.float_state]
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
            ns[f"NAME_{e}"] = _b.cstring(e)
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
        src += [f"    {k}_recording_start(env['{n}'])" for k, n in recorded]
        for f, n in self.pqueues.items():
            for k in range(n):
                src += [
                    f"    priorityqueue_recording_start(env['{f}'][{k}])"
                ]
        # Datasets tally over the measurement window only
        src += [f"    dataset_reset(env['{d}'])" for d in self.datasets]
        src += ["def _stop_rec(subject, obj):",
                "    env = carray(subject, 1)[0]"]
        src += [f"    {k}_recording_stop(env['{n}'])" for k, n in recorded]
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
            src += ["    h = buffer_create()",
                    f"    buffer_initialize(h, NAME_{q}, {cap_expr(cap)})",
                    f"    env['{q}'] = h"]
        for r in self.resources:
            src += ["    h = resource_create()",
                    f"    resource_initialize(h, NAME_{r})",
                    f"    env['{r}'] = h"]
        for p, cap in self.pools.items():
            src += ["    h = resourcepool_create()",
                    f"    resourcepool_initialize(h, NAME_{p}, "
                    f"{cap_expr(cap)})",
                    f"    env['{p}'] = h"]
        for s, cap in self.stores.items():
            src += ["    h = objectqueue_create()",
                    f"    objectqueue_initialize(h, NAME_{s}, "
                    f"{cap_expr(cap)})",
                    f"    env['{s}'] = h"]
        for d in self.datasets:
            src += ["    h = dataset_create()",
                    "    dataset_initialize(h)",
                    f"    env['{d}'] = h"]
        for c in self.conditions:
            src += ["    h = condition_create()",
                    f"    condition_initialize(h, NAME_{c})",
                    f"    env['{c}'] = h"]
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
            src += [f"    buffer_destroy(env['{q}'])"]
        for r in self.resources:
            src += [f"    resource_destroy(env['{r}'])"]
        for p in self.pools:
            src += [f"    resourcepool_destroy(env['{p}'])"]
        for s in self.stores:
            src += [f"    objectqueue_destroy(env['{s}'])"]
        for d in self.datasets:
            src += [f"    dataset_destroy(env['{d}'])"]
        for c in self.conditions:
            src += [f"    condition_destroy(env['{c}'])"]
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
        values (scalars are held fixed), replicated with distinct seeds."""
        compiled = self._compile()

        missing = set(self.params) - set(param_values)
        unknown = set(param_values) - set(self.params)
        if missing:
            raise ValueError(f"missing parameter values: {sorted(missing)}")
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

        rng = np.random.default_rng(
            seed if seed is not None else int(lib.cmb_random_hwseed())
        )
        trials["seed"] = rng.integers(1, np.iinfo(np.uint64).max,
                                      size=n_trials, dtype=np.uint64)
        return Experiment(self, trials, compiled["trial"].address)


class Experiment:
    model: Model
    #: One structured record per trial; outputs are filled in by run().
    trials: np.ndarray
    #: Number of failed trials in the last run(), or None before it.
    failures: int | None

    def __init__(self, model: Model, trials: np.ndarray, trial_addr: int):
        self.model = model
        self.trials = trials
        self._trial_addr = trial_addr
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
            self.failures = int(
                np.isnan(self.trials[self.model.outputs[0]]).sum()
            )
        return self.failures

    def __getitem__(self, field: str) -> np.ndarray:
        return self.trials[field]

    def __len__(self) -> int:
        return self.trials.size
