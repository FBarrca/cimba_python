"""Declaration markers and the typed field model behind them.

Model and Component classes declare env fields with marker annotations
(``queue: sim.Queue``). This module defines those markers and the typed
model they collect into:

* ``_FieldKind`` -- one static descriptor per declaration kind, holding
  everything the later stages need to know about fields of that kind
  (trial-record format, native entity binding, wirability, ...), so the
  flattening and codegen stages never hardcode per-kind special cases;
* ``_FieldDecl`` -- one declared field: a name, its kind, and the
  kind-specific extras (capacity, count, flattened shape);
* ``_Declarations`` -- the ordered collection of a class's declared
  fields plus the component metadata gathered alongside them.
"""

import keyword
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, get_type_hints

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


# --- Field kinds --------------------------------------------------------------

@dataclass(frozen=True)
class _FieldKind:
    """Static description of one declaration kind."""

    name: str
    #: numpy field format in the trial record
    fmt: str
    #: native entity prefix: "buffer" -> buffer_create()/_initialize()/
    #: _destroy() in the generated trial source (None: not an entity)
    binding: str | None = None
    #: recording_start/stop over the measurement window
    recordable: bool = False
    #: <binding>_initialize takes an interned entity-name cstring
    named: bool = True
    #: may be wired to another component instance's same-kind field
    wirable: bool = False
    #: the declaration default is a Queue/Pool/Store capacity
    capacitated: bool = False
    #: appears as an entity node in the inferred process DAG
    dag_entity: bool = False
    #: may be declared on Component classes
    on_component: bool = True


_KIND_LIST = [
    _FieldKind("param", "<f8"),
    _FieldKind("output", "<f8"),
    _FieldKind("state", "<i8", dag_entity=True),
    _FieldKind("fstate", "<f8", dag_entity=True),
    _FieldKind("queue", "<i8", binding="buffer", recordable=True,
               wirable=True, capacitated=True, dag_entity=True),
    _FieldKind("resource", "<i8", binding="resource", recordable=True,
               wirable=True, dag_entity=True),
    _FieldKind("pool", "<i8", binding="resourcepool", recordable=True,
               wirable=True, capacitated=True, dag_entity=True),
    _FieldKind("store", "<i8", binding="objectqueue", recordable=True,
               wirable=True, capacitated=True, dag_entity=True),
    _FieldKind("dataset", "<i8", binding="dataset", named=False),
    _FieldKind("condition", "<i8", binding="condition", wirable=True,
               dag_entity=True),
    _FieldKind("predicate", "<i8", on_component=False),
    _FieldKind("event", "<i8", dag_entity=True, on_component=False),
    _FieldKind("processes", "<i8"),
    # PQueues elements are created/recorded/destroyed per element, so the
    # trial codegen handles them apart from the scalar entity kinds.
    _FieldKind("pqueues", "<i8", binding="priorityqueue", dag_entity=True),
    _FieldKind("spawnable", "<i8"),
    _FieldKind("trace", "<i8"),
]

_FIELD_KINDS: dict[str, _FieldKind] = {kind.name: kind for kind in _KIND_LIST}


@dataclass(frozen=True)
class _FieldDecl:
    """One declared env field."""

    name: str
    kind: _FieldKind
    #: Queue/Pool/Store capacity: an int, a param name, or None (unbounded)
    capacity: int | str | None = None
    #: PQueues element count: an int or (components) a constant name;
    #: after flattening, the total element count
    count: int | str | None = None
    #: element count of flattened multi-instance fields, None for scalars
    shape: tuple[int, ...] | None = None


class _Declarations:
    """Ordered env-field declarations of one Model or Component class,
    plus the component metadata collected alongside them."""

    def __init__(self) -> None:
        self.fields: dict[str, _FieldDecl] = {}
        #: Ref / Refs annotations (Component classes only): name -> target
        self.refs: dict[str, Any] = {}
        self.ref_tables: dict[str, Any] = {}
        #: Const annotations (Component classes only): name -> declared type
        self.consts: dict[str, type] = {}
        #: component tree roots (Model classes only)
        self.components: list[Any] = []
        self.component_collections: list[Any] = []

    def add(self, field: _FieldDecl) -> None:
        if field.name in self.fields:
            raise ValueError(f"duplicate field name '{field.name}'")
        self.fields[field.name] = field

    def kind_of(self, name: str) -> str | None:
        field = self.fields.get(name)
        return None if field is None else field.kind.name

    def by_kind(self, *kinds: str) -> list[_FieldDecl]:
        """Fields of the given kinds, kind-major, declaration order
        within each kind."""
        return [field for kind in kinds for field in self.fields.values()
                if field.kind.name == kind]

    def names(self, *kinds: str) -> list[str]:
        return [field.name for field in self.by_kind(*kinds)]


# --- Declaration markers --------------------------------------------------------

if TYPE_CHECKING:
    from typing import Union

    import numpy as np
    from numpy.typing import NDArray

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
    #: reference to another declared component instance; Ref[Station]
    #: type-checks as Station
    Ref = Union
    #: indexable table of component references; Refs[Station][i]
    #: type-checks as Station
    Refs = Sequence
    #: per-instance constant read inside component code; Const[int]
    #: type-checks as int
    Const = Union

    class _RefHint:
        target: Any
        table: bool

    class _ConstHint:
        type: Any

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

    _DECL_KINDS: dict[Any, _FieldKind] = {}
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

    class _RefHint:
        """Annotation marker produced by Ref[Target] / Refs[Target]."""

        __slots__ = ("target", "table")

        def __init__(self, target, table):
            self.target = target
            self.table = table

    class Ref:
        """Reference to another declared component instance, e.g.
        ``downstream: sim.Ref[Station]``; the value is set on the instance
        (usually in ``__init__``) and resolved when the model is built."""

        def __class_getitem__(cls, target):
            return _RefHint(target, table=False)

    class Refs:
        """Indexable table of component references, e.g.
        ``routes: sim.Refs[Station]``; entries must be items of a single
        component collection so ``self.routes[i]`` can lower to an
        array lookup."""

        def __class_getitem__(cls, target):
            return _RefHint(target, table=True)

    class _ConstHint:
        """Annotation marker produced by ``Const[type]``."""

        __slots__ = ("type",)

        def __init__(self, type_):
            self.type = type_

    class Const:
        """A per-instance constant read inside component code, e.g.
        ``rate: sim.Const[int]``. The value is set on the instance
        (usually in ``__init__``) and baked into the compiled model --
        specialized to a literal when one instance is compiled, or read
        from a per-instance table when a collection shares one compiled
        body."""

        def __class_getitem__(cls, type_):
            return _ConstHint(type_)

    class _Capacity:
        def __init__(self, cap):
            self.cap = cap

    def capacity(cap):
        """Declare a Pool/Store capacity: an int or the name of a param."""
        return _Capacity(cap)

    def count(n):
        """Declare the number of elements in a PQueues field."""
        return n

    _DECL_KINDS = {Param: _FIELD_KINDS["param"],
                   Output: _FIELD_KINDS["output"],
                   State: _FIELD_KINDS["state"],
                   FloatState: _FIELD_KINDS["fstate"],
                   Queue: _FIELD_KINDS["queue"],
                   Resource: _FIELD_KINDS["resource"],
                   Pool: _FIELD_KINDS["pool"],
                   Store: _FIELD_KINDS["store"],
                   Dataset: _FIELD_KINDS["dataset"],
                   Condition: _FIELD_KINDS["condition"],
                   Predicate: _FIELD_KINDS["predicate"],
                   Event: _FIELD_KINDS["event"],
                   Processes: _FIELD_KINDS["processes"],
                   PQueues: _FIELD_KINDS["pqueues"],
                   Spawnable: _FIELD_KINDS["spawnable"],
                   Trace: _FIELD_KINDS["trace"]}

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


def _field_declarations(
    cls: type,
    *,
    allow_symbolic_pqueues: bool = False,
    allow_refs: bool = False,
) -> _Declarations:
    """Collect direct env field declarations from a Model/Component class."""
    decls = _Declarations()
    for fname, hint in get_type_hints(cls).items():
        if isinstance(hint, _RefHint):
            if not allow_refs:
                raise ValueError(
                    f"field '{fname}': Ref/Refs declarations are only "
                    "supported on Component classes")
            _check_name(fname, "ref")
            (decls.ref_tables if hint.table else decls.refs)[fname] = \
                hint.target
            continue
        if isinstance(hint, _ConstHint):
            if not allow_refs:
                raise ValueError(
                    f"field '{fname}': Const declarations are only "
                    "supported on Component classes")
            _check_name(fname, "const")
            decls.consts[fname] = hint.type
            continue
        try:
            kind = _DECL_KINDS.get(hint)
        except TypeError:
            kind = None
        if kind is None:
            continue
        default = getattr(cls, fname, None)
        if kind.capacitated:
            if isinstance(default, _Capacity):
                default = default.cap
            decls.add(_FieldDecl(fname, kind, capacity=default))
        elif kind.name == "pqueues":
            if isinstance(default, int) and default >= 1:
                decls.add(_FieldDecl(fname, kind, count=default))
            elif allow_symbolic_pqueues and isinstance(default, str):
                _check_name(default, "PQueues count constant")
                decls.add(_FieldDecl(fname, kind, count=default))
            else:
                raise ValueError(
                    f"field '{fname}': a PQueues declaration needs a "
                    "positive count default, e.g. "
                    "'qs: sim.PQueues = sim.count(4)'")
        else:
            method_binding = (
                kind.name in ("processes", "spawnable")
                and getattr(default, "__cimba_component_process__", None)
                is not None
            )
            if default is not None and not method_binding:
                raise ValueError(
                    f"field '{fname}': only Queue/Pool/Store declarations "
                    "may carry a capacity default")
            decls.add(_FieldDecl(fname, kind))
    return decls
