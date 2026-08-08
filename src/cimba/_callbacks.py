"""Shared callback declarations for Model and Component classes.

This module deliberately knows nothing about trial layouts or component
flattening.  It owns the public decorators and reduces a callback owner's MRO
to one immutable, effective declaration set.  Root and nested owners consume
that same set with different lowering policies later.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import inspect
import threading
from typing import Any, get_type_hints, Literal, TypeVar, overload
import weakref

from ._declarations import (
    _FIELD_KINDS,
    _MISSING,
    _Declarations,
    _FieldDecl,
    Spawnable,
    _check_name,
)

_F = TypeVar("_F", bound=Callable[..., Any])

_PROCESS_ATTR = "__cimba_process__"
_COLLECT_ATTR = "__cimba_collect__"
_PREDICATE_ATTR = "__cimba_predicate__"
_EVENT_ATTR = "__cimba_event__"
_FUNCTION_ATTR = "__cimba_function__"


class _DeclarationOwner:
    """Shared declaration and callback behavior for Models and Components.

    Both public owner types expose the same callback language.  Keeping the
    class-level validation and declaration binding here prevents the two
    implementations from growing subtly different rules.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "_validate_legacy_annotations", False):
            return
        for name, annotation in vars(cls).get("__annotations__", {}).items():
            if annotation is Spawnable:
                raise ValueError(
                    f"field '{name}': sim.Spawnable has been replaced by @sim.process(spawnable=True)"
                )

    @classmethod
    def _callbacks(cls) -> "_CallbackSet":
        return _callback_set(cls)

    @classmethod
    def _field_declarations(cls, **options: Any) -> _Declarations:
        """Collect this owner's typed field declarations."""
        from ._declarations import _field_declarations

        return _field_declarations(cls, **options)

    @classmethod
    def _bind_callbacks(
        cls,
        decls: _Declarations,
        *,
        owner: str,
        protected: frozenset[str] = frozenset(),
    ) -> "_CallbackSet":
        return _bind_callback_fields(cls, decls, owner=owner, protected=protected)


def _positional_parameters(
    method: Callable[..., Any],
    signature: str,
) -> tuple[inspect.Parameter, ...]:
    parameters = tuple(inspect.signature(method).parameters.values())
    if any(
        parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
        or parameter.default is not inspect.Parameter.empty
        for parameter in parameters
    ):
        raise ValueError(signature)
    return parameters


def _callback_arg_count(
    method: Callable[..., Any],
    allowed: tuple[int, ...],
    signature: str,
) -> int:
    count = len(_positional_parameters(method, signature))
    if count not in allowed:
        raise ValueError(signature)
    return count


def _process_signature(
    method: Callable[..., Any],
    receiver_count: int,
    is_struct_class: Callable[[Any], bool],
    label: str,
    signature: str,
    localns: Mapping[str, Any] | None = None,
) -> tuple[type | None, int]:
    """Validate the indexed and optional-Struct portion of a process."""
    parameters = _positional_parameters(method, signature)
    parameter_names = tuple(parameter.name for parameter in parameters)
    hints = get_type_hints(method, localns=localns)
    own = (hints.get(parameter_names[-1])
           if len(parameter_names) > receiver_count else None)
    struct_view = own if is_struct_class(own) else None
    injected = struct_view is not None
    end = len(parameter_names) - (1 if injected else 0)
    for name in parameter_names[receiver_count:end]:
        if is_struct_class(hints.get(name)):
            raise ValueError(
                f"{label}: the {hints[name].__name__} view must be the last "
                "parameter")

    base_arg_count = len(parameter_names) - (1 if injected else 0)
    if base_arg_count not in (receiver_count, receiver_count + 1):
        raise ValueError(signature)
    return struct_view, base_arg_count


@dataclass(frozen=True)
class _ProcessSpec:
    """Arguments captured by ``@sim.process``."""

    copies: int | str = 1
    priority: int = 0
    spawnable: bool = False
    struct: Any = None
    field: str | None = None

    def resolve_copies(self, owner: Any, label: str) -> int:
        """Resolve a component's literal or constant-named copy count."""
        if isinstance(self.copies, int):
            return self.copies
        value = getattr(owner, self.copies, _MISSING)
        if type(value) is not int or value < 1:
            raise ValueError(
                f"component process '{label}' copies constant "
                f"'{self.copies}' must be a positive int")
        return value


@dataclass(frozen=True)
class _CallbackFieldSpec:
    field: str | None = None


@dataclass(frozen=True)
class _CallbackDecl:
    name: str
    kind: str
    fn: Callable[..., Any]
    spec: Any
    order: int
    field: str | None = None


@dataclass(frozen=True)
class _CallbackSet:
    """Effective callbacks of one class after MRO replacement/removal."""

    declarations: tuple[_CallbackDecl, ...]
    processes: tuple[_CallbackDecl, ...] = ()
    collectors: tuple[_CallbackDecl, ...] = ()
    predicates: tuple[_CallbackDecl, ...] = ()
    events: tuple[_CallbackDecl, ...] = ()
    functions: tuple[_CallbackDecl, ...] = ()


_MARKERS = (
    ("process", _PROCESS_ATTR),
    ("collect", _COLLECT_ATTR),
    ("predicate", _PREDICATE_ATTR),
    ("event", _EVENT_ATTR),
    ("function", _FUNCTION_ATTR),
)

_CALLBACK_SET_CACHE: weakref.WeakKeyDictionary[type, _CallbackSet] = \
    weakref.WeakKeyDictionary()
_CALLBACK_SET_LOCK = threading.RLock()


def _callback_set(cls: type) -> _CallbackSet:
    """Normalize all callback markers in one base-first MRO pass.

    A decorated same-name override keeps the inherited declaration's
    position.  An undecorated override removes it; a later descendant that
    decorates that name again receives a new declaration position.
    """
    with _CALLBACK_SET_LOCK:
        cached = _CALLBACK_SET_CACHE.get(cls)
    if cached is not None:
        return cached

    effective: dict[str, _CallbackDecl] = {}
    next_order = 0
    for base in reversed(cls.__mro__):
        if base in (object, _DeclarationOwner):
            continue
        for name, value in vars(base).items():
            marked = [
                (kind, getattr(value, marker, None))
                for kind, marker in _MARKERS
                if getattr(value, marker, None) not in (None, False)
            ]
            if not marked:
                effective.pop(name, None)
                continue
            if len(marked) != 1:
                kinds = ", ".join(kind for kind, _spec in marked)
                raise ValueError(
                    f"callback '{cls.__name__}.{name}' has conflicting markers: {kinds}"
                )
            if not callable(value):
                raise TypeError(
                    f"callback '{cls.__name__}.{name}' is not callable")
            kind, spec = marked[0]
            previous = effective.get(name)
            if previous is None:
                order = next_order
                next_order += 1
            else:
                order = previous.order
            field = spec.field if kind in {"process", "predicate", "event"} \
                else None
            if kind == "predicate" and field is None:
                field = f"_pred_{name}"
            elif kind == "event" and field is None:
                field = f"_ev_{name}"
            elif kind == "process" and spec.spawnable and field is None:
                field = name
            effective[name] = _CallbackDecl(
                name=name, kind=kind, fn=value, spec=spec, order=order, field=field
            )

    declarations = tuple(sorted(effective.values(), key=lambda item: item.order))
    processes = tuple(decl for decl in declarations if decl.kind == "process")
    collectors_set = tuple(decl for decl in declarations
                           if decl.kind == "collect")
    predicates = tuple(decl for decl in declarations
                       if decl.kind == "predicate")
    events = tuple(decl for decl in declarations if decl.kind == "event")
    functions = tuple(decl for decl in declarations if decl.kind == "function")
    collectors = [decl.name for decl in collectors_set]
    if len(collectors) > 1:
        names = ", ".join(collectors)
        raise ValueError(
            f"callback owner '{cls.__name__}' has multiple collect callbacks: "
            f"{names}; override an inherited collector using the same method "
            "name")
    result = _CallbackSet(
        declarations, processes, collectors_set, predicates, events, functions)
    with _CALLBACK_SET_LOCK:
        existing = _CALLBACK_SET_CACHE.get(cls)
        if existing is not None:
            return existing
        _CALLBACK_SET_CACHE[cls] = result
    return result


def _bind_callback_fields(
    cls: type,
    decls: _Declarations,
    *,
    owner: str,
    protected: frozenset[str] = frozenset(),
) -> _CallbackSet:
    callbacks = _callback_set(cls)
    field_bindings: dict[tuple[str, str], str] = {}
    component = owner == "component"

    for callback in callbacks.declarations:
        if callback.name in protected:
            raise ValueError(
                f"{owner} {callback.kind} callback '{cls.__name__}."
                f"{callback.name}' shadows a public sim.Model operation")
        if callback.name in decls.fields:
            raise ValueError(
                f"{owner} callback '{cls.__name__}.{callback.name}' collides "
                "with a declared field; use a differently named callback "
                "and field= when binding callback fields")
        if callback.kind == "process":
            copies = callback.spec.copies
            if not component and type(copies) is not int:
                raise TypeError(
                    f"model process '{cls.__name__}.{callback.name}' copies "
                    "must be an int")

        if callback.kind in {"process", "predicate", "event"}:
            field = callback.spec.field
            if field is not None:
                kind = {
                    "process": "processes",
                    "predicate": "predicate",
                    "event": "event",
                }[callback.kind]
                field_decl = decls.fields.get(field)
                if field_decl is None or field_decl.kind.name != kind:
                    public = {
                        "processes": "Processes",
                        "predicate": "Predicate",
                        "event": "Event",
                    }[kind]
                    raise ValueError(
                        f"{owner} callback '{cls.__name__}.{callback.name}' "
                        f"field= must name a declared sim.{public} field")
                key = (kind, field)
                previous = field_bindings.get(key)
                if previous is not None:
                    if owner == "model":
                        raise ValueError(
                            f"{owner} {kind} field '{field}' already bound")
                    raise ValueError(
                        f"{owner} {kind} field '{field}' is bound by both "
                        f"'{previous}' and '{callback.name}'")
                field_bindings[key] = callback.name
            elif callback.kind in {"predicate", "event"}:
                decls.add(_FieldDecl(
                    callback.field,
                    _FIELD_KINDS[callback.kind],
                ))
            elif callback.spec.spawnable:
                decls.add(_FieldDecl(
                    callback.field, _FIELD_KINDS["spawnable"]))

    if component:
        for kind in ("processes", "predicate", "event"):
            unbound = [
                name for name in decls.names(kind)
                if not name.startswith(("_pred_", "_ev_"))
                and (kind, name) not in field_bindings
            ]
            if unbound:
                fields = ", ".join(unbound)
                raise ValueError(
                    f"component '{cls.__name__}' has unbound {kind} field(s): "
                    f"{fields}; bind each with field=")
    return callbacks


class SpawnableProcess:
    """Static marker returned by ``@sim.process(spawnable=True)``."""


def _reject_marker_conflict(fn: Callable[..., Any], kind: str) -> None:
    conflicts = [
        other for other, marker in _MARKERS
        if other != kind and getattr(fn, marker, None) not in (None, False)
    ]
    if conflicts:
        raise ValueError(
            f"'{fn.__qualname__}' cannot combine {kind} with "
            + ", ".join(conflicts))


@overload
def process(fn: _F) -> _F: ...


@overload
def process(fn: None = None, *, copies: Literal[1] = 1,
            priority: int = 0, spawnable: Literal[True],
            struct: Any = None, field: None = None,
            ) -> Callable[[_F], SpawnableProcess]: ...


@overload
def process(fn: None = None, *, copies: int | str = 1,
            priority: int = 0, spawnable: Literal[False] = False,
            struct: Any = None, field: str | None = None,
            ) -> Callable[[_F], _F]: ...


def process(fn=None, *, copies: int | str = 1, priority: int = 0,
            spawnable: bool = False, struct: Any = None,
            field: str | None = None):
    """Mark a Model or Component method as a process."""
    if isinstance(copies, int):
        if copies < 1:
            raise ValueError("copies must be >= 1")
    elif isinstance(copies, str):
        _check_name(copies, "copies constant")
    else:
        raise TypeError("copies must be an int or the name of an int constant")
    if field is not None:
        _check_name(field, "process field")
    if spawnable and copies != 1:
        raise ValueError("spawnable processes cannot take copies")
    if spawnable and field is not None:
        raise ValueError("spawnable processes cannot take field=")

    def decorate(f):
        _reject_marker_conflict(f, "process")
        setattr(f, _PROCESS_ATTR,
                _ProcessSpec(copies, priority, spawnable, struct, field))
        return f

    return decorate if fn is None else decorate(fn)


def collect(fn: _F) -> _F:
    """Mark a Model or Component end-of-trial collector."""
    _reject_marker_conflict(fn, "collect")
    setattr(fn, _COLLECT_ATTR, True)
    return fn


@overload
def predicate(fn: _F) -> _F: ...


@overload
def predicate(fn: None = None, *, field: str | None = None) -> Callable[[_F], _F]: ...


def predicate(fn=None, *, field: str | None = None):
    """Mark a Model or Component condition predicate callback."""
    return _signal(fn, field, "predicate", _PREDICATE_ATTR)


@overload
def event(fn: _F) -> _F: ...


@overload
def event(fn: None = None, *, field: str | None = None) -> Callable[[_F], _F]: ...


def event(fn=None, *, field: str | None = None):
    """Mark a Model or Component low-level event callback."""
    return _signal(fn, field, "event", _EVENT_ATTR)


def _signal(fn, field: str | None, kind: str, marker: str):
    if field is not None:
        _check_name(field, f"{kind} field")

    def decorate(f):
        _reject_marker_conflict(f, kind)
        setattr(f, marker, _CallbackFieldSpec(field))
        return f

    return decorate if fn is None else decorate(fn)


def function(fn: _F) -> _F:
    """Mark a read-only synchronous Model or Component helper."""
    _reject_marker_conflict(fn, "function")
    setattr(fn, _FUNCTION_ATTR, True)
    return fn
