"""Method syntax on entity handles, lowered to native helper calls.

Queue/Resource/Pool/Store/PQueues/Condition/Event/Dataset fields are
opaque int64 handles in the trial record, but model code calls methods on
them. Before Numba compiles a callback, ``lower_method_calls`` rewrites
every such call into a plain call of a native helper::

    env.queue.put(1)                -> _cimba_entity_queue_put(env.queue, 1)
    env.samples.mean()              -> _cimba_dataset_mean(env.samples)
    env.queue.history().mean()      -> _cimba_timeseries_mean(
                                           _cimba_history_buffer(env.queue))
    env.cond.wait_for(pred)         -> _cimba_entity_condition_wait(
                                           env.cond, pred, env)

The receiver may also be an indexed field (``env.lanes[i].put(...)``) or a
local alias (``q = env.queue``). ``Event.schedule()``/``.schedule_at()``
return a scheduled-instance handle with verbs of its own (``.cancel()``,
``.time()``, ...), covered by the synthetic ``"event_instance"`` kind.
Methods needing the caller's record (``wait_for``, ``schedule``) receive
the callback's own ``env`` implicitly.

``helper_namespace()`` supplies the helper names to the exec namespace of
lowered code; ``LOWERED_ENTITY_METHODS`` lets process-graph inference map
lowered helper calls back to the entity methods they came from.
"""

import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numba import njit

from . import _bindings as _b
from ._intrinsics import pq_get, record_addr, store_get


@dataclass(frozen=True)
class _MethodSpec:
    """One supported method: the helper it lowers to and its arguments."""

    helper_name: str
    helper: Any
    params: tuple[str, ...] = ()
    defaults: Mapping[str, int | float] = field(default_factory=dict)
    #: the helper takes the caller's env record as a final argument
    needs_env: bool = False

    def normalize_args(
        self,
        kind: str,
        method: str,
        args: Sequence[ast.expr],
        keywords: Sequence[ast.keyword],
        *,
        label: str,
    ) -> list[ast.expr]:
        call = f"{kind} {method}()"
        if len(args) > len(self.params):
            raise ValueError(f"{label} passes too many arguments to {call}")
        supplied = dict(zip(self.params, args))
        for kw in keywords:
            if kw.arg is None:
                raise ValueError(f"{label} cannot use **kwargs with {call}")
            if kw.arg not in self.params:
                raise ValueError(
                    f"{label} passes unknown {call} argument '{kw.arg}'")
            if kw.arg in supplied:
                raise ValueError(
                    f"{label} passes {call} argument '{kw.arg}' more than once")
            supplied[kw.arg] = kw.value

        keyword_order = [self.params.index(kw.arg) for kw in keywords]
        if keyword_order != sorted(keyword_order) and any(
            not isinstance(kw.value, (ast.Name, ast.Constant)) for kw in keywords
        ):
            raise ValueError(
                f"{label} would reorder expressions in {call}; assign keyword "
                "values to local variables first, or use parameter order")

        # Native external functions have no Python defaults. Supply every
        # argument even when the call has no keywords or omits trailing ones.
        result = []
        for param in self.params:
            if param in supplied:
                result.append(supplied[param])
            elif param in self.defaults:
                result.append(ast.Constant(self.defaults[param]))
            else:
                raise ValueError(
                    f"{label} is missing required {call} argument '{param}'")
        return result


# --- Helpers that the raw bindings don't expose directly ----------------------
#
# Native ``*_file`` reporters write to a path handle; path 0 with append=1
# means stdout.

_STDOUT = 0
_APPEND = np.uint64(1)


def _to_stdout(report_file: Any) -> Any:
    @njit
    def helper(handle):
        return report_file(handle, _STDOUT, _APPEND)
    return helper


def _histogram_to_stdout(histogram_file: Any) -> Any:
    @njit
    def helper(handle, bins, low, high):
        return histogram_file(handle, _STDOUT, _APPEND, np.uint64(bins),
                              low, high)
    return helper


def _correlogram_to_stdout(correlogram_file: Any) -> Any:
    @njit
    def helper(handle, lags):
        return correlogram_file(handle, _STDOUT, _APPEND, np.uint64(lags))
    return helper


@njit
def _store_get(store):
    return store_get(store)


@njit
def _pq_get(pqueue):
    return pq_get(pqueue)


@njit
def _condition_wait(condition, predicate, env):
    return _b.condition_wait(condition, predicate, record_addr(env))


@njit
def _event_schedule(event, delay, data, priority, env):
    return _b.event_schedule(event, record_addr(env), data,
                             _b.time() + delay, priority)


@njit
def _event_schedule_at(event, at, data, priority, env):
    return _b.event_schedule(event, record_addr(env), data, at, priority)


# --- Method tables ------------------------------------------------------------

_FILE_ARGS = ("path", "append")
_FILE_DEFAULTS = {"append": 1}


def _entity_spec(label: str, method: str, helper: Any,
                 params: tuple[str, ...] = (), *,
                 defaults: Mapping[str, int | float] | None = None,
                 needs_env: bool = False) -> _MethodSpec:
    return _MethodSpec(f"_cimba_entity_{label}_{method}", helper, params,
                       defaults or {}, needs_env)


def _reporting(label: str, report_file: Any) -> dict[str, _MethodSpec]:
    return {
        "report": _entity_spec(label, "report", _to_stdout(report_file)),
        "report_file": _entity_spec(label, "report_file", report_file,
                                    _FILE_ARGS, defaults=_FILE_DEFAULTS),
    }


#: declared field kind -> {method name: spec}
_ENTITY_METHODS: dict[str, dict[str, _MethodSpec]] = {
    "queue": {
        "put": _entity_spec("queue", "put", _b.buffer_put, ("amount",)),
        "get": _entity_spec("queue", "get", _b.buffer_get, ("amount",)),
        "level": _entity_spec("queue", "level", _b.buffer_level),
        "space": _entity_spec("queue", "space", _b.buffer_space),
        "mean_level": _entity_spec("queue", "mean_level",
                                   _b.buffer_mean_level),
        **_reporting("queue", _b.buffer_report_file),
    },
    "resource": {
        "acquire": _entity_spec("resource", "acquire", _b.resource_acquire),
        "release": _entity_spec("resource", "release", _b.resource_release),
        "preempt": _entity_spec("resource", "preempt", _b.resource_preempt),
        "available": _entity_spec("resource", "available",
                                  _b.resource_available),
        "in_use": _entity_spec("resource", "in_use", _b.resource_in_use),
        "held": _entity_spec("resource", "held", _b.resource_held,
                             ("process",)),
        "mean_in_use": _entity_spec("resource", "mean_in_use",
                                    _b.resource_mean_in_use),
        **_reporting("resource", _b.resource_report_file),
    },
    "pool": {
        "acquire": _entity_spec("pool", "acquire", _b.resourcepool_acquire,
                                ("amount",)),
        "release": _entity_spec("pool", "release", _b.resourcepool_release,
                                ("amount",)),
        "preempt": _entity_spec("pool", "preempt", _b.resourcepool_preempt,
                                ("amount",)),
        "available": _entity_spec("pool", "available",
                                  _b.resourcepool_available),
        "held": _entity_spec("pool", "held", _b.resourcepool_held,
                             ("process",)),
        "in_use": _entity_spec("pool", "in_use", _b.resourcepool_in_use),
        "mean_in_use": _entity_spec("pool", "mean_in_use",
                                    _b.resourcepool_mean_in_use),
        **_reporting("pool", _b.resourcepool_report_file),
    },
    "store": {
        "put": _entity_spec("store", "put", _b.objectqueue_put, ("obj",)),
        "get": _entity_spec("store", "get", _store_get),
        "take": _entity_spec("store", "take", _b.objectqueue_take),
        "length": _entity_spec("store", "length", _b.objectqueue_length),
        "space": _entity_spec("store", "space", _b.objectqueue_space),
        "position": _entity_spec("store", "position",
                                 _b.objectqueue_position, ("obj",)),
        "mean_length": _entity_spec("store", "mean_length",
                                    _b.objectqueue_mean_length),
        **_reporting("store", _b.objectqueue_report_file),
    },
    "pqueues": {
        "put": _entity_spec("pq", "put", _b.priorityqueue_put,
                            ("obj", "priority")),
        "get": _entity_spec("pq", "get", _pq_get),
        "take": _entity_spec("pq", "take", _b.priorityqueue_take),
        "length": _entity_spec("pq", "length", _b.priorityqueue_length),
        "space": _entity_spec("pq", "space", _b.priorityqueue_space),
        "position": _entity_spec("pq", "position",
                                 _b.priorityqueue_position, ("entry",)),
        "reprioritize": _entity_spec("pq", "reprioritize",
                                     _b.priorityqueue_reprioritize,
                                     ("entry", "priority")),
        "cancel": _entity_spec("pq", "cancel", _b.priorityqueue_cancel,
                               ("entry",)),
        "mean_length": _entity_spec("pq", "mean_length",
                                    _b.priorityqueue_mean_length),
        **_reporting("pq", _b.priorityqueue_report_file),
    },
    "condition": {
        "signal": _entity_spec("condition", "signal", _b.condition_signal),
        "wait_for": _entity_spec("condition", "wait", _condition_wait,
                                 ("predicate",), needs_env=True),
    },
    "event": {
        "schedule": _entity_spec(
            "event", "schedule", _event_schedule, ("delay", "data", "priority"),
            defaults={"data": 0, "priority": 0}, needs_env=True),
        "schedule_at": _entity_spec(
            "event", "schedule_at", _event_schedule_at,
            ("at", "data", "priority"),
            defaults={"data": 0, "priority": 0}, needs_env=True),
    },
    #: synthetic kind: the scheduled-instance handle returned by
    #: ``Event.schedule()``/``.schedule_at()``, never a declared field kind.
    "event_instance": {
        "cancel": _entity_spec("event", "cancel", _b.event_cancel),
        "reschedule": _entity_spec("event", "reschedule", _b.event_reschedule,
                                   ("at",)),
        "reprioritize": _entity_spec("event", "reprioritize",
                                     _b.event_reprioritize, ("priority",)),
        "scheduled": _entity_spec("event", "scheduled", _b.event_is_scheduled),
        "time": _entity_spec("event", "time", _b.event_time),
        "priority": _entity_spec("event", "priority", _b.event_priority),
        "wait_event": _entity_spec("event", "wait", _b.process_wait_event),
    },
}


def _statistics(prefix: str) -> dict[str, _MethodSpec]:
    """Summary and report methods shared by datasets (``dataset_*``
    bindings) and entity histories (``timeseries_*`` bindings)."""

    def native(name: str) -> Any:
        return getattr(_b, f"{prefix}_{name}")

    def spec(name: str, helper: Any, params: tuple[str, ...] = (),
             defaults: Mapping[str, int | float] | None = None) -> _MethodSpec:
        return _MethodSpec(f"_cimba_{prefix}_{name}", helper, params,
                           defaults or {})

    lags = {"lags": 20}
    bins = {"bins": 20, "low": 0.0, "high": 0.0}
    std = spec("std", native("std"))
    return {
        "count": spec("count", native("count")),
        "mean": spec("mean", native("mean")),
        "min": spec("min", native("min")),
        "max": spec("max", native("max")),
        "std": std,
        "stddev": std,
        "median": spec("median", native("median")),
        "print": spec("print", _to_stdout(native("print_file"))),
        "print_file": spec("print_file", native("print_file"), _FILE_ARGS,
                           _FILE_DEFAULTS),
        "fivenum": spec("fivenum", _to_stdout(native("fivenum_file"))),
        "fivenum_file": spec("fivenum_file", native("fivenum_file"),
                             _FILE_ARGS, _FILE_DEFAULTS),
        "histogram": spec("histogram",
                          _histogram_to_stdout(native("histogram_file")),
                          ("bins", "low", "high"), bins),
        "histogram_file": spec("histogram_file", native("histogram_file"),
                               (*_FILE_ARGS, "bins", "low", "high"),
                               {**_FILE_DEFAULTS, **bins}),
        "correlogram": spec("correlogram",
                            _correlogram_to_stdout(native("correlogram_file")),
                            ("lags",), lags),
        "correlogram_file": spec("correlogram_file",
                                 native("correlogram_file"),
                                 (*_FILE_ARGS, "lags"),
                                 {**_FILE_DEFAULTS, **lags}),
        "pacf_correlogram": spec(
            "pacf_correlogram",
            _correlogram_to_stdout(native("pacf_correlogram_file")),
            ("lags",), lags),
        "pacf_correlogram_file": spec("pacf_correlogram_file",
                                      native("pacf_correlogram_file"),
                                      (*_FILE_ARGS, "lags"),
                                      {**_FILE_DEFAULTS, **lags}),
    }


_DATASET_METHODS: dict[str, _MethodSpec] = {
    "add": _MethodSpec("_cimba_dataset_add", _b.dataset_add, ("value",)),
    "quantile": _MethodSpec("_cimba_dataset_quantile", _b.dataset_quantile,
                            ("q",)),
    **_statistics("dataset"),
}
_TIMESERIES_METHODS = _statistics("timeseries")

#: history binding (``_FieldKind.binding``) -> native history getter
_HISTORY_GETTERS = {
    "buffer": _b.buffer_history,
    "resource": _b.resource_history,
    "resourcepool": _b.resourcepool_history,
    "objectqueue": _b.objectqueue_history,
    "priorityqueue": _b.priorityqueue_history,
}
#: field kind -> history binding, for the kinds that record a history
_HISTORY_BINDINGS = {
    "queue": "buffer",
    "resource": "resource",
    "pool": "resourcepool",
    "store": "objectqueue",
    "pqueues": "priorityqueue",
}


def history_getter_name(binding: str) -> str:
    return f"_cimba_history_{binding}"


#: every method name that may appear in lowerable source, for cheap
#: ``co_names`` pre-checks
METHOD_NAMES = frozenset((
    *(method for methods in _ENTITY_METHODS.values() for method in methods),
    *_DATASET_METHODS,
    *_TIMESERIES_METHODS,
    "history",
))

#: lowered entity helper name -> (field kind, method) it implements;
#: scheduled-instance verbs report the ``"event"`` kind.
LOWERED_ENTITY_METHODS: dict[str, tuple[str, str]] = {
    spec.helper_name: ("event" if kind == "event_instance" else kind, method)
    for kind, methods in _ENTITY_METHODS.items()
    for method, spec in methods.items()
}

_NAMESPACE: dict[str, Any] = {
    **{spec.helper_name: spec.helper
       for methods in (*_ENTITY_METHODS.values(), _DATASET_METHODS,
                       _TIMESERIES_METHODS)
       for spec in methods.values()},
    **{history_getter_name(binding): getter
       for binding, getter in _HISTORY_GETTERS.items()},
}


def helper_namespace() -> dict[str, Any]:
    """Every helper name lowered code may call."""
    return dict(_NAMESPACE)


# --- Lowering -------------------------------------------------------------------

def _visit_expr(visit: Callable[[ast.AST], ast.AST], node: ast.expr,
                *, what: str) -> ast.expr:
    lowered = visit(node)
    if not isinstance(lowered, ast.expr):
        raise TypeError(f"{what} did not lower to an expression")
    return lowered


class _MethodCallLowerer(ast.NodeTransformer):
    """Rewrite method calls on the entity/dataset fields of one env name."""

    def __init__(self, *, env_name: str, fields: Mapping[str, str],
                 label: str):
        self.env_name = env_name
        self.fields = fields
        self.label = label
        self.changed = False
        #: local alias -> kind, for ``q = env.queue`` then ``q.put(1)``
        self.aliases: dict[str, str] = {}

    def _env(self) -> ast.Name:
        return ast.Name(id=self.env_name, ctx=ast.Load())

    def _is_field(self, node: ast.AST) -> bool:
        return (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == self.env_name
                and node.attr in self.fields)

    def _target(self, node: ast.AST) -> tuple[ast.expr, str] | None:
        """The lowered handle expression and kind of a method receiver."""
        if self._is_field(node):
            assert isinstance(node, ast.Attribute)
            return (ast.Attribute(value=self._env(), attr=node.attr,
                                  ctx=ast.Load()),
                    self.fields[node.attr])
        if isinstance(node, ast.Subscript) and self._is_field(node.value):
            assert isinstance(node.value, ast.Attribute)
            name = node.value.attr
            return (ast.Subscript(
                value=ast.Attribute(value=self._env(), attr=name,
                                    ctx=ast.Load()),
                slice=_visit_expr(self.visit, node.slice,
                                  what="entity field index"),
                ctx=ast.Load(),
            ), self.fields[name])
        if isinstance(node, ast.Name) and node.id in self.aliases:
            return ast.Name(id=node.id, ctx=ast.Load()), self.aliases[node.id]
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("schedule", "schedule_at")):
            receiver = self._target(node.func.value)
            if receiver is not None and receiver[1] == "event":
                return self._lower(node, receiver[0], "event"), \
                    "event_instance"
        return None

    def _history_target(self, node: ast.AST) -> tuple[ast.expr, str] | None:
        """The lowered ``getter(handle)`` of a bare ``<entity>.history()``
        call over a history-recording entity, with its binding."""
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "history"
                and not node.args and not node.keywords):
            return None
        target = self._target(node.func.value)
        if target is None:
            return None
        handle, kind = target
        binding = _HISTORY_BINDINGS.get(kind)
        if binding is None:
            return None
        getter = ast.Call(
            func=ast.Name(id=history_getter_name(binding), ctx=ast.Load()),
            args=[handle], keywords=[])
        return ast.copy_location(getter, node), binding

    def _lower(self, node: ast.Call, receiver: ast.expr, kind: str,
               ) -> ast.Call:
        assert isinstance(node.func, ast.Attribute)
        method = node.func.attr
        if kind == "dataset":
            spec = _DATASET_METHODS.get(method)
        elif kind == "timeseries":
            spec = _TIMESERIES_METHODS.get(method)
        else:
            spec = _ENTITY_METHODS.get(kind, {}).get(method)
        if spec is None:
            raise ValueError(
                f"{self.label} uses unsupported {kind} method {method}()")
        args = [_visit_expr(self.visit, arg, what=f"{kind} method argument")
                for arg in node.args]
        keywords = [
            ast.keyword(arg=kw.arg, value=_visit_expr(
                self.visit, kw.value, what=f"{kind} method keyword"))
            for kw in node.keywords
        ]
        call_args = [receiver, *spec.normalize_args(
            kind, method, args, keywords, label=self.label)]
        if spec.needs_env:
            call_args.append(self._env())
        self.changed = True
        return ast.copy_location(
            ast.Call(func=ast.Name(id=spec.helper_name, ctx=ast.Load()),
                     args=call_args, keywords=[]),
            node)

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            target = self._target(node.value)
            if target is not None:
                self.aliases[name] = target[1]
            else:
                self.aliases.pop(name, None)
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        # <entity>.history().method(...)
        if isinstance(node.func, ast.Attribute):
            history = self._history_target(node.func.value)
            if history is not None:
                return self._lower(node, history[0], "timeseries")
        # bare <entity>.history()
        history = self._history_target(node)
        if history is not None:
            self.changed = True
            return history[0]
        if isinstance(node.func, ast.Attribute):
            target = self._target(node.func.value)
            if target is not None:
                return self._lower(node, *target)
        return self.generic_visit(node)


def lower_method_calls(
    node: ast.FunctionDef,
    *,
    env_name: str,
    fields: Mapping[str, str],
    label: str,
) -> tuple[ast.FunctionDef, bool]:
    """Lower method calls on ``env.<field>`` receivers, where ``fields``
    maps each entity or dataset field name to its kind."""
    lowerer = _MethodCallLowerer(env_name=env_name, fields=fields, label=label)
    lowered = lowerer.visit(node)
    if not isinstance(lowered, ast.FunctionDef):
        raise TypeError("method lowering produced a non-function")
    return lowered, lowerer.changed
