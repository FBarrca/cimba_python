"""AST lowering support for compiled ``<entity>.history()`` calls."""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import helpers as _timeseries_helpers


@dataclass(frozen=True)
class _TimeseriesMethodSpec:
    helper_name: str
    helper_attr: str
    params: tuple[str, ...] = ()
    defaults: Mapping[str, object] = field(default_factory=dict)

    def normalize_args(
        self,
        method: str,
        args: Sequence[ast.expr],
        keywords: Sequence[ast.keyword],
        *,
        label: str,
    ) -> list[ast.expr]:
        if len(args) > len(self.params):
            raise ValueError(
                f"{label} passes too many arguments to timeseries "
                f"{method}()")
        call_args = list(args)
        if not keywords:
            return call_args

        by_name = {name: index for index, name in enumerate(self.params)}
        supplied = set(self.params[:len(call_args)])
        keyed: dict[int, ast.expr] = {}
        max_index = len(call_args) - 1
        for kw in keywords:
            if kw.arg is None:
                raise ValueError(
                    f"{label} cannot use **kwargs with timeseries "
                    f"{method}()")
            index = by_name.get(kw.arg)
            if index is None:
                raise ValueError(
                    f"{label} passes unknown timeseries {method}() "
                    f"argument '{kw.arg}'")
            if kw.arg in supplied or index in keyed:
                raise ValueError(
                    f"{label} passes timeseries {method}() argument "
                    f"'{kw.arg}' more than once")
            supplied.add(kw.arg)
            keyed[index] = kw.value
            max_index = max(max_index, index)

        for index in range(len(call_args), max_index + 1):
            if index in keyed:
                call_args.append(keyed[index])
            else:
                param = self.params[index]
                if param not in self.defaults:
                    raise ValueError(
                        f"{label} is missing required timeseries "
                        f"{method}() argument '{param}'")
                call_args.append(ast.Constant(self.defaults[param]))
        return call_args


def _spec(
    method: str,
    params: tuple[str, ...] = (),
    *,
    helper_attr: str | None = None,
    defaults: Mapping[str, object] | None = None,
) -> _TimeseriesMethodSpec:
    attr = helper_attr or method
    return _TimeseriesMethodSpec(
        helper_name=f"_cimba_timeseries_{attr}",
        helper_attr=attr,
        params=params,
        defaults=defaults or {},
    )


_METHOD_SPECS = {
    "count": _spec("count"),
    "mean": _spec("mean"),
    "min": _spec("min"),
    "max": _spec("max"),
    "std": _spec("std"),
    "stddev": _spec("stddev", helper_attr="std"),
    "median": _spec("median"),
    "print": _spec("print"),
    "print_file": _spec("print_file", ("path", "append"),
                        defaults={"append": 1}),
    "fivenum": _spec("fivenum"),
    "fivenum_file": _spec("fivenum_file", ("path", "append"),
                          defaults={"append": 1}),
    "histogram": _spec("histogram", ("bins", "low", "high"),
                       defaults={"bins": 20, "low": 0.0, "high": 0.0}),
    "histogram_file": _spec(
        "histogram_file",
        ("path", "append", "bins", "low", "high"),
        defaults={"append": 1, "bins": 20, "low": 0.0, "high": 0.0},
    ),
    "correlogram": _spec("correlogram", ("lags",),
                         defaults={"lags": 20}),
    "correlogram_file": _spec(
        "correlogram_file",
        ("path", "append", "lags"),
        defaults={"append": 1, "lags": 20},
    ),
    "pacf_correlogram": _spec("pacf_correlogram", ("lags",),
                              defaults={"lags": 20}),
    "pacf_correlogram_file": _spec(
        "pacf_correlogram_file",
        ("path", "append", "lags"),
        defaults={"append": 1, "lags": 20},
    ),
}

TIMESERIES_METHOD_NAMES = frozenset(_METHOD_SPECS)

#: field-kind binding name (``_FieldKind.binding``) -> history getter helper
HISTORY_GETTER_NAMES = {
    "buffer": "_cimba_history_buffer",
    "resource": "_cimba_history_resource",
    "resourcepool": "_cimba_history_resourcepool",
    "objectqueue": "_cimba_history_objectqueue",
    "priorityqueue": "_cimba_history_priorityqueue",
}


def timeseries_lowering_namespace() -> dict[str, Any]:
    namespace = {
        spec.helper_name: getattr(_timeseries_helpers, spec.helper_attr)
        for spec in _METHOD_SPECS.values()
    }
    namespace.update({
        helper_name: _timeseries_helpers.HISTORY_GETTERS[binding]
        for binding, helper_name in HISTORY_GETTER_NAMES.items()
    })
    return namespace


def _visit_expr(
    visit: Callable[[ast.AST], ast.AST],
    node: ast.expr,
    *,
    what: str,
) -> ast.expr:
    lowered = visit(node)
    if not isinstance(lowered, ast.expr):
        raise TypeError(f"{what} did not lower to an expression")
    return lowered


def lower_timeseries_method_call(
    node: ast.Call,
    entity: ast.expr,
    *,
    binding: str,
    visit: Callable[[ast.AST], ast.AST],
    label: str,
) -> ast.Call:
    """Lower ``<entity>.history.method(...)`` to a native timeseries call.
    ``entity`` is the already-lowered handle expression (the queue/resource/
    pool/store/priority-queue itself, not its ``.history``)."""
    if not isinstance(node.func, ast.Attribute):
        raise TypeError("timeseries method call must use attribute syntax")
    method = node.func.attr
    spec = _METHOD_SPECS.get(method)
    if spec is None:
        raise ValueError(
            f"{label} uses unsupported timeseries method {method}()")
    getter_name = HISTORY_GETTER_NAMES.get(binding)
    if getter_name is None:
        raise ValueError(f"{label} has no timeseries history for '{binding}'")

    args = [
        _visit_expr(visit, arg, what="timeseries method argument")
        for arg in node.args
    ]
    keywords = [
        ast.keyword(
            arg=kw.arg,
            value=_visit_expr(
                visit, kw.value, what="timeseries method keyword"),
        )
        for kw in node.keywords
    ]
    entity.ctx = ast.Load()
    history_call = ast.Call(
        func=ast.Name(id=getter_name, ctx=ast.Load()),
        args=[entity],
        keywords=[],
    )
    return ast.copy_location(
        ast.Call(
            func=ast.Name(id=spec.helper_name, ctx=ast.Load()),
            args=[history_call, *spec.normalize_args(
                method, args, keywords, label=label)],
            keywords=[],
        ),
        node,
    )


def lower_history_getter_call(
    node: ast.Call,
    entity: ast.expr,
    *,
    binding: str,
    label: str,
) -> ast.Call:
    """Lower a bare ``<entity>.history()`` call to its native getter."""
    getter_name = HISTORY_GETTER_NAMES.get(binding)
    if getter_name is None:
        raise ValueError(f"{label} has no timeseries history for '{binding}'")
    entity.ctx = ast.Load()
    return ast.copy_location(
        ast.Call(
            func=ast.Name(id=getter_name, ctx=ast.Load()),
            args=[entity],
            keywords=[],
        ),
        node,
    )


def _history_call_target(
    node: ast.expr,
    entity_target: Callable[[ast.AST], tuple[ast.expr, str] | None],
) -> tuple[ast.expr, str] | None:
    """If ``node`` is a bare, no-argument ``<entity>.history()`` call over
    a known history-capable entity, return (entity, binding)."""
    if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "history"
            and not node.args and not node.keywords):
        return entity_target(node.func.value)
    return None


class EnvHistoryMethodLowerer(ast.NodeTransformer):
    """Lower ``env.<entity>.history()`` (bare getter) and
    ``env.<entity>.history().method(...)`` (chained stat call), including
    the indexed ``env.<entity>[i].history()`` form, in a function body."""

    def __init__(
        self,
        *,
        env_name: str,
        history_fields: Mapping[str, str],
        label: str,
    ):
        self.env_name = env_name
        self.history_fields = dict(history_fields)
        self.label = label
        self.changed = False

    def _entity_target(self, node: ast.AST) -> tuple[ast.expr, str] | None:
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == self.env_name
                and node.attr in self.history_fields):
            return (ast.Attribute(
                value=ast.Name(id=self.env_name, ctx=ast.Load()),
                attr=node.attr,
                ctx=ast.Load(),
            ), self.history_fields[node.attr])
        if isinstance(node, ast.Subscript):
            value = node.value
            if (isinstance(value, ast.Attribute)
                    and isinstance(value.value, ast.Name)
                    and value.value.id == self.env_name
                    and value.attr in self.history_fields):
                return (ast.Subscript(
                    value=ast.Attribute(
                        value=ast.Name(id=self.env_name, ctx=ast.Load()),
                        attr=value.attr,
                        ctx=ast.Load(),
                    ),
                    slice=_visit_expr(
                        self.visit, node.slice,
                        what="history field index"),
                    ctx=ast.Load(),
                ), self.history_fields[value.attr])
        return None

    def visit_Call(self, node: ast.Call) -> ast.AST:
        # env.<entity>.history().method(...)
        if isinstance(node.func, ast.Attribute):
            target = _history_call_target(node.func.value, self._entity_target)
            if target is not None:
                entity, binding = target
                self.changed = True
                return lower_timeseries_method_call(
                    node, entity, binding=binding,
                    visit=self.visit, label=self.label)
        # bare env.<entity>.history()
        target = _history_call_target(node, self._entity_target)
        if target is not None:
            entity, binding = target
            self.changed = True
            return lower_history_getter_call(
                node, entity, binding=binding, label=self.label)
        return self.generic_visit(node)


def lower_env_history_method_calls(
    node: ast.FunctionDef,
    *,
    env_name: str,
    history_fields: Mapping[str, str],
    label: str,
) -> tuple[ast.FunctionDef, bool]:
    lowerer = EnvHistoryMethodLowerer(
        env_name=env_name,
        history_fields=history_fields,
        label=label,
    )
    lowered = lowerer.visit(node)
    if not isinstance(lowered, ast.FunctionDef):
        raise TypeError("timeseries method lowering produced a non-function")
    return lowered, lowerer.changed
