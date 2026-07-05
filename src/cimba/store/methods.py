"""AST lowering support for the ``env.<entity>.method(...)`` sugar.

Every Queue/Resource/Pool/Store/PQueues-element/Condition field supports a
fixed set of methods (``env.queue.put(1)``, ``env.server.acquire()``, and
so on). This module defines those per-kind method tables and the two
lowerers built on top of them:

* ``lower_entity_method_call`` -- rewrites one already-resolved
  ``<entity>.method(...)`` call (used by ``_components.py`` for
  component-owned scalar fields, mirroring ``lower_timeseries_method_call``);
* ``EnvEntityMethodLowerer`` / ``lower_env_entity_method_calls`` -- walks a
  function body rewriting every ``env.<field>.method(...)`` (and the
  indexed ``env.<field>[i].method(...)`` form, for PQueues elements and
  component-collection fields) call it finds.

``sim.Condition.wait_for`` is the one method needing the caller's own
``env`` (the underlying ``cmb_condition_wait`` takes it as a third
argument): the lowerers thread the function's own ``env`` name in and
append it themselves, so model authors just write ``env.cond.wait_for(pred)``.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import helpers as _entity_helpers


@dataclass(frozen=True)
class _EntityMethodSpec:
    helper_name: str
    helper_attr: str
    params: tuple[str, ...] = ()
    defaults: Mapping[str, object] = field(default_factory=dict)
    #: append the caller's own ``env`` expression as a final helper argument
    #: (sim.Condition.wait_for's implicit third argument)
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
        if len(args) > len(self.params):
            raise ValueError(
                f"{label} passes too many arguments to {kind} {method}()")
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
                    f"{label} cannot use **kwargs with {kind} {method}()")
            index = by_name.get(kw.arg)
            if index is None:
                raise ValueError(
                    f"{label} passes unknown {kind} {method}() argument "
                    f"'{kw.arg}'")
            if kw.arg in supplied or index in keyed:
                raise ValueError(
                    f"{label} passes {kind} {method}() argument '{kw.arg}' "
                    "more than once")
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
                        f"{label} is missing required {kind} {method}() "
                        f"argument '{param}'")
                call_args.append(ast.Constant(self.defaults[param]))
        return call_args


def _spec(
    kind: str,
    method: str,
    params: tuple[str, ...] = (),
    *,
    helper_attr: str | None = None,
    defaults: Mapping[str, object] | None = None,
    needs_env: bool = False,
) -> _EntityMethodSpec:
    attr = helper_attr or f"{kind}_{method}"
    return _EntityMethodSpec(
        helper_name=f"_cimba_entity_{attr}",
        helper_attr=attr,
        params=params,
        defaults=defaults or {},
        needs_env=needs_env,
    )


_REPORT_FILE = ("path", "append")
_REPORT_FILE_DEFAULTS = {"append": 1}


def _reporting_methods(kind: str) -> dict[str, _EntityMethodSpec]:
    return {
        "report": _spec(kind, "report"),
        "report_file": _spec(kind, "report_file", _REPORT_FILE,
                             defaults=_REPORT_FILE_DEFAULTS),
    }


#: declared field kind -> {method name: spec}
_KIND_SPECS: dict[str, dict[str, _EntityMethodSpec]] = {
    "queue": {
        "put": _spec("queue", "put", ("amount",)),
        "get": _spec("queue", "get", ("amount",)),
        "level": _spec("queue", "level"),
        "space": _spec("queue", "space"),
        "mean_level": _spec("queue", "mean_level"),
        **_reporting_methods("queue"),
    },
    "resource": {
        "acquire": _spec("resource", "acquire"),
        "release": _spec("resource", "release"),
        "preempt": _spec("resource", "preempt"),
        "available": _spec("resource", "available"),
        "in_use": _spec("resource", "in_use"),
        "held": _spec("resource", "held", ("process",)),
        "mean_in_use": _spec("resource", "mean_in_use"),
        **_reporting_methods("resource"),
    },
    "pool": {
        "acquire": _spec("pool", "acquire", ("amount",)),
        "release": _spec("pool", "release", ("amount",)),
        "preempt": _spec("pool", "preempt", ("amount",)),
        "available": _spec("pool", "available"),
        "held": _spec("pool", "held", ("process",)),
        "in_use": _spec("pool", "in_use"),
        "mean_in_use": _spec("pool", "mean_in_use"),
        **_reporting_methods("pool"),
    },
    "store": {
        "put": _spec("store", "put", ("obj",)),
        "get": _spec("store", "get"),
        "take": _spec("store", "take"),
        "length": _spec("store", "length"),
        "space": _spec("store", "space"),
        "position": _spec("store", "position", ("obj",)),
        "mean_length": _spec("store", "mean_length"),
        **_reporting_methods("store"),
    },
    "pqueues": {
        "put": _spec("pq", "put", ("obj", "priority")),
        "get": _spec("pq", "get"),
        "take": _spec("pq", "take"),
        "length": _spec("pq", "length"),
        "space": _spec("pq", "space"),
        "position": _spec("pq", "position", ("entry",)),
        "reprioritize": _spec("pq", "reprioritize", ("entry", "priority")),
        "cancel": _spec("pq", "cancel", ("entry",)),
        "mean_length": _spec("pq", "mean_length"),
        **_reporting_methods("pq"),
    },
    "condition": {
        "signal": _spec("condition", "signal"),
        "wait_for": _spec("condition", "wait", ("predicate",),
                          needs_env=True),
    },
}

#: every method name used by any kind, for the fast ``co_names`` pre-check
ENTITY_METHOD_NAMES = frozenset(
    method for methods in _KIND_SPECS.values() for method in methods)


def entity_lowering_namespace() -> dict[str, Any]:
    return {
        spec.helper_name: getattr(_entity_helpers, spec.helper_attr)
        for methods in _KIND_SPECS.values()
        for spec in methods.values()
    }


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


def lower_entity_method_call(
    node: ast.Call,
    target: ast.expr,
    *,
    kind: str,
    visit: Callable[[ast.AST], ast.AST],
    label: str,
    env_expr: ast.expr | None = None,
) -> ast.Call:
    """Lower ``<entity>.method(...)`` to a native helper call. ``target``
    is the already-lowered handle expression."""
    if not isinstance(node.func, ast.Attribute):
        raise TypeError("entity method call must use attribute syntax")
    method = node.func.attr
    spec = _KIND_SPECS.get(kind, {}).get(method)
    if spec is None:
        raise ValueError(f"{label} uses unsupported {kind} method {method}()")
    if spec.needs_env and env_expr is None:
        raise ValueError(f"{label} uses {kind} method {method}() outside a "
                         "context with its own env")

    args = [
        _visit_expr(visit, arg, what="entity method argument")
        for arg in node.args
    ]
    keywords = [
        ast.keyword(
            arg=kw.arg,
            value=_visit_expr(visit, kw.value, what="entity method keyword"),
        )
        for kw in node.keywords
    ]
    target.ctx = ast.Load()
    call_args = [target, *spec.normalize_args(
        kind, method, args, keywords, label=label)]
    if spec.needs_env:
        call_args.append(env_expr)
    return ast.copy_location(
        ast.Call(
            func=ast.Name(id=spec.helper_name, ctx=ast.Load()),
            args=call_args,
            keywords=[],
        ),
        node,
    )


class EnvEntityMethodLowerer(ast.NodeTransformer):
    """Lower ``env.<entity>.method(...)`` calls (including the indexed
    ``env.<entity>[i].method(...)`` form) in a function body.

    Also tracks plain ``local = env.<entity>`` (or ``env.<entity>[i]``)
    aliases -- a common way to shorten a repeated entity access, and one
    the old ``sim.put(local, ...)``-style free functions supported for
    free since they just took the handle as a plain int argument -- so
    ``local.method(...)`` keeps working after the alias."""

    def __init__(
        self,
        *,
        env_name: str,
        entity_fields: Mapping[str, str],
        label: str,
    ):
        self.env_name = env_name
        self.entity_fields = dict(entity_fields)
        self.label = label
        self.changed = False
        self.aliases: dict[str, str] = {}

    def _target(self, node: ast.AST) -> tuple[ast.expr, str] | None:
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == self.env_name
                and node.attr in self.entity_fields):
            return (ast.Attribute(
                value=ast.Name(id=self.env_name, ctx=ast.Load()),
                attr=node.attr,
                ctx=ast.Load(),
            ), self.entity_fields[node.attr])
        if isinstance(node, ast.Subscript):
            value = node.value
            if (isinstance(value, ast.Attribute)
                    and isinstance(value.value, ast.Name)
                    and value.value.id == self.env_name
                    and value.attr in self.entity_fields):
                return (ast.Subscript(
                    value=ast.Attribute(
                        value=ast.Name(id=self.env_name, ctx=ast.Load()),
                        attr=value.attr,
                        ctx=ast.Load(),
                    ),
                    slice=_visit_expr(
                        self.visit, node.slice, what="entity field index"),
                    ctx=ast.Load(),
                ), self.entity_fields[value.attr])
        if isinstance(node, ast.Name) and node.id in self.aliases:
            return (ast.Name(id=node.id, ctx=ast.Load()),
                    self.aliases[node.id])
        return None

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
        if isinstance(node.func, ast.Attribute):
            target = self._target(node.func.value)
            if target is not None:
                entity, kind = target
                self.changed = True
                return lower_entity_method_call(
                    node, entity, kind=kind, visit=self.visit,
                    label=self.label,
                    env_expr=ast.Name(id=self.env_name, ctx=ast.Load()))
        return self.generic_visit(node)


def lower_env_entity_method_calls(
    node: ast.FunctionDef,
    *,
    env_name: str,
    entity_fields: Mapping[str, str],
    label: str,
) -> tuple[ast.FunctionDef, bool]:
    lowerer = EnvEntityMethodLowerer(
        env_name=env_name,
        entity_fields=entity_fields,
        label=label,
    )
    lowered = lowerer.visit(node)
    if not isinstance(lowered, ast.FunctionDef):
        raise TypeError("entity method lowering produced a non-function")
    return lowered, lowerer.changed
