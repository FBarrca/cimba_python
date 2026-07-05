"""AST lowering support for compiled ``sim.Dataset`` method calls."""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import helpers as _dataset_helpers


@dataclass(frozen=True)
class _DatasetMethodSpec:
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
                f"{label} passes too many arguments to dataset {method}()")
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
                    f"{label} cannot use **kwargs with dataset {method}()")
            index = by_name.get(kw.arg)
            if index is None:
                raise ValueError(
                    f"{label} passes unknown dataset {method}() argument "
                    f"'{kw.arg}'")
            if kw.arg in supplied or index in keyed:
                raise ValueError(
                    f"{label} passes dataset {method}() argument "
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
                        f"{label} is missing required dataset {method}() "
                        f"argument '{param}'")
                call_args.append(ast.Constant(self.defaults[param]))
        return call_args


def _spec(
    method: str,
    params: tuple[str, ...] = (),
    *,
    helper_attr: str | None = None,
    defaults: Mapping[str, object] | None = None,
) -> _DatasetMethodSpec:
    attr = helper_attr or method
    return _DatasetMethodSpec(
        helper_name=f"_cimba_dataset_{attr}",
        helper_attr=attr,
        params=params,
        defaults=defaults or {},
    )


_METHOD_SPECS = {
    "add": _spec("add", ("value",)),
    "count": _spec("count"),
    "mean": _spec("mean"),
    "min": _spec("min"),
    "max": _spec("max"),
    "std": _spec("std"),
    "stddev": _spec("stddev", helper_attr="std"),
    "median": _spec("median"),
    "quantile": _spec("quantile", ("q",)),
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

DATASET_METHOD_NAMES = frozenset(_METHOD_SPECS)


def dataset_lowering_namespace() -> dict[str, Any]:
    return {
        spec.helper_name: getattr(_dataset_helpers, spec.helper_attr)
        for spec in _METHOD_SPECS.values()
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


def lower_dataset_method_call(
    node: ast.Call,
    target: ast.expr,
    *,
    visit: Callable[[ast.AST], ast.AST],
    label: str,
) -> ast.Call:
    if not isinstance(node.func, ast.Attribute):
        raise TypeError("dataset method call must use attribute syntax")
    method = node.func.attr
    spec = _METHOD_SPECS.get(method)
    if spec is None:
        raise ValueError(f"{label} uses unsupported dataset method {method}()")

    args = [
        _visit_expr(visit, arg, what="dataset method argument")
        for arg in node.args
    ]
    keywords = [
        ast.keyword(
            arg=kw.arg,
            value=_visit_expr(
                visit, kw.value, what="dataset method keyword"),
        )
        for kw in node.keywords
    ]
    target.ctx = ast.Load()
    return ast.copy_location(
        ast.Call(
            func=ast.Name(id=spec.helper_name, ctx=ast.Load()),
            args=[target, *spec.normalize_args(
                method, args, keywords, label=label)],
            keywords=[],
        ),
        node,
    )


class EnvDatasetMethodLowerer(ast.NodeTransformer):
    """Lower ``env.<dataset>.method(...)`` calls in a function body."""

    def __init__(
        self,
        *,
        env_name: str,
        dataset_fields: Iterable[str],
        label: str,
    ):
        self.env_name = env_name
        self.dataset_fields = set(dataset_fields)
        self.label = label
        self.changed = False

    def _target(self, node: ast.AST) -> ast.expr | None:
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == self.env_name
                and node.attr in self.dataset_fields):
            return ast.Attribute(
                value=ast.Name(id=self.env_name, ctx=ast.Load()),
                attr=node.attr,
                ctx=ast.Load(),
            )
        if isinstance(node, ast.Subscript):
            value = node.value
            if (isinstance(value, ast.Attribute)
                    and isinstance(value.value, ast.Name)
                    and value.value.id == self.env_name
                    and value.attr in self.dataset_fields):
                return ast.Subscript(
                    value=ast.Attribute(
                        value=ast.Name(id=self.env_name, ctx=ast.Load()),
                        attr=value.attr,
                        ctx=ast.Load(),
                    ),
                    slice=_visit_expr(
                        self.visit, node.slice,
                        what="dataset field index"),
                    ctx=ast.Load(),
                )
        return None

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if isinstance(node.func, ast.Attribute):
            target = self._target(node.func.value)
            if target is not None:
                self.changed = True
                return lower_dataset_method_call(
                    node, target, visit=self.visit, label=self.label)
        return self.generic_visit(node)


def lower_env_dataset_method_calls(
    node: ast.FunctionDef,
    *,
    env_name: str,
    dataset_fields: Iterable[str],
    label: str,
) -> tuple[ast.FunctionDef, bool]:
    lowerer = EnvDatasetMethodLowerer(
        env_name=env_name,
        dataset_fields=dataset_fields,
        label=label,
    )
    lowered = lowerer.visit(node)
    if not isinstance(lowered, ast.FunctionDef):
        raise TypeError("dataset method lowering produced a non-function")
    return lowered, lowerer.changed
