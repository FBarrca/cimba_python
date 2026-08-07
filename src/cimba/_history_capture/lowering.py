"""AST lowering for collect-declared history and dataset capture."""

from __future__ import annotations

import ast
import copy
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from .. import _bindings as _b
from .._components import (
    _closure_namespace,
    _compile_lowered,
    _function_def_from_source,
)
from .._timeseries.methods import timeseries_lowering_namespace
from .runtime import HISTORY_CAPTURE_STORE_FIELD, HISTORY_CAPTURE_TRIAL_FIELD

_F = TypeVar("_F", bound=Callable[..., Any])


class _HistoryCaptureLowerer(ast.NodeTransformer):
    """Rewrite scalar and bounded indexed history captures."""

    def __init__(
        self,
        *,
        env_name: str,
        history_fields: Mapping[str, str],
        indexed_history_fields: Mapping[str, int],
        register: Callable[[str, str, int | None], int],
        label: str,
    ):
        self.env_name = env_name
        self.history_fields = dict(history_fields)
        self.indexed_history_fields = dict(indexed_history_fields)
        self.register = register
        self.label = label
        self.changed = False
        self._bounded_indices: dict[str, int] = {}

    def _history_target(
        self,
        node: ast.AST,
    ) -> tuple[str, ast.expr, str, ast.expr | None, int | None]:
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "history"
                and not node.args and not node.keywords):
            raise ValueError(
                f"{self.label} uses capture() outside an entity history")
        target = node.func.value
        index: ast.expr | None = None
        base = target
        if isinstance(target, ast.Subscript):
            base = target.value
            index = target.slice
        if not (isinstance(base, ast.Attribute)
                and isinstance(base.value, ast.Name)
                and base.value.id == self.env_name):
            raise ValueError(
                f"{self.label} uses history capture on an unsupported target")
        field = base.attr
        binding = self.history_fields.get(field)
        if binding is None:
            raise ValueError(
                f"{self.label} captures unknown history field '{field}'")
        if binding == "priorityqueue":
            raise ValueError(
                f"{self.label} uses history capture on an indexed entity "
                "(priority queues); indexed histories are not supported")
        indexed_count = self.indexed_history_fields.get(field)
        if index is not None:
            if indexed_count is None:
                raise ValueError(
                    f"{self.label} uses history capture on an indexed "
                    "entity; only component collection fields are "
                    "supported")
            if (isinstance(index, ast.Constant)
                    and type(index.value) is int):
                if not 0 <= index.value < indexed_count:
                    raise ValueError(
                        f"{self.label} history capture index "
                        f"{index.value} is out of range for '{field}' "
                        f"(length {indexed_count})")
            elif (isinstance(index, ast.Name)
                  and index.id in self._bounded_indices
                  and self._bounded_indices[index.id] <= indexed_count):
                pass
            else:
                raise ValueError(
                    f"{self.label} uses an unbounded index for history "
                    f"capture on '{field}'")
        return field, copy.deepcopy(target), binding, index, indexed_count

    @staticmethod
    def _range_bound(node: ast.AST) -> int | None:
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "range"
                and len(node.args) == 1
                and not node.keywords):
            return None
        stop = node.args[0]
        if not (isinstance(stop, ast.Constant)
                and type(stop.value) is int
                and stop.value >= 0):
            return None
        return stop.value

    def visit_For(self, node: ast.For) -> ast.AST:
        node.target = self.visit(node.target)
        node.iter = self.visit(node.iter)
        name: str | None = None
        previous: int | None = None
        if isinstance(node.target, ast.Name):
            name = node.target.id
            previous = self._bounded_indices.get(name)
            bound = self._range_bound(node.iter)
            if bound is not None:
                self._bounded_indices[name] = bound
        node.body = [self.visit(statement) for statement in node.body]
        if name is not None:
            if previous is None:
                self._bounded_indices.pop(name, None)
            else:
                self._bounded_indices[name] = previous
        node.orelse = [self.visit(statement) for statement in node.orelse]
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if (isinstance(node.func, ast.Attribute)
                and node.func.attr == "capture"):
            value = node.func.value
            if not (isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and value.func.attr == "history"):
                return self.generic_visit(node)
            field, entity, binding, index, indexed_count = \
                self._history_target(node.func.value)
            if node.args or node.keywords:
                raise ValueError(
                    f"{self.label} history capture() takes no arguments")
            slot = self.register(field, binding, indexed_count)
            self.changed = True
            getter_name = {
                "buffer": "_cimba_history_buffer",
                "resource": "_cimba_history_resource",
                "resourcepool": "_cimba_history_resourcepool",
                "objectqueue": "_cimba_history_objectqueue",
            }.get(binding)
            if getter_name is None:
                raise ValueError(
                    f"{self.label} has no capture support for '{field}'")
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id="_cimba_capture_history",
                                  ctx=ast.Load()),
                    args=[
                        ast.Subscript(
                            value=ast.Name(id=self.env_name, ctx=ast.Load()),
                            slice=ast.Constant(HISTORY_CAPTURE_STORE_FIELD),
                            ctx=ast.Load(),
                        ),
                        ast.Subscript(
                            value=ast.Name(id=self.env_name, ctx=ast.Load()),
                            slice=ast.Constant(HISTORY_CAPTURE_TRIAL_FIELD),
                            ctx=ast.Load(),
                        ),
                        (ast.Constant(slot) if index is None else ast.BinOp(
                            left=ast.Constant(slot),
                            op=ast.Add(),
                            right=copy.deepcopy(index),
                        )),
                        ast.Call(
                            func=ast.Name(id=getter_name, ctx=ast.Load()),
                            args=[entity],
                            keywords=[],
                        ),
                    ],
                    keywords=[],
                ),
                node,
            )
        return self.generic_visit(node)


class _DatasetCaptureLowerer(ast.NodeTransformer):
    """Rewrite ``env.<dataset>.capture()`` in model collectors."""

    def __init__(
        self,
        *,
        env_name: str,
        dataset_fields: set[str],
        register: Callable[[str, str], int],
        label: str,
    ):
        self.env_name = env_name
        self.dataset_fields = dataset_fields
        self.register = register
        self.label = label
        self.changed = False

    def _dataset_target(self, node: ast.AST) -> tuple[str, ast.expr]:
        if isinstance(node, ast.Subscript):
            raise ValueError(
                f"{self.label} uses dataset capture on an indexed dataset; "
                "only scalar dataset fields are supported")
        if not (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == self.env_name):
            raise ValueError(
                f"{self.label} uses dataset capture on an unsupported target")
        field = node.attr
        if field not in self.dataset_fields:
            raise ValueError(
                f"{self.label} captures unknown dataset field '{field}'")
        return field, ast.copy_location(
            ast.Attribute(
                value=ast.Name(id=self.env_name, ctx=ast.Load()),
                attr=field,
                ctx=ast.Load(),
            ),
            node,
        )

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if (isinstance(node.func, ast.Attribute)
                and node.func.attr == "capture"):
            value = node.func.value
            if (isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and value.func.attr == "history"):
                return self.generic_visit(node)
            field, dataset = self._dataset_target(value)
            if node.args or node.keywords:
                raise ValueError(
                    f"{self.label} dataset capture() takes no arguments")
            slot = self.register(field, "dataset")
            self.changed = True
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id="_cimba_capture_dataset",
                                  ctx=ast.Load()),
                    args=[
                        ast.Subscript(
                            value=ast.Name(id=self.env_name, ctx=ast.Load()),
                            slice=ast.Constant(HISTORY_CAPTURE_STORE_FIELD),
                            ctx=ast.Load(),
                        ),
                        ast.Subscript(
                            value=ast.Name(id=self.env_name, ctx=ast.Load()),
                            slice=ast.Constant(HISTORY_CAPTURE_TRIAL_FIELD),
                            ctx=ast.Load(),
                        ),
                        ast.Constant(slot),
                        dataset,
                    ],
                    keywords=[],
                ),
                node,
            )
        return self.generic_visit(node)


def lower_history_capture_methods(
    fn: _F,
    *,
    model_name: str,
    history_fields: Mapping[str, str],
    indexed_history_fields: Mapping[str, int],
    register: Callable[[str, str, int | None], int],
) -> _F:
    names = set(fn.__code__.co_names)
    if "capture" not in names or "history" not in names:
        return fn
    try:
        node = copy.deepcopy(_function_def_from_source(fn))
    except (OSError, TypeError) as exc:
        raise ValueError(
            f"model '{model_name}' callback '{fn.__qualname__}' needs "
            "inspectable source to use history capture"
        ) from exc
    if not node.args.args:
        return fn

    env_name = node.args.args[0].arg
    label = f"model '{model_name}' callback '{fn.__name__}'"
    lowerer = _HistoryCaptureLowerer(
        env_name=env_name,
        history_fields=history_fields,
        indexed_history_fields=indexed_history_fields,
        register=register,
        label=label,
    )
    lowered = lowerer.visit(node)
    if not isinstance(lowered, ast.FunctionDef):
        raise TypeError("history capture lowering produced a non-function")
    if not lowerer.changed:
        return fn

    lowered.decorator_list = []
    lowered.returns = None
    lowered.type_comment = None
    for arg in lowered.args.args:
        arg.annotation = None
        arg.type_comment = None

    namespace = _closure_namespace(fn)
    namespace.update(timeseries_lowering_namespace())
    namespace["_cimba_capture_history"] = _b.history_capture_store_capture
    lowered_fn = _compile_lowered(
        lowered,
        filename=f"<cimba model callback '{model_name}.{fn.__name__}'>",
        fn_name=fn.__name__,
        qualname=fn.__qualname__,
        namespace=namespace,
        like=fn,
    )
    return lowered_fn


def lower_dataset_capture_methods(
    fn: _F,
    *,
    model_name: str,
    dataset_fields: set[str],
    register: Callable[[str, str], int],
) -> _F:
    names = set(fn.__code__.co_names)
    if "capture" not in names or not names.intersection(dataset_fields):
        return fn
    try:
        node = copy.deepcopy(_function_def_from_source(fn))
    except (OSError, TypeError) as exc:
        raise ValueError(
            f"model '{model_name}' callback '{fn.__qualname__}' needs "
            "inspectable source to use dataset capture"
        ) from exc
    if not node.args.args:
        return fn

    env_name = node.args.args[0].arg
    label = f"model '{model_name}' callback '{fn.__name__}'"
    lowerer = _DatasetCaptureLowerer(
        env_name=env_name,
        dataset_fields=dataset_fields,
        register=register,
        label=label,
    )
    lowered = lowerer.visit(node)
    if not isinstance(lowered, ast.FunctionDef):
        raise TypeError("dataset capture lowering produced a non-function")
    if not lowerer.changed:
        return fn

    lowered.decorator_list = []
    lowered.returns = None
    lowered.type_comment = None
    for arg in lowered.args.args:
        arg.annotation = None
        arg.type_comment = None

    namespace = _closure_namespace(fn)
    namespace["_cimba_capture_dataset"] = _b.dataset_capture_store_capture
    lowered_fn = _compile_lowered(
        lowered,
        filename=f"<cimba model callback '{model_name}.{fn.__name__}'>",
        fn_name=fn.__name__,
        qualname=fn.__qualname__,
        namespace=namespace,
        like=fn,
    )
    return lowered_fn


__all__ = ["lower_dataset_capture_methods", "lower_history_capture_methods"]
