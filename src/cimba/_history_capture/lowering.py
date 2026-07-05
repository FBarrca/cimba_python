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
    """Rewrite ``env.<field>.history().capture()`` in model collectors."""

    def __init__(
        self,
        *,
        env_name: str,
        history_fields: Mapping[str, str],
        register: Callable[[str, str], int],
        label: str,
    ):
        self.env_name = env_name
        self.history_fields = dict(history_fields)
        self.register = register
        self.label = label
        self.changed = False

    def _history_target(self, node: ast.AST) -> tuple[str, ast.expr, str]:
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "history"
                and not node.args and not node.keywords):
            raise ValueError(
                f"{self.label} uses capture() outside an entity history")
        target = node.func.value
        if isinstance(target, ast.Subscript):
            raise ValueError(
                f"{self.label} uses history capture on an indexed entity; "
                "only scalar entity fields are supported")
        if not (isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == self.env_name):
            raise ValueError(
                f"{self.label} uses history capture on an unsupported target")
        field = target.attr
        binding = self.history_fields.get(field)
        if binding is None:
            raise ValueError(
                f"{self.label} captures unknown history field '{field}'")
        if binding == "priorityqueue":
            raise ValueError(
                f"{self.label} uses history capture on priority queues; "
                "indexed histories are not supported")
        return field, ast.copy_location(
            ast.Attribute(
                value=ast.Name(id=self.env_name, ctx=ast.Load()),
                attr=field,
                ctx=ast.Load(),
            ),
            target,
        ), binding

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if (isinstance(node.func, ast.Attribute)
                and node.func.attr == "capture"):
            value = node.func.value
            if not (isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and value.func.attr == "history"):
                return self.generic_visit(node)
            field, entity, binding = self._history_target(node.func.value)
            if node.args or node.keywords:
                raise ValueError(
                    f"{self.label} history capture() takes no arguments")
            slot = self.register(field, binding)
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
                        ast.Constant(slot),
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
    register: Callable[[str, str], int],
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
