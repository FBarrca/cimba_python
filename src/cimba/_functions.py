"""Compilation of read-only ``@sim.function`` helpers.

Model and component methods marked ``@sim.function`` take explicitly typed
scalar arguments and return a scalar. Each one is validated (no mutation of
fields, no scheduling verbs), lowered against the flattened trial record,
and compiled into a Numba helper taking ``(env, receiver_index, *args)``;
calls such as ``env.policy.decide(level)`` lower to calls of that helper.
Polymorphic component collections get one helper per concrete layout plus a
dispatcher selecting on the receiver's variant.
"""

import ast
import copy
from collections.abc import Callable, Iterable, Mapping
from typing import Any, get_type_hints

from numba import njit, types

from ._callbacks import _callback_set
from ._components import _OwnerDecl
from ._lowering import (
    _FieldAccess,
    _FunctionSpec,
    _OwnerAccess,
    _OwnerPathLowerer,
    _closure_namespace,
    _compile_lowered,
    _component_method_source,
    _literal_int,
    _lowering_namespace,
    _strip_function_annotations,
)

_FUNCTION_SCALAR_TYPES = {
    bool: types.boolean,
    int: types.int64,
    float: types.float64,
}

_FORBIDDEN_FUNCTION_SIM_CALLS = frozenset({
    "hold", "interrupt", "stop", "wait_process", "wait_event", "resume",
    "spawn", "despawn", "suspend", "set_priority", "timer_set",
    "timer_add", "timer_cancel", "timers_clear", "clear_events",
})

def _function_scalar_type(annotation: Any, label: str) -> Any:
    numba_type = next(
        (candidate for scalar, candidate in _FUNCTION_SCALAR_TYPES.items()
         if annotation is scalar),
        None,
    )
    if numba_type is None:
        name = getattr(annotation, "__name__", repr(annotation))
        raise TypeError(
            f"{label} has unsupported type annotation {name}; expected "
            "bool, int/sim.Handle, or float")
    return numba_type


def _function_signature(
    node: ast.FunctionDef,
    method: Callable[..., Any],
    label: str,
    receiver: str,
    localns: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[Any, ...], Any]:
    args = node.args
    signature = (
        f"{label} must take {receiver} followed by explicitly annotated "
        "positional scalar arguments, without defaults or variadics, and "
        "declare a scalar return annotation")
    if (args.posonlyargs or args.vararg or args.kwonlyargs or args.kwarg
            or args.defaults or args.kw_defaults or not args.args):
        raise ValueError(signature)
    try:
        hints = get_type_hints(method, localns=localns)
    except Exception as exc:
        raise TypeError(f"{label} annotations could not be resolved") from exc

    parameter_names = tuple(arg.arg for arg in args.args[1:])
    argument_types = []
    for name in parameter_names:
        if name not in hints:
            raise TypeError(f"{label} argument '{name}' needs a type "
                            "annotation")
        argument_types.append(
            _function_scalar_type(hints[name], f"{label} argument '{name}'"))
    if "return" not in hints:
        raise TypeError(f"{label} needs a return type annotation")
    return_type = _function_scalar_type(
        hints["return"], f"{label} return value")

    returns = [item for item in ast.walk(node) if isinstance(item, ast.Return)]
    if not returns or any(item.value is None for item in returns):
        raise ValueError(f"{label} must return a scalar value")
    return parameter_names, tuple(argument_types), return_type


def _rooted_at_name(node: ast.AST, name: str) -> bool:
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return isinstance(node, ast.Name) and node.id == name


class _FunctionValidator(ast.NodeVisitor):
    """Reject side effects that must never enter a synchronous helper."""

    def __init__(self, *, receiver_name: str, method: Callable[..., Any],
                 label: str):
        self.receiver_name = receiver_name
        self.namespace = _closure_namespace(method)
        self.label = label

    def _check_target(self, node: ast.AST) -> None:
        if _rooted_at_name(node, self.receiver_name):
            raise ValueError(
                f"{self.label} cannot mutate component field {ast.unparse(node)}"
            )
        if isinstance(node, (ast.Tuple, ast.List)):
            for item in node.elts:
                self._check_target(item)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._check_target(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._check_target(node.target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._check_target(node.target)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            self._check_target(target)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        forbidden: str | None = None
        if (isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.attr in _FORBIDDEN_FUNCTION_SIM_CALLS):
            module = self.namespace.get(node.func.value.id)
            if getattr(module, "__name__", None) == "cimba.sim":
                forbidden = node.func.attr
        elif isinstance(node.func, ast.Name):
            obj = self.namespace.get(node.func.id)
            obj_name = getattr(obj, "__name__", node.func.id)
            if obj_name in _FORBIDDEN_FUNCTION_SIM_CALLS:
                module_name = getattr(obj, "__module__", "")
                if (module_name.startswith("cimba.")
                        or node.func.id in _FORBIDDEN_FUNCTION_SIM_CALLS):
                    forbidden = obj_name
        if forbidden is not None:
            raise ValueError(
                f"{self.label} cannot call scheduling/process operation sim.{forbidden}()"
            )
        self.generic_visit(node)


class _FunctionBodyLowerer(_OwnerPathLowerer):
    """Lower field paths while preserving reads inside the helper body."""

    def __init__(
        self,
        *,
        builder: "_FunctionBuilder",
        decl: _OwnerDecl,
        method_name: str,
        receiver_name: str,
        instance_indices: tuple[int, ...],
        variant: int | None,
    ):
        # Calls to other functions resolve through the builder (visit_Call),
        # not a precomputed function table.
        super().__init__(
            env_name="__cimba_env",
            label=f"component function '{decl.name}.{method_name}'",
            functions={},
        )
        self.builder = builder
        self.decl = decl
        self.method_name = method_name
        self.receiver_name = receiver_name
        self.instance_indices = instance_indices
        self.variant = variant
        self.reads: list[_FieldAccess] = []
        self.callees: list[str] = []
        self.helper_namespace: dict[str, Any] = {}

    def _root_namespace_ref(self, node: ast.AST) -> _OwnerAccess | None:
        if isinstance(node, ast.Name) and node.id == self.receiver_name:
            if self.variant is not None and len(self.instance_indices) == 1:
                index = (ast.Constant(self.instance_indices[0])
                         if self.decl.count > 1 else None)
            else:
                index = (
                    ast.Name(id="__cimba_receiver_index", ctx=ast.Load())
                    if self.decl.count > 1
                    else None
                )
            possible = (
                self.instance_indices if not isinstance(index, ast.Constant) else None
            )
            return _OwnerAccess(self.decl, index, self.receiver_name, possible)
        return None

    def _validate_scalar_field(self, access: _FieldAccess) -> None:
        if access.field in access.decl.constants:
            if access.field not in access.decl.decls.consts:
                raise ValueError(
                    f"{self.label} cannot read undeclared constant {access.text}; declare it as sim.Const"
                )
            ctype = access.decl.decls.consts[access.field]
            _function_scalar_type(
                ctype, f"{self.label} constant '{access.field}'"
            )
            return
        kind = access.decl.decls.kind_of(access.field)
        if kind not in ("param", "output", "state", "fstate"):
            raise ValueError(
                f"{self.label} cannot read non-scalar component field {access.text} ({kind})"
            )

    def visit_Call(self, node: ast.Call) -> ast.AST:
        lowered_len = self._lower_len_call(node)
        if lowered_len is not None:
            return lowered_len
        if isinstance(node.func, ast.Attribute):
            receiver = self._namespace_ref(node.func.value)
            if receiver is not None:
                candidates = self.builder.specs_for(
                    receiver.decl,
                    receiver.index,
                    node.func.attr,
                    receiver.possible_indices,
                )
                if candidates:
                    return self._dispatch_function_call(
                        node, receiver, candidates
                    )

            access = self._field_ref(node.func.value)
            if access is not None:
                operation = (
                    "entity or runtime operation"
                    if self.decl.owner_root
                    else "component field operation"
                )
                raise ValueError(
                    f"{self.label} cannot call {operation} {access.text}.{node.func.attr}()"
                )
            if receiver is not None:
                raise ValueError(
                    f"{self.label} cannot call unmarked component method {receiver.text}.{node.func.attr}()"
                )
        return self.generic_visit(node)

    def _bind_function(self, spec: _FunctionSpec) -> None:
        super()._bind_function(spec)
        self.helper_namespace[spec.symbol] = spec.helper
        if spec.graph_name not in self.callees:
            self.callees.append(spec.graph_name)
        for symbol, helper in spec.dispatchers.values():
            self.helper_namespace[symbol] = helper

    def _lower_attribute(self, access: _FieldAccess, node: ast.Attribute) -> ast.AST:
        if not isinstance(node.ctx, ast.Load):
            raise ValueError(
                f"{self.label} cannot mutate component field {access.text}"
            )
        self._validate_scalar_field(access)
        self.reads.append(copy.deepcopy(access))
        return super()._lower_attribute(access, node)

    def _direct_path_error(self, kind: str, text: str) -> ValueError:
        suffix = {
            "namespace": "directly; access a scalar field or marked function",
            "collection": "directly; index it",
            "table": "",
        }[kind]
        action = "must index" if kind == "table" else "cannot use"
        return ValueError(f"{self.label} {action} {text} {suffix}".rstrip())

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == self.receiver_name:
            raise ValueError(
                f"{self.label} cannot use self directly")
        return node


class _FunctionBuilder:
    """Validate and compile all synchronous functions for one model tree."""

    def __init__(self):
        self.specs: dict[str, _FunctionSpec] = {}
        self._building: list[str] = []

    def build(
        self,
        decl: _OwnerDecl,
        method_name: str,
        method: Callable[..., Any],
        variant: int | None,
        instance_indices: tuple[int, ...],
    ) -> _FunctionSpec:
        owner = "model" if decl.owner_root else "component"
        base_name = (
            f"model:{method_name}" if decl.owner_root else f"{decl.name}__{method_name}"
        )
        graph_name = base_name if variant is None else f"{base_name}__variant_{variant}"
        existing = self.specs.get(graph_name)
        if existing is not None:
            return existing
        if graph_name in self._building:
            start = self._building.index(graph_name)
            cycle = [*self._building[start:], graph_name]
            raise ValueError(f"recursive {owner} function call: " + " -> ".join(cycle))

        self._building.append(graph_name)
        try:
            node = copy.deepcopy(_component_method_source(method, "function"))
            display_name = decl.cls.__name__ if decl.owner_root else decl.name
            label = f"{owner} function '{display_name}.{method_name}'"
            parameter_names, argument_types, return_type = _function_signature(
                node,
                method,
                label,
                "self",
                {base.__name__: base for base in decl.cls.__mro__},
            )
            receiver_name = node.args.args[0].arg
            _FunctionValidator(
                receiver_name=receiver_name, method=method, label=label
            ).visit(node)

            symbol = f"_CIMBA_FUNCTION_{graph_name.replace(':', '_')}"
            lowerer = _FunctionBodyLowerer(
                builder=self,
                decl=decl,
                method_name=method_name,
                receiver_name=receiver_name,
                instance_indices=instance_indices,
                variant=variant,
            )
            lowered = lowerer.visit(node)
            if not isinstance(lowered, ast.FunctionDef):
                raise TypeError(f"{label} lowering produced a non-function")
            lowered.name = symbol
            _strip_function_annotations(lowered)
            lowered.args.args = [
                ast.arg(arg="__cimba_env"),
                ast.arg(arg="__cimba_receiver_index"),
                *lowered.args.args[1:],
            ]
            namespace = _closure_namespace(method)
            namespace.update(_lowering_namespace((decl,)))
            namespace.update(lowerer.helper_namespace)
            plain = _compile_lowered(
                lowered,
                filename=f"<cimba {owner} function '{decl.name}.{method_name}'>",
                fn_name=symbol,
                qualname=symbol,
                namespace=namespace,
                like=method,
            )
            # The complete record layout exists only after every callback has
            # been registered. Model.compile() supplies its explicit signature.
            helper = njit(plain)

            spec = _FunctionSpec(
                decl=decl,
                name=method_name,
                method=method,
                graph_name=graph_name,
                symbol=symbol,
                parameter_names=parameter_names,
                argument_types=argument_types,
                return_type=return_type,
                reads=tuple(lowerer.reads),
                helper=helper,
                callees=tuple(lowerer.callees),
                variant=variant,
                instance_indices=instance_indices,
            )
            self.specs[graph_name] = spec
            return spec
        finally:
            self._building.pop()

    def build_all(self, roots: Iterable[_OwnerDecl]) -> dict[str, _FunctionSpec]:
        for root in roots:
            for decl in root.walk():
                groups = decl.specialization_groups()
                for ordinal, instance_indices in enumerate(groups):
                    variant = None if len(groups) == 1 else ordinal
                    cls = decl.class_at(instance_indices[0])
                    for callback in _callback_set(cls).functions:
                        self.build(
                            decl, callback.name, callback.fn, variant, instance_indices
                        )
        return self.specs

    def specs_for(
        self,
        decl: _OwnerDecl,
        index: ast.expr | None,
        method_name: str,
        possible_indices: tuple[int, ...] | None,
    ) -> tuple[_FunctionSpec, ...]:
        """Resolve/build the concrete helper candidates for an access."""
        if (instance_index := _literal_int(index)) is not None:
            groups = decl.specialization_groups()
            variant = (
                None
                if len(groups) == 1
                else decl.specialization_slots()[instance_index]
            )
            group = groups[0] if variant is None else groups[variant]
            cls = decl.class_at(instance_index)
            method = next(
                (
                    callback.fn
                    for callback in _callback_set(cls).functions
                    if callback.name == method_name
                ),
                None,
            )
            if method is None:
                return ()
            return (self.build(
                decl, method_name, method, variant, group),)
        if not decl.polymorphic:
            method = next(
                (
                    callback.fn
                    for callback in _callback_set(decl.class_at()).functions
                    if callback.name == method_name
                ),
                None,
            )
            if method is None:
                return ()
            group = decl.specialization_groups()[0]
            return (self.build(decl, method_name, method, None, group),)
        candidates: list[_FunctionSpec] = []
        possible = set(possible_indices or range(decl.count))
        groups = decl.specialization_groups()
        for variant, group in enumerate(groups):
            if not possible.intersection(group):
                continue
            cls = decl.class_at(group[0])
            method = next(
                (
                    callback.fn
                    for callback in _callback_set(cls).functions
                    if callback.name == method_name
                ),
                None,
            )
            if method is None:
                raise ValueError(
                    f"dynamic component function call to '{decl.name}.{method_name}' requires every concrete component type to declare that @sim.function"
                )
            candidates.append(self.build(decl, method_name, method, variant, group))
        return tuple(candidates)


def _build_functions(roots: Iterable[_OwnerDecl]) -> dict[str, _FunctionSpec]:
    return _FunctionBuilder().build_all(roots)
