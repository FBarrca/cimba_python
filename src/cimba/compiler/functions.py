"""``@function`` methods: per-class implementations called through the dispatch table.

Every model record starts with a pointer to its class's dispatch table. A
``@function`` owns one slot in that table (see ``ClassSchema.slots``); each
class fills the slot with its own implementation, compiled for its own record
type, behind a C calling convention fixed by the annotated signature:

    ret f(intptr_t self, args...)     float -> double, int -> int64,
                                      bool -> uint8, model -> intptr_t

A call ``view.name(args)`` loads the slot from the *record's* table, not the
view's class, so a subclass override runs even when the caller only knows the
base class. The caller needs only the signature, never the implementation, so
compiled code still depends on classes alone.
"""

from __future__ import annotations

import inspect

from llvmlite import ir
from numba import from_dtype, types

from cimba.layout import RecordLayout
from cimba.modeling import Model
from cimba.schema import ClassSchema, FunctionSignature
from . import views
from .handles import _record_model_class
from .numba_compat import AbstractTemplate, cpu_target, lower_builtin, signature


def _record_type(model_class: type[Model]) -> types.Record:
    record = from_dtype(RecordLayout.of(ClassSchema.of(model_class)).dtype)
    assert isinstance(record, types.Record)
    return record


def _numba_type(kind):
    if kind is None:
        return types.none
    if kind is float:
        return types.float64
    if kind is bool:
        return types.boolean
    if kind is int:
        return types.int64
    return _record_type(kind)


def _ir_type(kind) -> ir.Type:
    if kind is None:
        return ir.VoidType()
    if kind is float:
        return ir.DoubleType()
    if kind is bool:
        return ir.IntType(8)
    return ir.IntType(64)


# ------------------------------------------------------------------ callee side
def function_signature(schema: ClassSchema, name: str, record_type):
    """The ``njit`` signature of this class's implementation of ``name``.

    Its native entry point (see ``compiler.entries``) adapts it to the slot ABI.
    """
    signature = schema.signature(name)
    return _numba_type(signature.returns)(
        record_type, *(_numba_type(kind) for kind in signature.types))


# ------------------------------------------------------------------ caller side
def _emit_call(context, builder, record_value, values, actual_types, slot: int,
               signature: FunctionSignature, return_type):
    """Load slot ``slot`` from the record's own dispatch table and call it."""
    kinds = signature.types
    returns = signature.returns
    i64 = ir.IntType(64)
    address = builder.ptrtoint(record_value, i64)
    descriptor = builder.load(builder.inttoptr(address, i64.as_pointer()))
    table = builder.inttoptr(descriptor, i64.as_pointer())
    pointer = builder.load(builder.gep(table, [ir.Constant(ir.IntType(32), slot)]))
    converted = [address]
    first_default = len(kinds) - len(signature.defaults)
    for index, kind in enumerate(kinds):
        if index >= len(values):
            default = signature.defaults[index - first_default]
            converted.append(ir.Constant(_ir_type(kind),
                                         float(default) if kind is float else int(default)))
            continue
        value, actual = values[index], actual_types[index]
        if kind is float:
            converted.append(context.cast(builder, value, actual, types.float64))
        elif kind is int:
            converted.append(context.cast(builder, value, actual, types.int64))
        elif kind is bool:
            flag = context.cast(builder, value, actual, types.boolean)
            converted.append(builder.zext(flag, ir.IntType(8)))
        else:
            converted.append(builder.ptrtoint(value, i64))
    function_type = ir.FunctionType(_ir_type(returns), [i64] + [_ir_type(k) for k in kinds])
    result = builder.call(builder.inttoptr(pointer, function_type.as_pointer()), converted)
    if returns is None:
        return context.get_dummy_value()
    if returns is bool:
        return builder.icmp_unsigned("!=", result, ir.Constant(ir.IntType(8), 0))
    if returns in (float, int):
        return result
    return builder.inttoptr(result, context.get_value_type(return_type))


def _check_arguments(owner: type[Model], name: str, signature: FunctionSignature, arguments):
    where = f"{owner.__name__}.{name}"
    required = len(signature.parameters) - len(signature.defaults)
    if not required <= len(arguments) <= len(signature.parameters):
        raise TypeError(f"{where}() takes {required} to {len(signature.parameters)} "
                        f"arguments, got {len(arguments)}")
    for (parameter, kind), actual in zip(signature.parameters, arguments):
        if kind in (float, int, bool):
            if not isinstance(actual, (types.Number, types.Boolean)):
                raise TypeError(f"{where}: {parameter} expects {kind.__name__}, got {actual}")
        else:
            model = _record_model_class(actual)
            if model is None or not issubclass(model, kind):
                raise TypeError(f"{where}: {parameter} expects a {kind.__name__} model, got {actual}")


# Calls are typed and lowered under a private key per function name, never
# under (types.Record, name): Numba's method-overload machinery would register
# the lowering for *every* record and shadow entity methods of the same name.
_lowered: set[str] = set()


def _call_key(name: str):
    return ("cimba.function", name)


def _register_lowering(name: str) -> None:
    if name in _lowered:
        return
    _lowered.add(name)

    @lower_builtin(_call_key(name), types.Record, types.VarArg(types.Any))
    def lower_call(context, builder, sig, arguments):
        record_type = sig.args[0]
        owner = _record_model_class(record_type)
        assert owner is not None
        schema = ClassSchema.of(owner)
        return _emit_call(context, builder, arguments[0], list(arguments[1:]),
                          list(sig.args[1:]), schema.slots.index(name),
                          schema.signature(name), sig.return_type)

    getattr(cpu_target.target_context, "refresh")()


def resolve_function(context, record, name: str):
    """Type ``record.name`` as a bound call when ``name`` is a ``@function``."""
    owner = _record_model_class(record)
    if owner is None:
        return None
    schema = ClassSchema.of(owner)
    if name not in schema.functions:
        method = getattr(owner, name, None)
        if inspect.isfunction(method) and not name.startswith("__"):
            role = getattr(method, "cimba_role", None)
            if role is None:
                raise TypeError(f"{owner.__name__}.{name} is not a @cb.function; decorate "
                                "it with @cb.function to call it from compiled model code")
            if role in {"process", "on_start", "on_end"}:
                raise TypeError(f"{owner.__name__}.{name} is a @cb.{role} method and "
                                "cannot be called directly; move shared logic into a "
                                "@cb.function")
        return None
    _register_lowering(name)
    function_signature = schema.signature(name)

    class FunctionCall(AbstractTemplate):
        key = _call_key(name)

        def generic(self, args, kws):
            if kws:
                raise TypeError(f"{owner.__name__}.{name}: pass arguments by position")
            _check_arguments(owner, name, function_signature, args)
            return signature(_numba_type(function_signature.returns), *args,
                             recvr=record)

    return types.BoundFunction(FunctionCall, record)


views.function_resolver = resolve_function


__all__ = ["function_signature", "resolve_function"]
