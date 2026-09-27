"""Native entry points: the C functions the engine calls for one model class.

``njit`` already emits, next to every compiled method, a C-callable wrapper
that takes the record pointer and reports errors the way ``@cfunc`` does. The
engine's calling conventions differ only in small ways: a process also receives
its process pointer, a predicate returns ``uint8``, and a ``@function`` passes
booleans as bytes and models as addresses. One LLVM module of thin adapters per
class bridges those differences and is optimized and emitted once. A ``@cfunc``
per method would instead run a second full Numba pipeline that relinks and
re-emits the method it wraps.

Entry ABI, by role (``self`` is the record address):

    process          intptr_t f(intptr_t process, intptr_t self)   returns 0
    event, hook      void     f(intptr_t self)
    predicate        uint8_t  f(intptr_t self)
    function         ret      f(intptr_t self, args...)     float -> double,
                     int -> int64, bool -> uint8, model -> intptr_t
"""

from __future__ import annotations

from itertools import count
from typing import Any

from llvmlite import ir
from numba import types

from .numba_compat import cpu_target, global_compiler_lock

_I64 = ir.IntType(64)
_U8 = ir.IntType(8)
# Entry names share one JIT symbol table across all classes and redefinitions.
_serial = count()


def _abi_type(context, value_type) -> ir.Type:
    if value_type is types.none:
        return ir.VoidType()
    if value_type == types.boolean:
        return _U8
    if isinstance(value_type, types.Record):
        return _I64
    return context.get_value_type(value_type)


def _to_native(context, builder, value, value_type):
    if value_type == types.boolean:
        return builder.icmp_unsigned("!=", value, ir.Constant(_U8, 0))
    if isinstance(value_type, types.Record):
        return builder.inttoptr(value, context.get_value_type(value_type))
    return value


def _to_abi(builder, value, value_type):
    if value_type == types.boolean:
        return builder.zext(value, _U8)
    if isinstance(value_type, types.Record):
        return builder.ptrtoint(value, _I64)
    return value


class EntryModule:
    """Collects one class's adapters; ``finalize`` emits them all at once."""

    def __init__(self, owner: str):
        # Numba types target_context as its cached-property descriptor.
        self._context: Any = cpu_target.target_context
        self._name = f"cimba.entries.{next(_serial)}.{owner}"
        self._module = self._context.create_module(self._name)
        self._symbols: dict[str, str] = {}
        # Methods own the machine code the adapters call; keep them alive.
        self._methods: list[Any] = []
        self._library = None

    def add(self, name: str, method, *, process: bool = False) -> None:
        """Adapt ``method``, an ``njit`` dispatcher with one signature."""
        (result,) = method.overloads.values()
        descriptor = result.fndesc
        context = self._context
        target_type = ir.FunctionType(
            context.get_value_type(descriptor.restype),
            [context.get_value_type(t) for t in descriptor.argtypes])
        target = result.library.get_pointer_to_function(
            descriptor.llvm_cfunc_wrapper_name)
        if not target:
            raise RuntimeError(f"{name}: Numba emitted no C wrapper")
        self._methods.append(method)

        record, *arguments = descriptor.argtypes
        returns = types.intp if process else descriptor.restype
        parameters = [_I64] * (2 if process else 1)
        parameters += [_abi_type(context, t) for t in arguments]
        symbol = f"{self._name}.{name}"
        entry = ir.Function(self._module,
                            ir.FunctionType(_abi_type(context, returns), parameters),
                            symbol)
        builder = ir.IRBuilder(entry.append_basic_block("entry"))
        values = list(entry.args[1:] if process else entry.args)
        native = [builder.inttoptr(values[0], context.get_value_type(record))]
        native += [_to_native(context, builder, value, value_type)
                   for value, value_type in zip(values[1:], arguments)]
        callee = builder.inttoptr(ir.Constant(_I64, target), target_type.as_pointer())
        value = builder.call(callee, native)
        if process:
            builder.ret(ir.Constant(_I64, 0))
        elif returns is types.none:
            builder.ret_void()
        else:
            builder.ret(_to_abi(builder, value, returns))
        self._symbols[name] = symbol

    def finalize(self) -> dict[str, int]:
        """Emit machine code for every adapter; return name -> address."""
        with global_compiler_lock:
            library = self._context.codegen().create_library(self._name)
            library.add_ir_module(self._module)
            library.finalize()
            self._library = library
            return {name: library.get_pointer_to_function(symbol)
                    for name, symbol in self._symbols.items()}


__all__ = ["EntryModule"]
