"""Numba intrinsics for the pointer and bit-level casts the bindings need.

cimba's C API traffics in raw pointers and opaque int64 objects; these
intrinsics let nopython code move between integer addresses, typed
pointers, records, and float/int bit patterns without round-tripping
through Python.
"""

from collections.abc import Callable
from typing import Any

from llvmlite import ir
from numba import types
from numba.core import cgutils
from numba.extending import intrinsic


@intrinsic
def addressof(typingctx, ptr):
    """Integer address of a typed pointer."""
    if not isinstance(ptr, types.CPointer):
        raise TypeError("addressof() expects a typed pointer")

    def codegen(context, builder, signature, args):
        return builder.ptrtoint(args[0], context.get_value_type(types.intp))

    return types.intp(ptr), codegen


@intrinsic
def record_addr(typingctx, rec):
    """Integer address of a record value."""
    if not isinstance(rec, types.Record):
        raise TypeError("record_addr() expects a record")

    def codegen(context, builder, signature, args):
        return builder.ptrtoint(args[0], context.get_value_type(types.intp))

    return types.intp(rec), codegen


def ptr_caster(pointee: Any) -> Callable[[int], Any]:
    """Build an intrinsic casting an integer address to ``pointee *``."""
    ptr_type = types.CPointer(pointee)

    @intrinsic
    def cast(typingctx, addr):
        if not isinstance(addr, types.Integer):
            raise TypeError("expected an integer address")

        def codegen(context, builder, signature, args):
            return builder.inttoptr(args[0], context.get_value_type(ptr_type))

        return ptr_type(addr), codegen

    return cast


@intrinsic
def store_get(typingctx, store):
    """Blocking objectqueue get returning (status, object)."""
    if not isinstance(store, types.Integer):
        raise TypeError("store_get() expects a store handle")

    ret_type = types.Tuple((types.int64, types.intp))

    def codegen(context, builder, signature, args):
        intp_t = context.get_value_type(types.intp)
        int64_t = context.get_value_type(types.int64)
        objloc = cgutils.alloca_once(builder, intp_t)
        builder.store(context.get_constant(types.intp, 0), objloc)

        fnty = ir.FunctionType(int64_t, [intp_t, intp_t.as_pointer()])
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, "cpy_objectqueue_get")
        status = builder.call(fn, [args[0], objloc])
        obj = builder.load(objloc)
        return context.make_tuple(builder, ret_type, [status, obj])

    return ret_type(store), codegen


@intrinsic
def pq_get(typingctx, pqueue):
    """Blocking priorityqueue get returning (status, object)."""
    if not isinstance(pqueue, types.Integer):
        raise TypeError("pq_get() expects a priority queue handle")

    ret_type = types.Tuple((types.int64, types.intp))

    def codegen(context, builder, signature, args):
        intp_t = context.get_value_type(types.intp)
        int64_t = context.get_value_type(types.int64)
        objloc = cgutils.alloca_once(builder, intp_t)
        builder.store(context.get_constant(types.intp, 0), objloc)

        fnty = ir.FunctionType(int64_t, [intp_t, intp_t.as_pointer()])
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, "cpy_priorityqueue_get")
        status = builder.call(fn, [args[0], objloc])
        obj = builder.load(objloc)
        return context.make_tuple(builder, ret_type, [status, obj])

    return ret_type(pqueue), codegen


@intrinsic
def f2i(typingctx, x):
    """Bit-cast a float64 to int64."""
    if x != types.float64:
        raise TypeError("f2i() expects a float64")

    def codegen(context, builder, signature, args):
        return builder.bitcast(args[0], context.get_value_type(types.int64))

    return types.int64(x), codegen


@intrinsic
def i2f(typingctx, i):
    """Bit-cast an int64 back to float64."""
    if not isinstance(i, types.Integer):
        raise TypeError("i2f() expects an int64")

    def codegen(context, builder, signature, args):
        return builder.bitcast(args[0], context.get_value_type(types.float64))

    return types.float64(i), codegen
