"""``cb.Array`` fields in compiled model code.

A record holds an array field as two words: the address of its element
buffer and its length (-1 once a spawned model is released and its buffer
freed). ``model.field`` is typed as an ``ArrayView``: the address of those two
words, never a copy of them, so every access reads the current buffer and a
view kept across ``release()`` can't reach freed memory. Every index is
checked against the length; a bad one fails the trial with a reason naming
the field, the index and the length.
"""

from __future__ import annotations

import operator

from llvmlite import ir
from numba import types
from numba.extending import intrinsic, overload

from cimba.engine.symbols import register
from .numba_compat import (
    RefType, cgutils, iternext_impl, lower_builtin, models, register_model,
)

register("cpy_array_index_fail")

_I64 = ir.IntType(64)


class ArrayViewType(types.IterableType):
    """``label``'s elements of type ``element``, as a field of a model."""

    def __init__(self, element: types.Type, label: str):
        self.element = element
        self.label = label
        super().__init__(name=f"ArrayView[{element}, {label}]")

    @property
    def key(self):
        return self.element, self.label

    @property
    def iterator_type(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        return ArrayIteratorType(self)


class ArrayIteratorType(types.SimpleIteratorType):
    def __init__(self, view: ArrayViewType):
        self.view = view
        super().__init__(f"iter({view})", view.element)


@register_model(ArrayViewType)
class ArrayViewModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, [("slot", types.intp)])


@register_model(ArrayIteratorType)
class ArrayIteratorModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, [("slot", types.intp),
                                        ("index", types.EphemeralPointer(types.int64))])


def array_view(context, builder, view_type: ArrayViewType, record, offset: int):
    """The view of the array field at ``offset`` of the record pointer ``record``."""
    base = builder.bitcast(record, ir.IntType(8).as_pointer())
    slot = builder.gep(base, [ir.Constant(ir.IntType(32), offset)])
    view = cgutils.create_struct_proxy(view_type)(context, builder)
    view.slot = builder.ptrtoint(slot, context.get_value_type(types.intp))
    return view._getvalue()


def _words(builder, slot):
    words = builder.inttoptr(slot, _I64.as_pointer())
    return builder.load(words), builder.load(builder.gep(words, [ir.Constant(_I64, 1)]))


def _element(context, builder, view_type, data, index):
    pointer = builder.inttoptr(data, context.get_data_type(view_type.element).as_pointer())
    return builder.gep(pointer, [index])


def _checked_element(context, builder, view_type, view_value, index_value, index_type):
    """Pointer to element ``index``, failing the trial when it is out of range."""
    view = cgutils.create_struct_proxy(view_type)(context, builder, value=view_value)
    index = context.cast(builder, index_value, index_type, types.int64)
    data, length = _words(builder, view.slot)
    outside = builder.or_(builder.icmp_signed("<", index, ir.Constant(_I64, 0)),
                          builder.icmp_signed(">=", index, length))
    with builder.if_then(outside, likely=False):
        fail = cgutils.get_or_insert_function(
            builder.module,
            ir.FunctionType(ir.VoidType(), [ir.IntType(8).as_pointer(), _I64, _I64]),
            "cpy_array_index_fail")
        builder.call(fail, [context.insert_const_string(builder.module, view_type.label),
                            index, length])
    return _element(context, builder, view_type, data, index)


_INDEX = (types.Integer, types.IntEnumMember)


@intrinsic
def _array_get(typing_context, view, index):
    if not isinstance(view, ArrayViewType) or not isinstance(index, _INDEX):
        return None

    def codegen(context, builder, signature, arguments):
        pointer = _checked_element(context, builder, view, arguments[0], arguments[1], index)
        return context.unpack_value(builder, view.element, pointer)

    return view.element(view, index), codegen


def _check_value(view: ArrayViewType, value) -> None:
    element = view.element
    accepted = {types.boolean: (types.Boolean,),
                types.int64: (types.Integer, types.Boolean, types.IntEnumMember),
                types.float64: (types.Number, types.Boolean, types.IntEnumMember)}[element]
    if not isinstance(value, accepted):
        name = {types.boolean: "bool", types.int64: "int", types.float64: "float"}[element]
        raise TypeError(f"{view.label} holds {name} values, got {value}"
                        + ("; convert with int()" if name == "int" else ""))


@intrinsic
def _array_set(typing_context, view, index, value):
    if not isinstance(view, ArrayViewType) or not isinstance(index, _INDEX):
        return None
    _check_value(view, value)

    def codegen(context, builder, signature, arguments):
        pointer = _checked_element(context, builder, view, arguments[0], arguments[1], index)
        stored = context.cast(builder, arguments[2], value, view.element)
        context.pack_value(builder, view.element, stored, pointer)
        return context.get_dummy_value()

    return types.none(view, index, value), codegen


@intrinsic
def _array_length(typing_context, view):
    if not isinstance(view, ArrayViewType):
        return None

    def codegen(context, builder, signature, arguments):
        proxy = cgutils.create_struct_proxy(view)(context, builder, value=arguments[0])
        _, length = _words(builder, proxy.slot)
        # A released model's length is -1: it has no elements.
        negative = builder.icmp_signed("<", length, ir.Constant(_I64, 0))
        return builder.select(negative, ir.Constant(_I64, 0), length)

    return types.intp(view), codegen


@overload(operator.getitem)
def array_getitem(view, index):
    if isinstance(view, ArrayViewType):
        if not isinstance(index, _INDEX):
            raise TypeError(f"{view.label}: index with an int, got {index}")
        return lambda view, index: _array_get(view, index)  # pyright: ignore[reportCallIssue]


@overload(operator.setitem)
def array_setitem(view, index, value):
    if isinstance(view, ArrayViewType):
        if not isinstance(index, _INDEX):
            raise TypeError(f"{view.label}: index with an int, got {index}")
        _check_value(view, value)

        def implementation(view, index, value):
            _array_set(view, index, value)  # pyright: ignore[reportCallIssue]
        return implementation


@overload(len)
def array_len(view):
    if isinstance(view, ArrayViewType):
        return lambda view: _array_length(view)  # pyright: ignore[reportCallIssue]


@lower_builtin("getiter", ArrayViewType)
def array_getiter(context, builder, signature, arguments):
    view = cgutils.create_struct_proxy(signature.args[0])(context, builder,
                                                           value=arguments[0])
    iterator = cgutils.create_struct_proxy(signature.return_type)(context, builder)
    iterator.slot = view.slot
    iterator.index = cgutils.alloca_once_value(builder, ir.Constant(_I64, 0))
    return iterator._getvalue()


@lower_builtin("iternext", ArrayIteratorType)
@iternext_impl(RefType.BORROWED)
def array_iternext(context, builder, signature, arguments, result):
    iterator_type = signature.args[0]
    iterator = cgutils.create_struct_proxy(iterator_type)(context, builder,
                                                           value=arguments[0])
    index = builder.load(iterator.index)
    # Read the buffer each step: a release during the loop ends it.
    data, length = _words(builder, iterator.slot)
    valid = builder.icmp_signed("<", index, length)
    result.set_valid(valid)
    with builder.if_then(valid):
        pointer = _element(context, builder, iterator_type.view, data, index)
        result.yield_(context.unpack_value(builder, iterator_type.view.element, pointer))
        builder.store(builder.add(index, ir.Constant(_I64, 1)), iterator.index)


__all__ = ["ArrayViewType", "array_view"]
