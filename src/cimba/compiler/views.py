"""Numba record views for child models and references.

The typing template is registered once, before Numba initializes its record
attribute tables. Each class then adds only exact-record lowering rules.
"""

from __future__ import annotations

from functools import lru_cache
import operator
from typing import get_args, get_origin

from llvmlite import ir
from numba import from_dtype, types
from numba.extending import (
    intrinsic, make_attribute_wrapper, overload,
)

from cimba.layout import RecordLayout
from cimba.modeling import Ref
from cimba.schema import ClassSchema
from cimba.engine.symbols import register
from .numba_compat import (
    AttributeTemplate, cgutils, cpu_target, infer_getattr, lower_getattr,
    models, register_model,
)

register("cimba_trial_abandon")


def _abandon_if(context, builder, condition):
    with builder.if_then(condition):
        function = cgutils.get_or_insert_function(
            builder.module, ir.FunctionType(ir.VoidType(), []),
            "cimba_trial_abandon")
        builder.call(function, [])


class CollectionViewType(types.Type):
    def __init__(self, element: types.Record):
        self.element = element
        super().__init__(name=f"CollectionView[{element}]")

    @property
    def key(self):
        return self.element


@register_model(CollectionViewType)
class CollectionViewModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type,
                         [("data", types.intp), ("length", types.uintp)])


make_attribute_wrapper(CollectionViewType, "length", "length")


@intrinsic
def collection_item(typing_context, collection, index):
    if not isinstance(collection, CollectionViewType) or not isinstance(index, types.Integer):
        return None

    def codegen(context, builder, signature, arguments):
        view = cgutils.create_struct_proxy(collection)(context, builder,
                                                        value=arguments[0])
        position = context.cast(builder, arguments[1], index, types.uintp)
        _abandon_if(context, builder,
                    builder.icmp_unsigned(">=", position, view.length))
        address = builder.inttoptr(
            view.data, context.get_value_type(types.intp).as_pointer())
        pointer = builder.load(builder.gep(address, [position]))
        return builder.inttoptr(
            pointer, context.get_value_type(collection.element))

    return collection.element(collection, index), codegen


@overload(operator.getitem)
def collection_getitem(collection, index):
    if isinstance(collection, CollectionViewType) and isinstance(index, types.Integer):
        def implementation(collection, index):
            return collection_item(collection, index)  # pyright: ignore[reportCallIssue]
        return implementation


@overload(len)
def collection_length(collection):
    if isinstance(collection, CollectionViewType):
        def implementation(collection):
            return collection.length
        return implementation


_view_fields: dict[tuple[types.Record, str], tuple[types.Type, int, int | None]] = {}
# Installed by compiler.functions: (typing context, record, attr) -> type | None
function_resolver = None


class EventHandleType(types.Type):
    def __init__(self):
        super().__init__(name="EventHandle")


class PredicateHandleType(types.Type):
    def __init__(self):
        super().__init__(name="PredicateHandle")


class ScheduledHandleType(types.Type):
    def __init__(self):
        super().__init__(name="ScheduledHandle")


class ProcessHandleType(types.Type):
    def __init__(self):
        super().__init__(name="ProcessHandle")


EVENT_HANDLE = EventHandleType()
PREDICATE_HANDLE = PredicateHandleType()
SCHEDULED_HANDLE = ScheduledHandleType()
PROCESS_HANDLE = ProcessHandleType()


@register_model(ProcessHandleType)
class ProcessHandleModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, [("pointer", types.intp)])


make_attribute_wrapper(ProcessHandleType, "pointer", "pointer")


@register_model(ScheduledHandleType)
class ScheduledHandleModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, [("handle", types.uint64)])


make_attribute_wrapper(ScheduledHandleType, "handle", "handle")


@register_model(EventHandleType)
@register_model(PredicateHandleType)
class CallbackHandleModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type,
                         [("callback", types.intp), ("context", types.intp)])


make_attribute_wrapper(EventHandleType, "callback", "callback")
make_attribute_wrapper(EventHandleType, "context", "context")
make_attribute_wrapper(PredicateHandleType, "callback", "callback")
make_attribute_wrapper(PredicateHandleType, "context", "context")


@infer_getattr
class ModelViewAttributes(AttributeTemplate):
    key = types.Record

    def generic_resolve(self, record, attr):
        entry = _view_fields.get((record, attr))
        if entry is not None:
            return entry[0]
        # ``@function`` methods: resolved here because this template runs
        # before Numba's own record-field lookup (see compiler.functions).
        if function_resolver is not None:
            return function_resolver(self.context, record, attr)
        return None


@lru_cache(maxsize=512)
def register_views(schema: ClassSchema) -> None:
    parent_layout = RecordLayout.of(schema)
    parent_type = from_dtype(parent_layout.dtype)
    assert isinstance(parent_type, types.Record)
    for field in schema.fields:
        if field.kind not in {"child", "ref", "collection"}:
            continue
        value_type = field.value_type
        if field.kind == "collection" and get_origin(value_type) is Ref:
            value_type = get_args(value_type)[0]
        child_type = from_dtype(RecordLayout.of(ClassSchema.of(value_type)).dtype)
        assert isinstance(child_type, types.Record)
        view_type = (CollectionViewType(child_type)
                     if field.kind == "collection" else
                     types.Optional(child_type) if field.optional else child_type)
        offset = parent_layout.offset(field.name)
        length_offset = (parent_layout.offset(f"{field.name}_length")
                         if field.kind == "collection" else None)
        key = (parent_type, field.name)
        previous = _view_fields.get(key)
        if previous is not None:
            if previous != (view_type, offset, length_offset):
                raise TypeError(
                    f"{schema.cls.__name__}.{field.name}: model classes with "
                    "identical native records have incompatible child views")
            continue
        _view_fields[key] = (view_type, offset, length_offset)

        def register_lowering(record_type, name, result_type, field_offset,
                              collection_length_offset):
            @lower_getattr(record_type, name)
            def lower(context, builder, typ, value):
                base = builder.bitcast(value, ir.IntType(8).as_pointer())
                address = builder.gep(
                    base, [ir.Constant(ir.IntType(32), field_offset)])
                location = builder.bitcast(
                    address, context.get_value_type(types.intp).as_pointer())
                pointer = builder.load(location)
                if collection_length_offset is None:
                    if isinstance(result_type, types.Optional):
                        present = builder.icmp_unsigned(
                            "!=", pointer,
                            ir.Constant(context.get_value_type(types.intp), 0))
                        with builder.if_else(present) as (yes, no):
                            with yes:
                                value = builder.inttoptr(
                                    pointer,
                                    context.get_value_type(result_type.type))
                                present_value = context.make_optional_value(
                                    builder, result_type.type, value)
                                yes_block = builder.block
                            with no:
                                missing_value = context.make_optional_none(
                                    builder, result_type.type)
                                no_block = builder.block
                        merged = builder.phi(context.get_value_type(result_type))
                        merged.add_incoming(present_value, yes_block)
                        merged.add_incoming(missing_value, no_block)
                        return merged
                    _abandon_if(context, builder,
                                builder.icmp_unsigned(
                                    "==", pointer,
                                    ir.Constant(context.get_value_type(types.intp), 0)))
                    return builder.inttoptr(
                        pointer, context.get_value_type(result_type))
                length_address = builder.gep(
                    base, [ir.Constant(ir.IntType(32), collection_length_offset)])
                length_location = builder.bitcast(
                    length_address, context.get_value_type(types.uintp).as_pointer())
                view = cgutils.create_struct_proxy(result_type)(context, builder)
                view.data = pointer
                view.length = builder.load(length_location)
                return view._getvalue()

        register_lowering(parent_type, field.name, view_type, offset, length_offset)
    for name in schema.events + schema.predicates:
        index = schema.slots.index(name)
        view_type = (EVENT_HANDLE if name in schema.events
                     else PREDICATE_HANDLE)
        key = (parent_type, name)
        if key in _view_fields:
            raise TypeError(f"{schema.cls.__name__}.{name}: callback name "
                            "conflicts with a field")
        _view_fields[key] = (view_type, index, None)

        def register_callback_lowering(record_type, callback_name,
                                       callback_type, slot):
            @lower_getattr(record_type, callback_name)
            def lower(context, builder, typ, value):
                base = builder.bitcast(value, ir.IntType(8).as_pointer())
                descriptor = builder.load(builder.bitcast(
                    base, context.get_value_type(types.intp).as_pointer()))
                callbacks = builder.inttoptr(
                    descriptor, context.get_value_type(types.intp).as_pointer())
                callback = builder.load(builder.gep(
                    callbacks, [ir.Constant(ir.IntType(32), slot)]))
                view = cgutils.create_struct_proxy(callback_type)(context, builder)
                view.callback = callback
                view.context = builder.ptrtoint(
                    value, context.get_value_type(types.intp))
                return view._getvalue()

        register_callback_lowering(parent_type, name, view_type, index)
    getattr(cpu_target.typing_context, "refresh")()
    getattr(cpu_target.target_context, "refresh")()
