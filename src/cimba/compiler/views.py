"""Numba record views for child models and references.

The typing template is registered once, before Numba initializes its record
attribute tables. Each class then adds only exact-record lowering rules.
"""

from __future__ import annotations

from functools import lru_cache
import operator
from typing import get_args, get_origin

import numpy as np
from llvmlite import ir
from numba import from_dtype, types
from numba.extending import (
    intrinsic, make_attribute_wrapper, overload,
)

from cimba.layout import MODEL_RECORD_CLASSES, RecordLayout
from cimba.modeling import Model, Ref
from cimba.schema import ClassSchema, is_int_enum
from cimba.engine.symbols import register
from .arrays import ArrayViewType, array_view
from .enums import member_type
from .numba_compat import (
    AttributeTemplate, NumbaTypeError, cgutils, cpu_target, infer_getattr, lower_cast, lower_getattr,
    lower_setattr, models, register_model,
)

register("cimba_trial_abandon")
register("cpy_reference_fail")

_ARRAY_ELEMENTS = {int: types.int64, float: types.float64, bool: types.boolean}


def _model_class(record):
    if not isinstance(record, types.Record):
        return None
    descriptor = record.fields.get("class_descriptor")
    return MODEL_RECORD_CLASSES.get(descriptor.title) if descriptor else None


# A model handle is a record pointer. Assigning or passing a subclass's handle
# where a base class is expected keeps the pointer: base-typed views read the
# base's fields at the same offsets and call through the record's own table.
@lower_cast(types.Record, types.Record)
def _upcast_model(context, builder, fromty, toty, value):
    actual, expected = _model_class(fromty), _model_class(toty)
    if actual is None or expected is None or not issubclass(actual, expected):
        raise NumbaTypeError(f"cannot use a {actual.__name__ if actual else fromty} model "
                        f"where a {expected.__name__ if expected else toty} is expected")
    return builder.bitcast(value, context.get_value_type(toty))


@lower_cast(types.Optional, types.Record)
def _present_model(context, builder, fromty, toty, value):
    """An optional reference used as a model: a None fails the trial."""
    optional = context.make_helper(builder, fromty, value=value)
    expected = _model_class(toty)
    with builder.if_then(builder.not_(cgutils.as_bool_bit(builder, optional.valid)),
                         likely=False):
        fail = cgutils.get_or_insert_function(
            builder.module, ir.FunctionType(ir.VoidType(), [ir.IntType(8).as_pointer()]),
            "cpy_reference_fail")
        builder.call(fail, [context.insert_const_string(
            builder.module, expected.__name__ if expected else str(toty))])
    return context.cast(builder, optional.data, fromty.type, toty)


@lower_cast(types.NoneType, types.Record)
def _none_model(context, builder, fromty, toty, value):
    expected = _model_class(toty)
    name = expected.__name__ if expected else str(toty)
    raise NumbaTypeError(f"None is not a {name} model; declare the reference "
                    f"cb.Ref[{name}] | None to allow None")


def _register_reference_assignment(record_type, name, view_type, offset, label):
    """``model.field = value`` for a ``Ref`` field: store the pointer, or 0 for None."""
    @lower_setattr(record_type, name)
    def lower(context, builder, signature, arguments):
        target, value = arguments
        if isinstance(view_type, types.Optional):
            optional = context.make_helper(builder, view_type, value=value)
            pointer = builder.select(
                cgutils.as_bool_bit(builder, optional.valid),
                builder.ptrtoint(optional.data, context.get_value_type(types.intp)),
                ir.Constant(context.get_value_type(types.intp), 0))
        else:
            pointer = builder.ptrtoint(value, context.get_value_type(types.intp))
        base = builder.bitcast(target, ir.IntType(8).as_pointer())
        location = builder.bitcast(
            builder.gep(base, [ir.Constant(ir.IntType(32), offset)]),
            context.get_value_type(types.intp).as_pointer())
        builder.store(pointer, location)
        return context.get_dummy_value()


def _register_structure_assignment(record_type, name, message):
    """Children, lists and arrays are structure: reject replacing them."""
    @lower_setattr(record_type, name)
    def lower(context, builder, signature, arguments):
        raise NumbaTypeError(message)


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
    if not isinstance(collection, CollectionViewType) or not isinstance(
            index, (types.Integer, types.IntEnumMember)):
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
    if isinstance(collection, CollectionViewType) and isinstance(
            index, (types.Integer, types.IntEnumMember)):
        def implementation(collection, index):
            return collection_item(collection, index)  # pyright: ignore[reportCallIssue]
        return implementation


@overload(len)
def collection_length(collection):
    if isinstance(collection, CollectionViewType):
        def implementation(collection):
            # Keep the storage ABI unsigned, but match Python's signed len().
            return np.intp(collection.length)
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
        descriptor = record.fields.get("class_descriptor")
        owner = MODEL_RECORD_CLASSES.get(descriptor.title) if descriptor else None
        if owner is not None:
            schema = ClassSchema.of(owner)
            if (attr in schema.events + schema.predicates or
                    any(field.name == attr and field.kind in
                        {"child", "ref", "collection", "array"} for field in schema.fields)):
                raise TypeError(f"no view registered for {owner.__name__}.{attr}")
        # ``@function`` methods: resolved here because this template runs
        # before Numba's own record-field lookup (see compiler.functions).
        if function_resolver is not None:
            return function_resolver(self.context, record, attr)
        return None


def register_reachable_views(schema: ClassSchema) -> None:
    """Register the declared model graph before typing any method.

    References may form cycles and name bases with no instances in the tree.
    Register views only; compiling implementations is still ``ensure``'s job.
    """
    pending = [schema]
    seen = set()
    while pending:
        current = pending.pop()
        if current.cls in seen:
            continue
        seen.add(current.cls)
        register_views(current)
        for field in current.fields:
            if field.kind not in {"child", "ref", "collection"}:
                continue
            target = field.value_type
            if field.kind == "collection" and get_origin(target) is Ref:
                target = get_args(target)[0]
            pending.append(ClassSchema.of(target))
        # Model parameters and results can expose bases without tree instances.
        for _, signature in current.signatures:
            for target in (*signature.types, signature.returns):
                if isinstance(target, type) and issubclass(target, Model):
                    pending.append(ClassSchema.of(target))


@lru_cache(maxsize=512)
def register_views(schema: ClassSchema) -> None:
    parent_layout = RecordLayout.of(schema)
    parent_type = from_dtype(parent_layout.dtype)
    assert isinstance(parent_type, types.Record)
    for field in schema.fields:
        if field.kind in {"param", "field", "output"} and is_int_enum(field.value_type):
            # Stored as int64 and typed as the enum's members: Numba's own
            # record lowering loads and stores the value; writes must be members.
            key = (parent_type, field.name)
            entry = (member_type(field.value_type), parent_layout.offset(field.name), None)
            previous = _view_fields.get(key)
            if previous is not None and previous != entry:
                raise TypeError(f"{schema.cls.__name__}.{field.name}: model classes with "
                                "identical native records have incompatible enum fields")
            _view_fields[key] = entry
            continue
        if field.kind == "array":
            label = f"{schema.cls.__name__}.{field.name}"
            view_type = ArrayViewType(_ARRAY_ELEMENTS[field.value_type], label)
            entry = (view_type, parent_layout.offset(field.name), None)
            key = (parent_type, field.name)
            previous = _view_fields.get(key)
            if previous is not None:
                if previous[1:] != entry[1:] or previous[0] != view_type:
                    raise TypeError(f"{label}: model classes with identical native "
                                    "records have incompatible array fields")
                continue
            _view_fields[key] = entry

            def register_array_lowering(record_type, name, result_type, field_offset):
                @lower_getattr(record_type, name)
                def lower(context, builder, typ, value):
                    return array_view(context, builder, result_type, value, field_offset)

            register_array_lowering(parent_type, field.name, view_type, entry[1])
            _register_structure_assignment(
                parent_type, field.name,
                f"{label} is an array field: assign its elements ({field.name}[i] = value); "
                "its length is configuration (cb.Array(n), or a spawn argument)")
            continue
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
        label = f"{schema.cls.__name__}.{field.name}"
        if field.kind == "ref":
            _register_reference_assignment(parent_type, field.name, view_type, offset, label)
        else:
            _register_structure_assignment(
                parent_type, field.name,
                f"{label} is a {'child model' if field.kind == 'child' else 'list'} field, "
                "part of the model structure, and cannot be reassigned in model code; "
                "declare a cb.Ref field to point at a model at run time")
    for name in schema.events + schema.predicates:
        index = schema.slots.index(name)
        view_type = (EVENT_HANDLE if name in schema.events
                     else PREDICATE_HANDLE)
        key = (parent_type, name)
        entry = (view_type, index, None)
        previous = _view_fields.get(key)
        if previous is not None:
            if previous != entry:
                raise TypeError(f"{schema.cls.__name__}.{name}: callback name "
                                "conflicts with a field")
            continue
        _view_fields[key] = entry

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
