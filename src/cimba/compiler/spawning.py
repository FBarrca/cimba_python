"""Numba typing and lowering for dynamic model allocation.

The call template reads keyword names directly from Numba's typed call. Model
source is never inspected or rewritten.
"""

from __future__ import annotations

import inspect
from math import nan

import numpy as np
from llvmlite import ir
from numba import from_dtype, types

from cimba.engine.symbols import register
from cimba.engine.abi import INPUT_EMPTY, INPUT_SLOT, INPUT_SLOT_BOOL, INPUT_SLOT_INT
from cimba.layout import MODEL_RECORD_CLASSES, RecordLayout, array_dtype
from cimba.modeling import ModelMeta, Sweep, spawn
from cimba.schema import ClassSchema, is_int_enum
from .numba_compat import (
    AbstractTemplate, cgutils, infer_global, lower_builtin, lower_constant,
    models, register_model, signature, typeof_impl,
)
from .enums import member_type
from .views import register_reachable_views

register("cpy_model_allocate")
register("cpy_model_start")
register("cpy_model_input_bind")
register("cpy_model_array")

# Initial elements of class-default arrays, kept alive for compiled spawns.
_array_defaults: dict[tuple[type, str], np.ndarray] = {}


class ModelClassType(types.Type):
    def __init__(self, model_class):
        self.model_class = model_class
        super().__init__(name=f"ModelClass[{model_class.__module__}.{model_class.__qualname__}]")

    @property
    def key(self):
        return self.model_class


def _input_type(field):
    return from_dtype({float: INPUT_SLOT, int: INPUT_SLOT_INT,
                       bool: INPUT_SLOT_BOOL}[field.value_type])


@register_model(ModelClassType)
class ModelClassModel(models.PrimitiveModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, types.intp)


@typeof_impl.register(ModelMeta)
def typeof_model_class(value, context):
    return ModelClassType(value)


@lower_constant(ModelClassType)
def lower_model_class(context, builder, typ, value):
    return context.get_constant(types.intp, 0)


@infer_global(spawn)
class SpawnTemplate(AbstractTemplate):
    key = spawn

    def generic(self, args, kws):
        if len(args) != 1 or not isinstance(args[0], ModelClassType):
            return None
        schema = ClassSchema.of(args[0].model_class)
        fields = {field.name: field for field in schema.fields}
        unsupported = [field.name for field in schema.fields
                       if field.kind not in {"param", "field", "output",
                                             "ref", "input", "series", "array"}]
        if unsupported:
            raise TypeError(f"{schema.cls.__name__}: dynamic models do not "
                            f"support these fields: {', '.join(unsupported)}. "
                            "Entities, child models and lists are static structure; "
                            "give a spawned model cb.Array fields for per-instance "
                            "tables, and cb.Ref fields to link it to other models")
        unknown = set(kws) - fields.keys()
        if unknown:
            raise TypeError(f"{schema.cls.__name__}: unknown spawn fields: "
                            f"{', '.join(sorted(unknown))}")
        for field in schema.fields:
            if field.kind in {"input", "series"}:
                source_type = kws.get(field.name)
                if (field.kind == "input" and field.optional and
                        (source_type is None or isinstance(source_type, types.NoneType))):
                    continue
                if source_type != _input_type(field):
                    raise TypeError(f"{schema.cls.__name__}.{field.name}: "
                                    "spawn requires an existing input handle")
            if field.kind in {"field", "param"}:
                if field.default is None and field.name not in kws:
                    raise TypeError(f"{schema.cls.__name__}.{field.name}: "
                                    "spawn value required")
            if (field.kind in {"field", "param", "output"} and field.name in kws and
                    is_int_enum(field.value_type) and
                    kws[field.name] != member_type(field.value_type)):
                raise TypeError(f"{schema.cls.__name__}.{field.name}: expected a "
                                f"{field.value_type.__name__} member, got {kws[field.name]}")
            if field.kind == "array":
                length_type = kws.get(field.name)
                if length_type is None:
                    if field.default is None:
                        raise TypeError(f"{schema.cls.__name__}.{field.name}: spawn "
                                        f"length required ({field.name}=<int length>), "
                                        "or give the field a class default cb.Array(n)")
                    if isinstance(field.default, Sweep):
                        raise TypeError(f"{schema.cls.__name__}.{field.name}: a spawned "
                                        "model's class default cannot be a sweep; pass "
                                        "the length at spawn")
                elif not isinstance(length_type, types.Integer):
                    raise TypeError(f"{schema.cls.__name__}.{field.name}: spawn takes the "
                                    f"array length as an int, got {length_type}")
            if field.kind == "ref":
                reference_type = kws.get(field.name)
                if reference_type is None:
                    if not field.optional:
                        raise TypeError(f"{schema.cls.__name__}.{field.name}: "
                                        "spawn reference required")
                elif isinstance(reference_type, types.NoneType):
                    if not field.optional:
                        raise TypeError(f"{schema.cls.__name__}.{field.name}: "
                                        "reference cannot be None")
                elif isinstance(reference_type, types.Record) or (
                        isinstance(reference_type, types.Optional) and
                        isinstance(reference_type.type, types.Record)):
                    # An optional for a required reference is checked when
                    # spawning: Numba keeps a narrowed optional's type.
                    record = (reference_type.type if isinstance(reference_type, types.Optional)
                              else reference_type)
                    assert isinstance(record, types.Record)
                    actual = next((model_class for marker, model_class
                                   in MODEL_RECORD_CLASSES.items()
                                   if marker in record.fields), None)
                    if actual is None or not issubclass(actual, field.value_type):
                        raise TypeError(f"{schema.cls.__name__}.{field.name}: "
                                        "incompatible model reference")
                else:
                    raise TypeError(f"{schema.cls.__name__}.{field.name}: "
                                    "expected a model reference")
        parameters = [inspect.Parameter(
            "model_class", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        parameters.extend(inspect.Parameter(
            name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in kws)
        register_reachable_views(schema)
        record_type = from_dtype(RecordLayout.of(schema).dtype)
        return signature(record_type, *args, *kws.values()).replace(
            pysig=inspect.Signature(parameters))


def _field_pointer(context, builder, record, offset, value_type):
    base = builder.bitcast(record, ir.IntType(8).as_pointer())
    address = builder.gep(base, [ir.Constant(ir.IntType(32), offset)])
    return builder.bitcast(address, context.get_value_type(value_type).as_pointer())


def _constant_pointer(builder, address):
    return builder.inttoptr(ir.Constant(ir.IntType(64), address),
                            ir.IntType(8).as_pointer())


def _spawn_array(context, builder, schema, layout, field, record, argument):
    """Allocate the element buffer of one array field of a new spawned model.

    An int spawn argument gives a zeroed buffer of that length; otherwise the
    class default ``cb.Array(n)`` or ``cb.Array([values])`` is copied.
    """
    i64 = ir.IntType(64)
    pointer = ir.IntType(8).as_pointer()
    element = array_dtype(field.value_type)
    initial = ir.Constant(pointer, None)
    if argument is not None:
        length = context.cast(builder, argument[1], argument[0], types.int64)
    else:
        default = field.default
        length = ir.Constant(i64, default.length)
        if default.values is not None and default.length:
            key = (schema.cls, field.name)
            if key not in _array_defaults:
                values = np.asarray(default.values, dtype=element)
                values.flags.writeable = False
                _array_defaults[key] = values
            initial = _constant_pointer(builder, _array_defaults[key].ctypes.data)
    allocate = cgutils.get_or_insert_function(
        builder.module,
        ir.FunctionType(ir.VoidType(), [pointer, i64, i64, i64, pointer, pointer]),
        "cpy_model_array")
    builder.call(allocate, [
        record, ir.Constant(i64, layout.offset(field.name)), length,
        ir.Constant(i64, element.itemsize), initial,
        context.insert_const_string(builder.module, f"{schema.cls.__name__}.{field.name}"),
    ])


@lower_builtin(spawn, types.VarArg(types.Any))
def lower_spawn(context, builder, sig, arguments):
    model_class = sig.args[0].model_class
    schema = ClassSchema.of(model_class)
    layout = RecordLayout.of(schema)
    from .classes import ensure
    compiled = ensure(schema)
    uintptr = context.get_value_type(types.intp)
    pointer = ir.IntType(8).as_pointer()
    allocate_type = ir.FunctionType(pointer, [uintptr, pointer])
    allocate = cgutils.get_or_insert_function(builder.module, allocate_type,
                                              "cpy_model_allocate")
    record = builder.call(allocate, [
        ir.Constant(uintptr, layout.dtype.itemsize),
        _constant_pointer(builder, compiled.dispatch.ctypes.data),
    ])
    supplied = dict(zip(tuple(sig.pysig.parameters)[1:],
                        zip(sig.args[1:], arguments[1:])))
    for field in schema.fields:
        if field.kind in {"input", "series"}:
            argument = supplied.get(field.name)
            if argument is None or isinstance(argument[0], types.NoneType):
                # calloc supplies an empty slot; only its kind needs setting.
                builder.store(context.get_constant(types.uint32, INPUT_EMPTY),
                              _field_pointer(context, builder, record,
                                             layout.offset(field.name), types.uint32))
                continue
            target_type = _input_type(field)
            value = arguments[tuple(sig.pysig.parameters).index(field.name)]
            location = _field_pointer(context, builder, record,
                                      layout.offset(field.name), target_type)
            cgutils.raw_memcpy(builder, location, value,
                               ir.Constant(uintptr, 1), INPUT_SLOT.itemsize)
            bind_type = ir.FunctionType(ir.VoidType(), [pointer])
            bind = cgutils.get_or_insert_function(builder.module, bind_type,
                                                 "cpy_model_input_bind")
            builder.call(bind, [builder.bitcast(location, pointer)])
            continue
        if field.kind == "array":
            _spawn_array(context, builder, schema, layout, field, record,
                         supplied.get(field.name))
            continue
        if field.kind == "ref":
            target_type = types.intp
            argument = supplied.get(field.name)
            if argument is None or isinstance(argument[0], types.NoneType):
                value = ir.Constant(uintptr, 0)
            elif isinstance(argument[0], types.Optional) and not field.optional:
                # A None fails the trial (see views._present_model)
                present = context.cast(builder, argument[1], argument[0], argument[0].type)
                value = builder.ptrtoint(present, uintptr)
            elif isinstance(argument[0], types.Optional):
                optional = context.make_helper(builder, argument[0], value=argument[1])
                value = builder.select(cgutils.as_bool_bit(builder, optional.valid),
                                       builder.ptrtoint(optional.data, uintptr),
                                       ir.Constant(uintptr, 0))
            elif isinstance(argument[0], types.Record):
                value = builder.ptrtoint(arguments[
                    tuple(sig.pysig.parameters).index(field.name)], uintptr)
            else:
                raise TypeError(f"{schema.cls.__name__}.{field.name}: "
                                "expected a model reference")
        else:
            target_type = (types.int64 if is_int_enum(field.value_type) else
                           {float: types.float64, int: types.int64,
                            bool: types.boolean}.get(field.value_type))
            if target_type is None:
                raise TypeError(f"{schema.cls.__name__}.{field.name}: "
                                "unsupported dynamic scalar type")
            argument = supplied.get(field.name)
            if argument is not None:
                value = context.cast(builder, argument[1], argument[0],
                                     target_type)
            else:
                default = field.default
                if default is None:
                    default = nan if field.kind == "output" and target_type == types.float64 else 0
                value = context.get_constant(
                    target_type, int(default) if target_type == types.int64 else default)
        builder.store(value, _field_pointer(
            context, builder, record, layout.offset(field.name), target_type))
    start_type = ir.FunctionType(ir.VoidType(), [pointer, pointer, uintptr,
                                                  pointer, uintptr, pointer,
                                                  uintptr, pointer, uintptr])
    start = cgutils.get_or_insert_function(builder.module, start_type,
                                           "cpy_model_start")
    builder.call(start, [record,
        _constant_pointer(builder, compiled.dynamic_processes.ctypes.data),
        ir.Constant(uintptr, len(compiled.dynamic_processes)),
        _constant_pointer(builder, compiled.dynamic_starts.ctypes.data),
        ir.Constant(uintptr, len(compiled.dynamic_starts)),
        _constant_pointer(builder, compiled.dynamic_ends.ctypes.data),
        ir.Constant(uintptr, len(compiled.dynamic_ends)),
        _constant_pointer(builder, compiled.dynamic_inputs.ctypes.data),
        ir.Constant(uintptr, len(compiled.dynamic_inputs)),
    ])
    return builder.bitcast(record, context.get_value_type(sig.return_type))
