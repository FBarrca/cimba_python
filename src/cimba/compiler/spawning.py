"""Numba typing and lowering for dynamic model allocation.

The call template reads keyword names directly from Numba's typed call. Model
source is never inspected or rewritten.
"""

from __future__ import annotations

import inspect
from math import nan

from llvmlite import ir
from numba import from_dtype, types

from cimba.engine.symbols import register
from cimba.engine.abi import INPUT_SLOT
from cimba.layout import MODEL_RECORD_CLASSES, RecordLayout
from cimba.modeling import ModelMeta, spawn
from cimba.schema import ClassSchema
from .numba_compat import (
    AbstractTemplate, cgutils, infer_global, lower_builtin, lower_constant,
    models, register_model, signature, typeof_impl,
)
from .views import register_reachable_views

register("cpy_model_allocate")
register("cpy_model_start")
register("cpy_model_input_bind")


class ModelClassType(types.Type):
    def __init__(self, model_class):
        self.model_class = model_class
        super().__init__(name=f"ModelClass[{model_class.__module__}.{model_class.__qualname__}]")

    @property
    def key(self):
        return self.model_class


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
                       if field.kind not in {"param", "state", "output",
                                             "constant", "ref", "input",
                                             "series"}]
        if unsupported:
            raise TypeError(f"{schema.cls.__name__}: dynamic models do not "
                            f"support these fields: {', '.join(unsupported)}")
        unknown = set(kws) - fields.keys()
        if unknown:
            raise TypeError(f"{schema.cls.__name__}: unknown spawn fields: "
                            f"{', '.join(sorted(unknown))}")
        for field in schema.fields:
            if field.kind in {"input", "series"}:
                source_type = kws.get(field.name)
                if source_type is None or source_type != from_dtype(INPUT_SLOT):
                    raise TypeError(f"{schema.cls.__name__}.{field.name}: "
                                    "spawn requires an existing input handle")
            if field.kind in {"state", "param", "constant"}:
                if field.default is None and field.name not in kws:
                    raise TypeError(f"{schema.cls.__name__}.{field.name}: "
                                    "spawn value required")
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
                elif isinstance(reference_type, types.Record):
                    actual = next((model_class for marker, model_class
                                   in MODEL_RECORD_CLASSES.items()
                                   if marker in reference_type.fields), None)
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
            target_type = from_dtype(INPUT_SLOT)
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
        if field.kind == "ref":
            target_type = types.intp
            argument = supplied.get(field.name)
            if argument is None or isinstance(argument[0], types.NoneType):
                value = ir.Constant(uintptr, 0)
            elif isinstance(argument[0], types.Record):
                value = builder.ptrtoint(arguments[
                    tuple(sig.pysig.parameters).index(field.name)], uintptr)
            else:
                raise TypeError(f"{schema.cls.__name__}.{field.name}: "
                                "expected a model reference")
        else:
            target_type = {float: types.float64, int: types.int64,
                           bool: types.boolean}.get(field.value_type)
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
                value = context.get_constant(target_type, default)
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
