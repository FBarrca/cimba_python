"""Physical record and trial-image plans derived from logical assemblies."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any, get_args, get_origin

import numpy as np

from cimba.engine.abi import (
    INPUT_SLOT, INPUT_SLOT_BOOL, INPUT_SLOT_INT, POINTER, TRIAL_HEADER,
)
from cimba.modeling import Condition, Container, Dataset, Model, PriorityStore, Resource, Store
from cimba.schema import Assembly, ClassSchema


def _scalar_dtype(annotation: Any) -> Any:
    if annotation is int:
        return np.int64
    if annotation is bool:
        return np.bool_
    if annotation is float:
        return np.float64
    raise TypeError(f"native records support scalar float, int and bool, got {annotation!r}")


CONTAINER_HANDLE = np.dtype([("container_handle", POINTER)], align=True)
RESOURCE_HANDLE = np.dtype([("resource_handle", POINTER)], align=True)
DATASET_HANDLE = np.dtype([("dataset_handle", POINTER)], align=True)
CONDITION_HANDLE = np.dtype([("condition_handle", POINTER)], align=True)
STORE_INT_HANDLE = np.dtype([("store_int_handle", POINTER)], align=True)
STORE_FLOAT_HANDLE = np.dtype([("store_float_handle", POINTER)], align=True)
PRIORITY_INT_HANDLE = np.dtype([("priority_int_handle", POINTER)], align=True)
PRIORITY_FLOAT_HANDLE = np.dtype([("priority_float_handle", POINTER)], align=True)
STORE_MODEL_TARGETS: dict[str, type[Model]] = {}
MODEL_RECORD_CLASSES: dict[str, type[Model]] = {}


def _model_store_handle(model_class: type[Model], priority: bool) -> np.dtype:
    marker = f"__cimba_store_target_{id(model_class):x}"
    STORE_MODEL_TARGETS[marker] = model_class
    name = "priority_model_handle" if priority else "store_model_handle"
    return np.dtype([((marker, name), POINTER)], align=True)


@dataclass(frozen=True)
class RecordLayout:
    schema: ClassSchema
    dtype: np.dtype

    @classmethod
    @lru_cache(maxsize=512)
    def of(cls, schema: ClassSchema) -> "RecordLayout":
        # The title is a zero-cost type tag: distinct model classes can have
        # identical field shapes while requiring different child view types.
        marker = f"__cimba_class_{id(schema.cls):x}"
        MODEL_RECORD_CLASSES[marker] = schema.cls
        members: list[tuple] = [((marker, "class_descriptor"), POINTER)]
        for field in schema.fields:
            if field.kind in {"param", "state", "output", "constant"}:
                members.append((field.name, _scalar_dtype(field.value_type)))
            elif field.kind in {"input", "series"}:
                handle = {float: INPUT_SLOT, int: INPUT_SLOT_INT,
                          bool: INPUT_SLOT_BOOL}.get(field.value_type)
                if handle is None:
                    raise TypeError(f"{schema.cls.__name__}.{field.name}: "
                                    "Input and Series require float, int or bool")
                members.append((field.name, handle))
            elif field.kind == "collection":
                members.append((field.name, POINTER))
                members.append((f"{field.name}_length", np.uint64))
            elif field.kind == "entity" and field.value_type is Container:
                members.append((field.name, CONTAINER_HANDLE))
            elif field.kind == "entity" and field.value_type is Resource:
                members.append((field.name, RESOURCE_HANDLE))
            elif field.kind == "entity" and field.value_type is Dataset:
                members.append((field.name, DATASET_HANDLE))
            elif field.kind == "entity" and field.value_type is Condition:
                members.append((field.name, CONDITION_HANDLE))
            elif field.kind == "entity" and get_origin(field.value_type) in (
                Store, PriorityStore
            ):
                entity_type = get_origin(field.value_type)
                arguments = get_args(field.value_type)
                if len(arguments) != 1 or not (
                    arguments[0] in (int, float) or
                    isinstance(arguments[0], type) and
                    issubclass(arguments[0], Model)
                ):
                    raise TypeError(f"{schema.cls.__name__}.{field.name}: "
                                    "Store value type must be int, float or Model")
                if arguments[0] in (int, float):
                    handle = {
                        (Store, int): STORE_INT_HANDLE,
                        (Store, float): STORE_FLOAT_HANDLE,
                        (PriorityStore, int): PRIORITY_INT_HANDLE,
                        (PriorityStore, float): PRIORITY_FLOAT_HANDLE,
                    }[(entity_type, arguments[0])]
                else:
                    model_class = arguments[0]
                    assert isinstance(model_class, type) and issubclass(model_class, Model)
                    handle = _model_store_handle(
                        model_class, entity_type is PriorityStore)
                members.append((field.name, handle))
            else:
                members.append((field.name, POINTER))
        record = cls(schema, np.dtype(members, align=True))
        # Base-typed views load fields by the base's offsets and scalar types.
        # Reject redeclarations or multiple inheritance that would break that
        # contract rather than silently reading the wrong bytes.
        fields = record.dtype.fields
        assert fields is not None
        for base in schema.cls.__mro__[1:]:
            if not issubclass(base, Model) or base is Model:
                continue
            inherited = cls.of(ClassSchema.of(base)).dtype
            assert inherited.names is not None and inherited.fields is not None
            for name in inherited.names[1:]:
                if fields[name][:2] != inherited.fields[name][:2]:  # pyright: ignore[reportArgumentType]
                    raise TypeError(f"{schema.cls.__name__}.{name}: native layout "
                                    f"must preserve the field declared by {base.__name__}")
        return record

    def offset(self, field_name: str) -> int:
        fields = self.dtype.fields
        assert fields is not None
        return fields[field_name][1]  # pyright: ignore[reportArgumentType]


@dataclass(frozen=True)
class Placement:
    model: Any
    label: str
    offset: int
    record: RecordLayout


@dataclass(frozen=True)
class TrialImageLayout:
    assembly: Assembly
    placements: tuple[Placement, ...]
    by_object: MappingProxyType
    size: int

    @classmethod
    def of(cls, assembly: Assembly) -> "TrialImageLayout":
        offset = TRIAL_HEADER.itemsize
        placements: list[Placement] = []
        index: dict[Any, Placement] = {}
        for instance in assembly.instances:
            record = RecordLayout.of(instance.schema)
            alignment = record.dtype.alignment
            offset = (offset + alignment - 1) // alignment * alignment
            placement = Placement(instance.model, instance.label, offset, record)
            placements.append(placement)
            index[instance.model] = placement
            offset += record.dtype.itemsize
        return cls(assembly, tuple(placements), MappingProxyType(index), offset)
