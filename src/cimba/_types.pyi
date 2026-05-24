"""Private typing aliases shared by the Cimba stubs."""

from collections.abc import Callable
from typing import Any, TypeAlias

_Amount: TypeAlias = int
_Count: TypeAlias = int
_EventHandle: TypeAlias = int
_LoggerFlags: TypeAlias = int
_Priority: TypeAlias = int
_ProcessSignal: TypeAlias = int
_ProcessStatus: TypeAlias = int
_QueueHandle: TypeAlias = int
_TimerHandle: TypeAlias = int

_ProcessFunc: TypeAlias = Callable[..., object]
_ConditionPredicate: TypeAlias = Callable[["Process | None", Any], bool]

_BufferPutResult: TypeAlias = tuple[_ProcessSignal, _Amount]
_BufferGetResult: TypeAlias = tuple[_ProcessSignal, _Amount]
_ObjectQueueGetResult: TypeAlias = tuple[_ProcessSignal, object | None]
_PriorityQueuePutResult: TypeAlias = tuple[_ProcessSignal, _QueueHandle]
_PriorityQueueGetResult: TypeAlias = tuple[_ProcessSignal, object | None]
_TimeSeriesRow: TypeAlias = tuple[float, float, float]
