"""Typed surface for Cimba's cmb_condition module."""

from typing import Literal, Self, TypeAlias

from ._types import _ConditionPredicate, _Count, _ProcessSignal
from .cmb_buffer import Buffer
from .cmb_objectqueue import ObjectQueue
from .cmb_priorityqueue import PriorityQueue
from .cmb_process import Process
from .cmb_resource import Resource
from .cmb_resourcepool import ResourcePool

_ConditionSignalSource: TypeAlias = Resource | ResourcePool | Buffer | ObjectQueue | PriorityQueue
_ConditionSignalSide: TypeAlias = Literal["front", "rear"]

class Condition:
    """Condition variable for arbitrary Python demand predicates."""

    def __init__(self, name: str) -> None:
        """Create a named condition variable."""
        ...

    def wait(
        self,
        predicate: _ConditionPredicate,
        context: object | None = None,
    ) -> _ProcessSignal:
        """Wait until predicate(process, context) returns true.

        The predicate is re-evaluated when the condition is signaled. Processes
        should recheck state after wakeup because another process may have acted
        first.
        """
        ...

    def signal(self) -> _Count:
        """Evaluate waiting predicates and reactivate those that are true."""
        ...

    def subscribe(
        self,
        *sources: _ConditionSignalSource,
        on: _ConditionSignalSide | None = None,
    ) -> Self:
        """Forward native resource signals from sources to this condition."""
        ...

    def unsubscribe(
        self,
        *sources: _ConditionSignalSource,
        on: _ConditionSignalSide | None = None,
    ) -> _Count:
        """Stop forwarding native resource signals from sources to this condition."""
        ...

    def cancel(self, process: Process) -> bool:
        """Remove process from this condition and wake it with CANCELLED."""
        ...

    def remove(self, process: Process) -> bool:
        """Remove process from this condition without waking it."""
        ...

    def close(self) -> None:
        """Destroy the native condition variable."""
        ...
