"""Typed surface for Cimba's cmb_condition module."""

from ._types import _ConditionPredicate, _Count, _ProcessSignal

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

    def close(self) -> None:
        """Destroy the native condition variable."""
        ...
