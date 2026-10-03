"""Host-side modeling language: declarations and configuration only."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from math import inf
from types import UnionType
from typing import Any, Generic, TypeVar, Union, cast, get_args, get_origin, get_type_hints

T = TypeVar("T")
M = TypeVar("M", bound="Model")
_axis_ids = count()


class NotInCompiledCode(RuntimeError):
    pass


type Param[T] = T
type State[T] = T
type Output[T] = T
type Ref[M] = M


class Input(Generic[T]):
    """A sequence input. In model code, consume it with ``next()``."""

    def next(self) -> T:
        raise NotInCompiledCode("Input.next() is available in compiled model code")

    def remaining(self) -> int:
        raise NotInCompiledCode("Input.remaining() is available in compiled model code")


class Series(Generic[T]):
    """A time-indexed input. ``step`` and ``origin`` define its buckets."""

    def __init__(self, *, step: float, origin: float = 0.0):
        if step <= 0:
            raise ValueError("Series.step must be positive")
        self.step = float(step)
        self.origin = float(origin)

    def now(self) -> T:
        raise NotInCompiledCode("Series.now() is available in compiled model code")

    def at(self, time: float) -> T:
        raise NotInCompiledCode("Series.at() is available in compiled model code")


@dataclass(frozen=True, eq=False)
class Sweep(Generic[T]):
    values: tuple[T, ...]
    axis: int

    def map(self, function):
        return Sweep(tuple(function(x) for x in self.values), self.axis)


def sweep(*values: T) -> Sweep[T]:
    """An independent design axis: ``sweep(a, b, c)`` or ``sweep([a, b, c])``."""
    if len(values) == 1 and isinstance(values[0], (list, tuple)):
        values = tuple(values[0])
    if not values:
        raise ValueError("sweep needs at least one value")
    return Sweep(tuple(values), next(_axis_ids))


def options(value: T | Sweep[T]) -> tuple[T, ...]:
    """Return a sweep's options, or a one-item tuple for an ordinary value."""
    return value.values if isinstance(value, Sweep) else (value,)


def sweeps(*columns):
    if not columns or any(not column for column in columns):
        raise ValueError("sweeps needs nonempty columns")
    lengths = {len(column) for column in columns}
    if len(lengths) != 1:
        raise ValueError("linked sweeps must have equal lengths")
    axis = next(_axis_ids)
    return tuple(Sweep(tuple(column), axis) for column in columns)


class Entity:
    def __init__(self):
        self.captured = False

    def capture(self) -> None:
        self.captured = True


class Container(Entity):
    """A counted level. ``initial`` units are present when each trial starts."""

    def __init__(self, initial: int = 0):
        if isinstance(initial, bool) or int(initial) != initial or initial < 0:
            raise ValueError("Container.initial must be a nonnegative integer")
        super().__init__()
        self.initial = int(initial)

    def put(self, amount: float) -> None:
        raise NotInCompiledCode("Container.put() requires compiled model code")

    def get(self, amount: float) -> None:
        raise NotInCompiledCode("Container.get() requires compiled model code")

    def mean_level(self) -> float:
        raise NotInCompiledCode("Container.mean_level() requires compiled model code")

    def max_level(self) -> float:
        raise NotInCompiledCode("Container.max_level() requires compiled model code")

    def level(self) -> int:
        raise NotInCompiledCode("Container.level() requires compiled model code")


class Store(Entity, Generic[T]):
    def __init__(self, *, capacity: int = 1_000_000_000):
        if capacity < 1:
            raise ValueError("Store.capacity must be positive")
        super().__init__()
        self.capacity = capacity

    def put(self, value: T) -> None:
        raise NotInCompiledCode("Store.put() requires compiled model code")

    def get(self) -> T:
        raise NotInCompiledCode("Store.get() requires compiled model code")

    def length(self) -> int:
        raise NotInCompiledCode("Store.length() requires compiled model code")


class PriorityStore(Store[T]):
    def put(self, value: T, priority: int = 0) -> None:
        raise NotInCompiledCode("PriorityStore.put() requires compiled model code")

    def enqueue(self, value: T, priority: int = 0) -> int:
        raise NotInCompiledCode("PriorityStore.enqueue() requires compiled model code")

    def cancel(self, handle: int) -> bool:
        raise NotInCompiledCode("PriorityStore.cancel() requires compiled model code")

    def position(self, handle: int) -> int:
        raise NotInCompiledCode("PriorityStore.position() requires compiled model code")


class Resource(Entity):
    def __init__(self, *, capacity: int = 1):
        if capacity < 1:
            raise ValueError("Resource.capacity must be positive")
        super().__init__()
        self.capacity = capacity

    def acquire(self, amount: int = 1) -> int:
        raise NotInCompiledCode("Resource.acquire() requires compiled model code")

    def preempt(self, amount: int = 1) -> int:
        raise NotInCompiledCode("Resource.preempt() requires compiled model code")

    def held(self, process: "Process") -> int:
        raise NotInCompiledCode("Resource.held() requires compiled model code")

    def release(self, amount: int = 1) -> None:
        raise NotInCompiledCode("Resource.release() requires compiled model code")

    def mean_in_use(self) -> float:
        raise NotInCompiledCode("Resource.mean_in_use() requires compiled model code")

    def available(self) -> int:
        raise NotInCompiledCode("Resource.available() requires compiled model code")


class Condition(Entity):
    def wait_until(self, predicate, timeout: float = inf) -> bool:
        """Wait for a predicate, returning whether it holds on return.

        A finite timeout bounds the wait in simulation time. Zero polls the
        predicate without blocking; infinity waits without a deadline.
        """
        raise NotInCompiledCode("Condition.wait_until() requires compiled model code")

    def signal(self) -> None:
        raise NotInCompiledCode("Condition.signal() requires compiled model code")


class Dataset(Entity):
    def record(self, value: float) -> None:
        raise NotInCompiledCode("Dataset.record() requires compiled model code")

    def sample_mean(self) -> float:
        raise NotInCompiledCode("Dataset.sample_mean() requires compiled model code")

    def sample_count(self) -> int:
        raise NotInCompiledCode("Dataset.sample_count() requires compiled model code")

    def sample_max(self) -> float:
        raise NotInCompiledCode("Dataset.sample_max() requires compiled model code")


class Process:
    pointer: int

    def __init__(self, pointer: int):
        raise NotInCompiledCode("Process handles are available only in compiled model code")

    def status(self) -> int:
        raise NotInCompiledCode("Process.status() requires compiled model code")

    def timer_set(self, delay: float, signal: int) -> int:
        raise NotInCompiledCode("Process.timer_set() requires compiled model code")

    def resume(self, signal: int = 0) -> None:
        raise NotInCompiledCode("Process.resume() requires compiled model code")

    def interrupt(self, signal: int = -1, priority: int = 0) -> None:
        raise NotInCompiledCode("Process.interrupt() requires compiled model code")

    def priority_set(self, priority: int) -> None:
        raise NotInCompiledCode("Process.priority_set() requires compiled model code")

    def timers_clear(self) -> None:
        raise NotInCompiledCode("Process.timers_clear() requires compiled model code")


class Scheduled:
    def cancel(self) -> bool:
        raise NotInCompiledCode("Scheduled.cancel() requires compiled model code")

    def pending(self) -> bool:
        raise NotInCompiledCode("Scheduled.pending() requires compiled model code")


def process(function=None, *, copies: int = 1, priority: int = 0):
    if copies < 1:
        raise ValueError("process copies must be positive")

    def decorate(method):
        method.cimba_role = "process"
        method.cimba_copies = copies
        method.cimba_priority = priority
        return method

    return decorate(function) if function is not None else decorate


def _role(name):
    def decorate(method):
        method.cimba_role = name
        return method
    return decorate


on_start = _role("on_start")
on_end = _role("on_end")
predicate = _role("predicate")
event = _role("event")


def function(method):
    """Make a method callable from compiled model code, with dynamic dispatch.

    Parameters and the return value must be annotated with ``float``, ``int``,
    ``bool`` or a model class (``None`` for no return value). Subclasses may
    override it with the same signature; calls through a base-typed reference
    or list run the implementation of the model's actual class. On the host
    the method stays an ordinary Python method.
    """
    method.cimba_role = "function"
    return method


def hold(duration: float) -> int:
    raise NotInCompiledCode("hold() is available only in compiled model code")


def now() -> float:
    raise NotInCompiledCode("now() is available only in compiled model code")


def suspend() -> int:
    raise NotInCompiledCode("suspend() is available only in compiled model code")


def spawn(*args, **kwargs):
    raise NotInCompiledCode("spawn() is available only in compiled model code")


def release(model: Model) -> None:
    """Stop this model's processes, whether it is static or spawned.

    Releasing the caller's own model ends the calling process. Static records,
    children and end hooks remain; spawned models are retired as before.
    """
    raise NotInCompiledCode("release() is available only in compiled model code")


def is_dynamic(model: Model) -> bool:
    """Whether this instance was spawned, even if it has been released.

    Available only in compiled model code. Static tree instances return False.
    This reports creation mode, not whether the model is still active.
    """
    raise NotInCompiledCode("is_dynamic() is available only in compiled model code")


def schedule(event: Any, delay: float, priority: int = 0) -> Scheduled:
    raise NotInCompiledCode("schedule() is available only in compiled model code")


def this_process() -> Process:
    raise NotInCompiledCode("this_process() is available only in compiled model code")


def log(flags: int, message: str, value: float | None = None) -> None:
    raise NotInCompiledCode("log() is available only in compiled model code")


def end_trial() -> None:
    raise NotInCompiledCode("end_trial() is available only in compiled model code")


class ModelMeta(type):
    """Marker used by the compiler for static model-class arguments."""


class Model(metaclass=ModelMeta):
    """A configured model node; constructing it does not compile or run code."""

    def attached(self, owner: Model, field: str) -> None:
        """Host hook called after this model is assigned to an owned field.

        Each child sweep option and each owned list item receives the same
        owner and field name. Override to bind a reference back to the owner.
        """

    def describe(self):
        from cimba.schema import Assembly
        return tuple({"label": instance.label,
                      "class": instance.schema.cls.__name__,
                      "fields": tuple((field.name, field.kind)
                                      for field in instance.schema.fields)}
                     for instance in Assembly.of(self).instances)

    def __setattr__(self, name: str, value: Any) -> None:
        from cimba.inputs.sources import Source

        annotations = {}
        for cls in reversed(type(self).__mro__):
            if cls is Model or not issubclass(cls, Model):
                continue
            try:
                annotations.update(get_type_hints(cls))
            except (NameError, AttributeError):
                annotations.update(getattr(cls, "__annotations__", {}))
        declaration = annotations.get(name)
        optional = False
        if get_origin(declaration) in (UnionType, Union):
            members = [member for member in get_args(declaration)
                       if member is not type(None)]
            if len(members) == 1:
                optional = len(members) != len(get_args(declaration))
                declaration = members[0]
        origin = get_origin(declaration) or declaration
        children: tuple[Model, ...] = ()
        if origin is Param:
            pass
        elif isinstance(origin, type) and issubclass(origin, Model):
            if value is not None or not optional:
                candidates = options(value)
                if not all(isinstance(child, origin) for child in candidates):
                    message = (f"sweep must contain {origin.__name__} models"
                               if isinstance(value, Sweep) else
                               f"expects a {origin.__name__} child")
                    raise TypeError(f"{type(self).__name__}.{name} {message}")
                children = cast(tuple[Model, ...], candidates)
        elif origin is list and get_args(declaration) and (
            isinstance(get_args(declaration)[0], type) and
            issubclass(get_args(declaration)[0], Model)
        ):
            expected = get_args(declaration)[0]
            if not isinstance(value, list) or not all(
                isinstance(child, expected) for child in value
            ):
                raise TypeError(f"{type(self).__name__}.{name} expects a list of "
                                f"{expected.__name__} models")
            children = tuple(value)
        elif origin in (Input, Series):
            if origin is Input and optional and value is None:
                object.__setattr__(self, name, value)
                return
            if not isinstance(value, (Source, Sweep)):
                raise TypeError(f"{type(self).__name__}.{name} expects an input source or sweep")
            if isinstance(value, Sweep) and not all(
                isinstance(x, Source) or (origin is Input and optional and x is None)
                for x in value.values
            ):
                raise TypeError(f"{type(self).__name__}.{name} sweep must contain sources")
        elif isinstance(value, Sweep):
            raise TypeError(f"{type(self).__name__}.{name} is not a Param, input or child model")
        object.__setattr__(self, name, value)
        for child in children:
            child.attached(self, name)
