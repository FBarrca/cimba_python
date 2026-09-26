"""Host-side modeling language: declarations and configuration only."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from typing import Any, Generic, TypeVar, get_args, get_origin, get_type_hints

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
    if not values:
        raise ValueError("sweep needs at least one value")
    return Sweep(tuple(values), next(_axis_ids))


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
    def __init__(self, initial: float = 0.0):
        super().__init__()
        self.initial = float(initial)

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
    def wait_until(self, predicate) -> None:
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


def hold(duration: float) -> int:
    raise NotInCompiledCode("hold() is available only in compiled model code")


def now() -> float:
    raise NotInCompiledCode("now() is available only in compiled model code")


def suspend() -> int:
    raise NotInCompiledCode("suspend() is available only in compiled model code")


def spawn(*args, **kwargs):
    raise NotInCompiledCode("spawn() is available only in compiled model code")


def release(model: Model) -> None:
    raise NotInCompiledCode("release() is available only in compiled model code")


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
        origin = get_origin(declaration) or declaration
        if origin is Param:
            pass
        elif origin in (Input, Series):
            if not isinstance(value, (Source, Sweep)):
                raise TypeError(f"{type(self).__name__}.{name} expects an input source or sweep")
            if isinstance(value, Sweep) and not all(isinstance(x, Source) for x in value.values):
                raise TypeError(f"{type(self).__name__}.{name} sweep must contain sources")
        elif isinstance(value, Sweep):
            raise TypeError(f"{type(self).__name__}.{name} is not a Param or input")
        object.__setattr__(self, name, value)
