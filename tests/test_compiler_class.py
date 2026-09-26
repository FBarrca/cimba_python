"""Compile model methods once per class with native typed record views."""

from cimba import inputs
from cimba.compiler import ensure
from cimba.modeling import Container, Input, Model, Output, hold, on_end, process
from cimba.schema import ClassSchema


class Queue(Model):
    gaps: Input[float] = inputs.dist.exponential(mean=2.0)
    buffer: Container
    mean: Output[float]

    @process
    def arrivals(self):
        for _ in range(4):
            hold(self.gaps.next())
            self.buffer.put(1)

    @on_end
    def measure(self):
        self.mean = self.buffer.mean_level()


def test_compile_is_per_class_and_source_agnostic():
    schema = ClassSchema.of(Queue)
    compiled = ensure(schema)
    assert compiled.processes["arrivals"].address
    assert compiled.ends["measure"].address
    assert ensure(schema) is compiled
