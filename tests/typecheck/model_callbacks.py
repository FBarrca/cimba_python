"""Pyright fixture for class-declared model callback inference."""

import cimba.sim as sim


class Visitor(sim.Struct):
    value: int


class Worker(sim.Component):
    handles: sim.Processes
    ready: sim.Predicate
    alarm: sim.Event
    result: sim.Output

    @sim.process(field="handles", struct=Visitor)
    def run(self, env: "Base") -> None:
        self.result = env.adjust(1.0)

    @sim.predicate(field="ready")
    def is_ready(self, env: "Base") -> bool:
        return self.result > 0

    @sim.event(field="alarm")
    def on_alarm(self, env: "Base", data: int) -> None:
        self.result = float(data)


class Base(sim.Model):
    workers: sim.Processes
    ready: sim.Predicate
    alarm: sim.Event
    result: sim.Output
    nested: Worker = Worker()

    @sim.function
    def adjust(self: "Base", value: float) -> float:
        return value + self.result

    @sim.process(copies=3, priority=1, field="workers")
    def worker(self: "Base", index: int) -> None:
        sim.hold(float(index))

    @sim.process(spawnable=True)
    def visitor(self: "Base", state: Visitor) -> None:
        self.result = state.value

    @sim.predicate(field="ready")
    def is_ready(self: "Base") -> bool:
        return self.result > 0

    @sim.event(field="alarm")
    def handle_alarm(self: "Base", data: int) -> None:
        self.result = data

    @sim.process
    def launch(self: "Base") -> None:
        handle: sim.Handle = sim.spawn(self.visitor, self)
        Visitor(handle).value = 1

    @sim.collect
    def stats(self: "Base") -> None:
        self.result = self.adjust(self.result) + 1


class Derived(Base):
    @sim.process
    def launch(self: "Derived") -> None:
        self.alarm.schedule(0.0, 2)
