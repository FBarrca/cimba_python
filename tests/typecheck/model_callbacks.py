"""Pyright fixture for class-declared model callback inference."""

import cimba.sim as sim


class Visitor(sim.Struct):
    value: int


class Base(sim.Model):
    workers: sim.Processes
    ready: sim.Predicate
    alarm: sim.Event
    result: sim.Output

    @sim.process(copies=3, priority=1, field="workers")
    def worker(env: "Base", index: int) -> None:
        sim.hold(float(index))

    @sim.process(spawnable=True)
    def visitor(env: "Base", state: Visitor) -> None:
        env.result = state.value

    @sim.predicate(field="ready")
    def is_ready(env: "Base") -> bool:
        return env.result > 0

    @sim.event(field="alarm")
    def handle_alarm(env: "Base", data: int) -> None:
        env.result = data

    @sim.process
    def launch(env: "Base") -> None:
        handle: sim.Handle = sim.spawn(env.visitor, env)
        Visitor(handle).value = 1

    @sim.collect
    def stats(env: "Base") -> None:
        env.result += 1


class Derived(Base):
    @sim.process
    def launch(env: "Derived") -> None:
        env.alarm.schedule(0.0, 2)
