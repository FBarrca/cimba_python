import gc
import importlib
import weakref

import pytest

import cimba

_native = importlib.import_module("cimba._cimba")


def test_event_queue_start_time_execute_next_and_stop_at():
    log = []

    def worker(ctx):
        log.append(("start", cimba.time(), ctx.now))
        cimba.hold(2.0)
        log.append(("after-hold", cimba.time()))

    with cimba.Simulation(start_time=3.0, seed=1) as sim:
        assert sim.now == 3.0
        assert sim.event_count == 0
        cimba.Process("Worker", worker, sim).start()
        assert sim.event_count == 1

        assert sim.execute_next() is True
        assert log == [("start", 3.0, 3.0)]
        assert sim.now == 3.0
        assert sim.event_count == 1

        handle = sim.stop_at(4.0)
        assert handle != 0
        assert sim.event_count == 2
        sim.execute()
        assert sim.now == 4.0

    assert log == [("start", 3.0, 3.0)]


def test_execute_next_on_empty_queue_returns_false():
    with cimba.Simulation(start_time=7.5, seed=1) as sim:
        assert sim.execute_next() is False
        assert sim.now == 7.5


def test_schedule_python_event_priority_metadata_and_current_event():
    log = []

    with cimba.Simulation(start_time=1.0, seed=1) as sim:
        low = sim.schedule(
            lambda subject, obj: log.append((subject, obj, cimba.time(), sim.current_event)),
            2.0,
            "low",
            "object",
            priority=0,
        )
        high = sim.schedule(
            lambda subject, obj: log.append((subject, obj, cimba.time(), sim.current_event)),
            2.0,
            "high",
            "object",
            priority=10,
        )

        assert sim.event_time(low) == 2.0
        assert sim.event_priority(high) == 10
        assert sim.is_event_scheduled(low) is True
        sim.execute()
        assert sim.current_event == low
        assert sim.is_event_scheduled(low) is False
        with pytest.raises(ValueError):
            sim.event_time(low)

    assert log == [
        ("high", "object", 2.0, high),
        ("low", "object", 2.0, low),
    ]


def test_cancel_reschedule_and_reprioritize_event_handles():
    log = []

    with cimba.Simulation(seed=1) as sim:
        canceled = sim.schedule(lambda subject, obj: log.append(subject), 5.0, "canceled")
        low = sim.schedule(lambda subject, obj: log.append(subject), 3.0, "low", priority=0)
        promoted = sim.schedule(lambda subject, obj: log.append(subject), 3.0, "promoted", priority=0)

        assert sim.reschedule_event(canceled, 2.0) is True
        assert sim.event_time(canceled) == 2.0
        assert sim.cancel_event(canceled) is True
        assert sim.cancel_event(canceled) is False
        assert sim.reschedule_event(canceled, 4.0) is False
        assert sim.reprioritize_event(canceled, 5) is False

        assert sim.reprioritize_event(promoted, 20) is True
        sim.execute()

    assert log == ["promoted", "low"]


def test_python_event_releases_references_after_execution_and_cancel():
    class Payload:
        pass

    def ignore(subject, obj):
        return None

    executed_payload = Payload()
    executed_ref = weakref.ref(executed_payload)
    with cimba.Simulation(seed=1) as sim:
        sim.schedule(ignore, 0.0, executed_payload)
        del executed_payload
        assert executed_ref() is not None
        sim.execute()
    gc.collect()
    assert executed_ref() is None

    canceled_payload = Payload()
    canceled_ref = weakref.ref(canceled_payload)
    with cimba.Simulation(seed=1) as sim:
        handle = sim.schedule(ignore, 1.0, canceled_payload)
        del canceled_payload
        assert canceled_ref() is not None
        assert sim.cancel_event(handle) is True
    gc.collect()
    assert canceled_ref() is None


def test_python_event_exception_propagates_and_clears_queue():
    def boom(subject, obj):
        raise ValueError("event failed")

    with cimba.Simulation(seed=1) as sim:
        sim.schedule(boom, 1.0)
        sim.schedule(lambda subject, obj: None, 2.0)
        with pytest.raises(ValueError, match="event failed"):
            sim.execute()
        assert sim.event_count == 0


def test_schedule_native_event_func_capsule():
    _native._test_event_count_reset()
    with cimba.Simulation(seed=1) as sim:
        handle = sim.schedule_native(_native._test_event_func_capsule(), 1.0)
        assert sim.is_event_scheduled(handle) is True
        sim.execute()

    assert _native._test_event_count() == 1


def test_wait_event_success_and_cancelled_signals():
    log = []

    def event_callback(subject, obj):
        log.append(("event", cimba.time()))

    def waiter(handle):
        sig = cimba.wait_event(handle)
        log.append(("waiter", cimba.time(), sig))

    with cimba.Simulation(seed=1) as sim:
        handle = sim.schedule(event_callback, 2.0)
        cimba.Process("Waiter", waiter, handle).start()
        sim.execute()

    assert log == [
        ("event", 2.0),
        ("waiter", 2.0, cimba.SUCCESS),
    ]

    log.clear()

    def cancel_waiter(handle):
        sig = cimba.wait_event(handle)
        log.append(("cancel-waiter", cimba.time(), sig))

    def canceller(ctx):
        cimba.hold(1.0)
        assert ctx["sim"].cancel_event(ctx["handle"]) is True

    with cimba.Simulation(seed=1) as sim:
        handle = sim.schedule(lambda subject, obj: None, 5.0)
        ctx = {"sim": sim, "handle": handle}
        cimba.Process("CancelWaiter", cancel_waiter, handle).start()
        cimba.Process("Canceller", canceller, ctx).start()
        sim.execute()

    assert log == [("cancel-waiter", 1.0, cimba.CANCELLED)]


def test_event_api_validation():
    with cimba.Simulation(start_time=5.0, seed=1) as sim:
        with pytest.raises(TypeError):
            sim.schedule(object(), 5.0)
        with pytest.raises(ValueError):
            sim.schedule(lambda subject, obj: None, 4.0)
        with pytest.raises(ValueError):
            sim.cancel_event(0)

        handle = sim.schedule(lambda subject, obj: None, 6.0)
        with pytest.raises(ValueError):
            sim.reschedule_event(handle, 4.0)
        with pytest.raises(OverflowError):
            sim.reprioritize_event(handle, 1 << 63)
        assert sim.cancel_event(handle) is True
        with pytest.raises(ValueError):
            sim.event_priority(handle)

        with pytest.raises(TypeError):
            sim.schedule_native(_native._test_trial_func_capsule(), 6.0)

    with pytest.raises(RuntimeError):
        cimba.wait_event(1)
