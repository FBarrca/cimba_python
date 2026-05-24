import cimba


def test_condition_wait_signal_and_spurious_recheck_pattern():
    state = {"ready": False}
    log = []

    def waiter(me, condition):
        while not state["ready"]:
            sig = condition.wait(lambda process, ctx: ctx["ready"], state)
            assert sig == cimba.SUCCESS
        log.append(("ready", cimba.time(), me.name))

    def signaler(condition):
        cimba.hold(1.0)
        assert condition.signal() == 0
        cimba.hold(1.0)
        state["ready"] = True
        assert condition.signal() == 1

    with cimba.Simulation(seed=1) as sim:
        condition = cimba.Condition("Ready")
        cimba.Process("Waiter", waiter, condition, pass_process=True).start()
        cimba.Process("Signaler", signaler, condition).start()
        sim.execute()

    assert log == [("ready", 2.0, "Waiter")]


def test_condition_predicate_can_gate_on_state_and_resources():
    state = {"wind": 20.0, "depth": 5.0}
    log = []

    def is_ready(process, ctx):
        assert process is not None
        return (
            ctx["depth"] >= 10.0
            and ctx["wind"] <= 5.0
            and ctx["tugs"].available >= 2
            and ctx["berths"].available >= 1
        )

    def ship(me, ctx):
        while not is_ready(me, ctx):
            assert ctx["condition"].wait(is_ready, ctx) == cimba.SUCCESS

        assert ctx["berths"].acquire(1) == cimba.SUCCESS
        assert ctx["tugs"].acquire(2) == cimba.SUCCESS
        log.append(("docked", cimba.time(), ctx["tugs"].held_by(me), ctx["berths"].held_by(me)))
        ctx["tugs"].release(2)
        ctx["berths"].release(1)

    def controller(ctx):
        cimba.hold(1.0)
        assert ctx["condition"].signal() == 0
        cimba.hold(1.0)
        ctx["wind"] = 3.0
        ctx["depth"] = 12.0
        assert ctx["condition"].signal() == 1

    with cimba.Simulation(seed=1) as sim:
        ctx = {
            "condition": cimba.Condition("Harbormaster"),
            "tugs": cimba.ResourcePool("Tugs", capacity=2),
            "berths": cimba.ResourcePool("Berths", capacity=1),
            **state,
        }
        cimba.Process("Ship", ship, ctx, pass_process=True).start()
        cimba.Process("Controller", controller, ctx).start()
        sim.execute()

    assert log == [("docked", 2.0, 2, 1)]


def test_condition_waiter_stop_does_not_leave_stale_native_waiter():
    state = {"ready": False}
    log = []

    def waiter(condition):
        try:
            condition.wait(lambda process, ctx: ctx["ready"], state)
        finally:
            log.append(("cancelled", cimba.time()))

    def stopper(target):
        cimba.hold(1.0)
        assert target.stop() == cimba.SUCCESS

    def signaler(condition):
        cimba.hold(2.0)
        state["ready"] = True
        log.append(("signalled", cimba.time(), condition.signal()))

    with cimba.Simulation(seed=1) as sim:
        condition = cimba.Condition("Ready")
        target = cimba.Process("Waiter", waiter, condition).start()
        cimba.Process("Stopper", stopper, target).start()
        cimba.Process("Signaler", signaler, condition).start()
        sim.execute()

    assert log == [("cancelled", 1.0), ("signalled", 2.0, 0)]


def test_condition_subscribe_resource_release_wakes_waiter_without_explicit_signal():
    log = []

    def is_available(process, ctx):
        return ctx["resource"].available == 1

    def holder(ctx):
        assert ctx["resource"].acquire() == cimba.SUCCESS
        cimba.hold(1.0)
        ctx["resource"].release()

    def waiter(ctx):
        while not is_available(None, ctx):
            assert ctx["condition"].wait(is_available, ctx) == cimba.SUCCESS
        log.append(("available", cimba.time()))

    with cimba.Simulation(seed=1) as sim:
        ctx = {
            "resource": cimba.Resource("Dock"),
            "condition": cimba.Condition("Dock available"),
        }
        ctx["condition"].subscribe(ctx["resource"])
        cimba.Process("Holder", holder, ctx).start()
        cimba.Process("Waiter", waiter, ctx).start()
        sim.execute()

    assert log == [("available", 1.0)]


def test_condition_subscribe_is_idempotent_and_unsubscribe_removes_observer():
    log = []

    def is_available(process, ctx):
        return ctx["resource"].available == 1

    def holder(ctx):
        assert ctx["resource"].acquire() == cimba.SUCCESS
        cimba.hold(1.0)
        ctx["resource"].release()

    def waiter(ctx):
        while not is_available(None, ctx):
            assert ctx["condition"].wait(is_available, ctx) == cimba.SUCCESS
        log.append(("available", cimba.time()))

    def signaler(ctx):
        cimba.hold(2.0)
        assert ctx["condition"].signal() == 1

    with cimba.Simulation(seed=1) as sim:
        ctx = {
            "resource": cimba.Resource("Dock"),
            "condition": cimba.Condition("Dock available"),
        }
        ctx["condition"].subscribe(ctx["resource"])
        ctx["condition"].subscribe(ctx["resource"])
        assert ctx["condition"].unsubscribe(ctx["resource"]) == 1
        cimba.Process("Holder", holder, ctx).start()
        cimba.Process("Waiter", waiter, ctx).start()
        cimba.Process("Signaler", signaler, ctx).start()
        sim.execute()

    assert log == [("available", 2.0)]


def test_condition_cancel_wakes_waiter_with_cancelled_signal():
    state = {"ready": False}
    log = []

    def waiter(condition):
        sig = condition.wait(lambda process, ctx: ctx["ready"], state)
        log.append(("waiter", cimba.time(), sig))

    def canceller(ctx):
        cimba.hold(1.0)
        assert ctx["condition"].cancel(ctx["target"])

    with cimba.Simulation(seed=1) as sim:
        condition = cimba.Condition("Ready")
        target = cimba.Process("Waiter", waiter, condition).start()
        cimba.Process("Canceller", canceller, {"condition": condition, "target": target}).start()
        sim.execute()

    assert log == [("waiter", 1.0, cimba.CANCELLED)]


def test_condition_remove_unlinks_waiter_without_waking_it():
    state = {"ready": False}
    log = []

    def waiter(condition):
        sig = condition.wait(lambda process, ctx: ctx["ready"], state)
        log.append(("waiter", cimba.time(), sig))

    def remover(ctx):
        cimba.hold(1.0)
        assert ctx["condition"].remove(ctx["target"])
        state["ready"] = True
        assert ctx["condition"].signal() == 0
        ctx["target"].resume(cimba.INTERRUPTED)

    with cimba.Simulation(seed=1) as sim:
        condition = cimba.Condition("Ready")
        target = cimba.Process("Waiter", waiter, condition).start()
        cimba.Process("Remover", remover, {"condition": condition, "target": target}).start()
        sim.execute()

    assert log == [("waiter", 1.0, cimba.INTERRUPTED)]


def test_condition_subscribe_buffer_front_wakes_on_put():
    log = []

    def has_content(process, ctx):
        return ctx["buffer"].level >= 1

    def waiter(ctx):
        while not has_content(None, ctx):
            assert ctx["condition"].wait(has_content, ctx) == cimba.SUCCESS
        log.append(("content", cimba.time()))

    def producer(ctx):
        cimba.hold(1.0)
        assert ctx["buffer"].put(1) == (cimba.SUCCESS, 0)

    with cimba.Simulation(seed=1) as sim:
        ctx = {
            "buffer": cimba.Buffer("Inventory", capacity=1),
            "condition": cimba.Condition("Has content"),
        }
        ctx["condition"].subscribe(ctx["buffer"], on="front")
        cimba.Process("Waiter", waiter, ctx).start()
        cimba.Process("Producer", producer, ctx).start()
        sim.execute()

    assert log == [("content", 1.0)]


def test_condition_subscribe_objectqueue_rear_wakes_on_get_space():
    log = []

    def has_space(process, ctx):
        return ctx["queue"].space >= 1

    def waiter(ctx):
        while not has_space(None, ctx):
            assert ctx["condition"].wait(has_space, ctx) == cimba.SUCCESS
        log.append(("space", cimba.time()))

    def consumer(ctx):
        cimba.hold(1.0)
        assert ctx["queue"].get() == (cimba.SUCCESS, "initial")

    with cimba.Simulation(seed=1) as sim:
        ctx = {
            "queue": cimba.ObjectQueue("Departed", capacity=1),
            "condition": cimba.Condition("Has space"),
        }
        assert ctx["queue"].put("initial") == cimba.SUCCESS
        ctx["condition"].subscribe(ctx["queue"], on="rear")
        cimba.Process("Waiter", waiter, ctx).start()
        cimba.Process("Consumer", consumer, ctx).start()
        sim.execute()

    assert log == [("space", 1.0)]
