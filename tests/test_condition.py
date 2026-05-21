import cimba


def test_condition_wait_signal_and_spurious_recheck_pattern():
    state = {"ready": False}
    log = []

    def waiter(me, condition):
        while not state["ready"]:
            sig = condition.wait(lambda process, ctx: ctx["ready"], state)
            assert sig == cimba.SUCCESS
        log.append(("ready", cimba.time(), me.name))

    def signaler(me, condition):
        cimba.hold(1.0)
        assert condition.signal() == 0
        cimba.hold(1.0)
        state["ready"] = True
        assert condition.signal() == 1

    with cimba.Simulation(seed=1) as sim:
        condition = cimba.Condition("Ready")
        cimba.Process("Waiter", waiter, condition).start()
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

    def controller(me, ctx):
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
        cimba.Process("Ship", ship, ctx).start()
        cimba.Process("Controller", controller, ctx).start()
        sim.execute()

    assert log == [("docked", 2.0, 2, 1)]
