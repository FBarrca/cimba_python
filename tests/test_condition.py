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
