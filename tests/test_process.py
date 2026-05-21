import cimba


def test_process_timer_interrupt_and_wait_process():
    log = []

    def timed(me, ctx):
        me.timer_set(2.0, 123)
        log.append(("yield", cimba.time()))
        sig = cimba.yield_process()
        log.append(("resume", cimba.time(), sig))
        return "timer-done"

    def waiter(me, target):
        sig = target.wait()
        log.append(("waited", cimba.time(), sig, target.exit_value()))

    with cimba.Simulation(seed=1) as sim:
        target = cimba.Process("Timed", timed).start()
        cimba.Process("Waiter", waiter, target).start()
        sim.execute()
        assert target.status == cimba.PROCESS_FINISHED

    assert log == [
        ("yield", 0.0),
        ("resume", 2.0, 123),
        ("waited", 2.0, cimba.SUCCESS, "timer-done"),
    ]


def test_process_resume_and_current_process():
    log = []

    def target(me, ctx):
        assert cimba.current_process() is me
        sig = cimba.yield_process()
        log.append(("target", cimba.time(), sig))
        return "resumed"

    def resumer(me, target_process):
        cimba.hold(1.0)
        target_process.resume(77)

    def waiter(me, target_process):
        assert target_process.wait() == cimba.SUCCESS
        log.append(("waiter", cimba.time(), target_process.exit_value()))

    with cimba.Simulation(seed=1) as sim:
        target_process = cimba.Process("Target", target).start()
        cimba.Process("Resumer", resumer, target_process).start()
        cimba.Process("Waiter", waiter, target_process).start()
        sim.execute()

    assert log == [
        ("target", 1.0, 77),
        ("waiter", 1.0, "resumed"),
    ]
