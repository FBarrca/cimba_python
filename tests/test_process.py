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


def test_stop_at_cooperatively_unwinds_finally_blocks():
    log = []

    def infinite(me, ctx):
        try:
            while True:
                cimba.hold(1.0)
        finally:
            log.append(("finally", cimba.time()))

    with cimba.Simulation(seed=1) as sim:
        cimba.Process("Infinite", infinite).start()
        sim.stop_at(3.0)
        sim.execute()

    assert log == [("finally", 3.0)]


def test_process_exit_unwinds_finally_and_preserves_exit_value():
    log = []

    def worker(me, ctx):
        try:
            cimba.process_exit("done")
        finally:
            log.append(("finally", cimba.time()))

    with cimba.Simulation(seed=1) as sim:
        proc = cimba.Process("Worker", worker).start()
        sim.execute()
        assert proc.exit_value() == "done"

    assert log == [("finally", 0.0)]


def test_simulation_close_cooperatively_cancels_running_processes():
    log = []

    def infinite(me, ctx):
        try:
            while True:
                cimba.hold(1.0)
        finally:
            log.append(("finally", cimba.time()))

    with cimba.Simulation(seed=1) as sim:
        cimba.Process("Infinite", infinite).start()
        assert sim.execute_next() is True

    assert log == [("finally", 0.0)]


def test_stopping_process_waiting_on_process_unwinds_cleanly():
    log = []

    def child(me, ctx):
        while True:
            cimba.hold(1.0)

    def waiter(me, target):
        try:
            target.wait()
        finally:
            log.append(("waiter-cancelled", cimba.time()))

    def stopper(me, target):
        cimba.hold(1.0)
        assert target.stop() == cimba.SUCCESS

    with cimba.Simulation(seed=1) as sim:
        child_proc = cimba.Process("Child", child).start()
        waiter_proc = cimba.Process("Waiter", waiter, child_proc).start()
        cimba.Process("Stopper", stopper, waiter_proc).start()
        sim.stop_at(2.0)
        sim.execute()

    assert log == [("waiter-cancelled", 1.0)]
