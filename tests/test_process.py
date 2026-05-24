from dataclasses import dataclass

import cimba


def test_process_target_accepts_normal_python_arguments():
    log = []

    def zero_arg():
        log.append(("zero", cimba.time()))

    def positional(name, delay):
        cimba.hold(delay)
        log.append(("positional", name, cimba.time()))

    def keyword(*, name, delay):
        cimba.hold(delay)
        log.append(("keyword", name, cimba.time()))

    with cimba.Simulation(seed=1) as sim:
        cimba.Process("Zero", zero_arg).start()
        cimba.Process("Positional", positional, "alpha", 1.0).start()
        cimba.Process("Keyword", keyword, name="bravo", delay=2.0).start()
        sim.execute()

    assert log == [
        ("zero", 0.0),
        ("positional", "alpha", 1.0),
        ("keyword", "bravo", 2.0),
    ]


def test_process_target_can_be_bound_method():
    @dataclass
    class Counter:
        value: int = 0

        def increment_after_hold(self, amount):
            cimba.hold(1.0)
            self.value += amount

    counter = Counter()

    with cimba.Simulation(seed=1) as sim:
        cimba.Process("Increment", counter.increment_after_hold, 3).start()
        sim.execute()

    assert counter.value == 3


def test_process_pass_process_opt_in_receives_running_process():
    log = []

    def worker(me, message):
        assert cimba.current_process() is me
        log.append((me.name, message))

    with cimba.Simulation(seed=1) as sim:
        cimba.Process("Worker", worker, "hello", pass_process=True).start()
        sim.execute()

    assert log == [("Worker", "hello")]


def test_process_timer_interrupt_and_wait_process():
    log = []

    def timed(me):
        me.timer_set(2.0, 123)
        log.append(("yield", cimba.time()))
        sig = cimba.yield_process()
        log.append(("resume", cimba.time(), sig))
        return "timer-done"

    def waiter(target):
        sig = target.wait()
        log.append(("waited", cimba.time(), sig, target.exit_value()))

    with cimba.Simulation(seed=1) as sim:
        target = cimba.Process("Timed", timed, pass_process=True).start()
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

    def target(me):
        assert cimba.current_process() is me
        sig = cimba.yield_process()
        log.append(("target", cimba.time(), sig))
        return "resumed"

    def resumer(target_process):
        cimba.hold(1.0)
        target_process.resume(77)

    def waiter(target_process):
        assert target_process.wait() == cimba.SUCCESS
        log.append(("waiter", cimba.time(), target_process.exit_value()))

    with cimba.Simulation(seed=1) as sim:
        target_process = cimba.Process("Target", target, pass_process=True).start()
        cimba.Process("Resumer", resumer, target_process).start()
        cimba.Process("Waiter", waiter, target_process).start()
        sim.execute()

    assert log == [
        ("target", 1.0, 77),
        ("waiter", 1.0, "resumed"),
    ]


def test_stop_at_cooperatively_unwinds_finally_blocks():
    log = []

    def infinite():
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

    def worker():
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

    def infinite():
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

    def child():
        while True:
            cimba.hold(1.0)

    def waiter(target):
        try:
            target.wait()
        finally:
            log.append(("waiter-cancelled", cimba.time()))

    def stopper(target):
        cimba.hold(1.0)
        assert target.stop() == cimba.SUCCESS

    with cimba.Simulation(seed=1) as sim:
        child_proc = cimba.Process("Child", child).start()
        waiter_proc = cimba.Process("Waiter", waiter, child_proc).start()
        cimba.Process("Stopper", stopper, waiter_proc).start()
        sim.stop_at(2.0)
        sim.execute()

    assert log == [("waiter-cancelled", 1.0)]
