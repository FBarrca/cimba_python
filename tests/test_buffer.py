import cimba


def test_buffer_partial_get_put_and_recorded_history():
    log = []

    def getter(me, queue):
        sig, got = queue.get(5)
        log.append(("got", cimba.time(), sig, got, queue.level))

    def putter(me, queue):
        cimba.hold(1.0)
        assert queue.put(3) == (cimba.SUCCESS, 0)
        cimba.hold(1.0)
        assert queue.put(2) == (cimba.SUCCESS, 0)

    with cimba.Simulation(seed=1) as sim:
        queue = cimba.Buffer("Buf", capacity=10)
        queue.start_recording()
        cimba.Process("Getter", getter, queue).start()
        cimba.Process("Putter", putter, queue).start()
        sim.execute()
        queue.stop_recording()
        history = queue.history()

    assert log == [("got", 2.0, cimba.SUCCESS, 5, 0)]
    assert history.count >= 2
    assert history.summary().mean >= 0.0


def test_buffer_interrupted_get_returns_partial_amount():
    log = []

    def getter(me, queue):
        sig, got = queue.get(5)
        log.append(("get", cimba.time(), sig, got, queue.level))

    def putter(me, queue):
        cimba.hold(1.0)
        assert queue.put(3) == (cimba.SUCCESS, 0)

    def interrupter(me, target):
        cimba.hold(2.0)
        target.interrupt(42)

    with cimba.Simulation(seed=1) as sim:
        queue = cimba.Buffer("Buf", capacity=10)
        target = cimba.Process("Getter", getter, queue).start()
        cimba.Process("Putter", putter, queue).start()
        cimba.Process("Interrupter", interrupter, target).start()
        sim.execute()

    assert log == [("get", 2.0, 42, 3, 0)]


def test_buffer_interrupted_put_returns_remaining_amount():
    log = []

    def putter(me, queue):
        assert queue.put(5) == (cimba.SUCCESS, 0)
        sig, remaining = queue.put(4)
        log.append(("put", cimba.time(), sig, remaining, queue.level))

    def getter(me, queue):
        cimba.hold(1.0)
        assert queue.get(2) == (cimba.SUCCESS, 2)

    def interrupter(me, target):
        cimba.hold(2.0)
        target.interrupt(88)

    with cimba.Simulation(seed=1) as sim:
        queue = cimba.Buffer("Buf", capacity=5)
        target = cimba.Process("Putter", putter, queue).start()
        cimba.Process("Getter", getter, queue).start()
        cimba.Process("Interrupter", interrupter, target).start()
        sim.execute()

    assert log == [("put", 2.0, 88, 2, 5)]


def test_buffer_waiter_stop_does_not_leave_stale_native_waiter():
    log = []

    def getter(me, queue):
        try:
            queue.get()
        finally:
            log.append(("cancelled", cimba.time()))

    def stopper(me, target):
        cimba.hold(1.0)
        assert target.stop() == cimba.SUCCESS

    def putter(me, queue):
        cimba.hold(2.0)
        assert queue.put(1) == (cimba.SUCCESS, 0)
        log.append(("put", cimba.time(), queue.level))

    with cimba.Simulation(seed=1) as sim:
        queue = cimba.Buffer("Buf", capacity=1)
        target = cimba.Process("Getter", getter, queue).start()
        cimba.Process("Stopper", stopper, target).start()
        cimba.Process("Putter", putter, queue).start()
        sim.execute()

    assert log == [("cancelled", 1.0), ("put", 2.0, 1)]
