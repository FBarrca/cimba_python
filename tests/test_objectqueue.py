import cimba


def test_objectqueue_blocking_order_and_identity():
    log = []

    def getter(me, queue):
        sig, got = queue.get()
        log.append((cimba.time(), sig, got))

    def putter(me, queue):
        cimba.hold(1.0)
        assert queue.put("fifo-item") == cimba.SUCCESS

    with cimba.Simulation(seed=1) as sim:
        queue = cimba.ObjectQueue("FIFO")
        queue.start_recording()
        cimba.Process("Getter", getter, queue).start()
        cimba.Process("Putter", putter, queue).start()
        sim.execute()
        queue.stop_recording()
        history = queue.history()

    assert log == [(1.0, cimba.SUCCESS, "fifo-item")]
    assert history.count >= 2


def test_objectqueue_position_and_interrupted_get():
    log = []

    def getter(me, queue):
        log.append(queue.get())

    def interrupter(me, target):
        cimba.hold(1.0)
        target.interrupt(55)

    with cimba.Simulation(seed=1) as sim:
        queue = cimba.ObjectQueue("FIFO", capacity=2)
        obj = object()
        assert queue.put(obj) == cimba.SUCCESS
        assert queue.position(obj) == 1
        assert queue.get() == (cimba.SUCCESS, obj)

        target = cimba.Process("Getter", getter, queue).start()
        cimba.Process("Interrupter", interrupter, target).start()
        sim.execute()

    assert log == [(55, None)]


def test_objectqueue_interrupted_put_on_full_queue_leaves_existing_item():
    log = []

    def putter(me, queue):
        sig = queue.put("second")
        log.append(("put", cimba.time(), sig, queue.length))

    def interrupter(me, target):
        cimba.hold(1.0)
        target.interrupt(66)

    with cimba.Simulation(seed=1) as sim:
        queue = cimba.ObjectQueue("FIFO", capacity=1)
        assert queue.put("first") == cimba.SUCCESS
        target = cimba.Process("Putter", putter, queue).start()
        cimba.Process("Interrupter", interrupter, target).start()
        sim.execute()

    assert log == [("put", 1.0, 66, 1)]
