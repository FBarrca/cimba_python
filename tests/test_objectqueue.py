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
        cimba.Process("Getter", getter, queue).start()
        cimba.Process("Putter", putter, queue).start()
        sim.execute()

    assert log == [(1.0, cimba.SUCCESS, "fifo-item")]
