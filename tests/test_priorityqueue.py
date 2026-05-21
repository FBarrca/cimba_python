import cimba


def test_priorityqueue_returns_highest_priority_first():
    log = []

    def getter(me, queue):
        log.append(queue.get())
        log.append(queue.get())

    def putter(me, queue):
        assert queue.put("low", priority=0)[0] == cimba.SUCCESS
        assert queue.put("high", priority=10)[0] == cimba.SUCCESS

    with cimba.Simulation(seed=1) as sim:
        queue = cimba.PriorityQueue("Priority")
        cimba.Process("Putter", putter, queue).start()
        cimba.Process("Getter", getter, queue).start()
        sim.execute()

    assert log == [(cimba.SUCCESS, "high"), (cimba.SUCCESS, "low")]


def test_priorityqueue_cancel_releases_item_and_removes_handle():
    with cimba.Simulation(seed=1):
        queue = cimba.PriorityQueue("Priority")
        obj = object()
        sig, handle = queue.put(obj, priority=5)
        assert sig == cimba.SUCCESS
        assert queue.position(handle) == 1
        assert queue.cancel(handle) is True
        assert queue.position(handle) == 0
        assert queue.length == 0
