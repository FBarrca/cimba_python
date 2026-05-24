import cimba


def test_priorityqueue_returns_highest_priority_first():
    log = []

    def getter(queue):
        log.append(queue.get())
        log.append(queue.get())

    def putter(queue):
        assert queue.put("low", priority=0)[0] == cimba.SUCCESS
        assert queue.put("high", priority=10)[0] == cimba.SUCCESS

    with cimba.Simulation(seed=1) as sim:
        queue = cimba.PriorityQueue("Priority")
        queue.start_recording()
        cimba.Process("Putter", putter, queue).start()
        cimba.Process("Getter", getter, queue).start()
        sim.execute()
        queue.stop_recording()
        history = queue.history()

    assert log == [(cimba.SUCCESS, "high"), (cimba.SUCCESS, "low")]
    assert history.count >= 2


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


def test_priorityqueue_reprioritize_changes_get_order():
    with cimba.Simulation(seed=1):
        queue = cimba.PriorityQueue("Priority")
        assert queue.put("first", priority=1)[0] == cimba.SUCCESS
        sig, handle = queue.put("second", priority=0)
        assert sig == cimba.SUCCESS
        queue.reprioritize(handle, 2)
        assert queue.get() == (cimba.SUCCESS, "second")
        assert queue.get() == (cimba.SUCCESS, "first")


def test_priorityqueue_interrupted_get_and_put():
    log = []

    def getter(queue):
        log.append(("get",) + queue.get())

    def putter(queue):
        sig, handle = queue.put("blocked", priority=5)
        log.append(("put", cimba.time(), sig, handle, queue.length))

    def interrupter(target):
        cimba.hold(1.0)
        target.interrupt(77)

    with cimba.Simulation(seed=1) as sim:
        queue = cimba.PriorityQueue("Priority", capacity=1)
        target = cimba.Process("Getter", getter, queue).start()
        cimba.Process("InterruptGetter", interrupter, target).start()
        sim.execute()

    with cimba.Simulation(seed=1) as sim:
        queue = cimba.PriorityQueue("Priority", capacity=1)
        assert queue.put("existing", priority=0)[0] == cimba.SUCCESS
        target = cimba.Process("Putter", putter, queue).start()
        cimba.Process("InterruptPutter", interrupter, target).start()
        sim.execute()

    assert log == [("get", 77, None), ("put", 1.0, 77, 0, 1)]


def test_priorityqueue_duplicate_object_handles_are_exact():
    with cimba.Simulation(seed=1):
        queue = cimba.PriorityQueue("Priority")
        obj = object()
        assert queue.put(obj, priority=1)[0] == cimba.SUCCESS
        sig, middle = queue.put(obj, priority=3)
        assert sig == cimba.SUCCESS
        sig, last = queue.put(obj, priority=2)
        assert sig == cimba.SUCCESS

        assert queue.cancel(middle) is True
        assert queue.position(middle) == 0
        assert queue.position(last) == 1
        assert queue.get() == (cimba.SUCCESS, obj)
        assert queue.get() == (cimba.SUCCESS, obj)
        assert queue.length == 0
