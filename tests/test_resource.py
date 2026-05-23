import cimba


def test_resource_preemption_wakes_lower_priority_holder():
    log = []

    def low_priority(me, resource):
        assert resource.acquire() == cimba.SUCCESS
        log.append(("low-acquired", cimba.time(), resource.held_by(me)))
        sig = cimba.hold(10.0)
        log.append(("low-resumed", cimba.time(), sig, resource.held_by(me)))

    def high_priority(me, resource):
        cimba.hold(1.0)
        sig = resource.preempt()
        log.append(("high-preempted", cimba.time(), sig, resource.held_by(me), resource.in_use))
        resource.release()
        log.append(("high-released", cimba.time(), resource.in_use))

    with cimba.Simulation(seed=1) as sim:
        resource = cimba.Resource("Resource")
        resource.start_recording()
        cimba.Process("Low", low_priority, resource, priority=0).start()
        cimba.Process("High", high_priority, resource, priority=5).start()
        sim.execute()
        resource.stop_recording()
        history = resource.history()

    assert log == [
        ("low-acquired", 0.0, 1),
        ("high-preempted", 1.0, cimba.SUCCESS, 1, 1),
        ("high-released", 1.0, 0),
        ("low-resumed", 1.0, cimba.PREEMPTED, 0),
    ]
    assert history.count >= 2


def test_resource_interrupted_waiting_acquire_does_not_take_resource():
    log = []

    def holder(me, resource):
        assert resource.acquire() == cimba.SUCCESS
        cimba.hold(2.0)
        resource.release()

    def waiter(me, resource):
        cimba.hold(0.1)
        sig = resource.acquire()
        log.append(("waiter", cimba.time(), sig, resource.held_by(me), resource.in_use))

    def interrupter(me, target):
        cimba.hold(1.0)
        target.interrupt(44)

    with cimba.Simulation(seed=1) as sim:
        resource = cimba.Resource("Resource")
        cimba.Process("Holder", holder, resource).start()
        target = cimba.Process("Waiter", waiter, resource).start()
        cimba.Process("Interrupter", interrupter, target).start()
        sim.execute()

    assert log == [("waiter", 1.0, 44, 0, 1)]


def test_resource_waiter_stop_does_not_leave_stale_native_waiter():
    log = []

    def holder(me, resource):
        assert resource.acquire() == cimba.SUCCESS
        cimba.hold(2.0)
        resource.release()
        log.append(("released", cimba.time(), resource.in_use))

    def waiter(me, resource):
        try:
            cimba.hold(0.1)
            resource.acquire()
        finally:
            log.append(("cancelled", cimba.time(), resource.held_by(me)))

    def stopper(me, target):
        cimba.hold(1.0)
        assert target.stop() == cimba.SUCCESS

    with cimba.Simulation(seed=1) as sim:
        resource = cimba.Resource("Resource")
        cimba.Process("Holder", holder, resource).start()
        target = cimba.Process("Waiter", waiter, resource).start()
        cimba.Process("Stopper", stopper, target).start()
        sim.execute()

    assert log == [("cancelled", 1.0, 0), ("released", 2.0, 0)]
