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
        cimba.Process("Low", low_priority, resource, priority=0).start()
        cimba.Process("High", high_priority, resource, priority=5).start()
        sim.execute()

    assert log == [
        ("low-acquired", 0.0, 1),
        ("high-preempted", 1.0, cimba.SUCCESS, 1, 1),
        ("high-released", 1.0, 0),
        ("low-resumed", 1.0, cimba.PREEMPTED, 0),
    ]
