import cimba


def test_resourcepool_acquire_release_and_wait_order():
    log = []

    def holder(me, pool):
        assert pool.acquire(5) == cimba.SUCCESS
        log.append(("holder-acquired", cimba.time(), pool.held_by(me), pool.available))
        cimba.hold(1.0)
        pool.release(5)
        log.append(("holder-released", cimba.time(), pool.available))

    def waiter(me, pool):
        assert pool.acquire(3) == cimba.SUCCESS
        log.append(("waiter-acquired", cimba.time(), pool.held_by(me), pool.available))
        pool.release(3)

    with cimba.Simulation(seed=1) as sim:
        pool = cimba.ResourcePool("Pool", capacity=5)
        cimba.Process("Holder", holder, pool).start()
        cimba.Process("Waiter", waiter, pool).start()
        sim.execute()

    assert log == [
        ("holder-acquired", 0.0, 5, 0),
        ("holder-released", 1.0, 5),
        ("waiter-acquired", 1.0, 3, 2),
    ]
