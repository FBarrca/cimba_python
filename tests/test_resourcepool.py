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
        pool.start_recording()
        cimba.Process("Holder", holder, pool).start()
        cimba.Process("Waiter", waiter, pool).start()
        sim.execute()
        pool.stop_recording()
        history = pool.history()

    assert log == [
        ("holder-acquired", 0.0, 5, 0),
        ("holder-released", 1.0, 5),
        ("waiter-acquired", 1.0, 3, 2),
    ]
    assert history.count >= 2


def test_resourcepool_interrupted_waiting_acquire_keeps_original_holdings():
    log = []

    def holder(me, pool):
        assert pool.acquire(5) == cimba.SUCCESS
        cimba.hold(2.0)
        pool.release(5)

    def waiter(me, pool):
        cimba.hold(0.1)
        sig = pool.acquire(3)
        log.append(("waiter", cimba.time(), sig, pool.held_by(me), pool.in_use))

    def interrupter(me, target):
        cimba.hold(1.0)
        target.interrupt(45)

    with cimba.Simulation(seed=1) as sim:
        pool = cimba.ResourcePool("Pool", capacity=5)
        cimba.Process("Holder", holder, pool).start()
        target = cimba.Process("Waiter", waiter, pool).start()
        cimba.Process("Interrupter", interrupter, target).start()
        sim.execute()

    assert log == [("waiter", 1.0, 45, 0, 5)]
