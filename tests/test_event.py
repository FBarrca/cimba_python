import cimba


def test_event_queue_start_time_execute_next_and_stop_at():
    log = []

    def worker(me, ctx):
        log.append(("start", cimba.time(), ctx.now))
        cimba.hold(2.0)
        log.append(("after-hold", cimba.time()))

    with cimba.Simulation(start_time=3.0, seed=1) as sim:
        assert sim.now == 3.0
        assert sim.event_count == 0
        cimba.Process("Worker", worker, sim).start()
        assert sim.event_count == 1

        assert sim.execute_next() is True
        assert log == [("start", 3.0, 3.0)]
        assert sim.now == 3.0
        assert sim.event_count == 1

        handle = sim.stop_at(4.0)
        assert handle != 0
        assert sim.event_count == 2
        sim.execute()
        assert sim.now == 4.0

    assert log == [("start", 3.0, 3.0)]


def test_execute_next_on_empty_queue_returns_false():
    with cimba.Simulation(start_time=7.5, seed=1) as sim:
        assert sim.execute_next() is False
        assert sim.now == 7.5
