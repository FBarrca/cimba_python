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
