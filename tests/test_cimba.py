import cimba


def test_single_mg1_style_trial_collects_buffer_history_after_warmup():
    utilization = 0.75
    service_cv = 0.5
    service_shape = 1.0 / (service_cv * service_cv)
    service_scale = service_cv * service_cv

    def arrival(queue):
        while True:
            cimba.hold(cimba.exponential(1.0 / utilization))
            assert queue.put(1) == (cimba.SUCCESS, 0)

    def service(queue):
        while True:
            assert queue.get(1)[0] == cimba.SUCCESS
            cimba.hold(cimba.gamma(service_shape, service_scale))

    def recorder(queue):
        cimba.hold(10.0)
        queue.start_recording()
        cimba.hold(100.0)
        queue.stop_recording()

    with cimba.Simulation(seed=0xC1A0) as sim:
        queue = cimba.Buffer("Queue")
        cimba.Process("Arrivals", arrival, queue).start()
        cimba.Process("Service", service, queue).start()
        cimba.Process("Recorder", recorder, queue).start()
        sim.stop_at(120.0)
        sim.execute()
        history = queue.history()

    summary = history.summary()
    assert summary.count > 0
    assert summary.mean >= 0.0
