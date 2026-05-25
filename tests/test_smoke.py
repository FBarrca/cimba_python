import cimba
from cimba.cmb_buffer import Buffer
from cimba.cmb_event import Simulation
from cimba.cmb_process import Process
from cimba.cmb_random import exponential


def test_c_module_shaped_imports_are_available():
    assert Buffer is cimba.Buffer
    assert Simulation is cimba.Simulation
    assert Process is cimba.Process
    assert exponential is cimba.exponential


def test_versions_are_available():
    assert cimba.__version__ == "0.2.1"
    assert cimba.native_version().startswith("3.")


def test_processes_can_block_on_buffer_and_resume():
    events = []

    def producer(queue):
        cimba.hold(1.0)
        queue.put(2)
        cimba.hold(1.0)
        queue.put(1)

    def consumer(queue):
        sig, got = queue.get(1)
        events.append((cimba.time(), sig, got, queue.level))
        sig, got = queue.get(2)
        events.append((cimba.time(), sig, got, queue.level))

    with cimba.Simulation(seed=123) as sim:
        queue = cimba.Buffer("Queue")
        cimba.Process("Producer", producer, queue).start()
        cimba.Process("Consumer", consumer, queue).start()
        sim.execute()

    assert events == [(1.0, cimba.SUCCESS, 1, 1), (2.0, cimba.SUCCESS, 2, 0)]


def test_python_object_queues_preserve_objects_and_priority():
    with cimba.Simulation(seed=1):
        fifo = cimba.ObjectQueue("FIFO")
        obj = {"payload": 1}
        assert fifo.put(obj) == cimba.SUCCESS
        assert fifo.length == 1
        sig, got = fifo.get()
        assert sig == cimba.SUCCESS
        assert got is obj

        pq = cimba.PriorityQueue("Priority")
        assert pq.put("low", priority=0)[0] == cimba.SUCCESS
        assert pq.put("high", priority=10)[0] == cimba.SUCCESS
        assert pq.get() == (cimba.SUCCESS, "high")
        assert pq.get() == (cimba.SUCCESS, "low")


def test_mm1_buffer_history_summary_runs_to_stop_time():
    def arrival(queue):
        while True:
            cimba.hold(cimba.exponential(1.0 / 0.75))
            queue.put(1)

    def service(queue):
        while True:
            queue.get(1)
            cimba.hold(cimba.exponential(1.0))

    with cimba.Simulation(seed=123) as sim:
        queue = cimba.Buffer("Queue")
        queue.start_recording()
        cimba.Process("Arrival", arrival, queue).start()
        cimba.Process("Service", service, queue).start()
        sim.stop_at(1000.0)
        sim.execute()
        queue.stop_recording()

        summary = queue.history().summary()
        assert summary.count > 0
        assert 0.5 < summary.mean < 6.0
