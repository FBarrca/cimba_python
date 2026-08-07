from importlib import import_module

import pytest

import cimba.sim as sim


def _take_from_helper(env):
    env.queue.get(1)


def test_process_dag_infers_queue_flow_with_aliases_and_helpers():
    class Line(sim.Model):
        queue: sim.Queue

        @sim.process
        def producer(env: "Line"):
            q = env.queue
            q.put(1)

        @sim.process
        def consumer(env: "Line"):
            _take_from_helper(env)

    model = Line()



    graph = model.process_dag()

    assert graph.nodes == (
        sim.ProcessDAGNode("producer"),
        sim.ProcessDAGNode("consumer"),
        sim.ProcessDAGNode("queue", kind="queue"),
    )
    assert graph.edges == (
        sim.ProcessDAGEdge("process:producer", "queue:queue", "put"),
        sim.ProcessDAGEdge("queue:queue", "process:consumer", "get"),
    )
    assert graph.topological_order() == (
        "process:producer",
        "queue:queue",
        "process:consumer",
    )


def test_process_dag_mermaid_and_dot_output_are_stable():
    class MM1(sim.Model):
        queue: sim.Queue

        @sim.process
        def arrivals(env):
            env.queue.put(1)

        @sim.process
        def service(env):
            env.queue.get(1)

    model = MM1("mm1")

    graph = model.process_dag()

    assert graph.to_mermaid() == "\n".join([
        "flowchart TD",
        '    n_process_arrivals["arrivals"]',
        '    n_process_service["service"]',
        '    n_queue_queue[("queue")]',
        "    n_process_arrivals -->|put| n_queue_queue",
        "    n_queue_queue -->|get| n_process_service",
    ])
    assert graph.to_dot() == "\n".join([
        "digraph ProcessDAG {",
        "    rankdir=TB;",
        '    "process:arrivals" [label="arrivals", shape=box];',
        '    "process:service" [label="service", shape=box];',
        '    "queue:queue" [label="queue", shape=ellipse];',
        '    "process:arrivals" -> "queue:queue" [label="put"];',
        '    "queue:queue" -> "process:service" [label="get"];',
        "}",
    ])


def test_process_dag_infers_spawn_store_pqueues_and_conditions():
    class Park(sim.Model):
        departed: sim.Store
        ride_queues: sim.PQueues = sim.count(1)
        ready: sim.Condition
        ready_pred: sim.Predicate

        @sim.process
        def arrivals(env: "Park"):
            sim.spawn(env.visitor, env)
            env.ready.signal()

        @sim.process(spawnable=True)
        def visitor(env: "Park"):
            env.ride_queues[0].put(sim.current(), 0)
            env.ready.wait_for(env.ready_pred)
            env.departed.put(sim.current())

        @sim.process
        def server(env: "Park"):
            q = env.ride_queues[0]
            q.take()

        @sim.process
        def departures(env: "Park"):
            sim.despawn(env.departed.take())

    model = Park()





    graph = model.process_dag()

    assert graph.edges == (
        sim.ProcessDAGEdge("process:arrivals", "process:visitor", "spawn"),
        sim.ProcessDAGEdge("process:arrivals", "condition:ready", "signal"),
        sim.ProcessDAGEdge("process:visitor", "pqueues:ride_queues", "pq_put"),
        sim.ProcessDAGEdge("condition:ready", "process:visitor", "wait_for"),
        sim.ProcessDAGEdge("process:visitor", "store:departed", "store_put"),
        sim.ProcessDAGEdge("pqueues:ride_queues", "process:server", "pq_take"),
        sim.ProcessDAGEdge("store:departed", "process:departures", "store_take"),
    )


def test_process_dag_infers_process_handle_interactions():
    class Work(sim.Model):
        worker: sim.Processes

        @sim.process(field="worker")
        def worker_process(env: "Work"):
            sim.suspend()

        @sim.process
        def supervisor(env: "Work"):
            target = env.worker[0]
            sim.interrupt(target, 42, 0)

    model = Work()



    graph = model.process_dag()

    assert sim.ProcessDAGEdge(
        "process:supervisor",
        "process:worker_process",
        "interrupt",
    ) in graph.edges


def test_process_dag_tracks_shared_resource_usage_without_fake_dependencies():
    class Shop(sim.Model):
        crew: sim.Pool = 2

        @sim.process
        def worker(env: "Shop"):
            env.crew.acquire(1)
            env.crew.release(1)

        @sim.process
        def inspector(env: "Shop"):
            env.crew.available()

    model = Shop()



    graph = model.process_dag()

    assert graph.edges == (
        sim.ProcessDAGEdge("process:worker", "pool:crew", "uses"),
        sim.ProcessDAGEdge("process:inspector", "pool:crew", "uses"),
    )
    assert all(
        not (
            edge.source == "process:worker"
            and edge.target == "process:inspector"
        )
        for edge in graph.edges
    )


def test_process_dag_infers_state_and_float_state_dependencies():
    class Telemetry(sim.Model):
        level: sim.FloatState
        count: sim.State

        @sim.process
        def sampler(env: "Telemetry"):
            env.level = 10.0
            env.count = 1

        @sim.process
        def controller(env: "Telemetry"):
            if env.level > 5.0 and env.count > 0:
                sim.suspend()

    model = Telemetry()



    graph = model.process_dag()

    assert graph.edges == (
        sim.ProcessDAGEdge("process:sampler", "fstate:level", "write"),
        sim.ProcessDAGEdge("process:sampler", "state:count", "write"),
        sim.ProcessDAGEdge("fstate:level", "process:controller", "read"),
        sim.ProcessDAGEdge("state:count", "process:controller", "read"),
    )


def test_process_dag_infers_events_waits_and_callback_state():
    class Clock(sim.Model):
        tick: sim.Event
        fired: sim.State

        @sim.event(field="tick")
        def on_tick(env: "Clock"):
            env.fired = env.fired + 1
            env.tick.schedule(1.0)

        @sim.process
        def timer(env: "Clock"):
            handle = env.tick.schedule(1.0)
            handle.wait_event()

    model = Clock()



    graph = model.process_dag()

    assert graph.nodes == (
        sim.ProcessDAGNode("timer"),
        sim.ProcessDAGNode("tick", kind="event"),
        sim.ProcessDAGNode("fired", kind="state"),
    )
    assert graph.edges == (
        sim.ProcessDAGEdge("process:timer", "event:tick", "schedule"),
        sim.ProcessDAGEdge("event:tick", "process:timer", "wait_event"),
        sim.ProcessDAGEdge("event:tick", "state:fired", "write"),
        sim.ProcessDAGEdge("state:fired", "event:tick", "read"),
        sim.ProcessDAGEdge("event:tick", "event:tick", "schedule"),
    )


def test_process_dag_renders_cycles_but_topological_order_rejects_them():
    class Cycle(sim.Model):
        queue: sim.Queue

        @sim.process
        def actor(env):
            env.queue.put(1)
            env.queue.get(1)

    model = Cycle("cycle")

    graph = model.process_dag()

    assert "n_process_actor" in graph.to_mermaid()
    with pytest.raises(ValueError, match="cycle"):
        graph.topological_order()


def test_tutorial_process_dags_render_from_inference():
    cases = [
        ("tut_1_1", lambda m: m.model),
        ("tut_1_2", lambda m: m.model),
        ("tut_1_3", lambda m: m.model),
        ("tut_1_4", lambda m: m.model),
        ("tut_1_5", lambda m: m.build_model()),
        ("tut_1_6", lambda m: m.build_model()),
        ("tut_1_7", lambda m: m.build_model()),
        ("tut_2_1", lambda m: m.game),
        ("tut_3_1", lambda m: m.park),
        ("tut_4_0", lambda m: m.model),
        ("tut_4_1", lambda m: m.harbor),
        ("tut_4_2", lambda m: m.harbor),
    ]

    for name, get_model in cases:
        module = import_module(f"tutorial.{name}")
        mermaid = get_model(module).process_dag().to_mermaid()
        assert mermaid.startswith("flowchart TD")
