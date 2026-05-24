import cimba

TIMER_JOCKEYING = 17
TIMER_RENEGING = 42


class Visitor:
    def __init__(self, name: str, patience: float = 1.0):
        self.name = name
        self.patience = patience
        self.entry_time_queue = 0.0
        self.riding_time = 0.0
        self.waiting_time = 0.0
        self.num_attractions_visited = 0
        self.status = "new"


def _server(ctx):
    while True:
        sig, visitor = ctx["queue"].get()
        assert sig == cimba.SUCCESS
        visitor.process.timers_clear()
        visitor.waiting_time += cimba.time() - visitor.entry_time_queue
        cimba.hold(ctx["ride_duration"])
        visitor.riding_time += ctx["ride_duration"]
        visitor.process.resume(cimba.SUCCESS)


def _visitor_proc(me, ctx):
    visitor = ctx["visitor"]
    visitor.process = me
    queue = ctx["queues"][0]
    visitor.entry_time_queue = cimba.time()
    sig, handle = queue.put(visitor, priority=me.priority)
    assert sig == cimba.SUCCESS

    me.timer_set(visitor.patience, TIMER_JOCKEYING)
    me.timer_add(10.0 * visitor.patience, TIMER_RENEGING)

    while True:
        sig = cimba.yield_process()
        if sig == TIMER_JOCKEYING:
            new_queue = ctx["queues"][1]
            if new_queue.length < queue.position(handle):
                assert queue.cancel(handle)
                queue = new_queue
                visitor.entry_time_queue = cimba.time()
                sig, handle = queue.put(visitor, priority=me.priority + 1)
                assert sig == cimba.SUCCESS
            continue
        if sig == TIMER_RENEGING:
            assert queue.cancel(handle)
            visitor.status = "reneged"
            break

        assert sig == cimba.SUCCESS
        visitor.num_attractions_visited += 1
        visitor.status = "served"
        break

    ctx["departed"].put(visitor)


def run_jockeying_demo() -> Visitor:
    with cimba.Simulation(seed=31) as sim:
        visitor = Visitor("Visitor_000001")
        ctx = {
            "visitor": visitor,
            "queues": [cimba.PriorityQueue("Queue_00"), cimba.PriorityQueue("Queue_01")],
            "departed": cimba.ObjectQueue("Departed"),
            "ride_duration": 2.0,
        }
        cimba.Process("Server_01", _server, {"queue": ctx["queues"][1], "ride_duration": 2.0}).start()
        cimba.Process(visitor.name, _visitor_proc, ctx, pass_process=True).start()
        sim.execute()

        sig, departed = ctx["departed"].get()
        assert sig == cimba.SUCCESS
        assert departed is visitor
        return visitor


def run_reneging_demo() -> tuple[Visitor, int]:
    def visitor_proc(me, ctx):
        visitor = ctx["visitor"]
        visitor.process = me
        sig, handle = ctx["queue"].put(visitor, priority=0)
        assert sig == cimba.SUCCESS
        me.timer_set(visitor.patience, TIMER_RENEGING)

        sig = cimba.yield_process()
        assert sig == TIMER_RENEGING
        assert ctx["queue"].cancel(handle)
        visitor.status = "reneged"
        ctx["departed"].put(visitor)

    with cimba.Simulation(seed=32) as sim:
        visitor = Visitor("Visitor_000002", patience=1.0)
        ctx = {
            "visitor": visitor,
            "queue": cimba.PriorityQueue("Queue_00"),
            "departed": cimba.ObjectQueue("Departed"),
        }
        cimba.Process(visitor.name, visitor_proc, ctx, pass_process=True).start()
        sim.execute()

        sig, departed = ctx["departed"].get()
        assert sig == cimba.SUCCESS
        assert departed is visitor
        return visitor, ctx["queue"].length


def main() -> None:
    visitor = run_jockeying_demo()
    print(visitor.status, visitor.num_attractions_visited)
    visitor, queue_length = run_reneging_demo()
    print(visitor.status, queue_length)


if __name__ == "__main__":
    main()
