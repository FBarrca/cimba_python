"""Tutorial 3.1: a nine-attraction park with visitor behavior.

Amusement park - tutorial 3.1 (subprojects/cimba/tutorial/tut_3_1.c)
through the Python bindings: an M/G/n network with balking, reneging,
and jockeying customer behaviors.

Visitors stream through a park of 9 attractions, choosing their next stop
from a per-attraction transition matrix and walking PERT-distributed
times between them. Each attraction has one or more priority queues
served by batch-loading ride servers (PERT ride durations). A quarter of
the visitors carry gold cards, queueing at priority 5 instead of 0. Each
visitor draws a patience factor scaling three behaviors:

* balking   - refuse to join if the shortest queue exceeds patience * 10;
* jockeying - after patience * 5 in queue, switch to a queue that is
              shorter than our current position;
* reneging  - after patience * 10 in queue, give up and walk away.

The queue wait runs on process timers: the visitor enqueues its own
process handle, arms a jockeying and a reneging timer, and sim.suspend()s.
It wakes either by a timer signal or by the ride server, which clears the
visitor's timers when boarding and resumes it after the ride.
"""

import time
import numpy as np
import cimba as cb
from cimba import inputs

# --- Park structure, hard-coded as in the C tutorial -------------------------
NUM_ATTRACTIONS = 9
IDX_ENTRANCE = 0
IDX_EXIT = NUM_ATTRACTIONS + 1

# Transition probabilities i -> j (row 0 entrance, row 10 exit)
TRANSITION_PROBS = np.array([
    [0.00, 0.30, 0.20, 0.20, 0.10, 0.05, 0.05, 0.00, 0.00, 0.00, 0.10],
    [0.00, 0.00, 0.30, 0.20, 0.10, 0.10, 0.05, 0.05, 0.00, 0.00, 0.20],
    [0.00, 0.10, 0.05, 0.20, 0.10, 0.15, 0.05, 0.05, 0.05, 0.05, 0.20],
    [0.00, 0.05, 0.10, 0.05, 0.20, 0.10, 0.10, 0.05, 0.05, 0.05, 0.25],
    [0.00, 0.05, 0.00, 0.10, 0.05, 0.20, 0.15, 0.10, 0.05, 0.05, 0.25],
    [0.00, 0.00, 0.00, 0.05, 0.05, 0.00, 0.20, 0.20, 0.10, 0.10, 0.30],
    [0.00, 0.00, 0.00, 0.05, 0.10, 0.05, 0.00, 0.30, 0.10, 0.10, 0.30],
    [0.00, 0.00, 0.00, 0.05, 0.05, 0.05, 0.05, 0.05, 0.20, 0.20, 0.35],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.05, 0.05, 0.10, 0.00, 0.30, 0.50],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.05, 0.10, 0.20, 0.00, 0.65],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
])

# Average walking times i -> j
TRANSITION_TIMES = np.array([
    [0.00, 3.00, 7.00, 8.00, 9.00, 12.0, 13.0, 15.0, 20.0, 25.0, 30.0],
    [3.00, 1.00, 3.00, 7.00, 8.00, 9.00, 12.0, 13.0, 15.0, 20.0, 25.0],
    [7.00, 3.00, 1.00, 3.00, 7.00, 8.00, 9.00, 12.0, 13.0, 15.0, 20.0],
    [8.00, 7.00, 3.00, 1.00, 3.00, 7.00, 8.00, 9.00, 12.0, 13.0, 15.0],
    [9.00, 8.00, 7.00, 3.00, 1.00, 3.00, 7.00, 8.00, 9.00, 12.0, 13.0],
    [12.0, 9.00, 8.00, 7.00, 3.00, 1.00, 3.00, 7.00, 8.00, 9.00, 12.0],
    [13.0, 12.0, 9.00, 8.00, 7.00, 3.00, 1.00, 3.00, 7.00, 8.00, 9.00],
    [15.0, 13.0, 12.0, 9.00, 8.00, 7.00, 3.00, 1.00, 3.00, 7.00, 8.00],
    [20.0, 15.0, 13.0, 12.0, 9.00, 8.00, 7.00, 3.00, 1.00, 3.00, 7.00],
    [25.0, 20.0, 15.0, 13.0, 12.0, 9.00, 8.00, 7.00, 3.00, 1.00, 3.00],
    [30.0, 25.0, 20.0, 15.0, 13.0, 12.0, 9.00, 8.00, 7.00, 3.00, 0.00],
])

NUM_QUEUES = np.array([0, 1, 1, 1, 3, 1, 1, 1, 1, 1, 0])
SERVERS_PER_Q = np.array([0, 1, 3, 2, 1, 1, 1, 1, 1, 1, 0])
BATCH_SIZES = np.array([0, 1, 5, 5, 1, 10, 5, 8, 1, 1, 0])
MIN_DUR = np.array([0.0, 3.0, 5.0, 4.0, 15.0, 8.0, 5.0, 5.0, 6.0, 3.0, 0.0])
MODE_DUR = np.array([0.0, 4.0, 6.0, 5.0, 20.0, 9.0, 6.0, 5.5, 7.0, 4.0, 0.0])
MAX_DUR = np.array([0.0, 5.0, 7.0, 6.0, 24.0, 12.0, 8.0, 6.0, 8.0, 5.0, 0.0])

TOTAL_QUEUES = int(NUM_QUEUES.sum())                      # 11
NUM_SERVERS = int(np.sum(NUM_QUEUES * SERVERS_PER_Q))     # 14
MAX_BATCH = int(BATCH_SIZES.max())

# Visitor behavior (all scaled by each visitor's patience)
ARRIVAL_RATE = 0.5
PERCENT_GOLDCARDS = 0.25
BALKING_THRESHOLD = 10.0
JOCKEYING_THRESHOLD = 5.0
RENEGING_THRESHOLD = 10.0

TIMER_JOCKEYING = 17
TIMER_RENEGING = 42

PARK_OPEN = 16 * 60.0       # minutes


class Visitor(cb.Model):
    park: cb.Ref["Park"]
    patience: cb.State[float]
    priority: cb.State[int]
    entry_park: cb.State[float]
    process_pointer: cb.State[int] = 0
    entry_queue: cb.State[float] = 0.0
    riding: cb.State[float] = 0.0
    waiting: cb.State[float] = 0.0
    walking: cb.State[float] = 0.0
    rides: cb.State[int] = 0

    @cb.process
    def visit(self):
        me = cb.this_process()
        self.process_pointer = me.pointer
        at = IDX_ENTRANCE
        while at != IDX_EXIT:
            draw = cb.random.uniform()
            cumulative = 0.0
            nxt = IDX_EXIT
            for candidate in range(IDX_EXIT + 1):
                cumulative += TRANSITION_PROBS[at, candidate]
                if draw < cumulative:
                    nxt = candidate
                    break
            mean_walk = TRANSITION_TIMES[at, nxt]
            walk = cb.random.pert(0.5 * mean_walk, mean_walk, 2.0 * mean_walk)
            cb.hold(walk)
            self.walking += walk
            at = nxt
            if at == IDX_EXIT:
                break

            chosen = self.park.shortest_line(at)
            shortest = self.park.ride_queues[chosen].line.length()
            if shortest > self.patience * BALKING_THRESHOLD:
                self.park.balks += 1
                continue

            line = self.park.ride_queues[chosen].line
            self.entry_queue = cb.now()
            ticket = line.enqueue(self, self.priority)
            me.timer_set(self.patience * JOCKEYING_THRESHOLD, TIMER_JOCKEYING)
            me.timer_set(self.patience * RENEGING_THRESHOLD, TIMER_RENEGING)
            while True:
                signal = cb.suspend()
                if signal == TIMER_JOCKEYING:
                    replacement = self.park.shortest_line(at)
                    replacement_length = self.park.ride_queues[replacement].line.length()
                    if replacement != chosen and (
                        replacement_length < line.position(ticket)
                    ):
                        if line.cancel(ticket):
                            chosen = replacement
                            line = self.park.ride_queues[chosen].line
                            ticket = line.enqueue(self, self.priority + 1)
                            self.park.jockeys += 1
                elif signal == TIMER_RENEGING:
                    if line.cancel(ticket):
                        self.park.reneges += 1
                    me.timers_clear()
                    break
                else:
                    self.rides += 1
                    break

        self.park.d_park.record(cb.now() - self.entry_park)
        self.park.d_riding.record(self.riding)
        self.park.d_waiting.record(self.waiting)
        self.park.d_walking.record(self.walking)
        self.park.d_rides.record(float(self.rides))
        cb.release(self)


class RideQueue(cb.Model):
    attraction: cb.Param[int]
    line: cb.PriorityStore[Visitor]
    duration: cb.Input[float] = inputs.dist.pert(low=3.0, mode=4.0, high=5.0)

    def __init__(self, attraction: int):
        self.attraction = attraction
        self.duration = inputs.dist.pert(
            low=float(MIN_DUR[attraction]),
            mode=float(MODE_DUR[attraction]),
            high=float(MAX_DUR[attraction]),
        )

    @cb.process
    def server(self):
        while True:
            visitor = self.line.get()
            visitor.waiting += cb.now() - visitor.entry_queue
            visitor_process = cb.Process(visitor.process_pointer)
            visitor_process.timers_clear()
            duration = self.duration.next()
            cb.hold(duration)
            visitor.riding += duration
            visitor_process.resume(0)


class RideQueueTwo(RideQueue):
    @cb.process(copies=2)
    def server(self):
        while True:
            visitor = self.line.get()
            visitor.waiting += cb.now() - visitor.entry_queue
            visitor_process = cb.Process(visitor.process_pointer)
            visitor_process.timers_clear()
            duration = self.duration.next()
            cb.hold(duration)
            visitor.riding += duration
            visitor_process.resume(0)


class RideQueueThree(RideQueue):
    @cb.process(copies=3)
    def server(self):
        while True:
            visitor = self.line.get()
            visitor.waiting += cb.now() - visitor.entry_queue
            visitor_process = cb.Process(visitor.process_pointer)
            visitor_process.timers_clear()
            duration = self.duration.next()
            cb.hold(duration)
            visitor.riding += duration
            visitor_process.resume(0)


class Park(cb.Model):
    closing: cb.Param[float] = PARK_OPEN
    arrival_gap: cb.Input[float] = inputs.dist.exponential(mean=1.0 / ARRIVAL_RATE)
    ride_queues: list[RideQueue]
    d_park: cb.Dataset
    d_riding: cb.Dataset
    d_waiting: cb.Dataset
    d_walking: cb.Dataset
    d_rides: cb.Dataset
    balks: cb.State[int] = 0
    jockeys: cb.State[int] = 0
    reneges: cb.State[int] = 0
    avg_rides: cb.Output[float]
    avg_time_in_park: cb.Output[float]
    avg_riding: cb.Output[float]
    avg_waiting: cb.Output[float]
    avg_walking: cb.Output[float]
    n_visitors: cb.Output[int]
    n_balks: cb.Output[int]
    n_jockeys: cb.Output[int]
    n_reneges: cb.Output[int]

    def __init__(self, closing: float = PARK_OPEN):
        self.closing = closing
        self.ride_queues = [
            (RideQueue if SERVERS_PER_Q[attraction] == 1 else
             RideQueueTwo if SERVERS_PER_Q[attraction] == 2 else
             RideQueueThree)(attraction)
            for attraction in range(1, NUM_ATTRACTIONS + 1)
            for _ in range(int(NUM_QUEUES[attraction]))
        ]

    @cb.function
    def shortest_line(self, attraction: int) -> int:
        """Index of the shortest line serving ``attraction``."""
        chosen = 0
        shortest = 1_000_000
        for index in range(len(self.ride_queues)):
            ride = self.ride_queues[index]
            if ride.attraction == attraction:
                length = ride.line.length()
                if length < shortest:
                    chosen = index
                    shortest = length
        return chosen

    @cb.process
    def arrivals(self):
        while True:
            cb.hold(self.arrival_gap.next())
            if cb.now() >= self.closing:
                break
            priority = 5 if cb.random.bernoulli(PERCENT_GOLDCARDS) else 0
            cb.spawn(Visitor, park=self,
                     priority=priority,
                     patience=cb.random.triangular(0.5, 1.0, 1.5),
                     entry_park=cb.now())
        while True:
            cb.suspend()

    @cb.on_end
    def park_stats(self):
        self.avg_rides = self.d_rides.sample_mean()
        self.avg_time_in_park = self.d_park.sample_mean()
        self.avg_riding = self.d_riding.sample_mean()
        self.avg_waiting = self.d_waiting.sample_mean()
        self.avg_walking = self.d_walking.sample_mean()
        self.n_visitors = self.d_park.sample_count()
        self.n_balks = self.balks
        self.n_jockeys = self.jockeys
        self.n_reneges = self.reneges


def main() -> None:
    park = Park()
    start = time.perf_counter()
    results = cb.Experiment(
        park, replications=20,
        window=cb.Window(duration=PARK_OPEN, cooldown=2_000.0),
        seed=20260613,
    ).run()
    if results.failed.any():
        raise RuntimeError(f"{results.failed.sum()} park trials failed")
    print(f"cimba {cb.engine_version()}; {NUM_ATTRACTIONS} attractions, "
          f"{TOTAL_QUEUES} queues, {NUM_SERVERS} ride servers")
    print(f"{results.failed.size} trials in {time.perf_counter() - start:.2f} s")
    for field in ("n_visitors", "avg_rides", "avg_time_in_park", "avg_riding",
                  "avg_waiting", "avg_walking", "n_balks", "n_jockeys",
                  "n_reneges"):
        samples = getattr(results[park], field).values
        print(f"{field}: {samples.mean():.3f}")


if __name__ == "__main__":
    main()
