"""
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

Translation notes (C -> cimba.sim):

* Visitors are dynamic processes, as in C: arrivals sim.spawn()s one per
  arrival through the sim.Spawnable `visitor` field and initializes its
  Visitor fields before it starts running (C's visitor_initialize). The
  per-visitor attributes are sim.Struct fields in the process's native
  allocation -- the Python form of the C tutorial deriving struct visitor
  from cmb_process -- and the process sees them through its annotated
  `vip: Visitor` parameter.
* As in C, the object a visitor posts in the ride queue is its own
  process handle. The server views it with Visitor(handle), adds the
  waiting and riding times to the visitor's fields, and resumes it with
  sim.SUCCESS -- the same data flow as the C server.
* A departing visitor tallies its statistics, hands its own handle to
  the departures process through the `departed` store, and returns; the
  departures process sim.despawn()s it (C's visitor_terminate/_destroy).
  Early despawning just recycles memory during the day -- any spawned
  process still alive at the end of the trial is reclaimed automatically.
* The park layout lives in module-level numpy arrays, baked into the
  compiled code as constants; the per-attraction queues are a flat
  sim.PQueues array indexed by Q_FIRST/Q_COUNT, and the 14 servers are
  copies of one indexed process parameterized by SRV_* tables.
* The C alias sampler (cmb_random_alias) becomes a linear scan over the
  cumulative transition row -- identical distribution, 11 outcomes.
* The C version stops arrivals at closing time and lets the day drain.
  Here arrivals stop emitting at start+duration and the trial's cooldown
  provides the draining window, so every visitor's day is complete.

Usage: uv run python examples/demo_park.py
"""

import time

import numpy as np
from numba import njit

import cimba as cp
import cimba.sim as sim

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

# Flat layout: queue k of attraction a is park_queues[Q_FIRST[a] + k]
Q_FIRST = np.concatenate(([0], np.cumsum(NUM_QUEUES)[:-1]))
TOTAL_QUEUES = int(NUM_QUEUES.sum())                      # 11

# One server process copy per physical server, each tied to a flat queue
SRV_Q = np.concatenate([
    np.repeat(np.arange(Q_FIRST[a], Q_FIRST[a] + NUM_QUEUES[a]),
              SERVERS_PER_Q[a])
    for a in range(1, NUM_ATTRACTIONS + 1)
])
SRV_ATTR = np.concatenate([
    np.full(NUM_QUEUES[a] * SERVERS_PER_Q[a], a)
    for a in range(1, NUM_ATTRACTIONS + 1)
])
NUM_SERVERS = int(SRV_Q.size)                             # 14
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


class Visitor(sim.Struct):
    """Per-visitor fields, as in the C tutorial's struct visitor."""
    patience: float
    priority: int
    entry_park: float
    entry_queue: float
    riding: float
    waiting: float
    walking: float
    rides: int


class Park(sim.Model):
    # Results (averages over each trial's visitors)
    avg_rides: sim.Output
    avg_time_in_park: sim.Output
    avg_riding: sim.Output
    avg_waiting: sim.Output
    avg_walking: sim.Output
    n_visitors: sim.Output
    n_balks: sim.Output
    n_jockeys: sim.Output
    n_reneges: sim.Output

    # Counters
    balks: sim.State
    jockeys: sim.State
    reneges: sim.State

    # Entities
    visitor: sim.Spawnable              # one spawned per arrival
    departed: sim.Store                 # finished visitors to reclaim
    park_queues: sim.PQueues = sim.count(TOTAL_QUEUES)
    d_park: sim.Dataset                 # time in park
    d_riding: sim.Dataset
    d_waiting: sim.Dataset
    d_walking: sim.Dataset
    d_rides: sim.Dataset                # attractions ridden per visitor


park = Park()


@njit
def _next_attraction(at):
    """Sample the next stop from the transition row (alias sampler in C)."""
    r = sim.random01()
    acc = 0.0
    for j in range(IDX_EXIT + 1):
        acc += TRANSITION_PROBS[at, j]
        if r < acc:
            return j
    return IDX_EXIT


@njit
def _shortest_queue(env, at):
    """Flat index of the shortest queue at the attraction."""
    base = Q_FIRST[at]
    best = base
    best_len = sim.pq_length(env.park_queues[base])
    for qi in range(1, NUM_QUEUES[at]):
        length = sim.pq_length(env.park_queues[base + qi])
        if length < best_len:
            best_len = length
            best = base + qi
    return best, best_len


@park.process(copies=NUM_SERVERS)
def server(env: Park, idx: int):
    q = env.park_queues[SRV_Q[idx]]
    attraction = SRV_ATTR[idx]
    batch_size = BATCH_SIZES[attraction]
    dmin = MIN_DUR[attraction]
    dmode = MODE_DUR[attraction]
    dmax = MAX_DUR[attraction]
    riders = np.empty(MAX_BATCH, dtype=np.int64)

    while True:
        # Wait for the first rider, then fill the ride as best possible
        riders[0] = sim.pq_take(q)
        cnt = 1
        while sim.pq_length(q) > 0 and cnt < batch_size:
            riders[cnt] = sim.pq_take(q)
            cnt = cnt + 1
        # Boarding: no more jockeying or reneging for this batch, and
        # the waiting is over -- log it into each visitor's record
        boarding = sim.now()
        for i in range(cnt):
            sim.timers_clear(riders[i])
            vip = Visitor(riders[i])
            vip.waiting += boarding - vip.entry_queue

        dur = sim.pert(dmin, dmode, dmax)
        sim.hold(dur)

        # Unload and send the riders on their merry way
        for i in range(cnt):
            Visitor(riders[i]).riding += dur
            sim.resume(riders[i], sim.SUCCESS)


@park.process
def arrivals(env: Park):
    closing = env.start_time + env.warmup_s + env.duration_s
    mean_interarr = 1.0 / ARRIVAL_RATE
    while True:
        sim.hold(sim.exponential(mean_interarr))
        if sim.now() >= closing:
            break
        # Spawn a new visitor and initialize it before it passes the
        # turnstile (it starts running once we block on the next hold)
        priority = 5 if sim.bernoulli(PERCENT_GOLDCARDS) == 1 else 0
        v = sim.spawn(env.visitor, env, priority)
        vip = Visitor(v)
        vip.entry_park = sim.now()
        vip.patience = sim.triangular(0.5, 1.0, 1.5)
        vip.priority = priority
    while True:
        sim.suspend()       # park entrance closed for today


@park.process
def visitor(env: Park, vip: Visitor):
    me = sim.current()
    at = IDX_ENTRANCE
    while at != IDX_EXIT:
        nxt = _next_attraction(at)

        # Walk there
        mwt = TRANSITION_TIMES[at, nxt]
        wt = sim.pert(0.5 * mwt, mwt, 2.0 * mwt)
        sim.hold(wt)
        vip.walking += wt
        at = nxt
        if at == IDX_EXIT:
            break

        # Join the shortest queue if several
        qi, qlen = _shortest_queue(env, at)

        # Balking?
        if qlen > vip.patience * BALKING_THRESHOLD:
            env.balks += 1
            continue        # too long a queue, go somewhere else

        # Arm the jockeying and reneging timeouts, then queue up
        sim.timer_set(me, vip.patience * JOCKEYING_THRESHOLD,
                      TIMER_JOCKEYING)
        sim.timer_add(me, vip.patience * RENEGING_THRESHOLD,
                      TIMER_RENEGING)
        q = env.park_queues[qi]
        vip.entry_queue = sim.now()
        entry = sim.pq_put(q, me, vip.priority)

        # Suspend until we have finished both queue and ride, trusting
        # the server to clear our timers at boarding and to update our
        # waiting and riding times, as in C
        while True:
            sig = sim.suspend()
            if sig == TIMER_JOCKEYING:
                my_pos = sim.pq_position(q, entry)
                new_qi, new_len = _shortest_queue(env, at)
                if new_len < my_pos:
                    sim.pq_cancel(q, entry)
                    q = env.park_queues[new_qi]
                    entry = sim.pq_put(q, me, vip.priority + 1)
                    env.jockeys += 1
            elif sig == TIMER_RENEGING:
                sim.pq_cancel(q, entry)
                sim.timers_clear(me)
                env.reneges += 1
                break       # give up, go somewhere else
            else:
                vip.rides += 1
                break       # yay! slightly dizzy, do it again?

    # Enough for today: tally up, then hand ourselves to departures
    sim.tally(env.d_park, sim.now() - vip.entry_park)
    sim.tally(env.d_riding, vip.riding)
    sim.tally(env.d_waiting, vip.waiting)
    sim.tally(env.d_walking, vip.walking)
    sim.tally(env.d_rides, 1.0 * vip.rides)
    sim.store_put(env.departed, me)


@park.process
def departures(env: Park):
    while True:
        sim.despawn(sim.store_take(env.departed))


@park.collect
def park_stats(env: Park):
    env.avg_rides = sim.dataset_mean(env.d_rides)
    env.avg_time_in_park = sim.dataset_mean(env.d_park)
    env.avg_riding = sim.dataset_mean(env.d_riding)
    env.avg_waiting = sim.dataset_mean(env.d_waiting)
    env.avg_walking = sim.dataset_mean(env.d_walking)
    env.n_visitors = sim.dataset_count(env.d_park)
    env.n_balks = env.balks
    env.n_jockeys = env.jockeys
    env.n_reneges = env.reneges


def ci95(vals: np.ndarray) -> tuple[float, float]:
    vals = vals[~np.isnan(vals)]
    return float(vals.mean()), float(1.96 * vals.std(ddof=1)
                                     / np.sqrt(vals.size))


def main() -> None:
    print(f"cimba {cp.version()}, using {cp.use_threads(0)} worker threads")
    print(f"{NUM_ATTRACTIONS} attractions, {TOTAL_QUEUES} queues, "
          f"{NUM_SERVERS} ride servers; {PARK_OPEN:.0f} min park day")

    # duration is the time the entrance stays open; the cooldown lets the
    # park drain so every visitor's day is complete, as in the C version
    exp = park.experiment(replications=20, duration=PARK_OPEN, warmup=0.0,
                          cooldown=2000.0, seed=20260613)
    t0 = time.perf_counter()
    fails = exp.run()
    wall = time.perf_counter() - t0
    print(f"{len(exp)} trials in {wall:.2f} s, {fails} failed\n")

    rows = [
        ("visitors / day", "n_visitors", "%6.0f"),
        ("rides taken", "avg_rides", "%6.2f"),
        ("time in park (min)", "avg_time_in_park", "%6.1f"),
        ("  riding", "avg_riding", "%6.1f"),
        ("  waiting in queues", "avg_waiting", "%6.1f"),
        ("  walking", "avg_walking", "%6.1f"),
        ("balks / day", "n_balks", "%6.0f"),
        ("jockey moves / day", "n_jockeys", "%6.0f"),
        ("reneges / day", "n_reneges", "%6.0f"),
    ]
    print(f"{'per-visitor averages':<22} {'mean':>7} {'+/-95%':>7}")
    for label, field, fmt in rows:
        m, w = ci95(exp[field])
        print(f"{label:<22} {fmt % m:>7} {w:7.2f}")


if __name__ == "__main__":
    main()
