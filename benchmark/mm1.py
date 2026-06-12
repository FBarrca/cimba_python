"""
Benchmark: M/M/1 queue via the Python bindings (cimba.sim).

An arrival process puts exactly NUM_OBJECTS timestamps into an object queue at
rate 0.9; a service process takes them, serves at rate 1.0, and accumulates
time-in-system. Expected mean system time: 1/(mu - lambda) = 10.

The trial ends when the event queue drains (the arrival process finishes after
NUM_OBJECTS puts and the server starves). Lifecycle events are pushed past that
point with a huge warmup so no recording overhead occurs during the run.

Usage: uv run python examples/benchmarks/mm1.py
"""

import time

import cimba as cp
import cimba.sim as sim

NUM_OBJECTS = 1_000_000
ARRIVAL_RATE = 0.9
SERVICE_RATE = 1.0
EXPECTED_MEAN_SYSTEM_TIME = 1.0 / (SERVICE_RATE - ARRIVAL_RATE)
REPS = 5

class MM1Bench(sim.Model):
    arr_mean: sim.Param
    srv_mean: sim.Param
    avg_wait: sim.Output
    sum_wait: sim.Output
    queue: sim.Store
    obj_cnt: sim.State


mm1 = MM1Bench("mm1_bench")


@mm1.process
def arrival(env: MM1Bench):
    for _ in range(NUM_OBJECTS):
        sim.hold(sim.exponential(env.arr_mean))
        sim.store_put(env.queue, sim.f2i(sim.now()))


@mm1.process
def service(env: MM1Bench):
    env.sum_wait = 0.0
    while True:
        job = sim.store_take(env.queue)
        sim.hold(sim.exponential(env.srv_mean))
        env.sum_wait = env.sum_wait + (sim.now() - sim.i2f(job))
        env.obj_cnt = env.obj_cnt + 1


@mm1.collect
def stats(env: MM1Bench):
    env.avg_wait = env.sum_wait / env.obj_cnt


def run_trial() -> tuple[float, float]:
    """Run one trial; return (wall seconds, measured mean system time)."""
    exp = mm1.experiment(arr_mean=1.0 / ARRIVAL_RATE,
                         srv_mean=1.0 / SERVICE_RATE,
                         duration=1.0,
                         warmup=1.0e15)
    t0 = time.perf_counter()
    exp.run()
    return time.perf_counter() - t0, float(exp["avg_wait"][0])


def main() -> None:
    print(f"cimba {cp.version()}, M/M/1 via cimba.sim")
    print(f"{NUM_OBJECTS:,} jobs, expected mean system time "
          f"{EXPECTED_MEAN_SYSTEM_TIME:.1f}\n")

    t0 = time.perf_counter()
    mm1.experiment(arr_mean=1.0, srv_mean=1.0, duration=1.0)
    print(f"one-time numba compile: {time.perf_counter() - t0:.1f} s\n")

    times: list[float] = []
    last_avg = 0.0
    for i in range(REPS):
        wall, avg = run_trial()
        times.append(wall)
        last_avg = avg
        print(f"run {i + 1}/{REPS}: {wall:.3f} s, "
              f"avg system time {avg:.4f}, "
              f"{NUM_OBJECTS / wall:,.0f} jobs/s")

    best = min(times)
    print(f"\nbest of {REPS}: {best:.3f} s ({NUM_OBJECTS / best:,.0f} jobs/s)")
    print(f"last avg system time: {last_avg:.4f} "
          f"(expected {EXPECTED_MEAN_SYSTEM_TIME:.1f})")


if __name__ == "__main__":
    main()
