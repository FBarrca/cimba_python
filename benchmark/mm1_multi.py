"""
Benchmark: M/M/1 queue via the Python bindings (cimba.sim), multi-core.

Runs NUM_TRIALS independent trials in parallel (one million jobs each) using
cimba_run_experiment through Experiment.run().

Usage: uv run python benchmark/mm1_multi.py
"""

import statistics
import time

import cimba as cp
import cimba.random as random
import cimba.sim as sim

NUM_OBJECTS = 1_000_000
ARRIVAL_RATE = 0.9
SERVICE_RATE = 1.0
EXPECTED_MEAN_SYSTEM_TIME = 1.0 / (SERVICE_RATE - ARRIVAL_RATE)
NUM_TRIALS = 100

class MM1Bench(sim.Model):
    arr_mean: sim.Param
    srv_mean: sim.Param
    avg_wait: sim.Output
    sum_wait: sim.Output
    queue: sim.Store
    obj_cnt: sim.State


mm1 = MM1Bench("mm1_bench_multi")


@mm1.process
def arrival(env: MM1Bench):
    for _ in range(NUM_OBJECTS):
        sim.hold(random.exponential(env.arr_mean))
        sim.store_put(env.queue, sim.f2i(sim.now()))


@mm1.process
def service(env: MM1Bench):
    env.sum_wait = 0.0
    while True:
        job = sim.store_take(env.queue)
        sim.hold(random.exponential(env.srv_mean))
        env.sum_wait = env.sum_wait + (sim.now() - sim.i2f(job))
        env.obj_cnt = env.obj_cnt + 1


@mm1.collect
def stats(env: MM1Bench):
    env.avg_wait = env.sum_wait / env.obj_cnt


def main() -> None:
    print(f"cimba {cp.version()}, M/M/1 via cimba.sim (multi-core)")
    print(f"{NUM_TRIALS} trials × {NUM_OBJECTS:,} jobs, "
          f"expected mean system time {EXPECTED_MEAN_SYSTEM_TIME:.1f}\n")

    t0 = time.perf_counter()
    mm1.experiment(arr_mean=1.0, srv_mean=1.0, duration=1.0)
    print(f"one-time numba compile: {time.perf_counter() - t0:.1f} s\n")

    exp = mm1.experiment(arr_mean=1.0 / ARRIVAL_RATE,
                         srv_mean=1.0 / SERVICE_RATE,
                         duration=1.0,
                         warmup=1.0e15,
                         replications=NUM_TRIALS)

    t0 = time.perf_counter()
    exp.run()
    wall = time.perf_counter() - t0

    avgs = [float(x) for x in exp["avg_wait"] if x == x]
    n = len(avgs)
    mean_tsys = statistics.mean(avgs)
    if n > 1:
        sdev = statistics.stdev(avgs)
        serr = sdev / (n ** 0.5)
        ci_w = 1.96 * serr
        ci_l = mean_tsys - ci_w
        ci_u = mean_tsys + ci_w
        print(f"wall time: {wall:.3f} s "
              f"({NUM_TRIALS * NUM_OBJECTS / wall:,.0f} jobs/s aggregate)")
        print(f"average system time: {mean_tsys:.6f} "
              f"(n {n}, conf int {ci_l:.6f} - {ci_u:.6f}, "
              f"expected {EXPECTED_MEAN_SYSTEM_TIME:.1f})")
    else:
        print(f"wall time: {wall:.3f} s")
        print(f"average system time: {mean_tsys:.6f}")


if __name__ == "__main__":
    main()
