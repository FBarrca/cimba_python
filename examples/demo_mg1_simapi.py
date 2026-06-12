"""
Demo of the SimPy-flavored API (cimba.sim): the M/G/1 model in ~25
lines of plain Python, plus an M/M/2 model showing multi-server stations
via copies=2. No yields - cimba processes are stackful fibers, so
sim.hold() and sim.get()/sim.put() simply block.


Usage: .venv/bin/python demo_mg1_simapi.py
"""

import math
import time

import numpy as np

import cimba as cp
import cimba.sim as sim

# --- M/G/1: arrivals -> queue -> single gamma server -------------------------

class MG1(sim.Model):
    utilization: sim.Param
    service_cv: sim.Param
    avg_queue_length: sim.Output
    queue: sim.Queue


mg1 = MG1()


@mg1.process
def arrivals(env: MG1):
    mean_interarr = 1.0 / env.utilization
    while True:
        sim.hold(sim.exponential(mean_interarr))
        sim.put(env.queue, 1)


@mg1.process
def service(env: MG1):
    shape = 1.0 / (env.service_cv * env.service_cv)
    scale = env.service_cv * env.service_cv
    while True:
        sim.hold(sim.gamma(shape, scale))
        sim.get(env.queue, 1)


@mg1.collect
def mg1_stats(env: MG1):
    env.avg_queue_length = sim.mean_level(env.queue)


# --- M/M/2: two identical servers pulling from one queue ---------------------

class MM2(sim.Model):
    utilization: sim.Param      # per-server utilization
    avg_queue_length: sim.Output
    queue: sim.Queue


mm2 = MM2()


@mm2.process
def mm2_arrivals(env: MM2):
    mean_interarr = 1.0 / (2.0 * env.utilization)
    while True:
        sim.hold(sim.exponential(mean_interarr))
        sim.put(env.queue, 1)


@mm2.process(copies=2)
def mm2_server(env: MM2):
    while True:
        sim.get(env.queue, 1)
        sim.hold(sim.exponential(1.0))


@mm2.collect
def mm2_stats(env: MM2):
    env.avg_queue_length = sim.mean_level(env.queue)


# --- Theory -------------------------------------------------------------------

def pk_queue_length(rho, cv):
    """Pollaczek-Khinchine mean waiting-queue length for M/G/1."""
    return rho ** 2 * (1.0 + cv ** 2) / (2.0 * (1.0 - rho))


def mm2_lq(rho):
    """M/M/2 mean waiting-queue length (Erlang C, c=2, mean service 1)."""
    a = 2.0 * rho
    p0 = 1.0 + a + a * a / (2.0 * (1.0 - rho))
    pw = (a * a / (2.0 * (1.0 - rho))) / p0
    return pw * rho / (1.0 - rho)


def report(exp, theory_func, *param_names):
    points = sorted(set(zip(*(exp[p] for p in param_names))))
    header = " ".join(f"{p[:10]:>10}" for p in param_names)
    print(f"{header} {'simulated':>10} {'+/-95%':>7} {'theory':>8}")
    for point in points:
        sel = np.ones(len(exp), dtype=bool)
        for p, v in zip(param_names, point):
            sel &= exp[p] == v
        vals = exp["avg_queue_length"][sel]
        vals = vals[~np.isnan(vals)]
        mean = vals.mean()
        ci95 = 1.96 * vals.std(ddof=1) / np.sqrt(vals.size)
        cols = " ".join(f"{v:10.2f}" for v in point)
        print(f"{cols} {mean:10.4f} {ci95:7.4f} {theory_func(*point):8.4f}")
    print()


def main() -> None:
    print(f"cimba {cp.version()}, using {cp.use_threads(0)} worker threads")

    exp1 = mg1.experiment(utilization=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9],
                          service_cv=[0.5, 1.0, 2.0],
                          replications=20,
                          duration=1.0e5,
                          warmup=1.0e3,
                          seed=20260611)
    t0 = time.perf_counter()
    fails1 = exp1.run()
    wall1 = time.perf_counter() - t0

    exp2 = mm2.experiment(utilization=[0.3, 0.6, 0.8, 0.9],
                          replications=20,
                          duration=1.0e5,
                          warmup=1.0e3,
                          seed=4711)
    t0 = time.perf_counter()
    fails2 = exp2.run()
    wall2 = time.perf_counter() - t0

    print(f"M/G/1: {len(exp1)} trials in {wall1:.2f} s ({fails1} failed); "
          f"M/M/2: {len(exp2)} trials in {wall2:.2f} s ({fails2} failed)\n")

    print("M/G/1 vs Pollaczek-Khinchine:")
    report(exp1, pk_queue_length, "utilization", "service_cv")

    print("M/M/2 vs Erlang C:")
    report(exp2, mm2_lq, "utilization")


if __name__ == "__main__":
    main()
