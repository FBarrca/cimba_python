"""
Demo of the cimba concept set through the SimPy-flavored API: three small
models exercising resources, datasets (tally statistics), stores
(object queues with timestamp objects), conditions with predicates, state
fields, and multi-copy processes with per-copy indices.

1. Machine repair shop - the classic finite-source M/M/1//N "machine
   interference" model: N machines fail and queue for a single repairman
   (a cmb_resource). Validated against birth-death theory and Little's
   law, including the repairman utilization and mean downtime tallied in
   a cmb_dataset.

2. M/M/1 waiting times - jobs carry their arrival time through a store
   (cmb_objectqueue) as bit-cast doubles; the server tallies each job's
   waiting time. Validated against Wq = rho/(1-rho).

3. Counting gate - a producer increments a state counter and signals a
   condition (cmb_condition); a waiter blocks on a @model.predicate until
   the count reaches a target, then records the time. Validated against
   the Erlang mean K * E[interarrival].

Usage: .venv/bin/python demo_repairshop.py
"""

import math
import time

import numpy as np

import cimba as cp
import cimba.sim as sim

N_MACHINES = 8

# --- 1. Machine repair shop ---------------------------------------------------

class RepairShop(sim.Model):
    mtbf: sim.Param
    repair_time: sim.Param
    avg_broken: sim.Output
    repair_util: sim.Output
    mean_downtime: sim.Output
    failures: sim.Output
    broken: sim.Queue           # counter of broken machines
    repairman: sim.Resource
    downtime: sim.Dataset


shop = RepairShop()


@shop.process(copies=N_MACHINES)
def machine(env: RepairShop, idx: int):
    # idx identifies the machine; here all machines are identical
    while True:
        sim.hold(sim.exponential(env.mtbf))
        t_fail = sim.now()
        sim.put(env.broken, 1)
        sim.acquire(env.repairman)
        sim.hold(sim.exponential(env.repair_time))
        sim.release(env.repairman)
        sim.get(env.broken, 1)
        sim.tally(env.downtime, sim.now() - t_fail)


@shop.collect
def shop_stats(env: RepairShop):
    env.avg_broken = sim.mean_level(env.broken)
    env.repair_util = sim.mean_in_use(env.repairman)
    env.mean_downtime = sim.dataset_mean(env.downtime)
    env.failures = sim.dataset_count(env.downtime)


def repairshop_theory(n, mtbf, repair_time):
    """Birth-death solution of the machine interference model."""
    rho = repair_time / mtbf
    weights = [math.perm(n, k) * rho ** k for k in range(n + 1)]
    total = sum(weights)
    p = [w / total for w in weights]
    avg_broken = sum(k * pk for k, pk in enumerate(p))
    repair_util = 1.0 - p[0]
    throughput = repair_util / repair_time          # repairs per time unit
    mean_downtime = avg_broken / throughput         # Little's law
    return avg_broken, repair_util, mean_downtime


# --- 2. M/M/1 waiting times through a store ----------------------------------

class MM1Waits(sim.Model):
    utilization: sim.Param
    avg_wait: sim.Output
    avg_jobs_waiting: sim.Output
    jobs: sim.Store             # carries arrival times as bit-cast doubles
    waits: sim.Dataset


mm1w = MM1Waits()


@mm1w.process
def mm1_arrivals(env: MM1Waits):
    mean_interarr = 1.0 / env.utilization
    while True:
        sim.hold(sim.exponential(mean_interarr))
        sim.store_put(env.jobs, sim.f2i(sim.now()))


@mm1w.process
def mm1_server(env: MM1Waits):
    while True:
        job = sim.store_take(env.jobs)
        sim.tally(env.waits, sim.now() - sim.i2f(job))
        sim.hold(sim.exponential(1.0))


@mm1w.collect
def mm1w_stats(env: MM1Waits):
    env.avg_wait = sim.dataset_mean(env.waits)
    env.avg_jobs_waiting = sim.store_mean_length(env.jobs)


# --- 3. Counting gate with a condition variable --------------------------------

class Gate(sim.Model):
    target: sim.Param
    t_done: sim.Output
    enough: sim.Condition
    count: sim.State
    reached: sim.Predicate      # bound by the @gate.predicate below


gate = Gate()


@gate.predicate
def reached(env: Gate) -> bool:
    return env.count >= env.target


@gate.process
def producer(env: Gate):
    while True:
        sim.hold(sim.exponential(1.0))
        env.count = env.count + 1
        sim.signal(env.enough)


@gate.process
def waiter(env: Gate):
    sim.wait_for(env.enough, env.reached, env)
    env.t_done = sim.now()
    while True:                 # idle until the trial ends
        sim.hold(1.0e12)


# --- Reports --------------------------------------------------------------------

def ci95(vals):
    vals = vals[~np.isnan(vals)]
    return vals.mean(), 1.96 * vals.std(ddof=1) / np.sqrt(vals.size)


def main() -> None:
    print(f"cimba {cp.version()}, using {cp.use_threads(0)} worker threads")

    exp1 = shop.experiment(mtbf=[10.0, 5.0], repair_time=1.0,
                           replications=20, duration=2.0e4, warmup=1.0e3,
                           seed=20260611)
    exp2 = mm1w.experiment(utilization=[0.5, 0.8, 0.9],
                           replications=20, duration=1.0e5, warmup=1.0e3,
                           seed=42)
    exp3 = gate.experiment(target=[100.0, 400.0],
                           replications=50, duration=1.0e4, warmup=0.0,
                           seed=7)

    t0 = time.perf_counter()
    fails = exp1.run() + exp2.run() + exp3.run()
    wall = time.perf_counter() - t0
    n = len(exp1) + len(exp2) + len(exp3)
    print(f"{n} trials across 3 models in {wall:.2f} s, {fails} failed\n")

    print(f"Repair shop, {N_MACHINES} machines, repair time 1.0 "
          f"(simulated vs theory):")
    print(f"{'mtbf':>5} {'broken':>7} {'+/-':>6} {'thry':>6}  "
          f"{'util':>5} {'+/-':>6} {'thry':>6}  "
          f"{'downtime':>8} {'+/-':>6} {'thry':>6}")
    for mtbf in (10.0, 5.0):
        s = exp1["mtbf"] == mtbf
        tb, tu, td = repairshop_theory(N_MACHINES, mtbf, 1.0)
        b, bw = ci95(exp1["avg_broken"][s])
        u, uw = ci95(exp1["repair_util"][s])
        d, dw = ci95(exp1["mean_downtime"][s])
        print(f"{mtbf:5.1f} {b:7.3f} {bw:6.3f} {tb:6.3f}  "
              f"{u:5.3f} {uw:6.3f} {tu:6.3f}  "
              f"{d:8.3f} {dw:6.3f} {td:6.3f}")

    print("\nM/M/1 waiting times via store (simulated vs theory):")
    print(f"{'util':>5} {'Wq':>7} {'+/-':>6} {'thry':>7}  "
          f"{'Lq':>7} {'+/-':>6} {'thry':>7}")
    for u in (0.5, 0.8, 0.9):
        s = exp2["utilization"] == u
        w, ww = ci95(exp2["avg_wait"][s])
        l, lw = ci95(exp2["avg_jobs_waiting"][s])
        print(f"{u:5.2f} {w:7.3f} {ww:6.3f} {u / (1 - u):7.3f}  "
              f"{l:7.3f} {lw:6.3f} {u * u / (1 - u):7.3f}")

    print("\nCounting gate via condition variable (simulated vs theory):")
    print(f"{'target':>6} {'t_done':>8} {'+/-':>6} {'theory':>7}")
    for k in (100.0, 400.0):
        s = exp3["target"] == k
        t, tw = ci95(exp3["t_done"][s])
        print(f"{k:6.0f} {t:8.2f} {tw:6.2f} {k:7.1f}")


if __name__ == "__main__":
    main()
