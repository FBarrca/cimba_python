"""
Tutorial 1.7 via the Python bindings (cimba.sim): M/M/1 queue sweep across
utilization (rho) from 0.025 to 0.975, replications per point, measuring mean
waiting queue length against M/M/1 theory (L_q = rho^2 / (1 - rho)).

Usage: uv run python examples/benchmarks/tut_1_7.py [-t] [-n N] [-d D] [-w W] [-s S]
       -t              Print wall-clock timing
       -n N            Replications per rho point (default 10)
       -d D            Duration per trial (default 1e6)
       -w W            Warmup time per trial (default 1000)
       -s S            Master seed (default hardware random)
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import cimba as cp
import cimba.sim as sim

N_RHOS = 39
RHO_START = 0.025
RHO_STEP = 0.025
SRV_RATE = 1.0
T_CRIT_95 = 2.228  # t-critical for df=9 (10 replications)

mm1 = sim.Model("mm1_tutorial",
                params=["arr_rate", "srv_rate", "warmup_time", "dur_s"],
                outputs=["avg_queue_length"],
                queues=["queue"],
                state=["seed_used"])


@mm1.process
def arrival(env):
    t_ia_mean = 1.0 / env.arr_rate
    while True:
        sim.hold(sim.exponential(t_ia_mean))
        sim.put(env.queue, 1)


@mm1.process
def service(env):
    t_srv_mean = 1.0 / env.srv_rate
    while True:
        sim.get(env.queue, 1)
        sim.hold(sim.exponential(t_srv_mean))


@mm1.collect
def collect_stats(env):
    env.avg_queue_length = sim.mean_level(env.queue)


def mm1_theory_lq(rho: float) -> float:
    return rho ** 2 / (1.0 - rho)


def run_experiment(*,
                   n_reps: int,
                   duration: float,
                   warmup_time: float,
                   seed: int | None) -> tuple[float, np.ndarray]:
    rhos = np.arange(N_RHOS) * RHO_STEP + RHO_START
    arr_rates = rhos * SRV_RATE

    t0 = time.perf_counter()
    exp = mm1.experiment(arr_rate=arr_rates,
                         srv_rate=SRV_RATE,
                         warmup_time=warmup_time,
                         dur_s=duration,
                         replications=n_reps,
                         seed=seed)
    exp.run()
    return time.perf_counter() - t0, exp["avg_queue_length"]


def print_report(rhos: np.ndarray,
                 queue_lengths: np.ndarray,
                 n_reps: int) -> None:
    print(f"{'rho':>8} {'simulated':>10} {'+/-95%':>7} {'theory':>8}")
    for i, rho in enumerate(rhos):
        vals = queue_lengths[i * n_reps:(i + 1) * n_reps]
        mean = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if n_reps > 1 else 0.0
        ci = T_CRIT_95 * sd if n_reps == 10 else 1.96 * sd / np.sqrt(n_reps)
        print(f"{rho:8.3f} {mean:10.4f} {ci:7.4f} {mm1_theory_lq(rho):8.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--timing", action="store_true",
                        help="Print wall-clock timing")
    parser.add_argument("-n", "--reps", type=int, default=10,
                        help="Replications per rho (default 10)")
    parser.add_argument("-d", "--duration", type=float, default=1e6,
                        help="Duration per trial (default 1e6)")
    parser.add_argument("-w", "--warmup", type=float, default=1000,
                        help="Warmup time per trial (default 1000)")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="Master seed (default hardware random)")
    args = parser.parse_args()

    n_trials = N_RHOS * args.reps
    print(f"cimba {cp.version()}")
    print(f"M/M/1 tutorial 1.7 (cimba.sim): {N_RHOS} rho values × "
          f"{args.reps} replications ({n_trials} trials)\n")

    cp.use_threads(0)

    print("Compiling model...")
    t0 = time.perf_counter()
    mm1.experiment(arr_rate=[0.5], srv_rate=SRV_RATE,
                   warmup_time=args.warmup, dur_s=args.duration,
                   replications=1)
    print(f"  compile: {time.perf_counter() - t0:.2f} s\n")

    print("Running experiment...")
    elapsed, queue_lengths = run_experiment(n_reps=args.reps,
                                            duration=args.duration,
                                            warmup_time=args.warmup,
                                            seed=args.seed)
    if args.timing:
        print(f"  elapsed: {elapsed:.2f} s ({n_trials / elapsed:,.0f} trials/s)\n")

    rhos = np.arange(N_RHOS) * RHO_STEP + RHO_START
    print_report(rhos, queue_lengths, args.reps)


if __name__ == "__main__":
    main()
