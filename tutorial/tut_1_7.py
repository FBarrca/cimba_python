"""Tutorial 1.7: command-line M/M/1 utilization sweep."""

from __future__ import annotations

import argparse
import time

import numpy as np

import cimba
import cimba.random as random
import cimba.sim as sim


class MM1Station(sim.Component):
    queue: sim.Queue

    @sim.process
    def arrival(self, env):
        while True:
            t_ia = random.exponential(1.0 / env.utilization)
            sim.hold(t_ia)
            self.queue.put(1)

    @sim.process
    def service(self, env):
        while True:
            self.queue.get(1)
            t_srv = random.exponential(1.0)
            sim.hold(t_srv)


class MM1(sim.Model):
    utilization: sim.Param
    avg_queue_length: sim.Output
    station: MM1Station = MM1Station()


def build_model() -> MM1:
    model = MM1("MM1")

    @model.collect
    def collect_stats(env: MM1):
        env.avg_queue_length = env.station.queue.mean_level()

    return model


def sweep_rho(
    *,
    replications: int,
    duration: float,
    warmup: float,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    rhos = np.arange(0.025, 1.0, 0.025)
    exp = build_model().experiment(
        utilization=rhos,
        replications=replications,
        duration=duration,
        warmup=warmup,
        seed=seed,
    )
    failures = exp.run()
    if failures:
        raise RuntimeError(f"{failures} trial(s) failed")
    return rhos, exp["avg_queue_length"].reshape(len(rhos), replications)


def print_sweep(rhos: np.ndarray, values: np.ndarray) -> None:
    print(f"cimba {cimba.native_version()}")
    print(f"{'rho':>8} {'simulated':>10} {'+/-95%':>10} {'theory':>10}")
    for rho, samples in zip(rhos, values):
        mean = float(samples.mean())
        ci = 0.0
        if samples.size > 1:
            ci = float(1.96 * samples.std(ddof=1) / np.sqrt(samples.size))
        theory = rho * rho / (1.0 - rho)
        print(f"{rho:8.3f} {mean:10.4f} {ci:10.4f} {theory:10.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--timing", action="store_true")
    parser.add_argument("-n", "--reps", type=int, default=10)
    parser.add_argument("-d", "--duration", type=float, default=1.0e6)
    parser.add_argument("-w", "--warmup", type=float, default=1.0e3)
    parser.add_argument("-s", "--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.perf_counter()
    rhos, values = sweep_rho(
        replications=args.reps,
        duration=args.duration,
        warmup=args.warmup,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - t0
    print_sweep(rhos, values)
    if args.timing:
        print(f"\n{len(rhos) * args.reps} trials in {elapsed:.2f} s")


if __name__ == "__main__":
    main()
