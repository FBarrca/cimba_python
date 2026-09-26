"""Finite M/M/1 queue benchmark using the 0.7 model and input APIs."""

from __future__ import annotations

import argparse
import statistics
import time

import cimba as cb
from cimba import inputs


class MM1Bench(cb.Model):
    jobs: cb.Param[int] = 1_000_000
    interarrival: cb.Input[float] = inputs.dist.exponential(mean=1 / 0.9)
    service_time: cb.Input[float] = inputs.dist.exponential(mean=1)
    arrivals: cb.Store[float]
    completed: cb.State[int] = 0
    total_system_time: cb.State[float] = 0.0
    mean_system_time: cb.Output[float]

    @cb.process
    def generate(self):
        for _ in range(self.jobs):
            cb.hold(self.interarrival.next())
            self.arrivals.put(cb.now())

    @cb.process
    def server(self):
        while True:
            arrived = self.arrivals.get()
            cb.hold(self.service_time.next())
            self.total_system_time += cb.now() - arrived
            self.completed += 1

    @cb.on_end
    def measure(self):
        self.mean_system_time = (self.total_system_time / self.completed
                                 if self.completed else 0.0)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1_000_000)
    parser.add_argument("--reps", type=int, default=10)
    args = parser.parse_args(argv)
    model = MM1Bench()
    model.jobs = args.jobs
    experiment = cb.Experiment(model, replications=1,
                               window=cb.Window.until_idle(), seed=42)
    times = []
    for index in range(args.reps):
        started = time.perf_counter()
        result = experiment.run(workers=1)
        elapsed = time.perf_counter() - started
        if result.failed.any():
            raise RuntimeError(result.failure_reasons)
        times.append(elapsed)
        print(f"{index + 1}: {elapsed:.3f}s, "
              f"mean system time {result[model].mean_system_time[0, 0]:.4f}")
    print(f"median: {statistics.median(times):.3f}s for {args.jobs:,} jobs")


if __name__ == "__main__":
    main()
