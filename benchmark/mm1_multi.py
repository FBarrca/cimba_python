"""Parallel M/M/1 trial benchmark using the 0.7 API."""

from __future__ import annotations

import argparse
import time

import cimba as cb
from mm1 import MM1Bench


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1_000_000)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    model = MM1Bench()
    model.jobs = args.jobs
    experiment = cb.Experiment(model, replications=args.trials,
                               window=cb.Window.until_idle(), seed=42)
    started = time.perf_counter()
    result = experiment.run(workers=args.workers)
    elapsed = time.perf_counter() - started
    if result.failed.any():
        raise RuntimeError(result.failure_reasons)
    values = result[model].mean_system_time.values
    print(f"{args.trials} trials x {args.jobs:,} jobs in {elapsed:.3f}s")
    print(f"mean system time: {values.mean():.4f} (theory: 10.0)")


if __name__ == "__main__":
    main()
