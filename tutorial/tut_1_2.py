"""Tutorial 1.2: stop the M/M/1 simulation at a fixed duration."""

import os
from pathlib import Path

import cimba
import cimba.random as random
import cimba.sim as sim

PLOT_PATH = Path("queue_history.png")
_NON_INTERACTIVE_BACKENDS = frozenset(
    {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}
)


class MM1(sim.Model):
    utilization: sim.Param
    avg_queue_length: sim.Output
    avg_interarrival_time: sim.Output
    queue: sim.Queue
    interarrival_times: sim.Dataset

    @sim.process
    def arrival(self: "MM1"):
        while True:
            t_ia = random.exponential(1.0 / self.utilization)
            self.interarrival_times.add(t_ia)
            sim.hold(t_ia)
            self.queue.put(1)

    @sim.process
    def service(self: "MM1"):
        while True:
            self.queue.get(1)
            t_srv = random.exponential(1.0)
            sim.hold(t_srv)

    @sim.collect
    def collect_stats(self: "MM1"):
        self.avg_queue_length = self.queue.history().mean()
        self.avg_interarrival_time = self.interarrival_times.mean()
        self.queue.history().capture()
        self.interarrival_times.capture()


model = MM1("MM1")

def main() -> None:
    cimba.logger_flags_on(cimba.LOGGER_INFO)
    exp = model.experiment(
        utilization=[0.75],
        replications=1,
        duration=10.0,
        warmup=0.0,
        seed=43,
    )
    failures = exp.run()
    if failures:
        raise RuntimeError(f"{failures} trial(s) failed")
    avg = float(exp.results.avg_queue_length[0])
    avg_interarrival = float(exp.results.avg_interarrival_time[0])
    queue_history = exp.history("queue")
    interarrivals = exp.dataset("interarrival_times")
    print(f"Simulation stopped at t=10.0, average queue length: {avg:.6f}")
    print(f"Average sampled interarrival time: {avg_interarrival:.6f}")
    print("First queue history rows: time, level, duration")
    print(queue_history.shape)
    print("Captured interarrival samples")
    print(interarrivals.shape)


if __name__ == "__main__":
    main()
