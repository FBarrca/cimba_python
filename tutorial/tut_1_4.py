"""Tutorial 1.4: collect queue statistics over a long run."""

import cimba.random as random
import cimba.sim as sim

class MM1(sim.Model):
    utilization: sim.Param
    avg_queue_length: sim.Output
    queue: sim.Queue

    @sim.process
    def arrival(self):
        while True:
            t_ia = random.exponential(1.0 / self.utilization)
            sim.hold(t_ia)
            self.queue.put(1)

    @sim.process
    def service(self):
        while True:
            self.queue.get(1)
            t_srv = random.exponential(1.0)
            sim.hold(t_srv)

    @sim.collect
    def collect_stats(self):
        self.avg_queue_length = self.queue.history().mean()
        self.queue.report()
        self.queue.history().pacf_correlogram(lags=20)

model = MM1("MM1")

def main() -> None:
    exp = model.experiment(
        utilization=[0.75],
        replications=1,
        duration=1.0e6,
        warmup=1.0e3,
        seed=45,
    )
    failures = exp.run()
    if failures:
        raise RuntimeError(f"{failures} trial(s) failed")
    avg = float(exp.results.avg_queue_length[0])
    print("Theory predicts an average M/M/1 waiting-queue length of 2.25")
    print(f"Simulation result: {avg:.6f}")


if __name__ == "__main__":
    main()
