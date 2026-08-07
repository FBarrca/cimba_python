"""Tutorial 1.1: a first M/M/1 queue model."""

import cimba
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
        self.avg_queue_length = self.queue.mean_level()


model = MM1("MM1")








def main() -> None:
    cimba.logger_flags_on(cimba.LOGGER_INFO)
    exp = model.experiment(
        utilization=[0.75],
        replications=1,
        duration=10.0,
        warmup=0.0,
        seed=42,
    )
    failures = exp.run()
    if failures:
        raise RuntimeError(f"{failures} trial(s) failed")
    avg = float(exp.results.avg_queue_length[0])
    print(f"Average queue length over the first 10 time units: {avg:.6f}")


if __name__ == "__main__":
    main()
