"""Tutorial 1.3: user logging from compiled processes."""

import cimba as cb
from cimba import inputs


USERFLAG1 = 1


class MM1(cb.Model):
    interarrival: cb.Input[float] = inputs.dist.exponential(mean=1.0 / 0.75)
    service_time: cb.Input[float] = inputs.dist.exponential(mean=1.0)
    queue: cb.Container
    avg_queue_length: cb.Output[float]

    @cb.process
    def arrival(self):
        while True:
            gap = self.interarrival.next()
            cb.log(USERFLAG1, "Holds for", gap)
            cb.hold(gap)
            cb.log(USERFLAG1, "Puts one into the queue")
            self.queue.put(1)

    @cb.process
    def service(self):
        while True:
            cb.log(USERFLAG1, "Gets one from the queue")
            self.queue.get(1)
            duration = self.service_time.next()
            cb.log(USERFLAG1, "Got one, services it for", duration)
            cb.hold(duration)

    @cb.on_end
    def collect_stats(self):
        self.avg_queue_length = self.queue.mean_level()


def main() -> None:
    cb.set_engine_log_level(USERFLAG1)
    try:
        model = MM1()
        results = cb.Experiment(
            model, window=cb.Window(duration=10.0), seed=44,
        ).run(workers=1)
        if results.failed.any():
            raise RuntimeError("M/M/1 trial failed")
        print(f"Average queue length with user logging enabled: "
              f"{results[model].avg_queue_length[0, 0]:.6f}")
    finally:
        cb.set_engine_log_level(0)


if __name__ == "__main__":
    main()
