"""Tutorial 1.2: fixed window, queue history and arrival observations."""

import cimba as cb
from cimba import inputs


class MM1(cb.Model):
    interarrival: cb.Input[float] = inputs.dist.exponential(mean=1.0 / 0.75)
    service_time: cb.Input[float] = inputs.dist.exponential(mean=1.0)
    queue: cb.Container
    interarrival_times: cb.Dataset
    avg_queue_length: cb.Output[float]
    avg_interarrival_time: cb.Output[float]

    @cb.process
    def arrival(self):
        while True:
            gap = self.interarrival.next()
            cb.hold(gap)
            self.interarrival_times.record(gap)
            self.queue.put(1)

    @cb.process
    def service(self):
        while True:
            self.queue.get(1)
            cb.hold(self.service_time.next())

    @cb.on_end
    def collect_stats(self):
        self.avg_queue_length = self.queue.mean_level()
        self.avg_interarrival_time = self.interarrival_times.sample_mean()


def main() -> None:
    model = MM1()
    model.queue = cb.Container()
    model.queue.capture()
    results = cb.Experiment(model, window=cb.Window(duration=25.0), seed=42).run()
    if results.failed.any():
        raise RuntimeError("M/M/1 trial failed")
    time, level = results[model].queue.trial(0, 0)
    print(f"Queue mean: {results[model].avg_queue_length[0, 0]:.3f}; "
          f"interarrival mean: {results[model].avg_interarrival_time[0, 0]:.3f}; "
          f"captured points: {len(time)}")


if __name__ == "__main__":
    main()
