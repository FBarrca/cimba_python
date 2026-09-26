"""Tutorial 4.1: weather-gated harbor with dynamic ships and resources."""

import time

import numpy as np

import cimba as cb
from cimba import inputs


SMALL, LARGE = 0, 1
TUGS_NEEDED = (1, 3)
MAX_WIND = (10.0, 12.0)
MIN_DEPTH = (8.0, 13.0)
HOURS_PER_YEAR = 24.0 * 7 * 52


class Ship(cb.Model):
    harbor: cb.Ref["Harbor"]
    size: cb.State[int]
    tugs_needed: cb.State[int]
    max_wind: cb.State[float]
    min_depth: cb.State[float]
    arrival: cb.State[float]

    @cb.process
    def voyage(self):
        sea = self.harbor.sea
        facilities = self.harbor.facilities
        if self.size == LARGE:
            berths = facilities.berths_large
        else:
            berths = facilities.berths_small

        while True:
            ready = (sea.water_depth >= self.min_depth and
                     sea.wind_mag <= self.max_wind and
                     facilities.tugs.available() >= self.tugs_needed and
                     berths.available() >= 1)
            if ready:
                break
            facilities.harbormaster.wait_until(self.harbor.should_call_harbormaster)

        berths.acquire(1)
        facilities.tugs.acquire(self.tugs_needed)
        facilities.comms.acquire()
        cb.hold(self.harbor.traffic.radio_delay.next())
        facilities.comms.release()
        cb.hold(self.harbor.traffic.movement.next())
        facilities.tugs.release(self.tugs_needed)
        facilities.harbormaster.signal()

        if self.size == LARGE:
            cb.hold(self.harbor.traffic.unload_large.next())
        else:
            cb.hold(self.harbor.traffic.unload_small.next())

        facilities.tugs.acquire(self.tugs_needed)
        facilities.comms.acquire()
        cb.hold(self.harbor.traffic.radio_delay.next())
        facilities.comms.release()
        cb.hold(self.harbor.traffic.movement.next())
        berths.release(1)
        facilities.tugs.release(self.tugs_needed)
        facilities.harbormaster.signal()
        if self.size == LARGE:
            self.harbor.traffic.time_large.record(cb.now() - self.arrival)
        else:
            self.harbor.traffic.time_small.record(cb.now() - self.arrival)
        self.harbor.traffic.departed.put(self)


class SeaConditions(cb.Model):
    harbor: cb.Ref["Harbor"]
    wind_source: cb.Input[float] = inputs.dist.rayleigh(scale=5.0)
    direction_source: cb.Input[float] = inputs.dist.pert(
        low=0.0, mode=225.0, high=360.0)
    wind_mag: cb.State[float] = 0.0
    wind_dir: cb.State[float] = 0.0
    water_depth: cb.State[float] = 0.0

    def __init__(self, harbor, mean_wind):
        self.harbor = harbor
        self.wind_source = inputs.dist.rayleigh(scale=mean_wind)

    @cb.process(priority=1)
    def weather(self):
        while True:
            self.wind_mag = 0.5 * self.wind_source.next() + 0.5 * self.wind_mag
            self.wind_dir = self.direction_source.next()
            cb.hold(1.0)

    @cb.process
    def tide(self):
        while True:
            t = cb.now()
            pi = np.pi
            astronomical = (self.harbor.reference_depth +
                            np.sin(2.0 * pi * t / 12.4) +
                            0.5 * np.sin(2.0 * pi * t / 24.0) +
                            0.25 * np.sin(2.0 * pi * t / (0.5 * 29.5 * 24)))
            wind_effect = (0.5 * self.wind_mag - 0.5 * self.wind_mag *
                           np.sin(self.wind_dir * pi / 180.0))
            self.water_depth = astronomical + wind_effect
            self.harbor.facilities.harbormaster.signal()
            cb.hold(1.0)


class HarborFacilities(cb.Model):
    tugs: cb.Resource
    berths_small: cb.Resource
    berths_large: cb.Resource
    comms: cb.Resource
    harbormaster: cb.Condition

    def __init__(self, num_tugs, num_berths_small, num_berths_large):
        self.tugs = cb.Resource(capacity=num_tugs)
        self.berths_small = cb.Resource(capacity=num_berths_small)
        self.berths_large = cb.Resource(capacity=num_berths_large)
        self.comms = cb.Resource(capacity=1)


class ShipTraffic(cb.Model):
    harbor: cb.Ref["Harbor"]
    arrival_gap: cb.Input[float] = inputs.dist.exponential(mean=2.0)
    radio_delay: cb.Input[float] = inputs.dist.gamma(shape=5.0, scale=0.01)
    movement: cb.Input[float] = inputs.dist.pert(low=0.4, mode=0.5, high=0.8)
    unload_small: cb.Input[float] = inputs.dist.pert(low=6.0, mode=8.0, high=16.0)
    unload_large: cb.Input[float] = inputs.dist.pert(low=9.0, mode=12.0, high=24.0)
    departed: cb.Store[Ship]
    time_small: cb.Dataset
    time_large: cb.Dataset

    def __init__(self, harbor, arrival_rate, unload_avg_small, unload_avg_large):
        self.harbor = harbor
        self.arrival_gap = inputs.dist.exponential(mean=1.0 / arrival_rate)
        self.unload_small = inputs.dist.pert(
            low=0.75 * unload_avg_small, mode=unload_avg_small,
            high=2.0 * unload_avg_small)
        self.unload_large = inputs.dist.pert(
            low=0.75 * unload_avg_large, mode=unload_avg_large,
            high=2.0 * unload_avg_large)

    @cb.process
    def arrivals(self):
        while True:
            cb.hold(self.arrival_gap.next())
            size = int(cb.random.bernoulli(self.harbor.percent_large))
            cb.spawn(Ship, harbor=self.harbor, size=size,
                     tugs_needed=TUGS_NEEDED[size],
                     max_wind=MAX_WIND[size], min_depth=MIN_DEPTH[size],
                     arrival=cb.now())

    @cb.process
    def departures(self):
        while True:
            cb.release(self.departed.get())


class Harbor(cb.Model):
    reference_depth: cb.Param[float] = 15.0
    percent_large: cb.Param[float] = 0.25
    sea: SeaConditions
    facilities: HarborFacilities
    traffic: ShipTraffic
    avg_time_small: cb.Output[float]
    avg_time_large: cb.Output[float]
    n_small: cb.Output[int]
    n_large: cb.Output[int]
    tug_util: cb.Output[float]
    berth_small_util: cb.Output[float]
    berth_large_util: cb.Output[float]

    def __init__(self, *, mean_wind=5.0, reference_depth=15.0,
                 arrival_rate=0.5, percent_large=0.25, num_tugs=10,
                 num_berths_small=6, num_berths_large=3,
                 unload_avg_small=8.0, unload_avg_large=12.0):
        self.reference_depth = reference_depth
        self.percent_large = percent_large
        self.facilities = HarborFacilities(
            int(num_tugs), int(num_berths_small), int(num_berths_large))
        self.sea = SeaConditions(self, mean_wind)
        self.traffic = ShipTraffic(
            self, arrival_rate, unload_avg_small, unload_avg_large)

    @cb.predicate
    def should_call_harbormaster(self):
        return True

    @cb.on_end
    def harbor_stats(self):
        self.avg_time_small = self.traffic.time_small.sample_mean()
        self.avg_time_large = self.traffic.time_large.sample_mean()
        self.n_small = self.traffic.time_small.sample_count()
        self.n_large = self.traffic.time_large.sample_count()
        self.tug_util = self.facilities.tugs.mean_in_use()
        self.berth_small_util = self.facilities.berths_small.mean_in_use()
        self.berth_large_util = self.facilities.berths_large.mean_in_use()


def main() -> None:
    harbor = Harbor()
    start = time.perf_counter()
    results = cb.Experiment(
        harbor, replications=20,
        window=cb.Window(warmup=24.0, duration=HOURS_PER_YEAR),
        seed=20260612,
    ).run()
    if results.failed.any():
        raise RuntimeError(f"{results.failed.sum()} harbor trials failed")
    print(f"cimba {cb.engine_version()}: {results.failed.size} harbor trials "
          f"in {time.perf_counter() - start:.2f} s")
    for field in ("n_small", "n_large", "avg_time_small", "avg_time_large",
                  "tug_util", "berth_small_util", "berth_large_util"):
        print(f"{field}: {getattr(results[harbor], field).values.mean():.3f}")


if __name__ == "__main__":
    main()
