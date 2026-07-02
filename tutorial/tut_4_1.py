"""
Harbor simulation - tutorial 4.1 (subprojects/cimba/tutorial/tut_4_1.c)
through the Python bindings.

Ships arrive at a harbor and wait offshore until the harbormaster clears
them to dock: the water must be deep enough, the wind calm enough, and a
berth of the right size plus enough tugboats available. The environment
is driven by a weather process (Rayleigh wind, smoothed) and a tide
process (astronomical + weather-driven components) that signals the
harbormaster every hour. Docked ships dismiss the tugs, unload, then call
the tugs back to depart. Every movement first claims the radio channel.

Translation notes (C -> cimba.sim):

* The harbor is grouped into components: SeaConditions owns weather/tide
  state and processes, HarborFacilities owns tugs/berths/radio/condition,
  and ShipTraffic owns arrivals, dynamic ships, departures, and datasets.
* Ships are dynamic processes, as in C: arrivals sim.spawn()s one per
  arrival through the component-owned sim.Spawnable `ship` field and
  initializes its Ship fields before it starts running (C's
  ship_initialize). The per-ship attributes are sim.Struct fields in the
  process allocation -- the Python form of the C tutorial deriving struct
  ship from cmb_process -- and the process sees them through its
  annotated `shp: Ship` parameter.
* A departing ship tallies its time in system, hands its own process
  handle to the departures process through the `departed` store, and
  returns; the departures process sim.despawn()s it, corresponding to
  the C davyjones/departed-ships recycling path. Spawned leftovers are
  stopped and reclaimed automatically at the end of the trial.
* The C version registers the tug/berth pools as resource guards of the
  harbormaster condition, so releases re-test waiting predicates
  automatically. Here, ships signal the harbormaster explicitly after
  releasing resources.
* Each ship reads its per-ship limits through the Ship struct, just as
  the C predicate reads struct ship. Python predicates do not receive the
  waiting process as an argument, so wait_for() uses a simple
  harbormaster wake-up predicate and the ship rechecks its own docking
  test in a loop, catching the same race as the C code (another ship
  grabbing the tugs between wakeup and resumption).
* Text reports, histograms, and time-series histories are exposed through
  sim.dataset_*(), sim.timeseries_*(), and sim.*_report() helpers. This
  runnable script keeps the console report compact over replicated trials;
  the docs tutorial shows the fuller single-trial report style.

Usage: uv run python tutorial/tut_4_1.py
"""

import time

import numpy as np

import cimba as cp
import cimba.sim as sim

# Ship classes (SMALL, LARGE), as hard-coded in the C tutorial
SMALL = 0
LARGE = 1
TUGS_NEEDED = (1, 3)
MAX_WIND = (10.0, 12.0)      # m/s
MIN_DEPTH = (8.0, 13.0)      # m

HOURS_PER_YEAR = 24.0 * 7 * 52


class Ship(sim.Struct):
    """Per-ship fields, as in the C tutorial's struct ship."""
    size: int
    tugs_needed: int
    max_wind: float
    min_depth: float
    arrival: float


class SeaConditions(sim.Component):
    wind_mag: sim.FloatState         # m/s
    wind_dir: sim.FloatState         # compass degrees
    water_depth: sim.FloatState      # m

    # Priority 1: each hour, the wind updates before the tide reads it
    @sim.process(priority=1)
    def weather(self, env):
        while True:
            # Wind magnitude in m/s, smoothed over the previous hour
            wmag = sim.rayleigh(env.mean_wind)
            self.wind_mag = 0.5 * wmag + 0.5 * self.wind_mag
            # Wind direction in compass degrees, dominant from the southwest
            self.wind_dir = sim.pert(0.0, 225.0, 360.0)
            sim.hold(1.0)

    @sim.process
    def tide(self, env):
        pi = np.pi
        while True:
            # A simple tide model with astronomical and weather-driven tides
            t = sim.now()
            da = (env.reference_depth
                  + 1.0 * np.sin(2.0 * pi * t / 12.4)
                  + 0.5 * np.sin(2.0 * pi * t / 24.0)
                  + 0.25 * np.sin(2.0 * pi * t / (0.5 * 29.5 * 24)))
            # Wind speed as a proxy for air pressure, assuming a west coast
            dw = (0.5 * self.wind_mag
                  - 0.5 * self.wind_mag
                  * np.sin(self.wind_dir * pi / 180.0))
            self.water_depth = da + dw
            # Request the harbormaster to read the tide dial
            sim.signal(env.facilities.harbormaster)
            sim.hold(1.0)


class HarborFacilities(sim.Component):
    tugs: sim.Pool = sim.capacity("num_tugs")
    berths_small: sim.Pool = sim.capacity("num_berths_small")
    berths_large: sim.Pool = sim.capacity("num_berths_large")
    comms: sim.Resource              # the radio channel
    harbormaster: sim.Condition      # gates docking


class ShipTraffic(sim.Component):
    ship: sim.Spawnable              # one spawned per arrival
    departed: sim.Store              # finished ships to reclaim
    time_small: sim.Dataset          # time in system
    time_large: sim.Dataset

    @sim.process
    def arrivals(self, env):
        mean_interarr = 1.0 / env.arrival_rate
        while True:
            sim.hold(sim.exponential(mean_interarr))
            h = sim.spawn(self.ship, env, 0)
            shp = Ship(h)
            shp.size = sim.bernoulli(env.percent_large)
            shp.tugs_needed = TUGS_NEEDED[shp.size]
            shp.max_wind = MAX_WIND[shp.size]
            shp.min_depth = MIN_DEPTH[shp.size]
            shp.arrival = sim.now()

    @sim.process
    def ship(self, env, shp: Ship):
        me = sim.current()
        if shp.size == LARGE:
            berths = env.facilities.berths_large
        else:
            berths = env.facilities.berths_small

        # Wait for suitable conditions to dock. The loop catches spurious
        # wakeups, such as several ships waiting for the tide and one of
        # them grabbing the tugs before we can react.
        while True:
            ready = (
                env.sea.water_depth >= shp.min_depth
                and env.sea.wind_mag <= shp.max_wind
                and sim.pool_available(env.facilities.tugs) >= shp.tugs_needed
                and sim.pool_available(berths) >= 1
            )
            if ready:
                break
            sim.wait_for(env.facilities.harbormaster,
                         env.harbormaster_called, env)

        # Cleared to dock: grab a berth and the tugs
        sim.pool_acquire(berths, 1)
        sim.pool_acquire(env.facilities.tugs, shp.tugs_needed)

        # Announce our intention to move
        sim.acquire(env.facilities.comms)
        sim.hold(sim.gamma(5.0, 0.01))
        sim.release(env.facilities.comms)

        # It takes a while to move into position
        sim.hold(sim.pert(0.4, 0.5, 0.8))

        # Safely at the quay, dismiss the tugs and unload
        sim.pool_release(env.facilities.tugs, shp.tugs_needed)
        sim.signal(env.facilities.harbormaster)
        if shp.size == LARGE:
            unload_avg = env.unload_avg_large
        else:
            unload_avg = env.unload_avg_small
        sim.hold(sim.pert(0.75 * unload_avg, unload_avg, 2.0 * unload_avg))

        # Need the tugs again to get out of here
        sim.pool_acquire(env.facilities.tugs, shp.tugs_needed)
        sim.acquire(env.facilities.comms)
        sim.hold(sim.gamma(5.0, 0.01))
        sim.release(env.facilities.comms)

        # Gently move out again, assisted by tugs
        sim.hold(sim.pert(0.4, 0.5, 0.8))

        # Cleared the berth, done with the tugs
        sim.pool_release(berths, 1)
        sim.pool_release(env.facilities.tugs, shp.tugs_needed)
        sim.signal(env.facilities.harbormaster)

        # Datasets are reset when the measurement window opens, which
        # replaces the C version's explicit warmup-time check.
        if shp.size == LARGE:
            sim.tally(self.time_large, sim.now() - shp.arrival)
        else:
            sim.tally(self.time_small, sim.now() - shp.arrival)
        sim.store_put(self.departed, me)

    @sim.process
    def departures(self, env):
        while True:
            sim.despawn(sim.store_take(self.departed))


class Harbor(sim.Model):
    # Model parameters
    mean_wind: sim.Param
    reference_depth: sim.Param
    arrival_rate: sim.Param          # ships per hour
    percent_large: sim.Param
    num_tugs: sim.Param
    num_berths_small: sim.Param
    num_berths_large: sim.Param
    unload_avg_small: sim.Param
    unload_avg_large: sim.Param

    # Results
    avg_time_small: sim.Output       # mean time in system, small ships
    avg_time_large: sim.Output
    n_small: sim.Output              # departures counted in the window
    n_large: sim.Output
    tug_util: sim.Output             # mean units in use over the window
    berth_small_util: sim.Output
    berth_large_util: sim.Output

    harbormaster_called: sim.Predicate
    sea: SeaConditions = SeaConditions()
    facilities: HarborFacilities = HarborFacilities()
    traffic: ShipTraffic = ShipTraffic()


harbor = Harbor()


@harbor.predicate
def harbormaster_called(env: Harbor) -> bool:
    return True


@harbor.collect
def harbor_stats(env: Harbor):
    env.avg_time_small = sim.dataset_mean(env.traffic.time_small)
    env.avg_time_large = sim.dataset_mean(env.traffic.time_large)
    env.n_small = sim.dataset_count(env.traffic.time_small)
    env.n_large = sim.dataset_count(env.traffic.time_large)
    env.tug_util = sim.pool_mean_in_use(env.facilities.tugs)
    env.berth_small_util = sim.pool_mean_in_use(env.facilities.berths_small)
    env.berth_large_util = sim.pool_mean_in_use(env.facilities.berths_large)


def ci95(vals: np.ndarray) -> tuple[float, float]:
    vals = vals[~np.isnan(vals)]
    return float(vals.mean()), float(1.96 * vals.std(ddof=1)
                                     / np.sqrt(vals.size))


def main() -> None:
    print(f"cimba {cp.version()}, using {cp.use_threads(0)} worker threads")

    # The parameter set of the C tutorial's load_params(), one year of
    # harbor operation per trial after a day of warmup
    exp = harbor.experiment(mean_wind=5.0,
                            reference_depth=15.0,
                            arrival_rate=0.5,
                            percent_large=0.25,
                            num_tugs=10.0,
                            num_berths_small=6.0,
                            num_berths_large=3.0,
                            unload_avg_small=8.0,
                            unload_avg_large=12.0,
                            replications=20,
                            warmup=24.0,
                            duration=HOURS_PER_YEAR,
                            seed=20260612)

    t0 = time.perf_counter()
    fails = exp.run()
    wall = time.perf_counter() - t0
    print(f"{len(exp)} trials of {HOURS_PER_YEAR:.0f} h in {wall:.2f} s, "
          f"{fails} failed\n")

    n_sm, _ = ci95(exp["n_small"])
    n_lg, _ = ci95(exp["n_large"])
    t_sm, t_sm_w = ci95(exp["avg_time_small"])
    t_lg, t_lg_w = ci95(exp["avg_time_large"])
    print("Time in system (hours):")
    print(f"  small ships: {t_sm:6.2f} +/- {t_sm_w:.2f}"
          f"   ({n_sm:,.0f} departures/trial)")
    print(f"  large ships: {t_lg:6.2f} +/- {t_lg_w:.2f}"
          f"   ({n_lg:,.0f} departures/trial)")

    tu, tu_w = ci95(exp["tug_util"])
    bs, bs_w = ci95(exp["berth_small_util"])
    bl, bl_w = ci95(exp["berth_large_util"])
    print("\nMean units in use:")
    print(f"  tugs:         {tu:5.2f} +/- {tu_w:.2f} of 10")
    print(f"  small berths: {bs:5.2f} +/- {bs_w:.2f} of 6")
    print(f"  large berths: {bl:5.2f} +/- {bl_w:.2f} of 3")


if __name__ == "__main__":
    main()
