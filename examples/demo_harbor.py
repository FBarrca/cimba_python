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

* The C version spawns one cmb_process per ship at arrival time and
  recycles it through the davyjones condition + departed-ships list.
  Process creation is static here, so each size class instead has a fixed
  crew of handler processes (more than can ever be in port at once)
  pulling arrival tickets from an anchorage store; the ticket is the
  arrival time, bit-cast with sim.f2i(). Handlers tally time-in-system
  directly, which retires the davyjones/exit-value machinery.
* The C version registers the tug/berth pools as resource guards of the
  harbormaster condition, so releases re-test waiting predicates
  automatically. Here, ships signal the harbormaster explicitly after
  releasing resources.
* is_ready_to_dock() reads per-ship limits through the process struct;
  here each size class has its own predicate over the shared env. The
  while-loop around sim.wait_for() catches the same race as the C code
  (another ship grabbing the tugs between wakeup and resumption).
* Logging and histogram printing are not exposed; the report prints
  dataset means and pool utilizations instead, over replicated trials
  running in parallel (the C tutorial runs a single trial).

Usage: uv run python examples/demo_harbor.py
"""

import time

import numpy as np
from numba import njit

import cimba as cp
import cimba.sim as sim

# Ship classes (SMALL, LARGE), as hard-coded in the C tutorial
TUGS_NEEDED = (1, 3)
MAX_WIND = (10.0, 12.0)      # m/s
MIN_DEPTH = (8.0, 13.0)      # m
N_HANDLERS = (16, 8)         # fixed crews of ship-handler processes

HOURS_PER_YEAR = 24.0 * 7 * 52


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

    # Environment, updated hourly by the weather and tide processes
    wind_mag: sim.FloatState         # m/s
    wind_dir: sim.FloatState         # compass degrees
    water_depth: sim.FloatState      # m

    # Entities
    tugs: sim.Pool = sim.capacity("num_tugs")
    berths_small: sim.Pool = sim.capacity("num_berths_small")
    berths_large: sim.Pool = sim.capacity("num_berths_large")
    comms: sim.Resource              # the radio channel
    harbormaster: sim.Condition      # gates docking
    anchorage_small: sim.Store       # arrival tickets (bit-cast times)
    anchorage_large: sim.Store
    time_small: sim.Dataset          # time in system
    time_large: sim.Dataset
    dockable_small: sim.Predicate
    dockable_large: sim.Predicate


harbor = Harbor()


# Shared docking test, used both by the predicates the harbormaster
# evaluates and by the handlers' race-catching loop.
@njit
def _can_dock(env, berths, tugs_needed, max_wind, min_depth):
    return (env.water_depth >= min_depth
            and env.wind_mag <= max_wind
            and sim.pool_available(env.tugs) >= tugs_needed
            and sim.pool_available(berths) >= 1)


@harbor.predicate
def dockable_small(env: Harbor) -> bool:
    return _can_dock(env, env.berths_small,
                     TUGS_NEEDED[0], MAX_WIND[0], MIN_DEPTH[0])


@harbor.predicate
def dockable_large(env: Harbor) -> bool:
    return _can_dock(env, env.berths_large,
                     TUGS_NEEDED[1], MAX_WIND[1], MIN_DEPTH[1])


# Priority 1: each hour, the wind updates before the tide reads it
@harbor.process(priority=1)
def weather(env: Harbor):
    while True:
        # Wind magnitude in m/s, smoothed over the previous hour
        wmag = sim.rayleigh(env.mean_wind)
        env.wind_mag = 0.5 * wmag + 0.5 * env.wind_mag
        # Wind direction in compass degrees, dominant from the southwest
        env.wind_dir = sim.pert(0.0, 225.0, 360.0)
        sim.hold(1.0)


@harbor.process
def tide(env: Harbor):
    pi = np.pi
    while True:
        # A simple tide model with astronomical and weather-driven tides
        t = sim.now()
        da = (env.reference_depth
              + 1.0 * np.sin(2.0 * pi * t / 12.4)
              + 0.5 * np.sin(2.0 * pi * t / 24.0)
              + 0.25 * np.sin(2.0 * pi * t / (0.5 * 29.5 * 24)))
        # Wind speed as a proxy for air pressure, assuming a west coast
        dw = (0.5 * env.wind_mag
              - 0.5 * env.wind_mag * np.sin(env.wind_dir * pi / 180.0))
        env.water_depth = da + dw
        # Request the harbormaster to read the tide dial
        sim.signal(env.harbormaster)
        sim.hold(1.0)


@harbor.process
def arrivals(env: Harbor):
    mean_interarr = 1.0 / env.arrival_rate
    while True:
        sim.hold(sim.exponential(mean_interarr))
        if sim.bernoulli(env.percent_large) == 1:
            sim.store_put(env.anchorage_large, sim.f2i(sim.now()))
        else:
            sim.store_put(env.anchorage_small, sim.f2i(sim.now()))


@njit
def _ship_lifecycle(env, anchorage, berths, dockable, time_in_system,
                    tugs_needed, max_wind, min_depth, unload_avg):
    while True:
        t_arr = sim.i2f(sim.store_take(anchorage))

        # Wait for suitable conditions to dock. The loop catches spurious
        # wakeups, such as several ships waiting for the tide and one of
        # them grabbing the tugs before we can react.
        while not _can_dock(env, berths, tugs_needed, max_wind, min_depth):
            sim.wait_for(env.harbormaster, dockable, env)

        # Cleared to dock: grab a berth and the tugs
        sim.pool_acquire(berths, 1)
        sim.pool_acquire(env.tugs, tugs_needed)

        # Announce our intention to move
        sim.acquire(env.comms)
        sim.hold(sim.gamma(5.0, 0.01))
        sim.release(env.comms)

        # It takes a while to move into position
        sim.hold(sim.pert(0.4, 0.5, 0.8))

        # Safely at the quay, dismiss the tugs and unload
        sim.pool_release(env.tugs, tugs_needed)
        sim.signal(env.harbormaster)
        sim.hold(sim.pert(0.75 * unload_avg, unload_avg, 2.0 * unload_avg))

        # Need the tugs again to get out of here
        sim.pool_acquire(env.tugs, tugs_needed)
        sim.acquire(env.comms)
        sim.hold(sim.gamma(5.0, 0.01))
        sim.release(env.comms)

        # Gently move out again, assisted by tugs
        sim.hold(sim.pert(0.4, 0.5, 0.8))

        # Cleared the berth, done with the tugs
        sim.pool_release(berths, 1)
        sim.pool_release(env.tugs, tugs_needed)
        sim.signal(env.harbormaster)

        # Datasets are reset when the measurement window opens, which
        # replaces the C version's explicit warmup-time check
        sim.tally(time_in_system, sim.now() - t_arr)


@harbor.process(copies=N_HANDLERS[0])
def ship_small(env: Harbor):
    _ship_lifecycle(env, env.anchorage_small, env.berths_small,
                    env.dockable_small, env.time_small,
                    TUGS_NEEDED[0], MAX_WIND[0], MIN_DEPTH[0],
                    env.unload_avg_small)


@harbor.process(copies=N_HANDLERS[1])
def ship_large(env: Harbor):
    _ship_lifecycle(env, env.anchorage_large, env.berths_large,
                    env.dockable_large, env.time_large,
                    TUGS_NEEDED[1], MAX_WIND[1], MIN_DEPTH[1],
                    env.unload_avg_large)


@harbor.collect
def harbor_stats(env: Harbor):
    env.avg_time_small = sim.dataset_mean(env.time_small)
    env.avg_time_large = sim.dataset_mean(env.time_large)
    env.n_small = sim.dataset_count(env.time_small)
    env.n_large = sim.dataset_count(env.time_large)
    env.tug_util = sim.pool_mean_in_use(env.tugs)
    env.berth_small_util = sim.pool_mean_in_use(env.berths_small)
    env.berth_large_util = sim.pool_mean_in_use(env.berths_large)


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
