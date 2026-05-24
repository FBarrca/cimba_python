import cimba

SMALL = 0
LARGE = 1


def is_ready_to_dock(process, ctx):
    ship = ctx["ship_by_process"][process]
    return (
        ctx["env"]["water_depth"] >= ship["min_depth"]
        and ctx["env"]["wind_magnitude"] <= ship["max_wind"]
        and ctx["tugs"].available >= ship["tugs_needed"]
        and ctx["berths"][ship["size"]].available >= 1
    )


def ship_proc(me, ctx):
    ship = ctx["ship_by_process"][me]
    t_arrival = cimba.time()

    while not is_ready_to_dock(me, ctx):
        assert ctx["harbormaster"].wait(is_ready_to_dock, ctx) == cimba.SUCCESS

    berth = ctx["berths"][ship["size"]]
    assert berth.acquire(1) == cimba.SUCCESS
    assert ctx["tugs"].acquire(ship["tugs_needed"]) == cimba.SUCCESS

    assert ctx["comms"].acquire() == cimba.SUCCESS
    cimba.hold(0.05)
    ctx["comms"].release()

    cimba.hold(0.5)
    ctx["tugs"].release(ship["tugs_needed"])

    cimba.hold(ship["unloading_time"])

    assert ctx["tugs"].acquire(ship["tugs_needed"]) == cimba.SUCCESS
    assert ctx["comms"].acquire() == cimba.SUCCESS
    cimba.hold(0.05)
    ctx["comms"].release()

    cimba.hold(0.5)
    berth.release(1)
    ctx["tugs"].release(ship["tugs_needed"])

    system_time = cimba.time() - t_arrival
    ctx["departed"].put((me.name, ship["size"], system_time))
    return system_time


def departure_proc(me, ctx):
    while True:
        sig, departed = ctx["departed"].get()
        assert sig == cimba.SUCCESS
        _name, size, system_time = departed
        ctx["time_in_system"][size].add(system_time)


def run_harbor_trial(seed: int = 41) -> dict[str, object]:
    def weather_and_tide(me, ctx):
        cimba.hold(1.0)
        ctx["env"]["wind_magnitude"] = 4.0
        ctx["env"]["water_depth"] = 12.0
        assert ctx["harbormaster"].signal() == 1

    with cimba.Simulation(seed=seed) as sim:
        ctx = {
            "env": {"wind_magnitude": 20.0, "water_depth": 5.0},
            "tugs": cimba.ResourcePool("Tugs", capacity=2),
            "berths": [cimba.ResourcePool("Small berth", 1), cimba.ResourcePool("Large berth", 1)],
            "comms": cimba.Resource("Comms"),
            "harbormaster": cimba.Condition("Harbormaster"),
            "departed": cimba.ObjectQueue("Departed ships"),
            "time_in_system": [cimba.Dataset(), cimba.Dataset()],
            "ship_by_process": {},
        }
        ctx["harbormaster"].subscribe(ctx["tugs"], *ctx["berths"])
        ship = {
            "size": SMALL,
            "tugs_needed": 1,
            "max_wind": 10.0,
            "min_depth": 8.0,
            "unloading_time": 2.0,
        }
        proc = cimba.Process("Ship_000001_small", ship_proc, ctx)
        ctx["ship_by_process"][proc] = ship
        proc.start()
        cimba.Process("Departures", departure_proc, ctx).start()
        cimba.Process("WeatherDepth", weather_and_tide, ctx).start()
        sim.execute()

        return {
            "small_system_times": ctx["time_in_system"][SMALL].values(),
            "large_system_times": ctx["time_in_system"][LARGE].values(),
            "tugs_available": ctx["tugs"].available,
            "small_berths_available": ctx["berths"][SMALL].available,
        }


def main() -> None:
    print(run_harbor_trial())


if __name__ == "__main__":
    main()
