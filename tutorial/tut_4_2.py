import cimba

from tutorial.tut_4_1 import LARGE, departure_proc, ship_proc


def run_two_large_ship_scenario(num_large_berths: int, seed: int = 42) -> float:
    with cimba.Simulation(seed=seed) as sim:
        ctx = {
            "env": {"wind_magnitude": 4.0, "water_depth": 20.0},
            "tugs": cimba.ResourcePool("Tugs", capacity=6),
            "berths": [
                cimba.ResourcePool("Small berth", 1),
                cimba.ResourcePool("Large berth", num_large_berths),
            ],
            "comms": cimba.Resource("Comms"),
            "harbormaster": cimba.Condition("Harbormaster"),
            "departed": cimba.ObjectQueue("Departed ships"),
            "time_in_system": [cimba.Dataset(), cimba.Dataset()],
            "ship_by_process": {},
        }

        for idx in range(2):
            ship = {
                "size": LARGE,
                "tugs_needed": 3,
                "max_wind": 12.0,
                "min_depth": 13.0,
                "unloading_time": 2.0,
            }
            proc = cimba.Process(f"Ship_{idx:06d}_large", ship_proc, ctx)
            ctx["ship_by_process"][proc] = ship
            proc.start()

        cimba.Process("Departures", departure_proc, ctx).start()
        sim.execute()

        return max(ctx["time_in_system"][LARGE].values())


def run_scenarios() -> dict[str, float]:
    return {
        "one_large_berth": run_two_large_ship_scenario(1),
        "two_large_berths": run_two_large_ship_scenario(2),
    }


def main() -> None:
    print(run_scenarios())


if __name__ == "__main__":
    main()
