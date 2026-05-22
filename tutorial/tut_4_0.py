import cimba


def run_template(duration: float = 10.0, seed: int = 40) -> dict[str, float | int]:
    with cimba.Simulation(start_time=0.0, seed=seed) as sim:
        sim.stop_at(duration, priority=-100)
        sim.execute()
        return {"seed": sim.seed_used, "now": sim.now, "event_count": sim.event_count}


def main() -> None:
    print(run_template())


if __name__ == "__main__":
    main()
