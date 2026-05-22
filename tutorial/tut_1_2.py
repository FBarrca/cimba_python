import cimba


def run(stop_time: float = 3.5, seed: int = 12) -> dict[str, object]:
    ticks = []

    def ticker(me, ctx):
        while True:
            cimba.hold(1.0)
            ticks.append(cimba.time())

    with cimba.Simulation(seed=seed) as sim:
        cimba.Process("Ticker", ticker).start()
        sim.stop_at(stop_time)
        sim.execute()
        return {"ticks": ticks, "now": sim.now, "event_count": sim.event_count}


def main() -> None:
    result = run()
    print(result)


if __name__ == "__main__":
    main()
