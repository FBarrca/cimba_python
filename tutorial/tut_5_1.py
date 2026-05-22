import cimba


class Target:
    def __init__(self, name: str):
        self.name = name
        self.x = 0.0
        self.y = 0.0
        self.visible = False


def run_awacs_demo(seed: int = 51, duration: float = 5.0, num_targets: int = 3) -> dict[str, float | int]:
    targets = [Target(f"Target_{idx:03d}") for idx in range(num_targets)]
    detections = cimba.DataSummary()

    def target_proc(me, target):
        target.x = cimba.uniform(-10.0, 10.0)
        target.y = cimba.uniform(-10.0, 10.0)
        while True:
            target.visible = cimba.bernoulli(0.5)
            cimba.hold(cimba.exponential(1.0))

    def sensor_proc(me, ctx):
        while True:
            for target in ctx["targets"]:
                detected = target.visible and cimba.bernoulli(0.8)
                ctx["detections"].add(1.0 if detected else 0.0)
            cimba.hold(1.0)

    with cimba.Simulation(seed=seed) as sim:
        for target in targets:
            cimba.Process(target.name, target_proc, target).start()
        cimba.Process("Radar", sensor_proc, {"targets": targets, "detections": detections}).start()
        sim.stop_at(duration)
        sim.execute()

    return {"count": detections.count, "mean": detections.mean}


def main() -> None:
    print(run_awacs_demo())


if __name__ == "__main__":
    main()
