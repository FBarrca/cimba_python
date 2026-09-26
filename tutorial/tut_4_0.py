"""Tutorial 4.0: empty harbor template for a model and experiment."""

import cimba as cb


class HarborTemplate(cb.Model):
    result: cb.Output[float]

    @cb.on_end
    def collect(self):
        self.result = 0.0


def main() -> None:
    model = HarborTemplate()
    results = cb.Experiment(model, window=cb.Window(duration=10.0)).run()
    print(f"Harbor template result: {results[model].result[0, 0]:.1f}")


if __name__ == "__main__":
    main()
