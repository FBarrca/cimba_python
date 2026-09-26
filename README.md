![Cimba logo](docs/static/cimba_logo_large.jpg)

# Cimba Python 0.7

Cimba Python compiles object-oriented discrete-event models to run on the [Cimba C engine](https://github.com/ambonvik/cimba). A model declares its variability as inputs. The same compiled model can use distributions, recorded traces, bootstrap resamples, or fitted sources.

Python 3.13 or newer is required. Linux x86_64, Windows AMD64, and macOS arm64 wheels include the native engine.

```bash
pip install cimba
```

## A queue model

```python
import cimba as cb
from cimba import inputs


class MM1(cb.Model):
    interarrival: cb.Input[float] = inputs.dist.exponential(mean=1 / 0.75)
    service_time: cb.Input[float] = inputs.dist.exponential(mean=1)
    queue: cb.Container
    mean_queue: cb.Output[float]

    @cb.process
    def arrivals(self):
        while True:
            cb.hold(self.interarrival.next())
            self.queue.put(1)

    @cb.process
    def server(self):
        while True:
            self.queue.get(1)
            cb.hold(self.service_time.next())

    @cb.on_end
    def measure(self):
        self.mean_queue = self.queue.mean_level()


model = MM1()
results = cb.Experiment(
    model, replications=100,
    window=cb.Window(warmup=100, duration=1000), seed=123,
).run()
print(cb.analysis.summary(results[model].mean_queue))
```

To replay observed arrival gaps, assign `model.interarrival = inputs.trace(gaps)`. To resample them, assign `inputs.bootstrap.stationary(gaps, mean_block=7)`. The process code and compiled class stay the same. Input sources are independent streams, and results record their provenance and consumption.

See the [documentation](https://fbarrca.github.io/cimba_python/), [standalone tutorials](tutorial/README.md), and [0.6 to 0.7 migration guide](MIGRATION.md).

## Development

```bash
git submodule update --init --recursive
uv sync --locked
uv run pytest
```

The Cimba C library is an unchanged submodule. Cimba Python is Apache-2.0 licensed; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
