# Migrating from 0.6 to 0.7

Version 0.7 replaces the Python modeling core. The Cimba engine submodule is unchanged. The old `cimba.sim` and `cimba.bootstrap` imports raise an error so a mixed model cannot run accidentally.

## The model and its inputs

| 0.6 | 0.7 |
| --- | --- |
| `import cimba.sim as sim` | `import cimba as cb` |
| `sim.Model`, `Component`, `Struct`, spawnable process descriptors | `cb.Model` for static and dynamic instances; `cb.spawn(ModelClass, ...)` inside processes |
| `Trace` declaration and a manual cursor | `Input[T]` with `.next()`; `Series[T]` with `.now()` or `.at(t)` |
| `random.*` for arrivals, services, demand, or delays | Declare an input and bind an `inputs.dist.*` source |
| `cimba.random` for routing and other model-internal choices | Still available; uses the trial default stream |
| `cimba.bootstrap.*(data, length=...)` returning a callable | `inputs.bootstrap.*(data, ...)` returning a source; row length is managed by the experiment |
| `trace_rng_name` and implicit joint RNG sharing | `RowSource.tag`; `inputs.bootstrap.joint(panel, ...)[key]` |
| Trace arrays or generators passed by field name to `experiment(...)` | Assign a source on the model object before constructing `Experiment` |
| External `trial_seeds` and trial-row generation | Built-in seeds and chunked generation; inspect `results[model].input.rows(point, replication)` |
| Exhaustion tripwire output | Explicit `on_exhausted`: trace `fail`, `wrap`, or `end_trial`; resamples extend by default |
| Traces used as fixed lookup tables | Plain constants or `Param[T]` on each model instance |

Class declarations now use typed `Param[T]`, `State[T]`, and `Output[T]`. Give state its initial value on the class or instance. Plain annotated attributes are constants. Use `Ref[ModelType]` and typed lists for links. Use `Container` for a counted level, `Store[T]` or `PriorityStore[T]` for values and model handles, and `Resource(capacity=n)` for capacity. The old opaque integer store, `f2i`/`i2f`, pools, and address-holder fields are gone. Predicates and events are decorated methods referenced directly. Replace `@collect` with `@on_end`, an initialization process with `@on_start`, and `@function` with a module-level `numba.njit` function that takes the model view as its first argument (undecorated model methods are not callable from compiled code).

## Experiments and results

```python
import cimba as cb
from cimba import inputs

model.base_stock = cb.sweep(250.0, 300.0, 350.0)
model.demand = inputs.bootstrap.stationary(history, mean_block=7)
model.orders.capture()

experiment = cb.Experiment(
    model, replications=100,
    window=cb.Window(warmup=30.0, duration=365.0), seed=7,
)
results = experiment.run(workers=4)
samples = results[model].service_level
print(cb.analysis.summary(samples))
print(results[model].demand.source)
print(results[model].demand.consumed)
```

Configure parameters, sources, and captures through their objects. `cb.sweep` creates independent axes; `cb.sweeps` links axes. `Experiment` accepts run settings only. Replace `exp.trials`, `exp["name"]`, flattened component names, and result namespaces with the immutable object-indexed `Results`. Outputs have `(design points, replications)` samples. `results.failed` and `results.failure_reasons` identify failed trials; `experiment.only(trials=[i])` reruns a trial. Use `analysis.summary`, `analysis.compare`, `analysis.check_input`, and `inputs.fit` for post-run and input-model analysis. Reports belong in Python after the run; `cb.log` handles compiled trial logging.

## Runtime and package changes

Model construction no longer compiles. Classes compile on first run and are reused when you change the instance graph, parameter values, or input sources. The old precompile plan, fork compiler, pickle cache, AST rewriting, private `_model` helpers, Cython/cffi entry points, and low-level experiment/thread-hook APIs were removed. `cimba.random` remains public. The native runner is a plain C library loaded through `ctypes`; model callbacks link directly to its verbs through Numba.

All scripts in [`tutorial/`](tutorial/README.md) use the 0.7 API and each is self-contained. Start with [`tut_1_1.py`](tutorial/tut_1_1.py), then the [multi-echelon inventory example](tutorial/multi_echelon_inventory.py) for joint, time-indexed data.
