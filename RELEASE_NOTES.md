# Cimba Python 0.6.1

Released 2026-09-06.

- Update the bundled Cimba fork to 3.0.0-RC2 while retaining Apple Silicon
  coroutine switching and recovery support.
- Fix native object initialization, termination, and cleanup after abandoned
  trials, including temporary statistics objects and spawned-process registries.
- Fix Windows recovery after a trial abandons an active coroutine.
- Add `duration=None, warmup=0.0` for finite workloads that run until the event
  queue empties. Collectors and explicit sampling still run; this opt-in mode
  has no automatic recording window. Existing timed simulation defaults remain.

# Cimba Python 0.6.0

Released 2026-08-23.

Version 0.6.0 moves model authoring to class-declared callbacks and adds a
model-shaped result namespace. Existing models need a small migration before
they can run on this release.

## API changes

### Breaking: callbacks are declared on model and component classes

The instance-bound callback decorators have been removed. The old form:

```python
model = Clinic()

@model.process
def arrivals(env: Clinic):
    ...

@model.collect
def collect_stats(env: Clinic):
    ...
```

must become class-body declarations using the exported `cimba.sim` markers:

```python
class Clinic(sim.Model):
    @sim.process
    def arrivals(self: "Clinic"):
        ...

    @sim.collect
    def collect_stats(self: "Clinic"):
        ...

model = Clinic()
```

The same migration applies to `@model.predicate` and `@model.event`, which
become `@sim.predicate` and `@sim.event`. Use `field="..."` when a callback
is bound to a differently named `sim.Processes`, `sim.Predicate`, or
`sim.Event` field. `@sim.function` can now also declare read-only helpers on a
model; component callbacks use the same shared declaration rules.

Callback bodies should use `self` for the root trial view. It is the
trial-local environment, not the Python model instance. Component methods also
use `self` for their component view.

### Structured experiment results

Retained outputs, captured datasets, and captured histories are now available
through `exp.results`, following the model's component tree:

```python
exp.run()
queue_means = exp.results.counters.mean_queue_length
wait_samples = exp.results.station.waits
queue_history = exp.results.station.queue
```

Output arrays preserve the existing trial and collection axes. Dataset and
history leaves have the same tuple-of-arrays shapes as `datasets()` and
`histories()`. `ExperimentResults` is exported from `cimba.sim`; parameterized
models can use `Model[YourResultsProtocol]` for static completion of the result
tree. The string-key API (`exp["..."]`) and the explicit dataset/history
accessors remain available for low-level, dynamic, and compatibility use.

### Component and callback behavior

Model and component declarations now share callback discovery, inheritance,
validation, and compilation semantics. Components can own `@sim.predicate` and
`@sim.event` callbacks, nested components retain their object paths in
structured results, and component `sim.Processes` fields are bound explicitly
with `@sim.process(field="...")` when the callback name differs from the
field name.
