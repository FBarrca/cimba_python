# Cimba Python 0.9.6

Released 2026-11-06.

- Structural design points (#29). `cb.sweep` now works on `cb.Ref` fields
  (`flow.target = cb.sweep(review, approve)`, for models or entities) and on
  list fields, as a sweep of lists (`hub.outgoing = cb.sweep([a], [a, b])`
  for `list[cb.Ref[Node]]`, or `pool.servers = cb.sweep([s1], [s1, s2])` for
  an owned `list[Server]`, whose options may share items). Like swept child
  models, these are structural axes: each design point runs on a tree that
  holds only its options, with the same trial seeds, so independent
  structural axes cross with each other and with parameter sweeps. A
  reference from a stable model into a swept child can be linked to it,
  `cb.sweeps([a, b], [a.entry, b.entry])`, and then follows the selected
  option. `cb.Experiment(...)` now builds and checks every design point's
  tree, so a dangling reference fails there, naming its design point,
  rather than inside `run()`.

# Cimba Python 0.9.5

Released 2026-11-03.

- Spawning processes on macOS is about 2.5× faster (#28). The macOS ports
  of the engine now keep a per-thread pool of guarded coroutine stacks, as
  the Linux ports do, so a stack is protected with `mprotect` once rather
  than on every spawn and retire. A churn of short-lived processes drops from
  1.75 to 0.71 µs per spawn on Apple Silicon. Linux and Windows are
  unchanged. The engine also skips fat LTO objects when the compiler rejects
  them together with `-flto`.
- `cb.release()` of a spawned model takes constant time. It used to walk
  the list of every model spawned in the trial, so trials that release
  models long after spawning them became quadratic: 400,000 late releases
  drop from 112 s to 1.4 s. A model releasing itself from its own process
  also ends through the fast exit of a returning process, so 200,000
  self-releases with 50,000 processes alive drop from 16.6 s to 0.86 s.
  Results are unchanged.

# Cimba Python 0.9.4

Released 2026-11-03.

- `cb.Ref` fields can reference entities (#26): `cb.Ref[cb.Resource]` for
  one, `list[cb.Ref[cb.Resource]]` for a list, and `| None` for an optional
  one. Every entity kind can be referenced (`Resource`, `Container`,
  `FractionalContainer`, `Store[T]`, `PriorityStore[T]`, `Condition`,
  `Dataset`). Configure them with entity objects on the host, such as
  `pool.resources = [role.res for role in roles]`. In compiled code a
  referenced entity is the entity itself: all its methods, `len()` and
  bounds-checked indexing of a list. Model code can assign an entity
  reference, and `cb.spawn` takes one for a spawned model's `Ref` field.
- Building an experiment checks every entity reference: a value that is not
  an entity of the same model tree (including one of a swept option the
  tree doesn't contain), one held by a field of another type, a `None` item
  in an entity list, or a missing required reference raises
  `ModelDefinitionError` naming the field.
- `cb.acquire_any` also takes a `list[cb.Ref[cb.Resource]]` field, such as
  `cb.acquire_any(self.pool.resources, priority=p)`, with the semantics of
  the tuple form. Its length is read at run time, so one compiled method
  serves pools of any size, with no maximum; an empty list fails the trial.
  The compile error for any other argument names both forms.
- Ending a spawned process no longer costs time proportional to the number
  of live processes (#27). The engine swept its whole event queue, which
  holds an event per live process, nine times per process, so trials with
  tens of thousands of processes alive took quadratic time: 60,000 live
  workers sharing a million holds took 12.2 s and now take 1.1 s, and a burst
  of 40,000 short-lived processes costs about 3 µs per process instead of
  hundreds. Results are unchanged. The vendored engine is not modified.
- `structure()` and `process_graph()` draw entity references: a reference
  is an edge to the model holding the entity, and `acquire_any` over a list
  draws an edge from every referenced resource.

# Cimba Python 0.9.3

Released 2026-11-03.

- `enum.IntEnum` members work in compiled model code (#25). They are
  frozen at compile time like other module-level constants, compare with
  members and integers, work in `if`/`elif` chains and `match`, convert with
  `int()`, `float()` and `bool()`, index NumPy arrays and model lists, and
  pass wherever an `int` or `float` is expected. Before, `float(member)`
  and similar conversions failed to compile.
- `cb.Param`, plain fields and `cb.Output` can be typed with an `IntEnum`
  (`kind: cb.Param[NodeKind] = NodeKind.TASK`). Trial memory holds the
  member's integer value. Host assignments are checked against the enum: an
  `int` that is a valid value is stored as its member, and anything else
  raises `ModelDefinitionError` naming the field. Params can be swept over
  members. In compiled code the field reads as a member and accepts writes of
  members only. Enum outputs are collected as integer values, like
  `Output[int]`. `cb.spawn` takes members for enum fields, and decisions
  can't be assigned to enum params.
- `@cb.function` parameters and return values can be typed with an
  `IntEnum`. An enum parameter takes members of that enum only; `int` and
  `float` parameters also accept members.
- A field or function typed with a plain `enum.Enum`, and converting a plain
  `Enum` member to a number in compiled code, are errors that say to derive
  the enum from `IntEnum`. Plain members still support `==`, `!=` and
  `.value` in compiled code.
- `cb.Array[T]` fields hold a per-instance table of `int`, `float` or `bool`
  whose length is configuration, on static and spawned models (#24). Static
  models take `cb.Array(n)` (zeros) or `cb.Array([values])` as a class
  default or host assignment, and a sweep of them. `cb.spawn(Case, joins=n)`
  takes an `int` length and gives a zeroed array, or uses the class default.
  Compiled code indexes, assigns, takes `len()` of and iterates arrays. Every
  index is checked: an index out of range, or a negative spawn length, fails
  the trial with a reason naming the field. A static array lives in the trial
  block; a spawned model's array is allocated at spawn and freed on release,
  at trial end or when the trial is abandoned.
- `Ref` fields can be assigned in compiled code (#24), including a model of a
  subclass, `None` for an optional reference, and an optional value, so model
  code can build linked structures of spawned models. Before, these failed to
  compile with `Cannot cast Record(...) to uint64`. Assigning another class,
  `None` to a required reference, or a child, list or array field is a
  compile error with the file and line of the assignment. An optional
  reference passed at spawn works the same way.
- Using an optional reference that is `None` as a model, for example reading
  `self.next.value` without a `None` check, now fails the trial with
  `a None reference was used as a Node model`. Before, the error was printed
  and the method returned early while the trial went on.
- `cb.spawn` is annotated to return an instance of the class it creates.
- Spawning a model with an entity, child or list field explains that
  spawned models use `cb.Array` and `cb.Ref` fields instead.

# Cimba Python 0.9.2

Released 2026-11-03.

- `Resource.acquire()` and `Resource.preempt()` take a `priority=` argument
  for the request, independent of the process priority. The request priority
  decides the queue position, which holders a `preempt` may take from, and
  the priority the granted units are held at, so later preempts rank victims
  by it. `priority_set()` no longer moves such a request or holding. The
  process priority still orders same-time events, including the wakeup that
  grants the request. The default `priority=None` behaves as before.
- The documentation now states the resource ordering rules: higher priority
  first and first come, first served among equals; no bypassing a request
  that can't be fully served; `preempt` takes only from strictly lower
  holders, lowest and most recent first, and the victim loses all its units.

- Time-varying `Resource` capacity. `set_capacity(n, mode="finish")`
  changes the capacity during a trial, from processes, functions or hooks;
  `n` may be zero, and a negative `n` fails the trial. Raising it grants
  waiting requests at once, in queue order. Lowering it with `"finish"` lets
  holders keep their units (`available()` is `0` until the units in use drop
  below `n`); with `"preempt"` it preempts holders in the `preempt` victim
  order until the units in use fit. `mode` must be a string literal and is
  checked when the model compiles. `capacity()` returns the current
  capacity. The guide on policies shows a shift-calendar process.
- `Resource.mean_capacity()` is the time-weighted capacity over the window,
  and `Resource.utilization()` is `mean_in_use() / mean_capacity()` (`nan`
  when the window offered no capacity).
- `Resource(capacity=0)` is allowed. A request larger than the current
  capacity now waits for the capacity to rise instead of aborting the engine.
- A partially served `Resource` request now keeps its place in line: each
  time it waits again for the rest, it returns to its original position, so
  a later request of equal priority is no longer served before it. Models
  whose multi-unit requests wait after taking part of their amount (such as
  the cheese tutorial) can give different results for the same seed; other
  models are unchanged. `Container` and `FractionalContainer` keep the
  engine's behaviour, now documented: a partially served `get` or `put`
  rejoins the line behind waiting calls of equal priority.
- `Resource` is now implemented in Cimba Python's own native layer
  (`resource.c`) on the engine's waiting-room and holding interfaces; the
  vendored Cimba engine is unmodified. Invalid amounts (`0`, negative) and
  releasing more than is held now fail the trial with a reason.
- `cb.acquire_any((r0, r1, ...), priority=None)` takes one unit from
  whichever of several resources grants first and returns its index (or the
  negative signal that ended the wait). A free unit is taken from the first
  listed resource that has one; otherwise the request waits in every
  resource's queue at the place an `acquire` would get, competing first
  come, first served with direct acquirers, and leaves the other queues the
  moment one grants. The unit is an ordinary, preemptable holding of that
  resource. Interrupts, stops, preemption and process timers leave no
  request behind. The guide on policies shows "any of several roles".
- `Condition(observe=...)` and `condition.observe(...)` make a condition
  signal itself after every change to the observed `Resource`, `Store`,
  `PriorityStore`, `Container` or `FractionalContainer`, so predicates over
  them need no manual `signal()`. Observed entities must belong to the same
  model tree, which the experiment checks. Use `acquire_any` for "one unit
  of any of these pools" and an observing condition for any other rule.
- Constructing a model now gives it its own copies of the entities declared
  as class defaults, before `__init__` runs (previously this happened when
  the experiment was set up), so `self.clerks` in `__init__` is the
  instance's own entity.
- Native ABI version 5: the trial descriptor carries the observation table.
- **Breaking:** the host-side configuration attribute of a `Resource` is
  `initial_capacity`; `capacity` is now the model-code method. Assigning
  `resource.capacity` raises `AttributeError`.

# Cimba Python 0.9.1

Released 2026-10-27.

- Python 3.11 is now supported. Requires-Python is `>=3.11`, and published
  wheels are `cp311-abi3` so one wheel per platform covers CPython 3.11 and
  later.
- `Param`, `Output`, and `Ref` use `typing_extensions.TypeAliasType` so the
  same declaration markers work on 3.11 without PEP 695 `type` aliases.

# Cimba Python 0.9.0

Released 2026-10-19.

- Simulation optimization. `cb.decision(low, high, step=..., log=...)`
  marks a `Param` as searchable, and `cb.Optimization(model, minimize=...)`
  (or `maximize=`) finds good values for a per-trial objective built from
  outputs. `run()` searches with differential evolution
  (`cb.optimize.DifferentialEvolution`), evaluating each generation as one
  native batch across all cores, with common random numbers and a cache of
  simulated candidates. The best few finalists are then re-run on fresh
  seeds to choose one, and the choice is estimated on independent seeds, so
  `best.estimate` is not biased by the search.
- `Optimization.evaluate()` scores candidates on the search trials, and
  `Optimization.experiment()` replays one as an ordinary `Experiment`.
  `best.apply()` writes the chosen values into the model.
- `Samples` support NumPy arithmetic (`results[m].cost + 2 * results[m].wait`),
  so objectives can combine outputs directly.
- Experiments now reject a `Param` or field value that doesn't fit its type,
  such as `2.5` for a `Param[int]`, with `ModelDefinitionError` naming the
  field. Before, NumPy truncated it.
- Finished spawned processes return their coroutine stacks for reuse, so
  models that spawn many short-lived processes allocate far less.
- New tutorial chapter, concepts, guide and reference pages on optimization.

# Cimba Python 0.8.0

Released 2026-10-15.

- **Breaking:** `cb.State` is removed. Annotate per-trial values as plain
  `T`: `on_hand: cb.State[float] = 0.0` becomes `on_hand: float = 0.0`.
  Plain annotated scalars were documented as constants but were always
  writable per-trial values; a static model must now give each one a
  value, on the class or in `__init__`.
- **Breaking:** plain fields are reported as kind `"field"` by `describe()`
  and in diagrams, and `process_graph(model, state=True)` is now
  `process_graph(model, fields=True)` (the docs directive option `:state:`
  is now `:fields:`).
- An `Output` starts each trial at its default, if it has one
  (`units: cb.Output[float] = 0.0`), so processes can count into it
  directly. Without one it starts as `nan` (`float`) or `0`, as before.
- **Breaking:** `FractionalContainer` keeps its level in fixed point, with
  `decimals` decimal places (default 6). Amounts are rounded half away from
  zero to that many places once, and every put and get after that is exact,
  so the level always equals the initial level plus everything put in minus
  everything taken out. This replaces the 0.7.6 tolerance, which let a `get`
  through with up to a billionth less than requested, and could still wait on
  drift made at a larger level: after `put(100_000_000)`, `put(0.3)` and
  `get(100_000_000)`, a `get(0.3)` waited forever.
- `FractionalContainer` takes a `capacity`: `put` fills to it and waits for
  room for the rest, as on a `Container`. The default is unlimited.
- `FractionalContainer.put()` returns the amount added, as `get()` returns
  the amount obtained.
- `FractionalContainer.round()` rounds an amount as the level rounds it, so
  model code can keep sums and ratios on the level's decimals:
  `tank.round(0.1 + 0.2) == 0.3`.
- A bad `FractionalContainer` amount fails the trial with a reason naming the
  container, such as `plant.silo: put(-1) is not a finite nonnegative amount`.
- The native ABI version is now 4.

# Cimba Python 0.7.6

Released 2026-10-13.

- `FractionalContainer.get()` now tolerates floating-point rounding error:
  a `get` succeeds once the level is within `math.isclose`-style tolerance of
  the amount requested, instead of waiting for a gap that is only rounding
  drift (e.g. `put(0.3)` then two `get(0.1)` no longer blocks a third
  `get(0.1)`). A real shortfall still waits, however small.
- **Breaking:** `FractionalContainer.get()` now returns the amount actually
  obtained (a `float`), instead of `None`. This can be less than requested
  when tolerance let the call through early, or when the wait was
  interrupted. Check the return value against the amount requested if your
  model needs to detect either case.

# Cimba Python 0.7.5

Released 2026-10-13.

- Add `FractionalContainer`, a level with `float` amounts. `get` takes what
  is present and waits for the rest, like `Container`; capacity is unlimited.
- **Breaking:** `Container.put()` and `get()` take `int` amounts. A `float`
  amount used to be truncated silently (`put(0.7)` added nothing) and is now a
  compile-time error.
- The native ABI version is now 3.

# Cimba Python 0.7.4

Released 2026-10-03.

- Fix compilation through model references and inherited fields.
- Add distribution tags for reproducible random streams across model refactors.
- Fix signed arithmetic on model collection lengths.
- Add condition wait timeouts and a boolean result.
- Support optional inputs as empty sources.
- Allow releasing static models while retaining their state and end hooks.
- Add child attachment hooks and `cb.options()` for configuring model sweeps.

# Cimba Python 0.7.3

Released 2026-09-27.

- Add `cb.is_dynamic(model)` for compiled model code. It reports whether an
  instance was spawned, remains true after release, and removes the need for
  constructor flags when a class supports both static and dynamic instances.

# Cimba Python 0.7.2

Released 2026-09-27.

## Faster compilation

Each model method is compiled once. The extra Numba `@cfunc` wrapper is gone;
one small native entry module per class adapts the engine's calling
conventions to the method wrappers. Fresh-process compile time drops 22–41%
on the tutorials and about 56% on classes with many processes. Simulation
runtime is unchanged.

## Fixes

- A window with warmup 0 opens as the processes start, so samples they record
  at time 0 are kept. Positive warmups are unchanged. Samples recorded in
  `@cb.on_start` hooks are still cleared when the window opens.

# Cimba Python 0.7.1

Released 2026-09-26.

The first published 0.7 release. Version 0.7 replaces the Python modeling
core; the Cimba engine is unchanged. Existing 0.6 models need migrating: see
[MIGRATION.md](MIGRATION.md).

## The 0.7 modeling core

- One `cb.Model` class for static and spawned instances, with typed `Param`,
  `State`, `Output`, `Input`, `Series`, `Ref`, child and list fields, and
  native entities (`Container`, `Store`, `PriorityStore`, `Resource`,
  `Condition`, `Dataset`).
- Processes, hooks, predicates and events are decorated methods, compiled once
  per class with Numba. Rewiring, resizing, sweeping or rebinding inputs never
  recompiles.
- Inputs are configured, not coded: `inputs.dist`, `inputs.trace`,
  `inputs.bootstrap` and `inputs.fitted` sources, each with its own random
  stream per trial, recorded provenance, and explicit exhaustion policies
  (`fail`, `wrap`, `end_trial`, `extend`).
- `cb.Experiment` with `cb.sweep`, `cb.sweeps` and `Window`, common random
  numbers by default, and `Results` read through the model objects
  themselves. `cimba.analysis` provides summaries, paired comparisons and
  input checks.

## New since the 0.7 rework

- `@cb.function`: annotated methods callable from compiled code, with dynamic
  dispatch on the model's actual class, so subclasses can override them.
  Functions can call each other, recurse, spawn models and block.
- Model sweeps: sweep a child model field over model objects
  (`store.policy = cb.sweep(OrderUpTo(), MinMax())`). Each option runs on its
  own model tree, so an unselected option's processes never run, and every
  option sees the same random numbers. `results.meta.variants` reports the
  number of trees run.
- `cb.sweep` accepts a single list.
- `cimba.diagrams`: `structure()` and `process_graph()` diagrams, rendered as
  Mermaid or Graphviz DOT.
- Rewritten documentation: a guided tutorial, concept pages, how-to guides,
  reference and an architecture section with diagrams.

## Fixes

- The `end_trial` exhaustion policy now ends the trial instead of failing it.
- `Container(initial=...)` sets the starting level.
- Processes spawned during a trial are stopped at the end of the window, and
  their pending starts are cancelled.

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
