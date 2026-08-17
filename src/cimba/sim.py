"""
cimba.sim - cimba's modeling concepts behind a SimPy-flavored API.

Processes are plain Python functions that block, in the style of SimPy --
but with no `yield`: cimba's processes are stackful fibers, so sim.hold()
and the acquire/get/wait verbs simply suspend the process, from any depth
of the call stack. Each process is compiled with Numba into machine code,
so models run at native speed on all cores; process bodies must stay in
nopython-compilable Python (numbers, loops, sim.* calls, and cimba.random
draws).

A model is a Model subclass whose annotated fields declare the trial
record (the `self` view seen by model callbacks); the subclass doubles as the
static type of `self`, so fields are checked and completed:

    class MG1(sim.Model):
        utilization: sim.Param          # swept input
        avg_queue_length: sim.Output    # result
        queue: sim.Queue                # cmb_buffer handle

        @sim.process
        def arrivals(self: "MG1"):
            while True:
                sim.hold(cimba.random.exponential(1.0 / self.utilization))
                self.queue.put(1)

    mg1 = MG1()

Every declared Queue/Resource/Pool/Store/PQueues-element/Condition/Event
field carries its verbs as methods: ``self.queue.put(1)``,
``self.queue.get(1)``, ``self.server.acquire()``, ``self.cond.wait_for(pred)``,
``self.tick.schedule(1.0)``, and so on. Component-owned fields support the
same sugar through the component's own ``self.<field>.<method>(...)``.

Concept translation (cimba -> sim API):

    cmb_process       @sim.process (copies=, priority=, struct=, field=),
                      sim.hold(), sim.current(), sim.interrupt(),
                      sim.stop(), sim.wait_process(), sim.wait_event(),
                      sim.resume(), sim.timer_set()/sim.timer_add()/
                      sim.timer_cancel(); dynamic creation via
                      @sim.process(spawnable=True), sim.spawn()/sim.despawn()
    derived structs   sim.Struct subclasses; a process declares its own
                      fields with a final `vip: Visitor` parameter, and
                      Visitor(handle) views any such process's fields
    cmb_buffer        sim.Queue, env.<queue>.put()/get()/level()/space()/
                      mean_level()
    cmb_resource      sim.Resource, env.<resource>.acquire()/release()/
                      preempt()/available()/in_use()/held()/mean_in_use()
    cmb_resourcepool  sim.Pool (= n or sim.capacity(param)),
                      env.<pool>.acquire()/release()/preempt()/
                      available()/held()/in_use()/mean_in_use()
    cmb_objectqueue   sim.Store, env.<store>.put()/get()/take()/length()/
                      space()/position()/mean_length()
                      (objects are opaque int64 values; sim.f2i()/
                      sim.i2f() bit-cast timestamps in and out)
    cmb_condition     sim.Condition + sim.Predicate + @sim.predicate,
                      env.<cond>.wait_for(predicate)/signal()
    cmb_event         sim.Event + @sim.event, env.<event>.schedule()/
                      schedule_at() (returns a scheduled-instance handle
                      with its own .cancel()/.reschedule()/.reprioritize()/
                      .scheduled()/.time()/.priority()/.wait_event()),
                      sim.current_event(), sim.event_count(),
                      sim.clear_events()
    cmb_dataset       sim.Dataset, env.waits.add(), env.waits.mean()/
                      count()/min()/max()/std()/median()/quantile()
    statistics        recorded over the measurement window (after warmup,
                      datasets are reset when it opens): env.<x>.mean_level()/
                      mean_in_use()/mean_length() (or env.<pq>[i].mean_length())

Data-driven generators replay per-trial trajectories generated outside
the simulation (bootstrap, fitted models, recorded traces): declare a
``sim.Trace`` field and pass the data to experiment() -- a 1-D array
shared by all trials, a 2-D array with one row per trial, a list of
1-D arrays, or a callable ``f(rng)`` / ``f(rng, trial_index)`` invoked
once per trial with a numpy Generator derived from that trial's own
seed (``sim.trace_rng(trial_seed, field_name)``), so the experiment
``seed`` also reproduces generated traces (bootstrap resamples, fitted
models); ``cimba.bootstrap`` provides ready-made iid/block/stationary
resamplers. Callables run serially before the parallel trial run; for
expensive generators, ``model.trial_seeds()`` exposes the same per-trial
seeds so rows can be generated in parallel outside cimba and passed in
precomputed. Inside a process body, ``values = sim.Trace(env.<field>)``
returns the trial's trace as a plain float64 array supporting len(),
indexing, slicing, and iteration. When a generator exhausts its trace
it simply finishes; the trial still runs to its configured window, so
traces should cover warmup + duration + cooldown.

Mutable per-trial counters are declared with sim.State. Multi-copy
processes may take a second argument to learn their index:
`def machine(self, idx)`. The trial function, recording lifecycle, and all
create/start/stop/destroy plumbing are generated and compiled by Model.

Related fields and process methods can be grouped with ``sim.Component``.
Component methods marked with top-level ``@sim.process`` are authoring-time
methods; Model lowers them into ordinary flat process functions before Numba
compilation. A component method marked with top-level ``@sim.collect`` runs
once per instance at the end of each trial (before the model-level
``@sim.collect``, which can then aggregate), typically assigning the
component's own Output fields from ``self``. Model callbacks use ``self`` as
the root trial environment and can access component fields with
``self.retailer.orders``; trial-table
fields remain flattened with names such as ``retailer__orders``. Components
can also expose explicitly typed, read-only synchronous methods with
``@sim.function``; calls such as ``env.policy.decide(level)`` compile to
nopython helpers whose component field reads are passed as flattened scalar
arguments. Models can declare root helpers with the same marker using
``def helper(self, ...)``; model callbacks call them through
``self.helper(...)`` and component callbacks through ``env.helper(...)``.
Components may also own ``@sim.predicate`` and
``@sim.event`` callbacks, with the same explicit ``field=`` binding used by
models.
Components may contain nested components; paths such as
``env.attraction.queues.line``
flatten to names such as ``attraction__queues__line``. A component process
marked ``@sim.process(spawnable=True)`` can be spawned with paths such as
``sim.spawn(self.visitor, env)`` or
``sim.spawn(env.flow.visitor, env)``.
Component-owned ``sim.Processes`` fields likewise publish handles for
fixed component process methods bound explicitly with
``@sim.process(field="...")``; component collections flatten
ragged per-item copy counts behind paths such as ``env.teams[i].worker[j]``.
Fixed collections of repeated components can be declared with standard
``list[ComponentType]`` annotations. Model callbacks can use indexed access
such as ``env.attractions[i].queues[j]``. Nested collections also work; Cimba
lowers paths like ``env.campus.zones[i].gates[j].queue`` to flattened fields
and generated offset tables before compilation.

Module layout: the verbs below alias the raw symbol bindings in
``_bindings``; the cast helpers live in ``_intrinsics``; declaration markers
live in ``_declarations``; shared callback declarations live in
``_callbacks``; Component lowering lives in ``_components``;
Model/Experiment and the trial codegen live in ``_model``.
"""

from typing import TYPE_CHECKING, Any

import numpy as _np

from numba import carray as _carray
from numba import njit
from numba import types as _nbtypes

from . import _bindings as _b
from ._intrinsics import ptr_caster as _ptr_caster
from ._intrinsics import record_addr as _record_addr
from ._callbacks import (SpawnableProcess, collect, event, function,
                         predicate, process)
from ._components import Component
from ._declarations import (Condition, Const, Dataset, Env, Event, FloatState,
                            Handle, Output, Param, Pool, PQueues, Predicate,
                            Processes, Queue, Ref, Refs, Resource, Spawnable,
                            State, Store, Trace, capacity, count)
from ._graph import (ProcessDAG, ProcessDAGBlock, ProcessDAGEdge,
                     ProcessDAGNode)
from ._model import (CompilationCacheStats, CompilationPlan,
                     CompilationStatus,
                     ComponentFieldSchema, Experiment, ExperimentResults,
                     Model, Struct, trace_rng)

__all__ = [
    "Model", "Component", "ComponentFieldSchema", "CompilationPlan",
    "CompilationStatus", "CompilationCacheStats", "Experiment",
    "ExperimentResults", "Env",
    "Handle",
    "Param", "Output", "State", "FloatState", "Queue", "Resource", "Pool",
    "Store", "Dataset", "Condition", "Predicate", "Event", "Processes",
    "PQueues", "Ref", "Refs", "Const", "Struct", "Trace",
    "SpawnableProcess",
    "capacity",
    "collect", "count", "event", "function", "predicate", "process",
    "trace_rng",
    "ProcessDAG", "ProcessDAGBlock", "ProcessDAGNode", "ProcessDAGEdge",
    "SUCCESS", "PREEMPTED", "INTERRUPTED", "STOPPED", "CANCELLED", "TIMEOUT",
    "LOGGER_FATAL", "LOGGER_ERROR", "LOGGER_WARNING", "LOGGER_INFO",
    "hold", "now", "current", "interrupt", "stop", "wait_process",
    "wait_event", "resume",
    "spawn", "despawn",
    "suspend", "status", "set_priority",
    "timer_set", "timer_add", "timer_cancel", "timers_clear",
    "current_event", "event_count", "clear_events",
    "log_text", "log_user", "log_user_i64", "log_user_f64",
    "f2i", "i2f",
]

# Signal values returned by the blocking verbs (cmb_process.h). Any other
# value is a user-defined signal passed via sim.interrupt()/sim.resume().
SUCCESS = 0       #: returned normally
PREEMPTED = -1    #: holdings were preempted; the process lost them all
INTERRUPTED = -2  #: interrupted with the generic signal
STOPPED = -3      #: the awaited process was stopped
CANCELLED = -4    #: a wait/request was cancelled
TIMEOUT = -5      #: conventional signal for timer wakeups

LOGGER_FATAL = 0x80000000
LOGGER_ERROR = 0x40000000
LOGGER_WARNING = 0x20000000
LOGGER_INFO = 0x10000000


def log_text(text: str) -> Handle:
    """Return a stable native string handle for process-body logging."""
    return _b.cstring(text)

if TYPE_CHECKING:
    # Typed declarations of the modeling verbs. At runtime (the `else`
    # branch) each is a Numba binding from _bindings/_intrinsics, callable
    # only inside nopython-compiled model code; entity handles are the
    # opaque ints stored in env fields. Blocking verbs return 0 on success
    # or, if the process was interrupted while waiting, the signal value.

    # --- Process verbs -------------------------------------------------------
    def hold(duration: float) -> int:
        """Suspend the calling process for `duration` simulated time."""
        ...

    def now() -> float:
        """Current simulation time."""
        ...

    def current() -> Handle:
        """Handle of the calling process."""
        ...

    def interrupt(process: Handle, signal: int, priority: int) -> None:
        """Interrupt a blocked process; it sees `signal` as return value."""
        ...

    def stop(process: Handle, retval: int) -> int:
        """Stop the target process."""
        ...

    def wait_process(process: Handle) -> int:
        """Block until the target process finishes (join)."""
        ...

    def wait_event(event: int) -> int:
        """Block until a scheduled event occurs or is cancelled."""
        ...

    def resume(process: Handle, signal: int) -> None:
        """Resume a process stopped with sim.stop()."""
        ...

    def spawn(process: SpawnableProcess, env: Env,
              priority: int = 0) -> Handle:
        """Create and start a new copy of a spawnable process; `process`
        is the descriptor published by a process decorated with
        ``spawnable=True``. The new process only begins running once the
        caller blocks, so its sim.Struct fields (zeroed at creation) can
        be initialized through the returned handle first."""
        ...

    def despawn(process: Handle) -> None:
        """Free a finished spawned process (its function returned,
        sim.status() == 2), recycling its memory during the trial.
        Optional: spawned processes still alive or unreclaimed at the
        end of the trial are stopped and freed automatically. Despawning
        the same handle twice is a no-op."""
        ...

    def suspend() -> int:
        """Suspend the calling process indefinitely; returns the signal
        of whatever wakes it (a timer, sim.resume(), sim.interrupt())."""
        ...

    def status(process: Handle) -> int:
        """Process status code (0 created, 1 running, 2 finished)."""
        ...

    def set_priority(process: Handle, priority: int) -> None:
        """Change a process's priority (queueing order in acquires)."""
        ...

    def timer_set(process: Handle, delay: float, signal: int) -> int:
        """Replace the process's pending timers with one waking it from
        sim.suspend() with `signal` after `delay`. Returns the timer id."""
        ...

    def timer_add(process: Handle, delay: float, signal: int) -> int:
        """Add a timer alongside any pending ones. Returns the timer id."""
        ...

    def timer_cancel(process: Handle, timer: int) -> int:
        """Cancel one pending timer. Returns 1 if found, else 0."""
        ...

    def timers_clear(process: Handle) -> None:
        """Cancel all pending timers of the process."""
        ...

    # --- Low-level events (cmb_event) -----------------------------------------
    def current_event() -> int:
        """Handle of the currently (or most recently) executed event,
        zero if none."""
        ...

    def event_count() -> int:
        """Number of events currently in the event queue."""
        ...

    def clear_events() -> None:
        """Cancel every scheduled event, ending the trial as soon as the
        caller blocks or returns. This also cancels the generated
        lifecycle events, so the recording window never closes and
        running processes are not stopped -- low-level escape hatch."""
        ...

    # --- Logging ---------------------------------------------------------------
    def log_user(flags: int, message: Handle) -> None:
        """Log a static message handle created by sim.log_text()."""
        ...

    def log_user_i64(flags: int, label: Handle, value: int) -> None:
        """Log a static label and int64 value."""
        ...

    def log_user_f64(flags: int, label: Handle, value: float) -> None:
        """Log a static label and float64 value."""
        ...

    # --- Bit-casts for store objects ----------------------------------------------
    def f2i(x: float) -> int:
        """Bit-cast a float64 to int64."""
        ...

    def i2f(i: int) -> float:
        """Bit-cast an int64 back to float64."""
        ...

else:
    from ._intrinsics import f2i, i2f

    # Process verbs
    hold = _b.process_hold
    now = _b.time
    current = _b.process_current
    interrupt = _b.process_interrupt
    stop = _b.process_stop
    wait_process = _b.process_wait_process
    wait_event = _b.process_wait_event
    resume = _b.process_resume
    suspend = _b.process_yield

    # Dynamic process creation: a spawnable process descriptor points at a
    # static descriptor [cfunc address, name cstring, allocation size]
    # built at model compile time (see _model._compile). Live spawns are
    # tracked in a per-trial native registry, so leftovers are stopped
    # and reclaimed at the end of the trial like the static processes.
    _spawn_desc = _ptr_caster(_nbtypes.int64)
    _process_create_sized = _b.process_create_sized
    _process_initialize = _b.process_initialize
    _process_start = _b.process_start
    _process_status = _b.process_status
    _process_stop = _b.process_stop
    _process_terminate = _b.process_terminate
    _process_destroy = _b.process_destroy
    _spawned_register = _b.spawned_register
    _spawned_unregister = _b.spawned_unregister

    @njit
    def spawn(process, env, priority=0):
        d = _carray(_spawn_desc(process), 3)
        p = _process_create_sized(_np.uint64(d[2]))
        _process_initialize(p, d[1], d[0], _record_addr(env), priority)
        _process_start(p)
        _spawned_register(p)
        return p

    @njit
    def despawn(process):
        if _spawned_unregister(process) != 0:
            if _process_status(process) == 1:
                _process_stop(process, 0)
            _process_terminate(process)
            _process_destroy(process)
    status = _b.process_status
    set_priority = _b.process_priority_set
    timer_set = _b.process_timer_set
    timer_add = _b.process_timer_add
    timer_cancel = _b.process_timer_cancel
    timers_clear = _b.process_timers_clear

    # Low-level events (cmb_event)
    current_event = _b.event_current
    event_count = _b.event_queue_count
    clear_events = _b.event_queue_clear

    # Logging
    log_user = _b.logger_user_msg
    log_user_i64 = _b.logger_user_i64
    log_user_f64 = _b.logger_user_f64
