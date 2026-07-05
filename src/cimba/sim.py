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
record (the `env` seen by process bodies); the subclass doubles as the
static type of `env`, so fields are checked and completed:

    class MG1(sim.Model):
        utilization: sim.Param          # swept input
        avg_queue_length: sim.Output    # result
        queue: sim.Queue                # cmb_buffer handle

    mg1 = MG1()

    @mg1.process
    def arrivals(env: MG1):
        while True:
            sim.hold(cimba.random.exponential(1.0 / env.utilization))
            env.queue.put(1)

Every declared Queue/Resource/Pool/Store/PQueues-element/Condition field
carries its verbs as methods: ``env.queue.put(1)``, ``env.queue.get(1)``,
``env.server.acquire()``, ``env.cond.wait_for(pred)``, and so on.
Component-owned fields support the same sugar through
``self.<field>.<method>(...)``.

Concept translation (cimba -> sim API):

    cmb_process       @model.process (copies=, priority=, struct=),
                      sim.hold(), sim.current(), sim.interrupt(),
                      sim.stop(), sim.wait_process(), sim.wait_event(),
                      sim.resume(), sim.timer_set()/sim.timer_add()/
                      sim.timer_cancel(); dynamic creation via
                      sim.Spawnable fields, sim.spawn()/sim.despawn()
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
    cmb_condition     sim.Condition + sim.Predicate + @model.predicate,
                      env.<cond>.wait_for(predicate)/signal()
    cmb_event         sim.Event + @model.event, sim.schedule()/
                      sim.schedule_at(), sim.event_cancel()/
                      _reschedule()/_reprioritize()/_scheduled()/
                      _time()/_priority(), sim.current_event(),
                      sim.event_count(), sim.clear_events()
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
`def machine(env, idx)`. The trial function, recording lifecycle, and all
create/start/stop/destroy plumbing are generated and compiled by Model.

Related fields and process methods can be grouped with ``sim.Component``.
Component methods marked with top-level ``@sim.process`` are authoring-time
methods; Model lowers them into ordinary flat process functions before Numba
compilation. A component method marked with top-level ``@sim.collect`` runs
once per instance at the end of each trial (before the model-level
``@model.collect``, which can then aggregate), typically assigning the
component's own Output fields from ``self``. Model callbacks can use ``env.retailer.orders``; trial-table
fields remain flattened with names such as ``retailer__orders``. Components
may contain nested components; paths such as ``env.attraction.queues.line``
flatten to names such as ``attraction__queues__line``. A component-owned
``sim.Spawnable`` field binds to the same-named component process method and
can be spawned with paths such as ``sim.spawn(self.visitor, env)`` or
``sim.spawn(env.flow.visitor, env)``.
Component-owned ``sim.Processes`` fields likewise publish handles for
same-named fixed component process methods; component collections flatten
ragged per-item copy counts behind paths such as ``env.teams[i].worker[j]``.
Fixed collections of repeated components can be declared with standard
``list[ComponentType]`` annotations. Model callbacks can use indexed access
such as ``env.attractions[i].queues[j]``. Nested collections also work; Cimba
lowers paths like ``env.campus.zones[i].gates[j].queue`` to flattened fields
and generated offset tables before compilation.

Module layout: the verbs below alias the raw symbol bindings in
``_bindings``; the cast helpers live in ``_intrinsics``; declaration markers
live in ``_declarations``; Component lowering lives in ``_components``;
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
from ._components import Component, collect, process
from ._declarations import (Condition, Const, Dataset, Env, Event, FloatState,
                            Handle, Output, Param, Pool, PQueues, Predicate,
                            Processes, Queue, Ref, Refs, Resource, Spawnable,
                            State, Store, Trace, capacity, count)
from ._graph import (ProcessDAG, ProcessDAGBlock, ProcessDAGEdge,
                     ProcessDAGNode)
from ._model import Experiment, Model, Struct, trace_rng

__all__ = [
    "Model", "Component", "Experiment", "Env", "Handle",
    "Param", "Output", "State", "FloatState", "Queue", "Resource", "Pool",
    "Store", "Dataset", "Condition", "Predicate", "Event", "Processes",
    "PQueues", "Ref", "Refs", "Const", "Spawnable", "Struct", "Trace",
    "capacity",
    "collect", "count", "process", "trace_rng",
    "ProcessDAG", "ProcessDAGBlock", "ProcessDAGNode", "ProcessDAGEdge",
    "SUCCESS", "PREEMPTED", "INTERRUPTED", "STOPPED", "CANCELLED", "TIMEOUT",
    "LOGGER_FATAL", "LOGGER_ERROR", "LOGGER_WARNING", "LOGGER_INFO",
    "hold", "now", "current", "interrupt", "stop", "wait_process",
    "wait_event", "resume",
    "spawn", "despawn",
    "suspend", "status", "set_priority",
    "timer_set", "timer_add", "timer_cancel", "timers_clear",
    "schedule", "schedule_at", "event_cancel", "event_reschedule",
    "event_reprioritize", "event_scheduled", "event_time",
    "event_priority", "current_event", "event_count", "clear_events",
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

    def spawn(process: int, env: Env, priority: int = 0) -> Handle:
        """Create and start a new copy of a spawnable process; `process`
        is a sim.Spawnable env field or a lowered component-owned
        sim.Spawnable path. The new process only begins running once the
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
    def schedule(event: int, env: Env, delay: float, data: int = 0,
                 priority: int = 0) -> int:
        """Schedule a @model.event callback `delay` time units from now.
        `event` is the env field of the same name; `data` is the int64
        word handed to the callback. Returns the event handle used by the
        other event verbs and sim.wait_event()."""
        ...

    def schedule_at(event: int, env: Env, at: float, data: int = 0,
                    priority: int = 0) -> int:
        """Schedule a @model.event callback at absolute time `at`
        (must not be before sim.now()). Returns the event handle."""
        ...

    def event_cancel(event: int) -> int:
        """Remove a scheduled event from the queue; 1 if found, else 0.
        Processes blocked on it in sim.wait_event() see CANCELLED."""
        ...

    def event_reschedule(event: int, at: float) -> int:
        """Move a scheduled event to absolute time `at`; 1 if found."""
        ...

    def event_reprioritize(event: int, priority: int) -> int:
        """Change a scheduled event's priority; 1 if found."""
        ...

    def event_scheduled(event: int) -> int:
        """1 if the event is currently in the event queue, else 0."""
        ...

    def event_time(event: int) -> float:
        """Scheduled time of an event; it must still be in the queue
        (check with event_scheduled() first if unsure)."""
        ...

    def event_priority(event: int) -> int:
        """Priority of an event; it must still be in the queue."""
        ...

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

    # Dynamic process creation: a sim.Spawnable env field points at a
    # static descriptor [cfunc address, name cstring, allocation size]
    # built at model compile time (see _model._compile). Live spawns are
    # tracked in a per-trial native registry, so leftovers are stopped
    # and reclaimed at the end of the trial like the static processes.
    _spawn_desc = _ptr_caster(_nbtypes.int64)
    _process_create_sized = _b.process_create_sized
    _process_initialize = _b.process_initialize
    _process_start = _b.process_start
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
            _process_terminate(process)
            _process_destroy(process)
    status = _b.process_status
    set_priority = _b.process_priority_set
    timer_set = _b.process_timer_set
    timer_add = _b.process_timer_add
    timer_cancel = _b.process_timer_cancel
    timers_clear = _b.process_timers_clear

    # Low-level events (cmb_event)
    event_cancel = _b.event_cancel
    event_reschedule = _b.event_reschedule
    event_reprioritize = _b.event_reprioritize
    event_scheduled = _b.event_is_scheduled
    event_time = _b.event_time
    event_priority = _b.event_priority
    current_event = _b.event_current
    event_count = _b.event_queue_count
    clear_events = _b.event_queue_clear

    _event_schedule = _b.event_schedule

    @njit
    def schedule(event, env, delay, data=0, priority=0):
        return _event_schedule(event, _record_addr(env), data,
                               now() + delay, priority)

    @njit
    def schedule_at(event, env, at, data=0, priority=0):
        return _event_schedule(event, _record_addr(env), data, at,
                               priority)

    # Logging
    log_user = _b.logger_user_msg
    log_user_i64 = _b.logger_user_i64
    log_user_f64 = _b.logger_user_f64
