"""
cimba.sim - cimba's modeling concepts behind a SimPy-flavored API.

Processes are plain Python functions that block, in the style of SimPy --
but with no `yield`: cimba's processes are stackful fibers, so sim.hold()
and the acquire/get/wait verbs simply suspend the process, from any depth
of the call stack. Each process is compiled with Numba into machine code,
so models run at native speed on all cores; process bodies must stay in
nopython-compilable Python (numbers, loops, and sim.* calls).

Concept translation (cimba -> sim API):

    cmb_process       @model.process (copies=, priority=), sim.hold(),
                      sim.current(), sim.interrupt(), sim.stop(),
                      sim.wait_process(), sim.resume()
    cmb_buffer        Model(queues=[...]), sim.put()/sim.get()/sim.level()
    cmb_resource      Model(resources=[...]), sim.acquire()/sim.release()/
                      sim.preempt()
    cmb_resourcepool  Model(pools={name: capacity}), sim.pool_acquire()/
                      sim.pool_release()/sim.pool_preempt()
    cmb_objectqueue   Model(stores={name: capacity}), sim.store_put()/
                      sim.store_take() (objects are opaque int64 values;
                      sim.f2i()/sim.i2f() bit-cast timestamps in and out)
    cmb_condition     Model(conditions=[...]), @model.predicate,
                      sim.wait_for()/sim.signal()
    cmb_dataset       Model(datasets=[...]), sim.tally(),
                      sim.dataset_mean()/sim.dataset_count()
    statistics        recorded automatically over the measurement window:
                      sim.mean_level(), sim.mean_in_use(),
                      sim.pool_mean_in_use(), sim.store_mean_length()

A minimal model:

    mg1 = sim.Model("mg1", params=["utilization"], queues=["queue"],
                    outputs=["avg_queue_length"])

    @mg1.process
    def arrivals(env):
        while True:
            sim.hold(sim.exponential(1.0 / env.utilization))
            sim.put(env.queue, 1)

The `env` argument is the trial record: params, outputs, entity handles,
and declared state fields as attributes. Multi-copy processes may take a
second argument to learn their index: `def machine(env, idx)`. The trial
function, recording lifecycle, and all create/start/stop/destroy plumbing
are generated and compiled by Model.

Module layout: the verbs below alias the raw symbol bindings in
``_bindings``; the cast helpers live in ``_intrinsics``; Model/Experiment
and the trial codegen live in ``_model``.
"""

from typing import TYPE_CHECKING, Any

from numba import njit

from . import _bindings as _b
from ._intrinsics import record_addr as _record_addr
from ._model import Env, Experiment, Handle, Model

__all__ = [
    "Model", "Experiment", "Env", "Handle",
    "hold", "now", "current", "interrupt", "stop", "wait_process", "resume",
    "put", "get", "level", "mean_level",
    "acquire", "release", "preempt", "in_use", "mean_in_use",
    "pool_acquire", "pool_release", "pool_preempt", "pool_in_use",
    "pool_mean_in_use",
    "store_put", "store_take", "store_length", "store_mean_length",
    "tally", "dataset_mean", "dataset_count",
    "wait_for", "signal",
    "exponential", "gamma", "uniform", "normal", "random01",
    "f2i", "i2f",
]

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

    def resume(process: Handle, signal: int) -> None:
        """Resume a process stopped with sim.stop()."""
        ...

    # --- Queues (cmb_buffer): counted amounts --------------------------------
    def put(queue: Handle, amount: int) -> int:
        """Add `amount` to the queue, blocking while it is full."""
        ...

    def get(queue: Handle, amount: int) -> int:
        """Remove `amount` from the queue, blocking until available."""
        ...

    def level(queue: Handle) -> int:
        """Current queue content."""
        ...

    def mean_level(queue: Handle) -> float:
        """Time-weighted mean queue content over the recording window."""
        ...

    # --- Resources (cmb_resource): single holder, priority-aware -------------
    def acquire(resource: Handle) -> int:
        """Acquire the resource, blocking until it is free."""
        ...

    def release(resource: Handle) -> None:
        """Release the resource."""
        ...

    def preempt(resource: Handle) -> int:
        """Acquire the resource, preempting a lower-priority holder."""
        ...

    def in_use(resource: Handle) -> int:
        """1 if the resource is currently held, else 0."""
        ...

    def mean_in_use(resource: Handle) -> float:
        """Time-weighted mean utilization over the recording window."""
        ...

    # --- Resource pools (cmb_resourcepool): capacity > 1 ----------------------
    def pool_acquire(pool: Handle, amount: int) -> int:
        """Acquire `amount` units from the pool, blocking until available."""
        ...

    def pool_release(pool: Handle, amount: int) -> None:
        """Return `amount` units to the pool."""
        ...

    def pool_preempt(pool: Handle, amount: int) -> int:
        """Acquire `amount` units, preempting lower-priority holders."""
        ...

    def pool_in_use(pool: Handle) -> int:
        """Number of pool units currently held."""
        ...

    def pool_mean_in_use(pool: Handle) -> float:
        """Time-weighted mean units in use over the recording window."""
        ...

    # --- Stores (cmb_objectqueue): FIFO of opaque int64 objects ---------------
    def store_put(store: Handle, obj: int) -> int:
        """Append an object (any nonzero int64), blocking while full."""
        ...

    def store_take(store: Handle) -> int:
        """Remove and return the oldest object, blocking while empty."""
        ...

    def store_length(store: Handle) -> int:
        """Current number of objects in the store."""
        ...

    def store_mean_length(store: Handle) -> float:
        """Time-weighted mean store length over the recording window."""
        ...

    # --- Datasets (cmb_dataset): tally statistics ------------------------------
    def tally(dataset: Handle, value: float) -> int:
        """Record an observation; returns the observation count."""
        ...

    def dataset_mean(dataset: Handle) -> float:
        """Mean of the observations tallied so far."""
        ...

    def dataset_count(dataset: Handle) -> int:
        """Number of observations tallied so far."""
        ...

    # --- Random draws -----------------------------------------------------------
    def exponential(mean: float) -> float:
        """Exponentially distributed draw with the given mean."""
        ...

    def gamma(shape: float, scale: float) -> float:
        """Gamma-distributed draw."""
        ...

    def uniform(low: float, high: float) -> float:
        """Uniform draw from [low, high)."""
        ...

    def normal(mu: float, sigma: float) -> float:
        """Normally distributed draw."""
        ...

    def random01() -> float:
        """Uniform draw from [0, 1)."""
        ...

    # --- Conditions (cmb_condition) ---------------------------------------------
    def signal(condition: Handle) -> int:
        """Wake the condition's waiters to re-evaluate their predicates."""
        ...

    def wait_for(condition: Handle, predicate: int, env: Env) -> int:
        """Block until the predicate is satisfied; re-evaluated on every
        sim.signal(condition). `predicate` is an env._pred_<name> field."""
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
    resume = _b.process_resume

    # Queues (cmb_buffer)
    put = _b.buffer_put
    get = _b.buffer_get
    level = _b.buffer_level
    mean_level = _b.buffer_mean_level

    # Resources (cmb_resource)
    acquire = _b.resource_acquire
    release = _b.resource_release
    preempt = _b.resource_preempt
    in_use = _b.resource_in_use
    mean_in_use = _b.resource_mean_in_use

    # Resource pools (cmb_resourcepool)
    pool_acquire = _b.resourcepool_acquire
    pool_release = _b.resourcepool_release
    pool_preempt = _b.resourcepool_preempt
    pool_in_use = _b.resourcepool_in_use
    pool_mean_in_use = _b.resourcepool_mean_in_use

    # Stores (cmb_objectqueue)
    store_put = _b.objectqueue_put
    store_take = _b.objectqueue_take
    store_length = _b.objectqueue_length
    store_mean_length = _b.objectqueue_mean_length

    # Datasets (cmb_dataset)
    tally = _b.dataset_add
    dataset_mean = _b.dataset_mean
    dataset_count = _b.dataset_count

    # Random draws
    exponential = _b.random_exponential
    gamma = _b.random_gamma
    uniform = _b.random_uniform
    normal = _b.random_normal
    random01 = _b.random01

    # Conditions (cmb_condition)
    signal = _b.condition_signal

    _condition_wait = _b.condition_wait

    @njit
    def wait_for(cond, pred, env):
        return _condition_wait(cond, pred, _record_addr(env))
