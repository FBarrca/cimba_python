"""
cimba.sim - cimba's modeling concepts behind a SimPy-flavored API.

Processes are plain Python functions that block, in the style of SimPy --
but with no `yield`: cimba's processes are stackful fibers, so sim.hold()
and the acquire/get/wait verbs simply suspend the process, from any depth
of the call stack. Each process is compiled with Numba into machine code,
so models run at native speed on all cores; process bodies must stay in
nopython-compilable Python (numbers, loops, and sim.* calls).

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
            sim.hold(sim.exponential(1.0 / env.utilization))
            sim.put(env.queue, 1)

Concept translation (cimba -> sim API):

    cmb_process       @model.process (copies=, priority=), sim.hold(),
                      sim.current(), sim.interrupt(), sim.stop(),
                      sim.wait_process(), sim.wait_event(), sim.resume(),
                      sim.timer_set()/sim.timer_add()/sim.timer_cancel()
    cmb_buffer        sim.Queue, sim.put()/sim.get()/sim.level()
    cmb_resource      sim.Resource, sim.acquire()/sim.release()/
                      sim.preempt(), sim.held()
    cmb_resourcepool  sim.Pool (= n or sim.capacity(param)),
                      sim.pool_acquire()/sim.pool_release()/
                      sim.pool_preempt()
    cmb_objectqueue   sim.Store, sim.store_put()/sim.store_take()/
                      sim.store_get()/sim.store_position()
                      (objects are opaque int64 values; sim.f2i()/
                      sim.i2f() bit-cast timestamps in and out)
    cmb_condition     sim.Condition + sim.Predicate + @model.predicate,
                      sim.wait_for()/sim.signal()
    cmb_event         sim.Event + @model.event, sim.schedule()/
                      sim.schedule_at(), sim.event_cancel()/
                      _reschedule()/_reprioritize()/_scheduled()/
                      _time()/_priority(), sim.current_event(),
                      sim.event_count(), sim.clear_events()
    cmb_dataset       sim.Dataset, sim.tally(), sim.dataset_mean()/
                      _count()/_min()/_max()/_std()
    statistics        recorded over the measurement window (after warmup,
                      datasets are reset when it opens): sim.mean_level(),
                      sim.mean_in_use(), sim.pool_mean_in_use(),
                      sim.store_mean_length(), sim.pq_mean_length()

Mutable per-trial counters are declared with sim.State. Multi-copy
processes may take a second argument to learn their index:
`def machine(env, idx)`. The trial function, recording lifecycle, and all
create/start/stop/destroy plumbing are generated and compiled by Model.

Module layout: the verbs below alias the raw symbol bindings in
``_bindings``; the cast helpers live in ``_intrinsics``; Model/Experiment
and the trial codegen live in ``_model``.
"""

from typing import TYPE_CHECKING, Any

from numba import njit

from . import _bindings as _b
from ._intrinsics import record_addr as _record_addr
from ._model import (Condition, Dataset, Env, Event, Experiment, FloatState,
                     Handle, Model, Output, Param, Pool, PQueues, Predicate,
                     Processes, Queue, Resource, State, Store, capacity,
                     count)

__all__ = [
    "Model", "Experiment", "Env", "Handle",
    "Param", "Output", "State", "FloatState", "Queue", "Resource", "Pool",
    "Store", "Dataset", "Condition", "Predicate", "Event", "Processes",
    "PQueues", "capacity", "count",
    "SUCCESS", "PREEMPTED", "INTERRUPTED", "STOPPED", "CANCELLED", "TIMEOUT",
    "LOGGER_FATAL", "LOGGER_ERROR", "LOGGER_WARNING", "LOGGER_INFO",
    "hold", "now", "current", "interrupt", "stop", "wait_process",
    "wait_event", "resume",
    "suspend", "status", "set_priority",
    "timer_set", "timer_add", "timer_cancel", "timers_clear",
    "schedule", "schedule_at", "event_cancel", "event_reschedule",
    "event_reprioritize", "event_scheduled", "event_time",
    "event_priority", "current_event", "event_count", "clear_events",
    "flip", "held", "pool_held",
    "pq_put", "pq_get", "pq_take", "pq_length", "pq_space", "pq_position",
    "pq_reprioritize", "pq_cancel", "pq_mean_length",
    "put", "get", "level", "space", "mean_level",
    "acquire", "release", "preempt", "available", "in_use", "mean_in_use",
    "pool_acquire", "pool_release", "pool_preempt", "pool_available",
    "pool_in_use", "pool_mean_in_use",
    "store_put", "store_get", "store_take", "store_length", "store_space",
    "store_position", "store_mean_length",
    "tally", "dataset_mean", "dataset_count", "dataset_min", "dataset_max",
    "dataset_std",
    "wait_for", "signal",
    "exponential", "gamma", "uniform", "normal", "random01",
    "rayleigh", "pert", "bernoulli", "triangular", "weibull", "lognormal",
    "erlang", "beta", "poisson", "dice",
    "std_normal", "std_exponential", "std_gamma", "std_beta", "pert_mod",
    "logistic", "cauchy", "pareto", "chisquared", "f_dist", "std_t",
    "t_dist", "geometric", "binomial", "negative_binomial", "pascal",
    "hypoexponential", "hyperexponential", "categorical", "loaded_dice",
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

    def space(queue: Handle) -> int:
        """Remaining queue capacity (huge for unbounded queues)."""
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

    def available(resource: Handle) -> int:
        """1 if the resource is currently free, else 0."""
        ...

    def in_use(resource: Handle) -> int:
        """1 if the resource is currently held, else 0."""
        ...

    def held(resource: Handle, process: Handle) -> int:
        """1 if `process` holds this resource, else 0."""
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

    def pool_available(pool: Handle) -> int:
        """Number of pool units currently free."""
        ...

    def pool_held(pool: Handle, process: Handle) -> int:
        """Number of pool units held by the given process."""
        ...

    def pool_in_use(pool: Handle) -> int:
        """Number of pool units currently held."""
        ...

    def pool_mean_in_use(pool: Handle) -> float:
        """Time-weighted mean units in use over the recording window."""
        ...

    # --- Stores (cmb_objectqueue): FIFO of opaque int64 objects ---------------
    def store_put(store: Handle, obj: int) -> int:
        """Append an object, blocking while full."""
        ...

    def store_get(store: Handle) -> tuple[int, int]:
        """Remove the oldest object, returning (status, object)."""
        ...

    def store_take(store: Handle) -> int:
        """Remove and return the oldest object; interrupted takes return 0."""
        ...

    def store_length(store: Handle) -> int:
        """Current number of objects in the store."""
        ...

    def store_space(store: Handle) -> int:
        """Remaining store capacity (huge for unbounded stores)."""
        ...

    def store_position(store: Handle, obj: int) -> int:
        """1-based position of an object in the store, or 0 if absent."""
        ...

    def store_mean_length(store: Handle) -> float:
        """Time-weighted mean store length over the recording window."""
        ...

    # --- Priority queues (cmb_priorityqueue) ------------------------------------
    def pq_put(pqueue: Handle, obj: int, priority: int) -> int:
        """Insert an object (any nonzero int64) ordered by priority;
        returns the entry handle for pq_position()/pq_cancel()."""
        ...

    def pq_get(pqueue: Handle) -> tuple[int, int]:
        """Remove the highest-priority object, returning (status, object)."""
        ...

    def pq_take(pqueue: Handle) -> int:
        """Remove and return the highest-priority object, blocking while
        the queue is empty."""
        ...

    def pq_length(pqueue: Handle) -> int:
        """Current number of entries in the queue."""
        ...

    def pq_space(pqueue: Handle) -> int:
        """Remaining priority queue capacity."""
        ...

    def pq_position(pqueue: Handle, entry: int) -> int:
        """1-based position of the entry in the queue."""
        ...

    def pq_reprioritize(pqueue: Handle, entry: int, priority: int) -> None:
        """Change an entry's priority, reshuffling queue order."""
        ...

    def pq_cancel(pqueue: Handle, entry: int) -> int:
        """Remove an entry from the queue; 1 if found, else 0."""
        ...

    def pq_mean_length(pqueue: Handle) -> float:
        """Time-weighted mean queue length over the recording window."""
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

    def dataset_min(dataset: Handle) -> float:
        """Smallest observation tallied so far."""
        ...

    def dataset_max(dataset: Handle) -> float:
        """Largest observation tallied so far."""
        ...

    def dataset_std(dataset: Handle) -> float:
        """Sample standard deviation of the observations (0 if < 2)."""
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

    def rayleigh(s: float) -> float:
        """Rayleigh-distributed draw with scale parameter `s`."""
        ...

    def pert(low: float, mode: float, high: float) -> float:
        """PERT-distributed draw (scaled beta) over [low, high]."""
        ...

    def pert_mod(low: float, mode: float, high: float,
                 lambda_: float) -> float:
        """Modified PERT draw with explicit peakiness parameter."""
        ...

    def bernoulli(p: float) -> int:
        """1 with probability `p`, else 0."""
        ...

    def flip() -> int:
        """Fair coin: 1 or 0, equivalent to bernoulli(0.5) but faster."""
        ...

    def triangular(low: float, mode: float, high: float) -> float:
        """Triangular-distributed draw over [low, high]."""
        ...

    def weibull(shape: float, scale: float) -> float:
        """Weibull-distributed draw."""
        ...

    def lognormal(m: float, s: float) -> float:
        """Log-normal draw; mean exp(m + s^2/2), median exp(m)."""
        ...

    def erlang(k: int, m: float) -> float:
        """Erlang draw: sum of k exponentials of mean m (mean k*m)."""
        ...

    def beta(a: float, b: float, low: float, high: float) -> float:
        """Beta(a, b) draw scaled to [low, high]."""
        ...

    def poisson(rate: float) -> int:
        """Poisson-distributed count with the given rate."""
        ...

    def dice(a: int, b: int) -> int:
        """Uniform integer draw from [a, b] inclusive."""
        ...

    def std_normal() -> float:
        """Standard normal draw with mean 0 and standard deviation 1."""
        ...

    def std_exponential() -> float:
        """Standard exponential draw with mean 1."""
        ...

    def std_gamma(shape: float) -> float:
        """Standard gamma draw with scale 1."""
        ...

    def std_beta(a: float, b: float) -> float:
        """Beta(a, b) draw over [0, 1]."""
        ...

    def logistic(m: float, s: float) -> float:
        """Logistic-distributed draw with location `m` and scale `s`."""
        ...

    def cauchy(mode: float, scale: float) -> float:
        """Cauchy-distributed draw."""
        ...

    def pareto(shape: float, mode: float) -> float:
        """Pareto-distributed draw on [mode, infinity)."""
        ...

    def chisquared(k: float) -> float:
        """Chi-squared draw with `k` degrees of freedom."""
        ...

    def f_dist(a: float, b: float) -> float:
        """F-distributed draw with numerator/denominator degrees."""
        ...

    def std_t(v: float) -> float:
        """Standard Student's t draw with `v` degrees of freedom."""
        ...

    def t_dist(m: float, s: float, v: float) -> float:
        """Location-scale Student's t draw."""
        ...

    def geometric(p: float) -> int:
        """Geometric draw: trials up to and including first success."""
        ...

    def binomial(n: int, p: float) -> int:
        """Binomial draw: successes in `n` Bernoulli trials."""
        ...

    def negative_binomial(m: int, p: float) -> int:
        """Failures before the `m`th success."""
        ...

    def pascal(m: int, p: float) -> int:
        """Alias for negative_binomial."""
        ...

    def hypoexponential(means: Any) -> float:
        """Hypoexponential draw from a non-empty sequence of means."""
        ...

    def hyperexponential(means: Any, weights: Any) -> float:
        """Hyperexponential draw from matching mean and weight sequences."""
        ...

    def categorical(weights: Any) -> int:
        """Return an index sampled in proportion to a non-empty sequence
        of nonnegative weights."""
        ...

    def loaded_dice(probabilities: Any) -> int:
        """Alias for categorical(probabilities)."""
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
    from ._intrinsics import f2i, i2f, pq_get, store_get

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

    # Queues (cmb_buffer)
    put = _b.buffer_put
    get = _b.buffer_get
    level = _b.buffer_level
    space = _b.buffer_space
    mean_level = _b.buffer_mean_level

    # Resources (cmb_resource)
    acquire = _b.resource_acquire
    release = _b.resource_release
    preempt = _b.resource_preempt
    available = _b.resource_available
    in_use = _b.resource_in_use
    held = _b.resource_held
    mean_in_use = _b.resource_mean_in_use

    # Resource pools (cmb_resourcepool)
    pool_acquire = _b.resourcepool_acquire
    pool_release = _b.resourcepool_release
    pool_preempt = _b.resourcepool_preempt
    pool_available = _b.resourcepool_available
    pool_held = _b.resourcepool_held
    pool_in_use = _b.resourcepool_in_use
    pool_mean_in_use = _b.resourcepool_mean_in_use

    # Stores (cmb_objectqueue)
    store_put = _b.objectqueue_put
    store_take = _b.objectqueue_take
    store_length = _b.objectqueue_length
    store_space = _b.objectqueue_space
    store_position = _b.objectqueue_position
    store_mean_length = _b.objectqueue_mean_length

    # Priority queues (cmb_priorityqueue)
    pq_put = _b.priorityqueue_put
    pq_take = _b.priorityqueue_take
    pq_length = _b.priorityqueue_length
    pq_space = _b.priorityqueue_space
    pq_position = _b.priorityqueue_position
    pq_reprioritize = _b.priorityqueue_reprioritize
    pq_cancel = _b.priorityqueue_cancel
    pq_mean_length = _b.priorityqueue_mean_length

    # Datasets (cmb_dataset)
    tally = _b.dataset_add
    dataset_mean = _b.dataset_mean
    dataset_count = _b.dataset_count
    dataset_min = _b.dataset_min
    dataset_max = _b.dataset_max
    dataset_std = _b.dataset_std

    # Random draws
    exponential = _b.random_exponential
    gamma = _b.random_gamma
    uniform = _b.random_uniform
    normal = _b.random_normal
    random01 = _b.random01
    rayleigh = _b.random_rayleigh
    pert = _b.random_pert
    pert_mod = _b.random_pert_mod
    bernoulli = _b.random_bernoulli
    flip = _b.random_flip
    triangular = _b.random_triangular
    weibull = _b.random_weibull
    lognormal = _b.random_lognormal
    erlang = _b.random_erlang
    beta = _b.random_beta
    poisson = _b.random_poisson
    dice = _b.random_dice
    std_normal = _b.random_std_normal
    std_exponential = _b.random_std_exponential
    std_gamma = _b.random_std_gamma
    std_beta = _b.random_std_beta
    logistic = _b.random_logistic
    cauchy = _b.random_cauchy
    pareto = _b.random_pareto
    chisquared = _b.random_chisquared
    f_dist = _b.random_f_dist
    std_t = _b.random_std_t
    t_dist = _b.random_t
    geometric = _b.random_geometric
    binomial = _b.random_binomial
    negative_binomial = _b.random_negative_binomial
    pascal = _b.random_pascal

    @njit
    def hypoexponential(means):
        """Hypoexponential draw from a non-empty sequence of means."""
        if len(means) == 0:
            raise ValueError("hypoexponential() expects at least one mean")
        x = 0.0
        for mean in means:
            x += exponential(mean)
        return x

    @njit
    def categorical(weights):
        """Return an index sampled in proportion to nonnegative weights."""
        if len(weights) == 0:
            raise ValueError("categorical() expects at least one weight")
        total = 0.0
        for weight in weights:
            total += weight
        if total <= 0.0:
            raise ValueError("categorical() expects positive total weight")

        target = random01() * total
        cumulative = 0.0
        last = 0
        for i, weight in enumerate(weights):
            cumulative += weight
            last = i
            if target < cumulative:
                return i
        return last

    @njit
    def loaded_dice(probabilities):
        """Alias for categorical(probabilities)."""
        return categorical(probabilities)

    @njit
    def hyperexponential(means, weights):
        """Hyperexponential draw from matching mean and weight sequences."""
        if len(means) != len(weights):
            raise ValueError(
                "hyperexponential() means and weights must match")
        return exponential(means[categorical(weights)])

    # Conditions (cmb_condition)
    signal = _b.condition_signal

    _condition_wait = _b.condition_wait

    @njit
    def wait_for(cond, pred, env):
        return _condition_wait(cond, pred, _record_addr(env))

    # Logging
    log_user = _b.logger_user_msg
    log_user_i64 = _b.logger_user_i64
    log_user_f64 = _b.logger_user_f64
