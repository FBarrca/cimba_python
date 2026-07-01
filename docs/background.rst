.. _background:

The Whys and Hows of Cimba Python, Explained
============================================

In this section, we explain the background for Cimba Python and the design
choices that shape the Python bindings. The structure mirrors the upstream
Cimba documentation, but the vocabulary here stays at the Python modeling layer:
``sim.Model`` classes, registered process functions, handles, queues,
resources, conditions, datasets, random draws, and experiments.

.. _background_history:

Project history and goals
-------------------------

Cimba Python exists to make Cimba's process-oriented discrete event simulation
style available from Python. The package keeps the same core goals as Cimba:
fast execution, portable builds, expressive simulation models, and robust
behavior that fails loudly when a model violates its own assumptions.

The Python binding adds another goal: the model should look like Python. A model
is declared as a ``sim.Model`` subclass. Parameters, outputs, state variables,
and simulation entities are annotations on that class. Process behavior is
written as ordinary Python functions registered with decorators such as
``@model.process`` and ``@model.collect``. A trial table is created with
``model.experiment(...)`` and run with ``Experiment.run()``.

The result is a modeling surface aimed at researchers and engineers who want
Python ergonomics without giving up compiled process bodies, independent
replications, and multicore experiment runs.

.. _background_coroutines:

Coroutines revisited
--------------------

Process-oriented simulation is built around active entities that can pause and
resume. In Cimba Python, a process function can call ``sim.hold()``,
``sim.get()``, ``sim.acquire()``, ``sim.wait_for()``, or another blocking
operation from inside normal Python control flow. There is no ``yield`` in the
model code and no need to split a process into callback fragments.

That detail matters for large models. A helper function called deep in the call
chain can block on simulation time or on a resource, then resume from the same
point. The model can be organized around the domain rather than around the
mechanics of scheduling.

.. _background_processes:

Cimba processes are asymmetric coroutines
-----------------------------------------

A Cimba Python process is a registered Python function that runs inside a
single trial. Only one process in a trial is active at a time. When the active
process blocks, control returns to the hidden dispatcher, which chooses the next
scheduled event or process wakeup.

Processes can hold for simulated time, wait for another process, wait for a
scheduled event, acquire or release resources, put to or get from queues, wait
for arbitrary model predicates, suspend, resume, stop, or interrupt other
processes by handle. Multi-copy processes receive an integer copy index, and
process handle arrays are exposed through ``sim.Processes`` fields.

The illustrations below show the same stack-switching idea used by the engine.
From Python's point of view, the important consequence is simpler: a process
continues exactly after the blocking ``sim`` call that paused it.

.. image:: ../subprojects/cimba/images/stack_1.png

.. image:: ../subprojects/cimba/images/stack_2.png

.. image:: ../subprojects/cimba/images/stack_3.png

.. image:: ../subprojects/cimba/images/stack_4.png

.. image:: ../subprojects/cimba/images/stack_5.png

.. image:: ../subprojects/cimba/images/stack_6.png

.. image:: ../subprojects/cimba/images/stack_7.png

Object-oriented modeling with Python declarations
-------------------------------------------------

Cimba Python uses Python classes for model declarations rather than a large
hierarchy of runtime objects. A ``sim.Model`` subclass declares the shape of a
trial:

.. code-block:: python

    import cimba.sim as sim

    class Clinic(sim.Model):
        arrival_rate: sim.Param
        wait_time: sim.Output
        queue: sim.Queue
        doctor: sim.Resource
        waits: sim.Dataset

    model = Clinic("clinic")

Model fields are typed by their simulation role. ``sim.Param`` values are
expanded into parameter combinations, ``sim.Output`` values are collected after
each trial, ``sim.State`` and ``sim.FloatState`` hold mutable trial-local state,
and entities such as ``sim.Queue``, ``sim.Resource``, ``sim.Pool``,
``sim.Store``, ``sim.PQueues``, ``sim.Condition``, and ``sim.Dataset`` are
created for each trial.

Processes are also model objects:

.. code-block:: python

    @model.process(copies=4, priority=0)
    def patient(env: Clinic, idx: int):
        sim.hold(sim.exponential(1.0 / env.arrival_rate))
        sim.put(env.queue, 1)

For per-process fields, declare a ``sim.Struct`` and attach it to a process.
The final annotated process parameter is a typed view of the current process's
fields:

.. code-block:: python

    class Visitor(sim.Struct):
        entered_at: float
        attraction: int

    @model.process(struct=Visitor)
    def visitor(env: Clinic, v: Visitor):
        v.entered_at = sim.now()

Dynamic processes use ``sim.Spawnable`` fields and ``sim.spawn()``. A spawned
process can be initialized through its handle, then reclaimed with
``sim.despawn()`` when it is no longer needed.

Events and the event queue
--------------------------

State changes happen at event times. Most model code schedules events
implicitly: ``sim.hold(5.0)`` schedules a wakeup five simulated time units from
now, ``sim.put()`` may wake a waiting consumer, and ``sim.release()`` may wake a
waiting resource user.

Explicit event callbacks are available when the model needs them. Register a
callback with ``@model.event``, expose it through a ``sim.Event`` field, and
schedule it with ``sim.schedule()`` or ``sim.schedule_at()``.

.. code-block:: python

    class Shutdown(sim.Model):
        done: sim.Event
        worker: sim.Processes

    model = Shutdown()

    @model.event
    def done(env: Shutdown):
        sim.stop(env.worker[0], 0)

    @model.process
    def controller(env: Shutdown):
        sim.schedule(env.done, env, 100.0)

Scheduled events can be cancelled, rescheduled, reprioritized, inspected, and
waited on with ``sim.wait_event()``. When several events have the same time, the
higher priority event runs first; ties at the same priority run in arrival
order.

The hash-heap - a binary heap meets a hash map
----------------------------------------------

The event queue needs two properties at the same time: fast access to the next
event in time order, and fast access to a specific event handle when a model
cancels or changes it. Cimba Python exposes the handle-oriented behavior through
``sim.schedule()``, ``sim.event_cancel()``, ``sim.event_reschedule()``,
``sim.event_reprioritize()``, ``sim.event_scheduled()``, ``sim.event_time()``,
and ``sim.event_priority()``.

Model code does not need to know the internal data structure, but it benefits
from it whenever timers, explicit events, and process wakeups are scheduled in
large numbers.

Resources, resource guards, demands and conditions
--------------------------------------------------

Many simulation entities have the same pattern: a process makes a demand, the
demand can either be satisfied immediately or the process waits, and a later
state change wakes one or more waiting processes.

Cimba Python exposes this as queues, resources, pools, stores, priority queues,
and conditions:

* ``sim.Queue`` stores counts and supports ``sim.put()`` and ``sim.get()``.
* ``sim.Resource`` represents one exclusive unit with ``sim.acquire()`` and
  ``sim.release()``.
* ``sim.Pool`` represents multiple interchangeable units with
  ``sim.pool_acquire()`` and ``sim.pool_release()``.
* ``sim.Store`` and ``sim.PQueues`` hold integer payloads that can be taken by
  handle or priority.
* ``sim.Condition`` combines ``sim.Predicate`` functions, ``sim.wait_for()``,
  and ``sim.signal()`` for arbitrary model-defined readiness checks.

Blocking calls return a signal. ``sim.SUCCESS`` means the operation completed.
Signals such as ``sim.PREEMPTED``, ``sim.INTERRUPTED``, ``sim.STOPPED``,
``sim.CANCELLED``, and ``sim.TIMEOUT`` let a process decide how to clean up and
what to do next.

Error handling: The loud crashing noise
---------------------------------------

Cimba Python prefers clear failures to quiet corruption. Invalid model
declarations, missing predicates, unsupported field types, impossible resource
requests, negative times, and invalid handles should fail early. During trial
execution, a failed trial is counted by ``Experiment.run()`` so the caller can
decide whether to reject the experiment, retry, or inspect the output.

This is especially important for compiled process bodies. Keep the process hot
path simple, use explicit state fields, and let exceptions during setup catch
configuration mistakes before the experiment fans out across many trials.

Logging flags and bit masks
---------------------------

Cimba Python exposes logger flags at the top level with
``cimba.logger_flags_on()`` and ``cimba.logger_flags_off()``. User process logs
use fixed-signature helpers:

.. code-block:: python

    import cimba
    import cimba.sim as sim

    USER = 0x00000001
    MSG = sim.log_text("customer arrived")
    LABEL = sim.log_text("queue level")

    @model.process
    def arrival(env):
        sim.log_user(USER, MSG)
        sim.log_user_i64(USER, LABEL, sim.level(env.queue))

Static text is registered once with ``sim.log_text()``. Process bodies then log
static messages, integer values, or floating-point values without allocating
formatted Python strings inside the simulation loop.

Pseudo-random number generators and distributions
-------------------------------------------------

Each trial receives its own pseudo-random stream. Passing ``seed=...`` to
``model.experiment(...)`` makes runs reproducible; omitting it lets the package
choose seeds for independent trials.

The ``cimba.sim`` namespace includes common continuous, discrete, and empirical
draws: ``sim.random01()``, ``sim.uniform()``, ``sim.exponential()``,
``sim.normal()``, ``sim.gamma()``, ``sim.beta()``, ``sim.triangular()``,
``sim.weibull()``, ``sim.lognormal()``, ``sim.poisson()``, ``sim.binomial()``,
``sim.geometric()``, ``sim.categorical()``, ``sim.loaded_dice()``, and more.

.. image:: ../subprojects/cimba/images/crossplot_random.png

Use random draws inside process bodies just like other ``sim`` functions. For
parameter sweeps, declare distribution parameters as ``sim.Param`` fields so
each combination becomes part of the experiment table.

Data sets and summaries
-----------------------

``sim.Dataset`` collects untimed samples with ``sim.tally()``. Summaries are
available with ``sim.dataset_count()``, ``sim.dataset_mean()``,
``sim.dataset_min()``, ``sim.dataset_max()``, and ``sim.dataset_std()``.
The same datasets can print raw values, five-number summaries, histograms,
autocorrelation correlograms, and partial-autocorrelation correlograms with
``sim.dataset_print()``, ``sim.dataset_fivenum()``,
``sim.dataset_histogram()``, ``sim.dataset_correlogram()``, and
``sim.dataset_pacf_correlogram()``. Matching ``*_file()`` helpers write the
same reports to path handles created with ``sim.log_text()``.

Time-weighted summaries are attached to simulation entities. For a queue,
``sim.mean_level()`` reports the duration-weighted mean level over the recording
window. Resources, pools, stores, and priority queues provide matching mean
utilization or length accessors. Their underlying histories are available with
``sim.queue_history()``, ``sim.resource_history()``, ``sim.pool_history()``,
``sim.store_history()``, and ``sim.pq_history()``. The returned time-series
handles can be summarized or reported with ``sim.timeseries_mean()``,
``sim.timeseries_std()``, ``sim.timeseries_median()``,
``sim.timeseries_fivenum()``, ``sim.timeseries_histogram()``,
``sim.timeseries_correlogram()``, and
``sim.timeseries_pacf_correlogram()``.

Entity-level text reports are exposed as ``sim.queue_report()``,
``sim.resource_report()``, ``sim.pool_report()``, ``sim.store_report()``, and
``sim.pq_report()``, again with ``*_file()`` variants for writing to files.
These are most useful in single-trial debugging and tutorial runs; scalar
outputs are usually better for large parallel experiments.

``Model.experiment(..., warmup=..., duration=...)`` controls the measurement
window. Warmup lets the model reach a representative state before summaries are
collected.

Experiments consist of multi-threaded trials
--------------------------------------------

A Cimba Python experiment is a table of independent trials. Parameter arrays
create combinations, ``replications`` repeats each combination, and
``Experiment.run()`` executes the trials. Outputs are read back by name:

.. code-block:: python

    exp = model.experiment(
        arrival_rate=[0.6, 0.7, 0.8],
        replications=20,
        duration=10000.0,
        warmup=1000.0,
        seed=42,
    )
    failures = exp.run()
    means = exp["wait_time"]

The independence of trials is the source of the parallelism. A single trial is
kept deterministic and sequential in simulated time, while many trials can run
at the same time across CPU cores.

Benchmarking Cimba against SimPy
--------------------------------

SimPy and Cimba Python both let modelers describe process-oriented simulations
in Python, but they make different tradeoffs. SimPy is pure Python and built
around generator processes. Cimba Python uses registered Python functions,
compiled process bodies, and the ``cimba.sim`` modeling API.

The result is that Cimba Python is aimed at experiments with many replications,
large parameter sweeps, and hot process loops where compiled execution and
parallel trial execution matter.

The repository's ``benchmark/`` directory contains M/M/1 queue benchmarks for
the Python API. The matching SimPy and native C benchmark sources are vendored
with the C library in ``subprojects/cimba/benchmark/``.

On an AMD Ryzen 7 9700X under WSL Ubuntu 24.04, averaged over 10 runs, with
Cimba Python timed after its one-time Numba compile:

.. list-table::
   :header-rows: 1

   * - Benchmark
     - SimPy
     - Cimba Python
     - Cimba C
   * - Single core, single trial
     - 2.612 s
     - 0.096 s
     - 0.083 s
   * - Multicore, 100 trials
     - 36.807 s
     - 1.131 s
     - 0.970 s

The benchmark data and charts are in
``benchmark/AMD_Ryzen_7_9700X_WSL.ods``.

Benchmark numbers depend on the model, machine, Python version, and build
configuration, so treat them as a reason to measure your model rather than a
universal constant.

How about the name 'Cimba'?
---------------------------

The Python package keeps the Cimba name because it exposes the same simulation
ideas through Python bindings. In Python code, users normally import
``cimba.sim`` and write models in terms of ``sim.Model`` and process functions.

If in doubt, read the source code
---------------------------------

Cimba Python is open source. The most useful starting points are the
``tutorial`` directory for complete runnable models, ``tests/test_smoke.py`` for
small focused API examples, and the generated :doc:`api/cimba` page for the
public Python surface.
