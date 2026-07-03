Public Surface Map
==================

Top-level package
-----------------

.. automodule:: cimba
   :members:

Simulation API
--------------

Use :mod:`cimba.sim` for modeling.

Model declarations:

``Model``, ``Component``, ``Experiment``, ``Env``, ``Handle``, ``Param``,
``Output``, ``State``, ``FloatState``, ``Queue``, ``Resource``, ``Pool``,
``Store``, ``Dataset``, ``Condition``, ``Predicate``, ``Event``,
``Processes``, ``PQueues``, ``Spawnable``, ``Struct``, ``Trace``,
``capacity()``, ``count()``, ``process()``, ``collect()``.

Components group related declarations and process methods. Methods decorated
with top-level ``@sim.process`` are lowered into ordinary model processes at
model construction, and model callbacks can read component fields with
``env.retailer.orders``. Component fields are exposed in experiments with
flattened names such as ``retailer__orders``. Methods decorated with
top-level ``@sim.collect`` run once per instance at the end of each trial,
before the model-level ``@model.collect`` callback, typically assigning the
component's ``sim.Output`` fields.

Components may contain other components, and flattened names follow the same
recursive convention, for example ``env.attraction.queues.line`` becomes
``attraction__queues__line``. Nested component process methods are also lowered
with their component path in the process name.

Components may declare ``sim.Spawnable`` fields. A component-owned spawnable
binds to a same-named ``@sim.process`` method on that component, and can be
spawned from component or model code with natural paths such as
``sim.spawn(self.visitor, env)`` or
``sim.spawn(env.park.entrance.visitor, env)``. Spawnable component processes
may receive a final ``sim.Struct`` view parameter.

Fixed repeated structures can be declared with standard ``list[Component]``
annotations, for example ``attractions: list[Attraction] = [...]``. Model
callbacks can use indexed access such as ``env.attractions[i].queues[j]``;
runtime fields remain flattened, for example ``attractions__queues``. Nested
collections are linearized behind the scenes, so
``env.campus.zones[i].gates[j].queue`` remains valid model source while the
trial table stores a one-dimensional ``campus__zones__gates__queue`` field.

Process DAGs:

Call ``model.process_dag()`` to infer a resource-aware graph from registered
process bodies. The returned ``ProcessDAG`` contains ``ProcessDAGNode`` and
``ProcessDAGEdge`` records for processes and model fields, and can render
Mermaid or Graphviz DOT text. The inference follows direct ``sim`` calls,
simple aliases, helper functions called with ``env``, spawnables, stores,
priority queues, conditions, events, mutable state, and shared resources:

.. code-block:: python

   graph = model.process_dag()
   print(graph.to_mermaid())
   print(graph.to_dot())

See :doc:`../topical_guides/process_graphs` for examples, inferred edge kinds,
and limitations.

Per-process fields:

Declare a ``sim.Struct`` subclass with ``float`` and ``int`` annotations. A
process can receive its own field view as a final annotated parameter:
``def visitor(env, view: Visitor)``. Multi-copy processes can also receive the
copy index: ``def visitor(env, idx, view: Visitor)``. ``Visitor(handle)``
returns a read/write view of another process's fields when model code already
has that process handle.

Data-driven traces:

Declare ``demand: sim.Trace`` and pass per-trial replay data to
``model.experiment(demand=...)``: a 1-D array is shared by every trial, a 2-D
array maps row *i* to trial *i* (trial order is design-point-major with
replications innermost), and a sequence of 1-D arrays gives ragged per-trial
traces.

A trace field also accepts a callable ``f(rng)`` or ``f(rng, trial_index)``
returning a 1-D array -- the idiom for bootstrap resampling and fitted
generators. It runs once per trial with a ``numpy.random.Generator`` seeded
from that trial's own cimba seed and the field name, so the single
``experiment(seed=...)`` argument reproduces the simulation streams and the
generated traces together, and distinct trace fields draw independent
streams. ``sim.trace_rng(trial_seed, field_name)`` rebuilds any trial's
generator from its recorded ``exp["seed"]``, e.g. to inspect the trace a
failed trial replayed.

Callable traces run serially inside ``experiment()``, before the parallel
trial run -- negligible for bootstrap resampling, but a bottleneck when a
single generation is expensive (fitted time-series or ML models).
``model.trial_seeds(seed=..., replications=..., **params)`` returns the exact
per-trial seeds that ``experiment()`` will assign, so such traces can be
generated in parallel outside cimba and passed in as finished rows, with
bit-identical results::

   seeds = model.trial_seeds(seed=42, scale=[1.0, 2.0], replications=100)
   rows = Parallel(n_jobs=-1)(
       delayed(slow_generator)(sim.trace_rng(s, "demand")) for s in seeds)
   exp = model.experiment(scale=[1.0, 2.0], demand=rows,
                          replications=100, seed=42)

``cimba.bootstrap`` provides ready-made trace generators that resample an
observed series: ``iid(data, length)`` for serially independent data,
``moving_block(data, length, block)`` and ``circular_block(data, length,
block)`` for stationary dependent series, and ``stationary(data, length,
mean_block)`` (random geometric block lengths -- a good default for
autocorrelated data such as demand histories). Each returns an ``f(rng)``
closure to pass directly as a trace field value::

   from cimba import bootstrap

   demand = bootstrap.stationary(history, length=horizon, mean_block=7)
   exp = model.experiment(demand=demand, replications=200, seed=42)

For trending, seasonal, or autoregressive data there are three model-based
factories that fit the structure internally from the raw series:
``residual(data, length, trend=1, period=None, mean_block=None)`` (polynomial
trend or, with a ``period``, STL decomposition; residuals resampled i.i.d. or
stationary-block), ``wild(data, length=None, trend=1, period=None,
weights="rademacher")`` (heteroskedastic residuals, weighted in place), and
``sieve(data, length, order=None)`` (AR(p) with AIC order selection and
Yule--Walker coefficients, simulated forward with resampled innovations).
``trend`` and ``period`` also accept ``"auto"``; all three take
``nonnegative=True`` (clip at zero, for demand data) and ``start`` (evaluate
the structure on ``start..start+length-1``, e.g. ``start=len(data)`` for the
horizon after the history).

For supply-chain demand there are two more:
``intermittent(data, length, jitter=False)`` (zero-inflated series:
Markov-chain occurrence plus resampled nonzero sizes) and
``joint(panel, length, name=..., mean_block=...)`` (a mapping of field name to
series, resampled with shared block draws so cross-correlation survives;
the returned generators carry a ``trace_rng_name`` attribute, which
``experiment()`` uses instead of the field name when deriving each trial's
generator -- callables sharing the tag receive identical rngs).

Size ``length`` to cover warmup + duration + cooldown.

Inside a process body, ``values = sim.Trace(env.demand)`` returns the
trial's trace as a ``float64`` NumPy view supporting ``len()``, indexing,
slicing, and iteration; treat it as read-only. A generator that exhausts its
trace simply finishes -- the trial still runs to its configured recording
window, so generate traces that cover ``warmup + duration + cooldown`` (or
derive the experiment duration from the trace span), and consider recording
``sim.now()`` into an ``Output`` when the loop ends as an exhaustion check.

Experiments:

``model.experiment(...)`` returns an ``Experiment``; ``exp.run()`` executes
the trial table in place and returns the number of failed trials, and
``exp["field"]`` reads any trial column as an array. ``exp.summary()``
condenses the outputs across replications: it returns a structured array with
one record per design point holding the swept parameter values and, for each
output, its replication mean (``name``) and Student-t confidence-interval
half-width (``name_hw``, 95% by default)::

   exp = model.experiment(utilization=[0.7, 0.8, 0.9], replications=20,
                          duration=10_000.0, seed=42)
   exp.run()
   for row in exp.summary("avg_wait"):
       print(f"rho={row['utilization']:.1f}  "
             f"wait={row['avg_wait']:.2f} +- {row['avg_wait_hw']:.2f}")

``exp.summary("a", "b", confidence=0.99)`` selects outputs and the confidence
level; failed trials (NaN) are excluded per output. ``exp.replications`` and
``exp.swept`` expose the layout (trial order is design-point-major with
replications innermost).

Process verbs:

``hold()``, ``now()``, ``current()``, ``interrupt()``, ``stop()``,
``wait_process()``, ``wait_event()``, ``resume()``, ``suspend()``,
``status()``, ``set_priority()``, ``timer_set()``, ``timer_add()``,
``timer_cancel()``, ``timers_clear()``, ``spawn()``, ``despawn()``.

Dynamic processes:

A process named in a ``sim.Spawnable`` field is created at runtime with
``sim.spawn(env.<name>, env, priority=0)``. The returned handle can be used to
initialize its ``sim.Struct`` fields before it first runs. Finished spawned
processes can be reclaimed with ``sim.despawn(handle)``. Component-owned
spawnables use the same call through the component namespace, for example
``sim.spawn(env.flow.visitor, env)``.

Low-level events:

Callbacks registered with ``@model.event`` are exposed in ``sim.Event`` fields.
Use ``schedule()``, ``schedule_at()``, ``event_cancel()``,
``event_reschedule()``, ``event_reprioritize()``, ``event_scheduled()``,
``event_time()``, ``event_priority()``, ``current_event()``,
``event_count()``, and ``clear_events()``.

Queues and resources:

``put()``, ``get()``, ``level()``, ``space()``, ``mean_level()``,
``acquire()``, ``release()``, ``preempt()``, ``available()``, ``in_use()``,
``held()``, ``mean_in_use()``, ``pool_acquire()``, ``pool_release()``,
``pool_preempt()``, ``pool_available()``, ``pool_held()``, ``pool_in_use()``,
``pool_mean_in_use()``, ``queue_history()``, ``resource_history()``,
``pool_history()``, ``queue_report()``, ``queue_report_file()``,
``resource_report()``, ``resource_report_file()``, ``pool_report()``,
``pool_report_file()``.

Stores, priority queues, datasets, and conditions:

``store_put()``, ``store_get()``, ``store_take()``, ``store_length()``,
``store_space()``, ``store_position()``, ``store_mean_length()``,
``pq_put()``, ``pq_get()``, ``pq_take()``, ``pq_length()``, ``pq_space()``,
``pq_position()``, ``pq_reprioritize()``, ``pq_cancel()``,
``pq_mean_length()``, ``tally()``, ``dataset_mean()``, ``dataset_count()``,
``dataset_min()``, ``dataset_max()``, ``dataset_std()``,
``dataset_median()``, ``dataset_quantile()``, ``store_history()``,
``pq_history()``, ``store_report()``, ``store_report_file()``,
``pq_report()``, ``pq_report_file()``, ``dataset_print()``,
``dataset_print_file()``, ``dataset_fivenum()``, ``dataset_fivenum_file()``,
``dataset_histogram()``, ``dataset_histogram_file()``,
``dataset_correlogram()``, ``dataset_correlogram_file()``,
``dataset_pacf_correlogram()``, ``dataset_pacf_correlogram_file()``,
``timeseries_count()``, ``timeseries_min()``, ``timeseries_max()``,
``timeseries_mean()``, ``timeseries_std()``, ``timeseries_median()``,
``timeseries_print()``, ``timeseries_print_file()``, ``timeseries_fivenum()``,
``timeseries_fivenum_file()``, ``timeseries_histogram()``,
``timeseries_histogram_file()``, ``timeseries_correlogram()``,
``timeseries_correlogram_file()``, ``timeseries_pacf_correlogram()``,
``timeseries_pacf_correlogram_file()``, ``wait_for()``, ``signal()``.

For text reports and text-mode plots, the no-suffix helpers print to stdout for
console and notebook use; the ``*_file()`` variants write to a path handle
created with ``sim.log_text()``.

Random draws:

``exponential()``, ``gamma()``, ``uniform()``, ``normal()``, ``random01()``,
``rayleigh()``, ``pert()``, ``pert_mod()``, ``bernoulli()``, ``flip()``,
``triangular()``, ``weibull()``, ``lognormal()``, ``erlang()``, ``beta()``,
``poisson()``, ``dice()``, ``std_normal()``, ``std_exponential()``,
``std_gamma()``, ``std_beta()``, ``logistic()``, ``cauchy()``, ``pareto()``,
``chisquared()``, ``f_dist()``, ``std_t()``, ``t_dist()``, ``geometric()``,
``binomial()``, ``negative_binomial()``, ``pascal()``, ``hypoexponential()``,
``hyperexponential()``, ``categorical()``, ``loaded_dice()``.

Logging:

``log_text()``, ``log_user()``, ``log_user_i64()``, ``log_user_f64()``.
Top-level ``cimba`` also exposes ``logger_flags_on()`` and
``logger_flags_off()``.

Signal constants and casts:

``SUCCESS``, ``PREEMPTED``, ``INTERRUPTED``, ``STOPPED``, ``CANCELLED``,
``TIMEOUT``, ``LOGGER_FATAL``, ``LOGGER_ERROR``, ``LOGGER_WARNING``,
``LOGGER_INFO``, ``f2i()``, ``i2f()``.
