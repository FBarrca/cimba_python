Cimba Python API Reference
==========================

Top-level package
-----------------

.. automodule:: cimba
   :members:

Simulation API
--------------

Use :mod:`cimba.sim` for modeling.

Model declarations:

``Model``, ``Experiment``, ``Env``, ``Handle``, ``Param``, ``Output``,
``State``, ``FloatState``, ``Queue``, ``Resource``, ``Pool``, ``Store``,
``Dataset``, ``Condition``, ``Predicate``, ``Event``, ``Processes``,
``PQueues``, ``Spawnable``, ``Struct``, ``capacity()``, ``count()``.

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

Process verbs:

``hold()``, ``now()``, ``current()``, ``interrupt()``, ``stop()``,
``wait_process()``, ``wait_event()``, ``resume()``, ``suspend()``,
``status()``, ``set_priority()``, ``timer_set()``, ``timer_add()``,
``timer_cancel()``, ``timers_clear()``, ``spawn()``, ``despawn()``.

Dynamic processes:

A process named in a ``sim.Spawnable`` field is created at runtime with
``sim.spawn(env.<name>, env, priority=0)``. The returned handle can be used to
initialize its ``sim.Struct`` fields before it first runs. Finished spawned
processes can be reclaimed with ``sim.despawn(handle)``.

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
``dataset_min()``, ``dataset_max()``, ``dataset_std()``, ``store_history()``,
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
