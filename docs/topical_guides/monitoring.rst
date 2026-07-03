Monitoring and Statistics
=========================

Statistics in a ``sim.Model`` flow through three layers:

1. **Inside the trial**, entities record time-weighted state and datasets
   collect tallied observations, both over the measurement window.
2. **At the end of each trial**, the ``@model.collect`` callback reduces
   those recordings to ``sim.Output`` fields.
3. **Across trials**, the experiment table holds one record per trial;
   ``exp["field"]`` reads aligned arrays and ``exp.summary()`` condenses
   them into means and confidence intervals.

This guide walks the layers in order. The classic in-process objects
(``cimba.DataSummary`` and friends) are covered at the end -- they belong to
the classic API, not to ``sim.Model`` code.

The measurement window
----------------------

``model.experiment(warmup=..., duration=..., cooldown=...)`` defines the
window. Recording starts when the warmup ends: time-weighted statistics
begin accumulating and datasets are reset, so anything tallied during warmup
is discarded. Recording stops after ``duration``; the optional cooldown then
lets the model drain before processes are stopped.

Everything below reports over that window only -- no manual start/stop calls
are needed.

Time-weighted state
-------------------

Shared entities track their own occupancy. One verb per entity kind returns
the duration-weighted mean over the window:

* ``sim.mean_level(queue)`` -- mean content of a ``sim.Queue``
* ``sim.mean_in_use(resource)`` -- utilization of a ``sim.Resource``
* ``sim.pool_mean_in_use(pool)`` -- mean units held of a ``sim.Pool``
* ``sim.store_mean_length(store)`` -- mean length of a ``sim.Store``
* ``sim.pq_mean_length(pqueue)`` -- mean length of a ``sim.PQueues`` entry

Duration weighting makes these the right tool for average queue lengths,
utilization, and occupancy: a level held for ten hours counts ten times as
much as one held for one hour.

Tally statistics
----------------

For per-observation data -- waiting times, cycle times, batch sizes --
declare a ``sim.Dataset`` field and tally into it from process code:

.. code-block:: python

   class Clinic(sim.Model):
       avg_wait: sim.Output
       p95_wait: sim.Output
       waits: sim.Dataset
       ...

   @model.process
   def patient(env: Clinic):
       ...
       t0 = sim.now()
       sim.acquire(env.doctor)
       sim.tally(env.waits, sim.now() - t0)
       ...

A dataset keeps every value, so both moment and order statistics are
available: ``sim.dataset_mean()``, ``sim.dataset_count()``,
``sim.dataset_min()``, ``sim.dataset_max()``, ``sim.dataset_std()``,
``sim.dataset_median()``, and ``sim.dataset_quantile(dataset, q)``.

The median and quantile getters sort a copy of the data, so call them once
per trial rather than per observation -- the collector is the natural place.

Collecting outputs
------------------

``@model.collect`` runs once per trial, after the window closes and before
entities are torn down. It reads the recordings and writes ``sim.Output``
fields:

.. code-block:: python

   @model.collect
   def stats(env: Clinic):
       env.avg_wait = sim.dataset_mean(env.waits)
       env.p95_wait = sim.dataset_quantile(env.waits, 0.95)
       env.doctor_util = sim.mean_in_use(env.doctor)

Entities are destroyed when the trial function returns, so anything worth
keeping must land in an output here (or be written to a file with the report
verbs below).

Components can own their statistics the same way they own processes: a
method marked with top-level ``@sim.collect`` runs once per instance at the
end of each trial, before ``@model.collect`` (which can then aggregate over
the component outputs). See :doc:`../advanced/components`.

Across trials
-------------

After ``exp.run()``, each output is a per-trial array aligned with the
parameter columns, and ``exp.summary()`` reduces the table to one record per
design point with means and Student-t confidence half-widths:

.. code-block:: python

   exp = model.experiment(arrival_rate=[0.7, 0.8, 0.9],
                          replications=20, duration=10_000.0, seed=42)
   exp.run()
   for row in exp.summary("p95_wait"):
       print(f"rate={row['arrival_rate']:.1f}: "
             f"p95 wait {row['p95_wait']:.2f} +- {row['p95_wait_hw']:.2f}")

See :doc:`../concepts/experiments_results` for the full experiment
lifecycle, failure handling, and summary options.

Histories and text reports
--------------------------

For inspection beyond a mean, every recorded entity keeps a native time
series of its state over the window. ``sim.queue_history()``,
``sim.resource_history()``, ``sim.pool_history()``, ``sim.store_history()``,
and ``sim.pq_history()`` return a handle accepted by the ``timeseries_*``
getters: ``count``, ``min``, ``max``, and the duration-weighted ``mean``,
``std``, and ``median``, typically called from the collector:

.. code-block:: python

   @model.collect
   def stats(env: Clinic):
       h = sim.queue_history(env.waiting_room)
       env.median_queue = sim.timeseries_median(h)

Datasets and histories can also render text reports from inside compiled
code: ``sim.queue_report()``, ``sim.dataset_fivenum()``,
``sim.dataset_histogram()``, ``sim.timeseries_histogram()``,
``sim.dataset_correlogram()``, and their siblings print to stdout, and each
has a ``*_file()`` variant that appends to a path handle created with
``sim.log_text("waits.txt")``. These are diagnostic tools: with parallel
trials the stdout variants interleave, so print from single-replication
runs, or write per-trial files.

The classic in-process layer
----------------------------

The objects in this section belong to the classic API
(:class:`cimba.Simulation` and Python callback processes) and to offline
analysis. They are not usable inside ``sim.Model`` process bodies, which
compile to machine code.

* :class:`cimba.DataSummary` -- running summary of equally weighted samples.
* :class:`cimba.WeightedSummary` -- samples with explicit weights.
* :class:`cimba.Dataset` -- keeps all values for medians and quantiles.
* :class:`cimba.TimeSeries` -- a recorded state history.

``cimba.reporting`` builds on them with ``summarize()``, ``five_number()``,
``histogram()``, ``correlogram()``, and matplotlib helpers such as
``plot_history()`` -- see :doc:`../api_reference/reporting`. In classic
models, recording is manual (``queue.start_recording()`` /
``queue.stop_recording()``), and replications run through
``cimba.run_experiment(trial, n=..., seed=...)`` with a trial function that
returns plain Python values.
