Datasets, Summaries and Reporting
=================================

``sim.Dataset`` is a trial-local list of untimed numeric observations. Add one
observation with ``env.waits.add(value)`` each time the measured event happens.
At collection time, call methods such as ``env.waits.mean()`` or
``env.waits.quantile(0.95)`` to turn that list into the metric you want to
report for the current replication. Time-weighted histories are attached to
simulation entities and read back through the ``*_history()`` accessors as
time-series handles.

Datasets
--------

Declare datasets as model or component fields:

.. code-block:: python

   class Clinic(sim.Model):
       waits: sim.Dataset
       avg_wait: sim.Output
       p95_wait: sim.Output

Inside process code, each ``add()`` appends one sample to the current trial's
dataset:

.. code-block:: python

   env.waits.add(wait_time)

At collection time, summarize that trial-local list into the outputs you care
about:

.. code-block:: python

   @model.collect
   def collect_stats(env: Clinic):
       env.avg_wait = env.waits.mean()
       env.p95_wait = env.waits.quantile(0.95)

A dataset field is created separately for every trial row. If an experiment
uses ``replications=50``, then ``env.waits`` is 50 independent datasets, not
one dataset shared by all replications. Each replication records its own
samples. The model-level ``@model.collect`` callback, or a component-level
``@sim.collect`` callback, writes one output value per replication for each
metric you choose. For example, ``exp["avg_wait"]`` contains 50
per-replication averages, while ``exp["p95_wait"]`` contains 50
per-replication percentile estimates. Use ``exp.summary()`` or ordinary Python
after ``exp.run()`` to summarize those output values across replications.

Datasets are reset when the measurement window opens after warmup. Values
added before warmup are discarded, so collectors see the samples for the
measured part of that trial.

Datasets support method-style compiled calls: ``add()``, ``count()``,
``mean()``, ``min()``, ``max()``, ``std()``, ``median()``, ``quantile()``,
``print()``, ``print_file()``, ``fivenum()``, ``fivenum_file()``,
``histogram()``, ``histogram_file()``, ``correlogram()``,
``correlogram_file()``, ``pacf_correlogram()``, and
``pacf_correlogram_file()``.

Time series
-----------

Entity histories come from ``queue_history()``, ``resource_history()``,
``pool_history()``, ``store_history()``, and ``pq_history()``. The returned
handles are summarized and reported with:

``timeseries_count()``, ``timeseries_min()``, ``timeseries_max()``,
``timeseries_mean()``, ``timeseries_std()``, ``timeseries_median()``,
``timeseries_print()``, ``timeseries_print_file()``, ``timeseries_fivenum()``,
``timeseries_fivenum_file()``, ``timeseries_histogram()``,
``timeseries_histogram_file()``, ``timeseries_correlogram()``,
``timeseries_correlogram_file()``, ``timeseries_pacf_correlogram()``,
``timeseries_pacf_correlogram_file()``.

For text reports and text-mode plots, the no-suffix helpers print to stdout for
console and notebook use; the ``*_file()`` variants write to a path handle
created with ``sim.log_text()``. A dataset report is produced from the current
trial's dataset. In multi-replication experiments, writing every replication to
the same path can interleave or overwrite raw samples depending on the append
flag and parallel execution. These helpers are most useful in single-trial
debugging; scalar ``sim.Output`` fields are usually better for large parallel
experiments.

``Model.experiment(..., warmup=..., duration=...)`` controls the measurement
window: warmup lets the model reach a representative state before summaries are
collected.
