Experiments and Results
=======================

A model declaration becomes data through an experiment. ``model.experiment()``
builds a trial table, ``exp.run()`` executes those trials, and outputs are read
back as arrays.

.. code-block:: python

   exp = model.experiment(
       arrival_rate=[0.7, 0.9],
       mean_service=1.0,
       replications=20,
       warmup=1000.0,
       duration=10000.0,
       seed=42,
   )

This creates every parameter combination and repeats each combination for the
requested number of replications. Each trial receives its own seed and its own
simulation entities.

Trial-local entities
--------------------

Every row in the experiment table is a separate trial: one parameter
combination, one replication, one ``env`` record. Entity fields declared on the
model, including ``sim.Dataset``, are created fresh for that trial. They do not
accumulate samples across other replications.

Think of a dataset as a trial-local list of numeric observations. Process code
adds observations to that list with ``sim.tally()``. At the end of the same
trial, a collector chooses the metrics you want to report from that list and
writes them to scalar ``sim.Output`` fields. Those metrics can be counts,
means, minima, maxima, standard deviations, medians, quantiles, or whatever
summary is relevant to the model.

That gives datasets and outputs two distinct levels of meaning:

* during a trial, ``sim.tally(env.waits, value)`` adds another observation to
  that trial's dataset;
* after the trial, ``@model.collect`` or a component ``@sim.collect`` usually
  reduces that dataset to scalar ``sim.Output`` fields, such as
  ``sim.dataset_mean(env.waits)`` or ``sim.dataset_count(env.waits)``;
* after ``exp.run()``, ``exp["avg_wait"]`` is an array of those scalar outputs,
  one value per trial row.

For example, with ``replications=20``, a ``waits: sim.Dataset`` field is not
one dataset containing all waits from all 20 replications. It is 20 separate
trial-local datasets. If the collector writes ``env.avg_wait`` from
``sim.dataset_mean(env.waits)``, then ``exp["avg_wait"]`` contains 20 means:
one mean computed from the waits observed in each replication.

The measurement window
----------------------

``warmup`` lets the model run before summary statistics are recorded.
``duration`` is the measured part of the trial. Together they are the usual
way to stop steady-state models with infinite process loops.

The generated trial starts the registered processes, opens the recording window
after warmup, closes it after duration, runs the collector, and stops remaining
processes.

Collectors
----------

Use ``@model.collect`` to write outputs after the measured run:

.. code-block:: python

   @model.collect
   def collect_stats(env: Clinic):
       env.completed = float(env.served)
       env.avg_waiting = sim.mean_level(env.waiting_room)

The collector is still part of the compiled model. It should read trial-local
state and write ``sim.Output`` fields. Distribution statistics of tallied
datasets are available here too -- ``sim.dataset_median(env.waits)`` and
``sim.dataset_quantile(env.waits, 0.95)`` reduce a per-trial distribution to
an output without leaving the trial. Those reductions are per replication; the
cross-replication statistics happen later from the output arrays or from
``exp.summary()``. Do broader Python analysis after ``exp.run()`` returns.

Running and reading results
---------------------------

``exp.run()`` runs the trial table in place and returns the number of failed
trials:

.. code-block:: python

   failures = exp.run()
   if failures:
       raise RuntimeError(f"{failures} trial(s) failed")

   average_waiting = exp["avg_waiting"]
   completions = exp["completed"]
   arrival_rates = exp["arrival_rate"]

The arrays are aligned by trial row. That means ``average_waiting[i]`` belongs
to the same trial as ``arrival_rates[i]`` and ``completions[i]``.

Summarizing across replications
-------------------------------

``exp.summary()`` reduces the trial table to one record per design point,
with the swept parameter values and, for each output, the mean over its
replications plus a Student-t confidence-interval half-width:

.. code-block:: python

   for row in exp.summary("avg_waiting"):
       print(f"arrival_rate={row['arrival_rate']:.2f}: "
             f"{row['avg_waiting']:.2f} +- {row['avg_waiting_hw']:.2f}")

Positional arguments select outputs (default: all of them) and
``confidence=`` sets the interval (default 0.95). Failed trials are excluded
output by output, and the half-width is NaN when fewer than two replications
survive. For anything beyond means and intervals, fall back to the aligned
per-trial arrays above.

What belongs outside the model
------------------------------

Use ordinary Python around the experiment for tasks that are not simulation
behavior:

* preparing parameter grids,
* calling ``model.experiment(...)``,
* checking the failure count,
* grouping or summarizing output arrays,
* plotting and exporting results.

Keep the model focused on the simulated world. Keep the analysis code focused
on the experiment table that comes back from that world.

For a complete worked build, continue with :ref:`the tutorial <tutorial>`.
For the full API reference, see :doc:`../api_reference/index`.
