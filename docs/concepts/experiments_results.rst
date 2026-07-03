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
state and write ``sim.Output`` fields. Do broader Python analysis after
``exp.run()`` returns.

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
For the full public surface map, see :doc:`../api/cimba`.
