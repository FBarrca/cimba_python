Capture and plot histories
==========================

**Goal:** look at the raw behavior of a queue, resource or dataset, not just
its summary.

Mark entities before the run
----------------------------

Capturing is configured on the entity **object**. If the field has no value
yet, create the entity first:

.. code-block:: python

   line = AssemblyLine()
   line.system = cb.Container()     # work in progress
   line.system.capture()
   line.cycle_time = cb.Dataset()   # one sample per finished part
   line.cycle_time.capture()

A model can also do this in its own ``__init__``, as the assembly-line
tutorial does.

Read them after the run
-----------------------

.. code-block:: python

   results = cb.Experiment(line, replications=10,
                           window=cb.Window(duration=10_000.0), seed=45).run()

   times, wip = results[line].system.trial(0, 3)     # point 0, replication 3
   _, cycle_times = results[line].cycle_time.trial(0, 3)

Both arrays are read-only NumPy arrays. For containers, resources and stores
you get a step function: the value after each change. For datasets you get
the samples and an empty time array.

Plot
----

.. code-block:: python

   import matplotlib.pyplot as plt       # pip install "cimba[plot]"

   fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
   left.step(times, wip, where="post")
   left.set(xlabel="minutes", ylabel="parts in system", title="Work in progress")
   right.hist(cycle_times, bins=30)
   right.set(xlabel="minutes", title="Cycle time")
   fig.tight_layout()

To overlay replications, loop over ``range(replications)``. To pool samples
across replications, concatenate them:

.. code-block:: python

   pooled = np.concatenate([results[line].cycle_time.trial(0, r)[1]
                            for r in range(10)])

Time series statistics
----------------------

Captured datasets are ordinary arrays, so the SciPy and statsmodels tools
apply directly: histograms, ``np.quantile``, ``statsmodels.tsa.stattools.acf``
for autocorrelation, and so on. For a time-weighted mean from a captured step
history:

.. code-block:: python

   window_end = 10_000.0                     # warmup + duration
   durations = np.diff(np.append(times, window_end))
   time_weighted_mean = np.sum(wip * durations) / durations.sum()

Memory
------

A capture keeps every change of every captured entity in every trial. A busy
queue over a long window can record millions of points per trial. Capture
the entities you're investigating, with few replications, and use in-trial
summaries (``mean_level()``, ``sample_mean()``) for production runs.
