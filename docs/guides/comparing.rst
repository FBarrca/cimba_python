Compare scenarios fairly
========================

**Goal:** decide whether design A beats design B, with a confidence interval
on the difference.

The recipe
----------

1. Put the alternatives in **one experiment** as a sweep, if you can.
2. Keep the default ``seeding="common"``.
3. Compare with :func:`cimba.analysis.compare`, which is a *paired*
   comparison.

.. code-block:: python

   dc.base_stock = cb.sweep(2500.0, 3000.0, 3500.0)
   results = cb.Experiment(network, replications=50,
                           window=cb.Window(duration=360.0), seed=1).run()

   service = results[dc].service_level
   print(results.levels(dc.base_stock))
   print(cb.analysis.summary(service).mean)
   print(cb.analysis.compare(service, a=0, b=2))   # point 2 minus point 0

``compare`` returns ``difference``, ``lower`` and ``upper`` (a Student-*t*
interval, 95% by default) plus ``n``, the number of pairs where both trials
succeeded. If the interval excludes zero, the difference is statistically
clear. Whether it's *large enough to matter* is a domain question.

Why pairing works
-----------------

Under common random numbers, replication *r* sees the same arrivals, the
same demand and the same failures at every design point, because each input
has its own stream keyed by its place in the model. Most of the run-to-run
noise is shared, so it cancels in the difference. Comparing two independent
summaries would carry all of that noise.

Comparing things you can't sweep
--------------------------------

Structure, such as entity capacities, the number of stations or a different
class, can't be swept. Build one model per variant and run each with the
**same seed**. As long as the inputs you want synchronized keep the same path
in both models, they get the same streams. Stack the samples and compare:

.. code-block:: python

   import numpy as np

   rows = []
   for tugs in (8, 10):
       harbor = Harbor(num_tugs=tugs)
       results = cb.Experiment(harbor, replications=20, window=window, seed=7).run()
       rows.append(results[harbor].avg_time_small.values[0])

   cb.analysis.compare(np.vstack(rows), a=0, b=1)

Comparing input models
----------------------

Sources can be swept like any value. To compare a fitted model with a
bootstrap:

.. code-block:: python

   shop.demand = cb.sweep(inputs.fitted.sieve(history, nonnegative=True),
                          inputs.bootstrap.stationary(history, mean_block=7))

When several inputs must switch together, for example every site's demand
switching from one model to another at once, use **linked** sweeps so you
get 2 design points instead of 2\ :sup:`n`:

.. code-block:: python

   axes = cb.sweeps(*[[joint[n], independent[n]] for n in sites])
   for n, axis in zip(sites, axes):
       network.facilities[n].demand = axis

:doc:`../tutorial/inventory` works through this example.

Many scenarios
--------------

``compare`` handles one pair at a time. For a grid, use ``summary`` for the
overview and ``compare`` against a baseline:

.. code-block:: python

   baseline = 0
   for point, level in enumerate(results.levels(dc.base_stock)):
       c = cb.analysis.compare(service, a=baseline, b=point)
       print(level, round(c.difference, 4), (round(c.lower, 4), round(c.upper, 4)))

When you compare many pairs, widen each interval (for example with a
Bonferroni correction, ``confidence=1 - 0.05 / k``) to keep the overall error
rate honest.
