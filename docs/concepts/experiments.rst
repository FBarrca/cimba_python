Experiments
===========

An :class:`~cimba.Experiment` turns a configured model into a set of
independent **trials** and runs them in parallel.

.. code-block:: python

   experiment = cb.Experiment(
       model,                       # the configured root model
       replications=100,            # trials per design point
       window=cb.Window(warmup=30.0, duration=365.0),
       seed=7,                      # the whole experiment is reproducible from this
       seeding="common",            # or "independent"
   )
   results = experiment.run(workers=None)   # None = every core

Creating an ``Experiment`` takes a **snapshot** of the model tree and its
configuration: instances, values, sources, sweeps and captures. It validates
everything (missing parameters or sources, dangling references, series too
short for the window) and reports problems with the instance path. Nothing is
compiled until ``run()``.

Trials, replications and design points
--------------------------------------

.. code-block:: text

                    replication 0   replication 1   …   replication R-1
   design point 0   trial           trial               trial
   design point 1   trial           trial               trial
   …
   design point P-1 trial           trial               trial

* A **design point** is one combination of sweep values. With no sweeps there
  is exactly one.
* A **replication** is one independent repetition with its own seed.
* A **trial** is one cell: a complete, independent simulation run. Trials
  share nothing mutable, so they can run on any core in any order.

Everything in the results keeps this ``(points, replications)`` shape.

Sweeps
------

``cb.sweep(*values)`` creates a **design axis**. You can assign a sweep to:

* a ``Param`` field: ``model.base_stock = cb.sweep(250.0, 300.0, 350.0)``;
* a distribution parameter:
  ``inputs.dist.exponential(mean=cb.sweep(1.0, 2.0))``;
* an ``Input`` or ``Series`` field, as a sweep of **sources**:
  ``model.demand = cb.sweep(trace_source, bootstrap_source)``.

Assigning a sweep anywhere else (``State``, constants, entity capacities) is
a ``TypeError`` at that line. Those define the model's *structure*, which is
fixed for an experiment. To compare structural variants, build one model per
variant and run each with the same seed (see :doc:`../guides/comparing`).

**Crossing.** Independent sweeps combine as a full factorial. Two sweeps of 3
and 4 values give 12 design points. Points are ordered like nested loops, the
first sweep found in the model tree being the outermost.

**Deriving.** ``axis.map(f)`` makes a new sweep on the *same* axis, with
``f`` applied to each value. It doesn't add design points:

.. code-block:: python

   rho = cb.sweep(0.5, 0.7, 0.9)
   model.interarrival = inputs.dist.exponential(mean=rho.map(lambda r: 1 / r))
   model.label_rho = rho            # a Param that records the level

**Linking.** ``cb.sweeps(*columns)`` returns several sweeps that share one
axis and move together, like columns of a table:

.. code-block:: python

   stock, reorder = cb.sweeps([2500.0, 3000.0, 3500.0],
                              [ 800.0, 1000.0, 1200.0])
   dc.base_stock = stock            # 3 design points, not 9
   dc.reorder_point = reorder

**Reading the levels.** ``results.levels(sweep)`` returns the sweep's value at
every design point, in order, for any axis you created. You never need to
work out the point order by hand.

The measurement window
----------------------

:class:`~cimba.Window` sets how long each trial runs and when it measures:

.. raw:: html
   :file: ../static/diagrams/timeline.svg.html

.. code-block:: python

   cb.Window(warmup=100.0, duration=1_000.0, cooldown=50.0)
   cb.Window.until_idle()           # same as cb.Window(), the default

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Phase
     - What happens
   * - warmup
     - Processes run, but entity statistics aren't recorded. Use it to get
       past the empty-and-idle start of a steady-state system.
   * - duration
     - The measurement window. Time-weighted statistics are recorded;
       datasets are cleared at its start.
   * - cooldown
     - Processes keep running and datasets keep recording, but time-weighted
       statistics are frozen. Use it to let in-flight work finish.
   * - after
     - The static model's processes are stopped, remaining events drain,
       and the ``on_end`` hooks run.

``until_idle()`` runs until the event queue is empty, recording from time
0. It suits finite workloads (a batch of jobs, one day's orders). It can't
have a warmup or cooldown. A model that keeps scheduling events forever
never goes idle, so use a finite duration for those.

Seeds and common random numbers
-------------------------------

Every trial gets a 64-bit seed. Everything random in the trial is derived
from it: each input's own stream, the default ``cimba.random`` stream and
the host-generated rows for resampled inputs.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - ``seeding=``
     - Trial seeds
   * - ``"common"`` (default)
     - Derived from ``(seed, replication)``. Replication *r* has the **same**
       seed at every design point.
   * - ``"independent"``
     - Derived from ``(seed, replication, point)``. All trials differ.
   * - ``seeds=array``
     - Explicit seeds, shaped ``(replications,)`` or
       ``(points, replications)``.

**Why common is the default.** With common random numbers (CRN), the 50th
customer of replication 3 has the same arrival and service draws at every
design point. A difference between points is then caused by the change
between points, not by different luck. Paired statistics
(:func:`cimba.analysis.compare`) exploit this and typically give much
narrower confidence intervals for differences than independent runs would.
Because every input has its own stream, CRN keeps holding even when points
consume different amounts of an input.

Use ``"independent"`` when you need design points to be statistically
independent, for example to fit a metamodel with standard regression
assumptions.

Running
-------

.. code-block:: python

   results = experiment.run(
       workers=None,           # None: all cores; an int limits worker threads
       on_failure="record",    # or "raise": raise TrialsFailed if any trial fails
       input_memory=1 << 30,   # bytes of host-generated input rows per chunk
   )

What happens inside ``run()``:

1. Each model class is compiled with Numba, **once per Python process**. A
   later run of the same class (with any structure, values or sources) reuses
   the compiled code. ``cb.cache_info()`` and ``cb.clear_cache()`` inspect
   and reset that cache.
2. Sweeps are expanded into design points and trial seeds are derived.
3. Rows for resampled and fitted inputs are generated per replication,
   in chunks that fit in ``input_memory``.
4. One memory block per trial is laid out and filled: parameters, initial
   state, references, input slots.
5. The native runner executes all trials on worker threads, with the GIL
   released.
6. Trials that exhausted an extendable input are rerun with longer rows.
7. Outputs, input records and captures are copied into an immutable
   :class:`~cimba.Results`.

Results are identical for any ``workers`` and ``input_memory``.

Reproducing a single trial
--------------------------

``experiment.only(trials=[i, ...])`` reruns selected trials with their
original seeds and configuration. Trial indices are flattened:
``i = point * replications + replication``. That's the tool for debugging a
trial that failed or produced an odd value, especially combined with
``workers=1`` and logging.

.. code-block:: python

   bad = np.argwhere(results.failed)[0]                  # (point, replication)
   i = bad[0] * experiment.replications + bad[1]
   again = experiment.only(trials=[i])

Describing an experiment
------------------------

``experiment.describe()`` lists every field of every instance with its kind
and, for inputs, the bound source's description, sweeps included. It's a
quick check that you've bound what you think you've bound.
