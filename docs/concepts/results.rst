Results and analysis
====================

``run()`` returns an immutable :class:`~cimba.Results`. You read it through the
**same model objects** you configured, so there are no field-name strings, no
flattened paths and no index bookkeeping.

.. code-block:: python

   results = experiment.run()

   results[model].mean_queue          # an Output          -> Samples
   results[model].interarrival        # an Input / Series  -> InputRecord
   results[model].queue               # a captured entity  -> Signal
   results[model.station].utilization # any model in the tree works

``results[obj]`` accepts any static model instance in the tree.

Samples: one output across the design
-------------------------------------

A :class:`~cimba.Samples` wraps a read-only NumPy array shaped
``(design points, replications)``:

.. code-block:: python

   samples = results[model].mean_queue
   samples.shape              # (3, 100)
   samples[1, 7]              # design point 1, replication 7
   samples.values             # the underlying ndarray (read-only)
   np.asarray(samples)        # works anywhere NumPy does

Values are ``float64``, even for ``int`` and ``bool`` outputs, so that failed
trials can be masked as ``nan``.

InputRecord: what each input did
--------------------------------

.. list-table::
   :widths: 28 72

   * - ``source``
     - Tuple with one ``describe()`` dict per design point: method,
       parameters, data fingerprints, tag and policy.
   * - ``consumed``
     - ``(points, replications)`` integer array: values read (or the highest
       bucket read + 1 for a ``Series``) in each trial.
   * - ``extended``
     - ``(points, replications)``: how many times each trial's row was
       regenerated longer.
   * - ``rows(point, replication)``
     - Regenerates the exact values the trial consumed.

Signal: captured histories
--------------------------

For an entity marked with ``capture()`` before the run,
``results[model].entity`` is a :class:`~cimba.Signal`.
``signal.trial(point, replication)`` returns read-only ``(times, values)``
arrays (see :doc:`entities` for what each entity records).

Trial status and failures
-------------------------

A trial that hits an engine error (for example an input that runs out under
``"fail"``, or an invalid operation in model code) is **abandoned**. The
engine cleans up everything it allocated, records a reason, and moves on. One
bad trial never stops the experiment.

.. list-table::
   :widths: 28 72

   * - ``results.failed``
     - ``(points, replications)`` bool array.
   * - ``results.failure_reasons``
     - ``(points, replications)`` string array; empty for successful trials.
   * - ``results.seeds``
     - ``(points, replications)`` seeds actually used.

Outputs of failed trials are ``nan``, and the analysis functions ignore
``nan``, so summaries use the successful trials. Always glance at
``results.failed.any()``. Pass ``run(on_failure="raise")`` to turn any
failure into a ``TrialsFailed`` exception instead.

Run metadata
------------

``results.meta`` is a read-only mapping (also readable as attributes):

.. list-table::
   :widths: 28 72

   * - ``meta.wall_time``
     - Seconds for the whole run.
   * - ``meta.compile.misses``, ``meta.compile.wall_time``
     - Classes compiled during this run and the time spent.
   * - ``meta.extensions``
     - Total input extensions (reruns).
   * - ``meta.workers``
     - The ``workers`` argument.
   * - ``meta.chunks``
     - Present when replications were processed in several memory chunks.

Tables
------

``results.to_table()`` returns one dict per trial, with ``point``,
``replication``, ``seed``, ``failed``, ``reason`` and one column per output
labelled ``ClassName.field``. It's ready for ``pandas.DataFrame(...)``.

Statistics: cimba.analysis
--------------------------

.. code-block:: python

   from cimba import analysis

   s = analysis.summary(results[model].mean_queue, confidence=0.95)
   s.n, s.mean, s.std, s.lower, s.upper    # one entry per design point

   c = analysis.compare(results[model].mean_queue, a=0, b=2)
   c.difference, c.lower, c.upper          # paired: point b minus point a

:func:`~cimba.analysis.summary`
   Per design point: count of finite samples, mean, standard deviation, and
   a Student-*t* confidence interval for the mean.

:func:`~cimba.analysis.compare`
   A **paired** comparison of two design points: replication *r* of ``a``
   against replication *r* of ``b``. With common random numbers this is the
   right test, and usually far tighter than comparing two independent
   summaries. It accepts any ``(points, replications)`` array, so you can
   stack samples from separate experiments that shared a seed.

:func:`~cimba.analysis.check_input`
   Compares a source against reference data *before* a run: mean, variance,
   quantiles, autocorrelation, partial autocorrelation and a two-sample
   Kolmogorov–Smirnov test. For a joint bootstrap it also compares
   cross-correlation matrices. See :doc:`../guides/input_models`.
