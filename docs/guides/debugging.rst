Debug a failed trial
====================

**Goal:** find out why trials failed and reproduce one in isolation.

See what failed
---------------

A failing trial never stops the experiment. It is recorded:

.. code-block:: python

   results = experiment.run()
   if results.failed.any():
       print(results.failed.sum(), "trials failed")
       print(set(results.failure_reasons[results.failed]))

.. code-block:: text

   3 trials failed
   {'bad.gaps exhausted after 3 values'}

Reasons name the input or operation involved, using the canonical path of
the model instance. Outputs of failed trials are ``nan`` and are skipped by
:func:`~cimba.analysis.summary` and :func:`~cimba.analysis.compare`.

If any failure should stop you immediately, ask for an exception:

.. code-block:: python

   experiment.run(on_failure="raise")
   # cimba.experiments.run.TrialsFailed: 3 trials failed: bad.gaps exhausted after 3 values

Reproduce one trial
-------------------

Every trial is a pure function of its seed and configuration, so you can rerun
exactly one:

.. code-block:: python

   import numpy as np

   point, replication = np.argwhere(results.failed)[0]
   index = point * experiment.replications + replication
   again = experiment.only(trials=[index])          # workers=1 by default
   again.failure_reasons                            # same reason, same trial

Watch it with logging
---------------------

Add ``cb.log`` calls to the processes involved (they cost nothing while
disabled), enable them, and rerun the single trial:

.. code-block:: python

   TRACE = 1

   # in the model:
   #     cb.log(TRACE, "next gap", gap)

   cb.set_engine_log_level(TRACE)
   try:
       experiment.only(trials=[index])
   finally:
       cb.set_engine_log_level(0)

Each line shows the trial index, simulated time, process name, your message
and value.

Typical causes
--------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Symptom
     - Likely cause and fix
   * - ``… exhausted after N values``
     - A trace shorter than the run needs. Use ``on_exhausted="wrap"``,
       a longer recording, a shorter window, or a resampling source.
   * - ``… extension limit reached`` / ``source maximum length reached``
     - A resample was extended six times (64× its first length) or reached
       its hard limit (``fitted.wild``). Look for a loop that reads an input
       far more often than intended.
   * - ``ModelDefinitionError`` when creating the experiment
     - A missing parameter, source, initial state or reference, a reference
       to a model outside the tree, or a model used twice. The message names
       the path.
   * - ``ModelCompileError`` from ``run()``
     - Unsupported code in a compiled method. See
       :doc:`../concepts/compiled_code`.
   * - The run never finishes
     - A spawned model's process loops forever (the window doesn't stop
       spawned processes), or ``Window.until_idle()`` on a model that always
       has future events.
   * - Outputs are all ``nan``
     - The ``on_end`` hook didn't set them, or every trial failed.

If the Python process itself crashes (a segmentation fault), run with
``python -X faulthandler`` to see where, and please report it with the model.
