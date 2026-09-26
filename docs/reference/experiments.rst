Experiments and results
=======================

.. currentmodule:: cimba

Sweeps
------

.. function:: sweep(*values)

   Create an independent design axis. Assign it to a ``Param`` field, a
   distribution parameter, or an ``Input``/``Series`` field (as a sweep of
   sources). Independent sweeps cross.

   The returned object has ``values`` and one method:

   .. method:: map(function)
      :no-index:

      A new sweep on the **same** axis with ``function`` applied to each
      value.

.. function:: sweeps(*columns)

   Create linked sweeps that share one axis, one per column. All columns
   must have the same, nonzero length. Returns a tuple of sweeps.

Experiment
----------

.. class:: Experiment(model, *, replications=1, window=None, seed=0, seeding="common", seeds=None)

   Snapshot and validate a configured model for running.

   :param model: The root :class:`Model` instance.
   :param replications: Trials per design point (≥ 1).
   :param window: A :class:`Window`; default :meth:`Window.until_idle`.
   :param seed: Base seed of the experiment.
   :param seeding: ``"common"`` (same seed per replication across design
      points) or ``"independent"``.
   :param seeds: Explicit seeds, shaped ``(replications,)`` or
      ``(points, replications)``; overrides ``seed``/``seeding``.
   :raises cimba.schema.ModelDefinitionError: for missing or invalid
      configuration (the message names the instance path).

   .. method:: run(*, workers=None, on_failure="record", input_memory=1 << 30)

      Compile (if needed) and run every trial; return :class:`Results`.

      :param workers: Worker threads; ``None`` uses all cores.
      :param on_failure: ``"record"``, or ``"raise"`` to raise
         ``TrialsFailed`` if any trial fails.
      :param input_memory: Byte budget for host-generated input rows;
         replications are chunked to fit.

   .. method:: only(*, trials, workers=1, on_failure="record", input_memory=1 << 30)

      Rerun selected trials with their original seeds. ``trials`` holds
      flattened indices ``point * replications + replication``. Returns a
      :class:`Results` with one design point per selected trial.

   .. method:: describe()

      A list of ``{"instance", "field", "kind", "source"}`` rows.

   .. attribute:: replications
   .. attribute:: window
   .. attribute:: seed
   .. attribute:: seeding

.. class:: Window(warmup=0.0, duration=inf, cooldown=0.0)

   The measurement window of every trial. ``duration`` must be positive.
   An infinite ``duration`` means run until idle, with no warmup or cooldown.

   .. classmethod:: until_idle()

      ``Window()``: run until no events remain, recording from time 0.

Results
-------

.. class:: Results

   Immutable outcome of :meth:`Experiment.run`.

   .. method:: __getitem__(model)

      Results for a static model instance, with one attribute per field:
      :class:`Samples` for outputs, an ``InputRecord`` for inputs and series,
      a :class:`Signal` for captured entities.

   .. attribute:: failed

      ``(points, replications)`` bool array.

   .. attribute:: failure_reasons

      ``(points, replications)`` string array.

   .. attribute:: seeds

      ``(points, replications)`` uint64 array of trial seeds.

   .. attribute:: meta

      Read-only mapping: ``wall_time``, ``workers``,
      ``compile`` (``misses``, ``wall_time``), ``extensions``,
      ``native_failed`` and, when chunked, ``chunks``.

   .. method:: levels(sweep)

      The sweep's value at each design point, as a tuple.

   .. method:: to_table()

      One dict per trial: ``point``, ``replication``, ``seed``, ``failed``,
      ``reason`` and ``"Class.output"`` columns.

.. class:: Samples

   One output across the design.

   .. attribute:: values

      Read-only ``float64`` array ``(points, replications)``; failed trials
      are ``nan``.

   .. attribute:: shape

   Supports indexing (``samples[p, r]``) and ``numpy.asarray``.

.. class:: Signal

   A captured entity history.

   .. method:: trial(point, replication)

      Read-only ``(times, values)`` arrays for one trial.

.. py:class:: cimba.results.InputRecord

   An input's record in the results.

   .. attribute:: source

      Tuple of provenance dicts, one per design point.

   .. attribute:: consumed

      ``(points, replications)`` values consumed per trial.

   .. attribute:: extended

      ``(points, replications)`` extensions per trial.

   .. method:: rows(point, replication)

      Regenerate the exact values the trial consumed (read-only array).

Errors
------

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Exception
     - Raised when
   * - ``cimba.schema.ModelDefinitionError``
     - the configured model is incomplete or inconsistent (at
       ``Experiment(...)``).
   * - ``cimba.experiments.run.ExperimentConfigError``
     - run settings are invalid (replications, window, seeds, memory).
   * - ``cimba.compiler.ModelCompileError``
     - a model method can't be compiled; the message gives ``file:line``.
   * - ``cimba.experiments.run.TrialsFailed``
     - trials failed and ``on_failure="raise"``.
   * - ``cimba.inputs.InputError``
     - a source has invalid parameters or data.
   * - ``cimba.modeling.NotInCompiledCode``
     - a compiled-only call is made on the host.
   * - ``TypeError``
     - an assignment of the wrong kind (for example a sweep on a ``State``).
