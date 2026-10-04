Optimization API
================

``cb.decision`` and ``cb.Optimization`` are in ``cimba``. Settings and result
types are in ``cimba.optimize``. For explanations see
:doc:`../concepts/optimization`; for recipes, :doc:`../guides/optimizing`.

.. currentmodule:: cimba

Decisions
---------

.. function:: decision(low, high, *, step=None, log=False)

   Return a :class:`Decision`: the inclusive range a ``Param`` may take in an
   optimization. ``low < high``, both finite. ``step`` must be positive and
   divide ``high - low`` exactly. ``log=True`` needs ``low > 0`` and can't be
   combined with ``step``. Invalid arguments raise ``ValueError``.

The annotation of the field receiving the decision sets the domain:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Values
   * - ``Param[float]``
     - Reals in ``[low, high]``; with ``step``, only ``low + k * step``; with
       ``log=True``, searched on a log scale.
   * - ``Param[int]``
     - Integers in ``[low, high]``. Bounds (and ``step``) must be integral and
       within ±2\ :sup:`53`. ``log`` is not supported.
   * - ``Param[bool]``
     - ``cb.decision(False, True)``, without ``step`` or ``log``.

Decisions can only be assigned to ``Param`` fields of models in the tree
(otherwise ``TypeError``). A plain ``Experiment`` raises
``ModelDefinitionError`` if any decision remains. Domain errors that depend
on the field type are raised, with the field's path, when the study is
created.

.. class:: Decision

   One search dimension, identified by the object. Fields assigned the same
   object always take the same value and must have the same type.

   .. method:: map(function)

      A decision on the same dimension whose value is ``function(value)``.
      Mapped decisions can be mapped again and are valid keys for
      ``Optimum``; the result is validated against the receiving field.

A **candidate** is a mapping from each original (unmapped) decision to a
value in its domain, on its step lattice. ``initial``, ``evaluate`` and
``experiment`` take candidates.

Optimization
------------

.. class:: Optimization(model, *, minimize=None, maximize=None, replications=32, window=None, seed=0)

   A search over the decisions in ``model``. Creating it snapshots and
   validates the configuration; later changes to the model don't affect it.
   The model must contain at least one decision and no sweeps.

   :param minimize: Objective to minimize. Give exactly one of ``minimize``
      and ``maximize``.
   :param maximize: Objective to maximize. Reported values keep its sign.
   :param replications: Search trials per candidate (≥ 1).
   :param window: The measurement :class:`Window`; default
      ``Window.until_idle()``.
   :param seed: Base seed (≥ 0) for every random stream of the study.

   The objective is called once per batch with its ``Results`` and must
   return real numbers that broadcast to ``(points, replications)``. Scalars
   and ``(points,)`` arrays are rejected with ``OptimizationError``. Values of
   failed trials are ignored; a non-finite value of a successful trial marks
   that trial failed (``"objective is not finite"``).

   .. method:: run(*, evaluations=None, optimizer=None, initial=None, finalists=5, validation=None, workers=None, input_memory=1 << 30, on_failure="reject", progress=None)

      Search, select a finalist on fresh seeds, estimate it on independent
      seeds, and return an :class:`~cimba.optimize.Optimum`.

      :param evaluations: Maximum distinct candidates simulated in the search.
         Must cover the initial population. Cached candidates are free. The
         search stops before a generation that might exceed it.
      :param optimizer: A :class:`~cimba.optimize.DifferentialEvolution`;
         ``None`` uses the defaults.
      :param initial: Candidates that replace the first members of the
         initial population (at most the population size).
      :param finalists: Maximum candidates re-run for selection (≥ 1).
      :param validation: Replications for selection and for the estimate
         (≥ 2). Default ``max(100, 4 * replications)``.
      :param workers: Worker threads; ``None`` uses every logical CPU.
      :param input_memory: Byte limit for generated input rows, as in
         :meth:`Experiment.run`.
      :param on_failure: ``"reject"``, ``"ignore"`` or ``"raise"`` (below).
      :param progress: Called with a :class:`~cimba.optimize.Progress` after
         each evolved generation; a truthy return stops the search.

      Repeated runs reuse the study's cache of simulated candidates.

   .. method:: evaluate(candidates, *, workers=None, input_memory=1 << 30, on_failure="reject")

      Evaluate candidates on the search seeds as one batch, reusing the
      cache. Returns a tuple of :class:`~cimba.optimize.Evaluation`, in order.

   .. method:: experiment(values)

      A plain :class:`Experiment` at one candidate, with the study's saved
      configuration, seed, replications and window. The model is left
      unchanged. Its trials are the search trials of that candidate.

   .. method:: describe()

      One dict per field of the saved model: ``instance``, ``field``,
      ``kind``, ``domain``, ``source`` and ``objective``.

.. list-table:: Failure policies (``on_failure``)
   :header-rows: 1
   :widths: 18 82

   * - Policy
     - Effect on a candidate with failed trials
   * - ``"reject"``
     - Excluded from the search and from the finalists.
   * - ``"ignore"``
     - Scored on its successful trials; rejected if none succeeded.
   * - ``"raise"``
     - ``TrialsFailed``, naming the candidate and the first reason.

The policy applies to search, selection and estimation. ``OptimizationError``
is raised if the whole initial population or every finalist is rejected, or
if the chosen candidate fails its estimation trials.

Differential evolution
----------------------

.. module:: cimba.optimize

.. class:: DifferentialEvolution(*, popsize=15, mutation=(0.5, 1.0), recombination=0.7, strategy="best1bin", tol=0.01, atol=0.0, maxiter=1000, init="latinhypercube")

   Settings for SciPy's ``differential_evolution``, with SciPy's meanings.

   :param popsize: Population per decision (≥ 1), or ``"fill"``. With *D*
      decisions the population is ``max(5, D * popsize)`` (at least 6 for
      ``rand2`` strategies; rounded up to a power of two for ``sobol``).
   :param mutation: In ``[0, 2)``, or an increasing pair for dithering.
   :param recombination: Crossover probability in ``[0, 1]``.
   :param strategy: ``best1``, ``best2``, ``rand1``, ``rand2``,
      ``randtobest1`` or ``currenttobest1``, followed by ``bin`` or ``exp``.
   :param tol: Relative convergence tolerance (≥ 0).
   :param atol: Absolute convergence tolerance (≥ 0). The search has
      converged when the population's energies have a standard deviation of
      at most ``atol + tol * abs(mean)``.
   :param maxiter: Maximum evolved generations (≥ 0).
   :param init: ``"latinhypercube"``, ``"random"``, ``"sobol"`` or ``"halton"``.

   ``popsize="fill"`` resolves once per run to
   ``max(15, ceil(256 * W / (D * R)))`` for *W* workers and *R*
   replications, recorded in ``meta.optimizer.popsize``.

   Cimba supplies the bounds, integrality, random generator, callback and a
   vectorized objective, and always uses ``updating="deferred"``,
   ``workers=1`` and ``polish=False``: each generation is evaluated as one
   native batch.

Results
-------

.. class:: Optimum

   The result of :meth:`~cimba.Optimization.run`. ``best[decision]`` returns
   a chosen value; mapped decisions are valid keys.

   .. attribute:: values

      Read-only mapping from decisions to chosen values.

   .. attribute:: estimate

      :class:`Estimate` of the chosen candidate from the estimation trials.

   .. attribute:: finalists

      Tuple of :class:`Finalist`, ordered by search mean.

   .. attribute:: results

      Full ``Results`` of the estimation run, including input records and
      captured entities.

   .. attribute:: history

      Every search proposal, by generation, including cache hits.

   .. attribute:: batches

      One row per native runner call.

   .. attribute:: meta

      Read-only run metadata (below), by key or attribute.

   .. method:: apply()

      Replace every decision in the model with its chosen value.

.. class:: Estimate

   ``n`` successful trials, ``mean``, ``std`` (``ddof=1``) and the 95%
   Student-*t* interval ``lower``, ``upper`` (NaN with fewer than two trials).

.. class:: Evaluation

   One candidate's ``values``, read-only per-trial ``samples`` and ``failed``
   arrays, failure ``reasons`` and ``estimate``.

.. class:: Finalist

   ``values``, ``search_mean``, ``selection`` (an :class:`Estimate`) and
   ``selection_mean``. ``difference`` is an ``analysis.Comparison`` of this
   finalist minus the chosen one on the selection trials, at confidence
   ``1 - 0.05 / max(1, F - 1)`` for *F* finalists. ``rejected`` reports a
   failure under the policy; ``indistinguishable`` is true when the interval
   contains zero.

.. class:: Progress

   ``generation``, simulated ``evaluations`` so far, and ``best``, the best
   accepted :class:`Evaluation`.

``history`` and ``batches`` are sequences of rows with ``to_table()``, which
returns a list of dicts ready for ``pandas.DataFrame``.

.. list-table:: Row fields
   :header-rows: 1
   :widths: 15 85

   * - Table
     - Fields
   * - ``history``
     - ``generation``, ``values``, ``mean``, ``std``, ``n``, ``failed`` (count),
       ``reasons``, ``cached``.
   * - ``batches``
     - ``phase`` (``search``, ``selection``, ``estimation``), ``generation``,
       ``proposed``, ``cached``, ``simulated``, ``candidates``, ``trials``,
       ``native_wall_time``, ``host_wall_time``, ``busy_cpus`` (native CPU
       time ÷ native wall time), ``native_calls``, ``compile_misses``,
       ``compile_wall_time``.

Memory chunking or input extension can split one batch into several native
calls; the extra rows carry only native work. A generation made only of
cache hits has history rows but no batch row.

``meta`` holds ``evaluations``, ``generations``, ``trials`` (per phase),
``wall_time``, ``stop_reason`` (``converged``, ``budget``, ``maxiter`` or
``progress``), ``compile``, ``optimizer`` (resolved settings),
``population_size``, ``numpy_version``, ``scipy_version``, ``workers``,
``on_failure``, ``replications``, ``validation``, ``seed``,
``selection_seed`` and ``estimation_seed``.

Seeds
-----

Search replication *r* uses ``trial_seed(seed, r)``, the seed of a common
seeded ``Experiment``. Selection and estimation use
``trial_seed(selection_seed, r)`` and ``trial_seed(estimation_seed, r)``.
Those base seeds, and DE's own generator, are drawn from
``inputs.trace_rng(seed, tag)`` with the tags ``"cimba.optimize.selection"``,
``"cimba.optimize.estimation"`` and ``"cimba.optimize.de"``. NumPy's global
random state is never used.

Errors
------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Exception
     - Raised when
   * - ``ValueError``
     - invalid decision bounds, DE settings or candidate mappings.
   * - ``TypeError``
     - a decision is assigned to a field that isn't a ``Param``.
   * - ``ModelDefinitionError``
     - the model has no decisions, contains sweeps, or a decision doesn't fit
       its field (message names the field).
   * - ``ExperimentConfigError``
     - invalid study or run arguments.
   * - ``OptimizationError``
     - the objective returns the wrong shape or type, or every candidate is
       rejected (see failure policies).
   * - ``TrialsFailed``
     - a trial fails and ``on_failure="raise"``.
