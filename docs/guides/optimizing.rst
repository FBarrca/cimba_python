Find the best parameters
========================

**Goal:** search parameter values that minimize a cost (or maximize a
reward), and get an honest estimate of how good the answer is.

New to optimization in Cimba? Read :doc:`../tutorial/optimization` first.

The recipe
----------

.. code-block:: python

   import cimba as cb

   policy = MinMax()
   policy.minimum = cb.decision(0.0, 100.0, step=1.0)
   policy.maximum = cb.decision(40.0, 250.0, step=1.0)
   store = Store()
   store.policy = policy

   def cost(results):                    # one value per trial
       s = results[store]
       return s.mean_stock + 20 * s.orders_per_week + 400 * (1 - s.fill_rate)

   study = cb.Optimization(store, minimize=cost, replications=32,
                           window=cb.Window(warmup=30, duration=365), seed=11)
   best = study.run(evaluations=500)

   print(best[policy.minimum], best[policy.maximum], best.estimate)
   best.apply()                          # write the answer into the model

1. Assign ``cb.decision(low, high)`` to each ``Param`` that may change.
2. Write an objective that returns **per-trial** values from ``results``.
   Never average inside it.
3. Create the study with the run settings you would give an ``Experiment``.
4. ``run()``, read the answer through the decision objects, and ``apply()`` it.

Quote ``best.estimate``, not the search's best mean: the estimate comes from
fresh trials and isn't flattered by the search's luck.

Start from your current settings
--------------------------------

Put the configuration you use today into the first generation, so the search
is sure to consider it:

.. code-block:: python

   best = study.run(initial=[{policy.minimum: 40.0, policy.maximum: 120.0}])

Each candidate is a mapping from decision objects to values, and must give a
value for every decision.

Choose replications and a budget
--------------------------------

``replications`` is the number of trials per candidate during the search.
More replications make each comparison more reliable, but every candidate
costs more. Start with about 32. If ``best.finalists`` shows the leading
candidates are all tied, the answer is already as good as these trials can
tell. If the choice changes when you change ``seed``, add replications.

``evaluations`` caps how many distinct candidates are simulated. Without it,
the search runs until the population converges (or 1,000 generations). The
search never exceeds the cap; it stops early if the next generation would not
fit.

The check after the search costs ``(finalists + 1) × validation`` more
trials. By default ``finalists=5`` and ``validation`` is four times
``replications`` (at least 100).

To keep searching longer before convergence is declared, lower the
tolerance: ``optimizer=optimize.DifferentialEvolution(tol=1e-4)``.

Use every core
--------------

Each generation runs as one batch on all worker threads. A batch holds
``population × replications`` trials, where the population is ``popsize``
(default 15) per decision. Small batches can leave cores idle. Check with the
batch report:

.. code-block:: python

   for batch in best.batches:
       print(batch.phase, batch.trials, f"{batch.busy_cpus:.1f} busy CPUs")

``busy_cpus`` is how many logical CPUs were working, on average, while the
batch ran. If it is well below your core count, enlarge the population:

.. code-block:: python

   from cimba import optimize

   best = study.run(optimizer=optimize.DifferentialEvolution(popsize="fill"))

``"fill"`` sizes the population so each worker gets about 256 trials per
generation. A larger population simulates more candidates, but they run side
by side, so wall time grows much more slowly than the candidate count, and
the search covers more ground. The default is fixed so that a study gives
the same result on every machine. ``"fill"`` depends on the core count, so
the integer it picked is stored in ``best.meta.optimizer.popsize``; pass
that integer to repeat the run elsewhere.

When trials fail
----------------

A trial can fail, for example when a recorded input runs out. ``on_failure``
decides what that means for its candidate:

* ``"reject"`` (default): the candidate is out. This is the safe choice.
* ``"ignore"``: average the trials that succeeded. Be careful: a policy that
  fails exactly in the hardest scenarios then looks better than it is.
* ``"raise"``: stop with ``TrialsFailed`` naming the candidate.

A non-finite objective value, such as ``0/0``, counts as a failed trial.
``best.history`` lists every candidate with its failure count and reasons.

Check or replay a candidate
---------------------------

Score a configuration on the search's trials without searching:

.. code-block:: python

   values = {policy.minimum: 40.0, policy.maximum: 120.0}
   evaluation = study.evaluate([values])[0]
   print(evaluation.estimate, evaluation.reasons)

Turn a candidate into an ordinary experiment, to capture histories or debug
one trial:

.. code-block:: python

   experiment = study.experiment(values)       # the model itself is unchanged
   results = experiment.run()
   one = experiment.only(trials=[3])

The search trials are those of an ``Experiment`` with the study's seed, so
this replays them exactly. To replay the final estimate, use
``seed=best.meta.estimation_seed`` and ``replications=best.meta.validation``.

Compare policies or designs
---------------------------

An optimization tunes one configured model. To choose between policy
classes, tune each one, ``apply()`` the answers, then compare the tuned
models in an experiment with a new seed. :doc:`comparing` explains the
paired comparison, and :doc:`../tutorial/optimization` shows the whole
pattern.

What can't be a decision
------------------------

* Only scalar ``Param`` fields (``float``, ``int``, ``bool``) of models in the
  tree. Capacities and structure are fixed; to search a number of servers,
  use a ``Param[int]`` that puts tokens in a ``Container``, as in the
  tutorial's call centre.
* A study can't contain sweeps. Run one study per scenario.
* Relations between decisions, such as ``maximum > minimum``, aren't
  supported. Reparametrize instead: decide ``minimum`` and a ``gap``, and
  compute the maximum in the model.
