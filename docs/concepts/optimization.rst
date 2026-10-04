Optimization
============

An experiment runs the configurations you list. An **optimization study**
looks for the configuration that makes an objective as small (or as large)
as possible. You say which parameters may change and how to score a trial;
Cimba searches, then estimates the chosen configuration on fresh trials.

:doc:`../tutorial/optimization` shows a complete study. This page explains
the ideas behind it.

Decisions
---------

A **decision** gives a ``Param`` a range instead of a value:

.. code-block:: python

   policy.level = cb.decision(40.0, 250.0)          # any value in [40, 250]
   centre.staff = cb.decision(1, 20)                # Param[int]: 1, 2, ..., 20

The field's annotation sets the kind of value: real for ``Param[float]``,
integer for ``Param[int]``, ``False``/``True`` for ``Param[bool]``.
``step=`` restricts a real range to evenly spaced values, and ``log=True``
searches a positive range on a log scale.

Each decision object is one **dimension** of the search. Assign the same
object to several fields and they always take the same value. Use ``map`` to
derive another field from it without adding a dimension:

.. code-block:: python

   level = cb.decision(40.0, 250.0)
   policy.level = level
   policy.reorder_point = level.map(lambda value: 0.35 * value)

Decisions are configuration, like sweeps. The compiled model code never
changes, and a plain ``cb.Experiment`` refuses a model that still holds
decisions.

The objective scores each trial
-------------------------------

The objective is a function of a batch's ``Results`` that returns one
number per trial, shaped ``(candidates, replications)``:

.. code-block:: python

   def cost(results):
       outputs = results[store]
       return outputs.mean_stock + 20 * outputs.orders_per_week

The study minimizes (``minimize=``) or maximizes (``maximize=``) the
objective's **expected value**, estimated by the mean over replications.
Per-trial values are what let Cimba compute confidence intervals, compare
candidates trial by trial and set failed trials aside.

Every candidate sees the same scenarios
---------------------------------------

All candidates run on the same replication seeds: replication *r* of every
candidate sees the same demand, arrivals and delays. This is the
:term:`common random numbers` idea from experiments, and it matters even
more here. When two candidates differ, the difference comes from their
settings, not from one of them drawing easier scenarios.

Because the seeds never change during the search, evaluating a candidate
twice would give exactly the same trials. Cimba remembers every candidate it
has simulated and never runs it again.

The search: differential evolution
----------------------------------

**Differential evolution** keeps a population of candidates. Each generation
it builds a challenger for every member by combining other members, and the
better of each pair survives. While the population is spread out it takes
large steps; as it converges, the steps shrink.

All challengers of a generation are independent, so Cimba runs a generation
as **one batch**: every new candidate × every replication, in one native call
across all cores. The population size sets how much work each batch holds.
The search stops when the population has converged, when the evaluation
budget is spent, or when a generation limit is reached.

Why the answer is checked on fresh trials
-----------------------------------------

The best mean found during a search is too optimistic. The search picked it
*because* it looked good on those particular scenarios, and part of looking
good was luck. This is sometimes called the **optimizer's curse**.

So after the search, Cimba uses two more sets of seeds that the search never
saw:

1. **Selection.** The best few candidates, the **finalists**, run on fresh
   shared seeds. The best one is chosen, and each other finalist is compared
   with it trial by trial.
2. **Estimation.** The chosen candidate runs once more, on yet another set of
   seeds. Those trials played no part in the choice, so their mean and
   confidence interval, ``best.estimate``, are an honest measure of how good
   the answer is.

A finalist whose comparison interval contains zero is statistically tied
with the choice. That doesn't prove they are equal, only that these trials
can't tell them apart.

Fresh seeds give fresh draws from distributions, bootstraps and fitted
inputs. A recorded **trace** replays the same data on every seed, so for that
input the check is not on unseen data.

Reproducibility
---------------

A study's search, choice and estimate depend only on the model, its
configuration, the seed and the optimizer settings (and the NumPy and SciPy
versions). The number of worker threads changes how fast it runs, not what
it finds.

The search trials are exactly those of an ``Experiment`` with the same seed,
so any candidate can be replayed as an ordinary experiment. The selection and
estimation seeds are recorded in ``best.meta``.
