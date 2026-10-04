.. _tut_7:

7. Tuning a policy
==================

In :doc:`../guides/policies` a store compared inventory policies with
hand-picked settings: ``MinMax`` with a minimum of 40 and a maximum of 120.
Why 40 and 120? They were guesses. This chapter lets Cimba find good values
instead. You'll mark the settings that may change, say what "better" means,
and let a search try hundreds of candidates in parallel. Then you'll check
the answer on trials the search never saw.

The script is ``tutorial/policy_optimization.py`` and runs in a few seconds:

.. code-block:: console

   $ uv run python -m tutorial.policy_optimization

.. contents:: In this chapter
   :local:
   :depth: 1

The store
---------

The model comes from ``tutorial/policy_comparison.py``. Each day the store
sells what it can from stock; demand it can't meet is lost. Then it asks its
policy how much to order. Orders arrive after a random lead time of two to
five days. Three outputs describe a year of trading: ``mean_stock``,
``orders_per_week`` and ``fill_rate``, the share of demand served.

The policy we will tune waits until the inventory position (stock on hand
plus stock on order) falls to ``minimum``, then orders back up to
``maximum``:

.. literalinclude:: ../../tutorial/policy_comparison.py
   :pyobject: MinMax

Step 1: say what may change
---------------------------

A **decision** replaces a parameter's value with a range of allowed values:

.. literalinclude:: ../../tutorial/policy_optimization.py
   :pyobject: tunable_min_max

The search may choose any ``minimum`` from 0 to 100 and any ``maximum`` from
40 to 250, bounds included. ``step=1.0`` keeps both on whole units. The
field's annotation decides the kind of value: these are ``Param[float]``
fields, so they stay floats; a ``Param[int]`` field would get integers.

Nothing else about the model changes. ``MinMax`` is still an ordinary model
class, and its compiled code is the same.

Step 2: say what "better" means
-------------------------------

More stock protects sales but costs money to hold. Ordering more often keeps
stock low but each order costs something. The **objective** turns that
trade-off into one number per trial:

.. literalinclude:: ../../tutorial/policy_optimization.py
   :pyobject: store_cost

Each unit of average stock costs 1, each weekly order costs 20, and losing
all demand would cost 400. In a real study these weights come from your
business.

The function receives the results of many candidates at once.
``outputs.mean_stock`` is shaped ``(candidates, replications)``, and the
arithmetic keeps that shape, so the function returns a cost **for every
trial**. Don't average it yourself: Cimba needs the individual trials to
compute confidence intervals and compare candidates fairly.

Step 3: run the search
----------------------

.. literalinclude:: ../../tutorial/policy_optimization.py
   :pyobject: tune

``cb.Optimization`` takes the model and the objective, plus the same run
settings as an experiment. Every candidate runs 32 replications of a year of
trading, after a 30-day warmup.

All candidates use the **same 32 seeds**. Replication 5 of one candidate sees
exactly the demand and lead times that replication 5 of any other candidate
sees. Differences in cost are then caused by the policy settings, not by
luck, which makes the comparison sharp.

``run()`` searches with **differential evolution**. It keeps a population of
candidates (30 here, 15 per decision) and improves it generation by
generation, combining good candidates into new ones. Each generation is
handed to Cimba's native runner as one batch: 30 candidates × 32
replications = 960 trials, spread over every core. ``evaluations=300`` caps
how many distinct candidates are simulated.

Step 4: read the answer
-----------------------

.. code-block:: python

   policy = tunable_min_max()
   best = tune(policy)

   best[policy.minimum], best[policy.maximum]   # (60.0, 118.0)
   best.estimate                                 # cost and its confidence interval

You read the answer through the decision objects you assigned. Here is what
``main()`` prints for this study:

.. code-block:: text

   MinMax: minimum=60, maximum=118, cost 88.07 (95% CI 87.18 to 88.95)
     searched 254 candidates in 8 generations
     finalists (minimum, maximum, cost, difference from the choice):
         65  120   87.07  +0.38 (-0.54 to +1.29)
         66  121   87.36  +0.67 (-0.23 to +1.57)
         60  129   87.00  +0.31 (-0.79 to +1.40)
         64  119   86.74  +0.05 (-0.85 to +0.95)
         60  118   86.69  chosen

The search stopped on its own after 8 generations, when the population had
converged. Three things happened after it stopped:

1. **Finalists.** The five best candidates were run again, on 128 fresh
   replications they all share, and the cheapest was chosen.
2. **Comparison.** Every other finalist was compared with the choice, trial
   by trial. All five intervals contain zero: on these trials, the five
   settings are statistically tied. Any of them is a good answer, and
   that is worth knowing before arguing about 60 versus 66.
3. **Estimate.** The chosen settings were run once more, on another 128 fresh
   replications. That gives the cost you can quote: **88.07**, with a 95%
   confidence interval from 87.18 to 88.95.

Why not just report 86.69, the choice's cost among the finalists? Because
picking the lowest of several noisy numbers favors the ones that were lucky.
The estimate comes from trials that played no part in the choice, so it isn't
flattered by that luck. Here it's about 1.4 higher.

To use the answer, write it back into the model:

.. code-block:: python

   best.apply()
   policy.minimum, policy.maximum               # (60.0, 118.0): plain values again

The policy is now an ordinary configured model, ready for any experiment.

Step 5: was MinMax the right policy?
------------------------------------

The search tuned one policy. To choose *between* policies, tune each one,
then compare the tuned versions on fresh trials. ``OrderUpTo`` orders every
day, back up to a single ``level``:

.. literalinclude:: ../../tutorial/policy_optimization.py
   :pyobject: tunable_order_up_to

.. literalinclude:: ../../tutorial/policy_optimization.py
   :pyobject: tune_policies

After ``apply()``, both policies hold their chosen values. ``compare`` puts
them in a model sweep and runs a plain experiment with a new seed:

.. literalinclude:: ../../tutorial/policy_optimization.py
   :pyobject: compare

.. code-block:: text

   Tuned policies on fresh trials:
     MinMax     cost 87.52
     OrderUpTo  cost 185.87
     OrderUpTo - MinMax: +98.36 (95% CI +97.65 to +99.06)

Even at its best level, ``OrderUpTo`` places an order every day, and seven
orders a week cost 140 on their own. Tuned ``MinMax`` is far cheaper, and the
paired interval leaves no doubt. Tuning first and comparing second is the
fair way to judge policies: an untuned policy can lose just because of bad
settings.

Step 6: an integer decision
---------------------------

Decisions work the same way for whole numbers. A call centre pays 20 per
agent and 400 per unit of average waiting time. How many agents should it
employ?

.. literalinclude:: ../../tutorial/policy_optimization.py
   :pyobject: CallCentre

The agents are tokens in a ``Container``: a call takes one while it is being
handled and puts it back afterwards (see ``Caller`` in the script). The start
hook adds ``staff`` tokens, so the number of agents is an ordinary
``Param[int]`` and can be a decision. The cost is computed inside the model,
as an output, so the objective just returns it:

.. literalinclude:: ../../tutorial/policy_optimization.py
   :pyobject: tune_staff

.. code-block:: text

   Call centre: 7 agents, cost 157.85 (95% CI 156.82 to 158.88)

``staff`` is a ``Param[int]``, so the search only proposes whole numbers of
agents. Six agents leave callers waiting too long; eight cost more in wages
than they save.

What you learned
----------------

* ``cb.decision(low, high)`` makes a ``Param`` searchable. The field's type
  decides whether values are real, integer or boolean.
* The objective returns a value **per trial**, built from outputs with
  ordinary arithmetic.
* ``cb.Optimization(...).run()`` searches with differential evolution. Every
  candidate shares the same seeds, and each generation runs as one parallel
  batch.
* The reported ``best.estimate`` comes from fresh trials, so it is honest.
  ``best.finalists`` tells you which alternatives are statistically tied.
* ``best.apply()`` writes the answer back. Tune each policy, then compare the
  tuned policies on fresh trials.

Where next
----------

* :doc:`../guides/optimizing` covers the practical choices: replications,
  budgets, using every core, failed trials and reproducing a candidate.
* :doc:`../concepts/optimization` explains why the search and the estimate
  use different seeds.
* :doc:`../reference/optimize` lists every option and report field.
