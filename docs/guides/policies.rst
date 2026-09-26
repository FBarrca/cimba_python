Compare policies with polymorphic functions
===========================================

**Goal:** keep a process unchanged while swapping the *decision logic* inside
it, and compare the alternatives fairly in one experiment.

Many studies are about policies: how much to reorder, which job to serve
next, where to route a part, when to call in extra staff. The process around
the decision stays the same; only the decision changes. Cimba Python
expresses this with two features working together:

* **polymorphic functions:** the decision is a ``@cb.function`` on a policy
  model, and each policy class overrides it;
* **reference sweeps:** the process reaches its policy through a ``cb.Ref``,
  and sweeping that reference runs every candidate as its own design point.

The complete example is ``tutorial/policy_comparison.py``.

The policy interface
--------------------

The base class declares the decision and its signature. Every parameter and
the return value are annotated:

.. literalinclude:: ../../tutorial/policy_comparison.py
   :pyobject: Policy

Each policy is a subclass that overrides the function with the **same
signature**, and can have its own parameters:

.. literalinclude:: ../../tutorial/policy_comparison.py
   :pyobject: MinMax

.. literalinclude:: ../../tutorial/policy_comparison.py
   :pyobject: FixedQuantity

The process that doesn't change
-------------------------------

The store holds a ``cb.Ref[Policy]``, any subclass will do, and asks it
what to order. It never mentions a concrete policy:

.. literalinclude:: ../../tutorial/policy_comparison.py
   :pyobject: Store.trade
   :dedent: 4
   :emphasize-lines: 12

Calls through a reference or a list use the implementation of the model's
**actual** class, even though the store's code only knows ``Policy``. This
is dynamic dispatch, as in ordinary Python. It's implemented with a per-class
function table, so the store is compiled once and never recompiled when the
policy changes.

One experiment, every policy
----------------------------

A reference is just a pointer, and a pointer is data. So you can sweep it,
as long as every candidate is part of the model tree. Here a small ``Study``
model owns the candidates and the store:

.. literalinclude:: ../../tutorial/policy_comparison.py
   :pyobject: Study

.. literalinclude:: ../../tutorial/policy_comparison.py
   :pyobject: swept_study
   :emphasize-lines: 4

.. literalinclude:: ../../tutorial/policy_comparison.py
   :pyobject: compare_policies

.. cimba-diagram:: tutorial.policy_comparison:swept_study
   :kind: structure
   :direction: LR

The dotted ``policy (sweep)`` edges show the three pointers the store can
take, one per design point.

.. code-block:: console

   $ uv run python tutorial/policy_comparison.py
   policy          fill rate  mean stock  orders/wk
   OrderUpTo           0.985        42.7       6.99
   MinMax              0.898        41.9       0.85
   FixedQuantity       0.914        41.1       0.96
   fill rate, MinMax - OrderUpTo: -0.0878 (95% CI -0.0922 .. -0.0834)
   fill rate, FixedQuantity - OrderUpTo: -0.0715 (95% CI -0.0759 .. -0.0670)

``results.levels(choice)`` returns the policy objects, one per design point.
Ordering every day keeps the fill rate at 98.5%, but it costs seven orders a
week. The two reorder-point policies order less than once a week and hold
about the same stock, but lose 7–9 points of fill rate. Because all three
design points see the **same demand and lead times** (common random numbers,
with one stream per input), ``compare`` gives tight, paired intervals.

Tuning within a policy
----------------------

Policy parameters are ordinary ``Param`` fields, so they sweep too. To tune
one policy, sweep its parameter. To tune while comparing, create one
candidate per setting:

.. code-block:: python

   study.candidates = [OrderUpTo(), MinMax(), MinMax()]
   study.candidates[2].minimum = 60.0              # a second MinMax setting
   study.store.policy = cb.sweep(*study.candidates)

Or cross a policy sweep with any other sweep, for example of the demand
source, to see whether the ranking holds under a different input model.

Rules for ``@cb.function``
--------------------------

* **Explicit.** Only methods decorated with ``@cb.function`` can be called
  from compiled code. Calling an undecorated method gives a compile error
  that says so.
* **Annotated.** Parameters are ``float``, ``int``, ``bool`` or a model
  class. The return type is one of those or ``None``. Defaults are allowed
  for scalar parameters. Arguments are passed by position.
* **Same signature when overriding**, and the override must also carry
  ``@cb.function``. Both are checked when the class is first used.
* **Can do anything a process can:** read and write fields, use entities
  and inputs, call other functions (recursion works), spawn models, and even
  block with ``cb.hold`` or ``get``. Don't block inside a function called
  from a predicate or event.
* **Plain Python on the host.** ``OrderUpTo().order_quantity(0.0, 40.0)``
  runs as a normal method, which is handy for unit-testing policy logic.
* Names reserved by ``cimba.Model`` (such as ``describe``) can't be used.

Rules for reference sweeps
--------------------------

* Only ``cb.Ref`` fields can be swept this way, and every choice must be a
  model **in the tree**, owned somewhere (here by ``Study.candidates``). An
  optional reference (``cb.Ref[M] | None``) may include ``None``.
* Candidates that aren't selected still exist in the trial. If they have
  processes, those processes run too. Keep policy models passive (functions
  and parameters) or make their processes check whether they're in use.
* Linked sweeps (``cb.sweeps``) switch several references together, for
  example the same policy class for every store in a network.
