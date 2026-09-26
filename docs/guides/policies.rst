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
* **model sweeps:** the policy is a child model, and sweeping that child runs
  every policy as its own design point, in one experiment.

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

A policy can also have its own processes. This one reviews on a weekly
clock and orders only when a review is due:

.. literalinclude:: ../../tutorial/policy_comparison.py
   :pyobject: PeriodicReview

The process that doesn't change
-------------------------------

The store holds its policy as a **child model**, ``policy: Policy``, so any
subclass fits, and it asks the policy what to order. It never mentions a
concrete policy:

.. literalinclude:: ../../tutorial/policy_comparison.py
   :pyobject: Store.trade
   :dedent: 4
   :emphasize-lines: 12

Calls through a child, a reference or a list use the implementation of the
model's **actual** class, even though the store's code only knows
``Policy``. This is dynamic dispatch, as in ordinary Python. It's implemented
with a per-class function table, so the store is compiled once and never
recompiled when the policy changes.

One experiment, every policy
----------------------------

Sweep the child. Pass the policy objects to ``cb.sweep``, as separate
arguments or as one list:

.. literalinclude:: ../../tutorial/policy_comparison.py
   :pyobject: policy_study

.. literalinclude:: ../../tutorial/policy_comparison.py
   :pyobject: compare_policies

A swept child model gives **one model tree per option**. The trials of a
design point contain the store and *only* that point's policy. The other
policies don't exist there at all, so their processes (like
``PeriodicReview``'s clock) and hooks never run anywhere but in their own
design point.

.. cimba-diagram:: tutorial.policy_comparison:policy_study
   :kind: structure
   :direction: LR

In diagrams, the options appear as ``policy (option i)`` edges to
``store.policy#i``. Inside a trial, each option is simply ``store.policy``.

.. code-block:: console

   $ uv run python tutorial/policy_comparison.py
   policy          fill rate  mean stock  orders/wk
   OrderUpTo           0.983        42.7       6.98
   MinMax              0.897        42.1       0.85
   FixedQuantity       0.913        40.9       0.96
   PeriodicReview      0.932        52.4       1.00
   fill rate, MinMax - OrderUpTo: -0.0864 (95% CI -0.0915 .. -0.0814)
   fill rate, FixedQuantity - OrderUpTo: -0.0699 (95% CI -0.0742 .. -0.0656)
   fill rate, PeriodicReview - OrderUpTo: -0.0513 (95% CI -0.0550 .. -0.0476)

``results.levels(store.policy)`` returns the policy objects, one per design
point. Ordering every day keeps the fill rate at 98.3%, but it costs seven
orders a week. The two reorder-point policies order less than once a week at
about the same stock, but lose 7–9 points of fill rate. The weekly review
orders exactly once a week and sits in between, at the price of more stock.
Because all four design points see the **same demand and lead times**
(common random numbers, with one stream per input), ``compare`` gives tight,
paired intervals.

Reading results per policy
~~~~~~~~~~~~~~~~~~~~~~~~~~

Results for the store cover every design point. Results for a policy object
cover only the design points where it was selected. Elsewhere its outputs
are ``nan``, because it wasn't there:

.. code-block:: python

   weekly = results.levels(store.policy)[3]
   results[weekly].some_output          # values in row 3, nan in rows 0-2

Tuning within a policy
----------------------

Policy parameters are ordinary ``Param`` fields, so they sweep too, and
sweeps inside an option cross with the model sweep:

.. code-block:: python

   daily, weekly = OrderUpTo(), PeriodicReview()
   weekly.level = cb.sweep(110.0, 130.0, 150.0)
   store = Store(cb.sweep(daily, weekly))       # 2 x 3 = 6 design points

Sweeps cross, so ``OrderUpTo`` also runs once per ``level`` value even
though that parameter doesn't affect it. Those three design points repeat
the same trials. When that matters, use separate options instead:

.. code-block:: python

   lean, generous = MinMax(), MinMax()
   generous.minimum = 60.0
   store = Store(cb.sweep(OrderUpTo(), lean, generous))

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

Rules for sweeping a model
--------------------------

* Sweep a **child** field (``policy: Policy``) with model objects of that
  type. Each option gets its own model tree, and each trial contains only its
  design point's option.
* Options must not be used anywhere else in the tree; a reference into an
  option from outside would dangle in the other trees. An option may
  reference the rest of the model, for example a policy with a
  ``store: cb.Ref["Store"]`` back-reference.
* Classes are compiled once, however many options and design points there
  are. ``results.meta.variants`` reports how many model trees were run.
* The model sweep crosses with every other sweep, including sweeps of
  parameters inside the options.
