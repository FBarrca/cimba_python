.. _tut_6:

6. A data-driven supply chain
=============================

Every model so far invented its randomness: an exponential here, a PERT
there. Real studies start from **data**, such as a few years of daily sales
or a log of delivery delays. The hard questions are about that data. Does a
distribution fit it? Should we replay history or resample it? Does it matter
that the sites' demands move together?

Cimba Python treats these as first-class questions. This chapter builds a
multi-echelon inventory network driven by recorded demand and lead times,
**checks** the input model before trusting it, and then **compares input
models head to head** in one experiment. The model code never changes along
the way.

The full script is ``tutorial/multi_echelon_inventory.py``; the data is in
``tutorial/data/multi_echelon_inventory/``.

.. contents:: In this chapter
   :local:
   :depth: 1

The network
-----------

Six facilities form a tree. An external **source** (node 0) has unlimited
stock. A **distribution centre** (node 1) orders from it and supplies two
downstream sites. One of them, node 3, is a regional warehouse that in turn
supplies nodes 4 and 5:

.. code-block:: text

   0 source ──► 1 DC ──┬──► 2 store
                       └──► 3 warehouse ──┬──► 4 store
                                          └──► 5 store

Every stocking node (1–5) faces daily customer demand (node 3's is zero: it
only serves other nodes) and follows a *base-stock* policy. When its
inventory position falls to the reorder point, it orders up to the base
stock from its upstream node. Shipments take a base lead time plus a random
number of extra days.

Two kinds of input
------------------

The model consumes two very different kinds of real-world variability:

* **Daily demand** is *indexed by time*. There is one value per day per
  facility, and "today's demand" means the value for today's date, however
  often or rarely the model asks.
* **Extra lead-time days** form a *sequence*. Each shipment takes the next
  value, whenever it happens.

Cimba has a field type for each:

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Declaration
     - Read with
     - Meaning
   * - ``cb.Input[T]``
     - ``.next()``
     - A sequence. Each call consumes the next value.
   * - ``cb.Series[T] = cb.Series(step=..., origin=...)``
     - ``.now()``, ``.at(t)``
     - Time-indexed buckets. The value at time *t* belongs to bucket
       ``floor((t - origin) / step)``.

A ``Series`` never "skips" a value because the model read it twice, or
"loses" one because the model didn't read it on a given day. The bucket
depends only on the time.

Base classes and specializations
--------------------------------

All facilities share fields, so they share a base class:

.. literalinclude:: ../../tutorial/multi_echelon_inventory.py
   :pyobject: Facility
   :lines: 1-19

The source and the stocking facilities are subclasses. A subclass inherits
every field and may add its own. Only stocking facilities have demand, so
only they declare the ``Series``:

.. literalinclude:: ../../tutorial/multi_echelon_inventory.py
   :pyobject: StockingFacility
   :lines: 1-10

``cb.Series(step=1.0, origin=1.0)`` makes buckets one day wide, starting at
time 1. The customer-service process holds for a day and then reads
``self.demand.now()``: on day 1 bucket 0, on day 2 bucket 1, and so on.

.. literalinclude:: ../../tutorial/multi_echelon_inventory.py
   :pyobject: StockingFacility.serve_customer
   :dedent: 4

Here is the whole model as :func:`cimba.diagrams.process_graph` sees it,
before any data is bound (hence ``demand · no source``). Each stocking
facility's ``place_order`` spawns an ``Order`` and puts it into its
upstream's ``orders`` store. ``fulfill_orders`` gets orders and spawns
``Shipment`` models, which read the lead-time input and hand themselves back
to the network:

.. cimba-diagram:: tutorial.multi_echelon_inventory:MultiEchelonInventory

Look at what this code does **not** contain: no array, no day counter, no
check for the end of the data. It asks for today's demand and gets it.

The network keeps its facilities in a heterogeneous list,
``facilities: list[Facility]``, and the global lead-time input lives on the
root model because every shipment uses it:

.. literalinclude:: ../../tutorial/multi_echelon_inventory.py
   :pyobject: MultiEchelonInventory
   :lines: 1-5

The class default, ``inputs.trace([0.0], on_exhausted="wrap")``, means "no
extra days" until we bind real data. It gives the model a sensible behavior
out of the box.

Step 1: look at the data
------------------------

.. code-block:: python

   demand, lead_time_delay = load_data()
   demand.shape            # (10000, 5): 10,000 days × stocking nodes 1..5
   lead_time_delay[:10]    # [1. 0. 1. 2. 1. 2. 1. 1. 2. 0.]

Two observations shape our input model:

* The extra lead-time days are small **integers** (0 to 6). A continuous
  distribution would be a poor description. The simplest faithful model is to
  resample the observed values: ``inputs.bootstrap.iid(lead_time_delay)``.
  (For continuous data, ``inputs.fit(data)`` ranks exponential, gamma,
  lognormal, normal and Weibull fits by AIC. See
  :doc:`../guides/input_models`.)
* Demand is a **panel**: five series over the same days. If a heat wave lifts
  sales everywhere, resampling each site independently would break that
  link. A **joint** resample keeps it: it picks the same calendar blocks for
  every site.

Step 2: build and check the demand model
----------------------------------------

.. code-block:: python

   nodes = range(1, NUM_NODES)
   panel = {node: demand[:, node - 1] for node in nodes}
   mean_block = round(demand.shape[0] ** (1 / 3))       # 22 days

   joint = inputs.bootstrap.joint(panel, mean_block=mean_block,
                                  tag="facility-demand")

``bootstrap.joint`` is a *stationary block bootstrap* over the whole panel. It
builds new histories from random runs of consecutive days, with an average
run of 22 days, and uses the same runs for every member. Indexing it,
``joint[2]``, gives the source for node 2.

Before trusting it, **check** it. :func:`cimba.analysis.check_input`
generates a sample from a source and compares it with the reference data:

.. code-block:: python

   report = cb.analysis.check_input(joint[1], reference=panel[1])
   report.reference_mean, report.generated_mean           # 49.54, 48.67
   report.reference_variance, report.generated_variance   # 2426.4, 2312.9
   report.ks_statistic                                    # 0.011
   report.reference_acf[:4], report.generated_acf[:4]     # both ≈ (1, 0, 0, 0)

Means, variances and quantiles agree. The Kolmogorov–Smirnov distance
between the marginal distributions is about 1%, and the autocorrelation
structure (none to speak of here) is preserved. For the panel as a whole,
pass the joint source and the dictionary. The report adds the reference and
generated **cross-correlation** matrices:

.. code-block:: python

   panel_report = cb.analysis.check_input(joint, reference=panel)
   panel_report.reference_correlation
   panel_report.generated_correlation

(Node 3's row is ``nan`` because its demand is constantly zero.)

Step 3: bind the sources and run
--------------------------------

Binding is plain assignment to the model objects:

.. literalinclude:: ../../tutorial/multi_echelon_inventory.py
   :pyobject: main
   :emphasize-lines: 5-11

.. code-block:: console

   $ uv run python tutorial/multi_echelon_inventory.py
   Average on-hand by node: [   0.    1755.096  321.631  363.721   56.403   84.804]
   Service level by node: [1.     1.     0.9995 0.     0.3082 0.2617]

(Node 3's "service level" is 0 only because it has no customers of its own.)

Step 4: ask the inputs what they did
------------------------------------

Every input reports its provenance and consumption:

.. code-block:: python

   store = network.facilities[2]
   record = result[store].demand

   record.source[0]
   # {'method': 'bootstrap.joint', 'parameters': {'mean_block': 22, 'member': '2',
   #   'panel_sha256': '094a…'}, 'sha256': '756b…', 'tag': 'facility-demand',
   #  'prefix_stable': True, 'on_exhausted': 'extend', ...}

   record.consumed[0, :5]     # [359 359 359 359 359]  days read per trial
   record.extended.max()      # 0: no trial ran out
   record.rows(0, 0)[:5]      # the exact demand trial (0, 0) saw

   result[network].lead_time_delay.consumed[0, :5]   # [40 49 41 40 41] shipments

``source`` records what produced the values: method, parameters and SHA-256
fingerprints of the reference data, so a result can always be traced back to
the exact data it used. ``rows(point, replication)`` regenerates the exact
values any trial saw. Rows are regenerated on demand rather than stored, so
this costs nothing until you ask.

How much data is enough?
~~~~~~~~~~~~~~~~~~~~~~~~

We never said how many days of demand to generate. For a ``Series``, Cimba
computes the number of buckets the window needs,
``ceil((warmup + duration + cooldown − origin) / step)``, and generates at
least that many rows per replication. For sequence inputs like the lead
time, it starts with a default length (1,024 values) and, if a trial reads
past the end, **extends** it. The row is regenerated at double the length and
the trial rerun. Resampling sources are *prefix-stable*: the longer row starts
with exactly the same values. So an extended trial gives exactly the result
it would have given with enough data from the start.
``results.meta.extensions`` counts how often this happened.

Step 5: compare input models head to head
-----------------------------------------

Does keeping the sites' days aligned actually matter for the answer? Put the
two input models side by side in one experiment. Each facility's demand
becomes a two-valued sweep: joint member or independent stationary
bootstrap. These five sweeps must move **together**. We want "all joint" versus
"all independent", not 2⁵ mixtures. That's what ``cb.sweeps`` (plural) is
for:

.. code-block:: python

   independent = {n: inputs.bootstrap.stationary(panel[n], mean_block=mean_block)
                  for n in nodes}
   axes = cb.sweeps(*[[joint[n], independent[n]] for n in nodes])   # linked
   for n, axis in zip(nodes, axes):
       network.facilities[n].demand = axis

   network.lead_time_delay = inputs.bootstrap.iid(lead_time_delay)
   result = cb.Experiment(network, replications=50,
                          window=cb.Window(duration=360.0), seed=123).run()

   for n in (1, 2, 4, 5):
       store = network.facilities[n]
       print(n, cb.analysis.compare(result[store].service_level, a=0, b=1))

.. code-block:: text

   1 Comparison(a=0, b=1, n=50, difference= 0.0005, lower=-0.0001, upper= 0.0011)
   2 Comparison(a=0, b=1, n=50, difference=-0.0007, lower=-0.0018, upper= 0.0004)
   4 Comparison(a=0, b=1, n=50, difference=-0.0467, lower=-0.0891, upper=-0.0044)
   5 Comparison(a=0, b=1, n=50, difference=-0.0376, lower=-0.0738, upper=-0.0015)

For the DC and node 2 the choice makes no detectable difference. For nodes 4
and 5, which share the regional warehouse, resampling each site
independently **lowers the estimated service level by about 4 percentage
points**, and the paired intervals exclude zero. The input model is a
modeling decision with consequences, and now you can measure them.

Two details make this comparison sharp:

* Both design points run with the same replication seeds, and the lead-time
  input has its own stream, identified by its place in the model. Every
  shipment delay is therefore identical across the two points. Only the
  demand model differs.
* ``compare`` pairs replication *r* of point 0 with replication *r* of point
  1, which removes most of the run-to-run noise.

The same pattern compares any input models: a fitted AR model
(``inputs.fitted.sieve(history)``) against a block bootstrap, a trace
against a resample, or a fitted distribution against the data it was fitted
to.

What you learned
----------------

* ``Input[T]`` is a sequence (``next()``); ``Series[T]`` is time-indexed
  (``now()``, ``at(t)``). Model code never manages cursors, lengths or the end
  of the data.
* The input workflow is **look → choose → check → bind → run → inspect →
  compare**.
* ``inputs.bootstrap.joint`` resamples a panel while keeping its members
  aligned; ``check_input`` compares a source with reference data before a run.
* Results carry provenance (``source``), consumption (``consumed``,
  ``extended``) and replay (``rows``) for every input.
* Linked sweeps (``cb.sweeps``) compare input models head to head under
  common random numbers.

Where next
----------

You've now seen every major feature. From here:

* :doc:`../concepts/index` explains each idea more precisely, including edge
  cases the tutorial glossed over.
* :doc:`../guides/index` has focused recipes for recurring jobs.
* :doc:`../architecture/index` shows how the pieces fit together inside.
