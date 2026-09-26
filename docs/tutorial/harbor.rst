.. _tut_4:

4. A weather-gated harbor
=========================

Larger models are built from parts: the physical plant, the environment, the
flow of work. Each part has its own state and processes, and the parts need to
see each other. This chapter builds a harbor from four cooperating models and
introduces the most flexible way to wait in Cimba: **waiting for a
condition**.

The scenario, from the C library's tutorial 4:

* Small and large ships arrive at random. To dock, a ship needs a free berth
  of its size, enough **tugboats** (one for a small ship, three for a large
  one), a free radio channel, **and** safe conditions: wind below its limit
  and water deep enough for its draught.
* **Wind** changes hourly. **Water depth** follows the tides, pushed up or
  down by the wind.
* After unloading, the ship needs the tugs and the radio again to leave.

We want the time ships spend in the harbor, and how busy the tugs and berths
are. The full script is ``tutorial/tut_4_1.py``.

.. contents:: In this chapter
   :local:
   :depth: 1

Designing the model tree
------------------------

Start with the parts and how they relate, not the code. The harbor owns three
parts, and ships are spawned into it during the trial:

.. cimba-diagram:: tutorial.tut_4_1:Harbor
   :kind: structure
   :direction: LR

Solid arrows mean *owns*; dotted arrows are references. Each part is a model class. The root owns the three parts as **children** and
builds them in its constructor:

.. literalinclude:: ../../tutorial/tut_4_1.py
   :pyobject: Harbor.__init__
   :dedent: 4

Notice ``SeaConditions(self, mean_wind)``: the children get a **reference
back to the harbor**, because the tide process needs the harbor's
``reference_depth`` parameter and the ships need everything. References may
point anywhere in the tree: up, down or sideways. Cycles are fine, since a
``Ref`` is just a pointer, not ownership.

The constructor also turns plain Python arguments into configuration. Some
become ``Param`` values (``reference_depth``, ``percent_large``), some become
entity capacities, and some become input sources. That's the pattern for
reusable models: take domain arguments in ``__init__``, turn them into fields.

The facilities are just entities with capacities:

.. literalinclude:: ../../tutorial/tut_4_1.py
   :pyobject: HarborFacilities

The environment
---------------

Weather and tide are two processes on the ``SeaConditions`` model:

.. literalinclude:: ../../tutorial/tut_4_1.py
   :pyobject: SeaConditions

A few things are worth pointing out:

* ``wind_source`` and ``direction_source`` are **inputs**, with default
  sources on the class and a per-instance override in ``__init__``. Wind is
  exactly the kind of variability you might later drive with measured
  weather data.
* The weather process has ``priority=1``. Both processes wake up every hour.
  When two events fall at the same instant, the higher priority runs first,
  so the tide always sees *this* hour's wind.
* The tide calls NumPy (``np.sin``, ``np.pi``) inside compiled code. Numba
  supports most of NumPy's math.
* After updating the depth, the tide calls
  ``self.harbor.facilities.harbormaster.signal()``. We'll see why next.

Waiting for a condition
-----------------------

A ship can dock only when *five* things are true at once: depth, wind, tugs,
berth, and (a moment later) radio. You can't express that as a single
``acquire``. What we want is: *sleep until the world might have changed, then
check again.*

That's a :class:`~cimba.Condition`. Processes wait on it with
``wait_until(predicate)``; anyone who changes something relevant calls
``signal()``. When a condition is signalled, the engine re-evaluates each
waiting process's **predicate**, a method marked ``@cb.predicate`` that
returns ``True`` or ``False``, and wakes those whose predicate holds.

The ship's voyage starts like this:

.. literalinclude:: ../../tutorial/tut_4_1.py
   :pyobject: Ship.voyage
   :lines: 1-18
   :dedent: 4

The harbor's predicate is deliberately simple: it always says "yes, wake up",
and the ship re-checks its own detailed condition in the ``while`` loop. That
keeps the ship-specific logic next to the ship.

.. literalinclude:: ../../tutorial/tut_4_1.py
   :pyobject: Harbor.should_call_harbormaster
   :dedent: 4

The other half of the contract is signalling. The tide signals every hour, and
every ship signals when it releases tugs or a berth. A missing ``signal()`` is
the classic condition bug: waiters sleep forever because nobody told them the
world changed.

.. tip::

   Predicates can also be selective. If ``is_ready`` checks the actual
   condition (``return self.ready``), waiters wake only when it's true.
   Predicates are compiled like processes and follow the same rules.

   A related tool is the **event**: a method marked ``@cb.event`` that you
   schedule with ``cb.schedule(self.method, delay)``. It runs once at that
   time, outside any process, and returns a ``Scheduled`` handle you can
   ``cancel()``. Events are handy for one-off happenings such as a shift
   change or a deadline.

Once cleared, the ship grabs what it needs in order, and releases in order:

.. literalinclude:: ../../tutorial/tut_4_1.py
   :pyobject: Ship.voyage
   :lines: 19-45
   :dedent: 4

Every duration comes from an input on ``ShipTraffic``. The ship reaches them
through its reference: ``self.harbor.traffic.unload_large.next()``. Inputs are
fields like any other, so they can live wherever they belong conceptually.

The whole picture
-----------------

With all the processes written, :func:`cimba.diagrams.process_graph` shows
how they meet. Each ship's single ``voyage`` process touches every facility,
every traffic input and both datasets. Weather and tide run on their own and
talk to the ships only through the harbormaster condition.

.. cimba-diagram:: tutorial.tut_4_1:Harbor
   :direction: LR

Ships as values in a store
--------------------------

The last line of the voyage hands the ship itself to a store:
``self.harbor.traffic.departed.put(self)``. On the other side, a departures
process releases each ship:

.. literalinclude:: ../../tutorial/tut_4_1.py
   :pyobject: ShipTraffic
   :lines: 22-35
   :dedent: 4

``departed: cb.Store[Ship]`` is a FIFO store of ship handles. Stores are typed
(``Store[int]``, ``Store[float]`` or ``Store[SomeModel]``), so what comes out
of ``get()`` is fully usable. Here the store is a tidy way to hand the ship's
lifetime back to the traffic model.

Running and reading the results
-------------------------------

.. code-block:: console

   $ uv run python tutorial/tut_4_1.py
   cimba 3.0.0: 20 harbor trials in 2.24 s
   n_small: 3278.750
   n_large: 1092.400
   avg_time_small: 10.861
   avg_time_large: 17.148
   tug_util: 0.876
   berth_small_util: 3.810
   berth_large_util: 1.827

A year of harbor operations, 20 times, in about two seconds. Large ships
spend 17 hours in port against 11 for small ones: they unload longer, and
their stricter depth and wind limits keep them waiting for the weather.

Comparing designs fairly
------------------------

The point of a model like this is to ask *what if*. Two kinds of change are
possible, and Cimba handles them slightly differently.

**Changing a parameter.** ``percent_large`` is a ``Param``, so it can be
swept inside one experiment:

.. code-block:: python

   harbor = Harbor()
   share = cb.sweep(0.15, 0.25, 0.35)
   harbor.percent_large = share

   window = cb.Window(warmup=24.0, duration=HOURS_PER_YEAR)
   results = cb.Experiment(harbor, replications=20, window=window, seed=7).run()

   print(results.levels(share))
   print(cb.analysis.summary(results[harbor].avg_time_large).mean)
   print(cb.analysis.compare(results[harbor].avg_time_large, a=0, b=2))

.. code-block:: text

   (0.15, 0.25, 0.35)
   [15.17 17.13 26.46]
   Comparison(a=0, b=2, n=20, difference=11.29, lower=10.28, upper=12.31)

Going from 15% to 35% large ships adds 11.3 hours (95% CI 10.3–12.3) to each
large ship's stay.

**Changing structure.** The number of tugs is a resource *capacity*, part of
the model's structure, so it is set when the model is built, not swept. Build
one model per variant and run each with the **same seed**:

.. code-block:: python

   import numpy as np

   samples = []
   for tugs in (8, 10):
       variant = Harbor(num_tugs=tugs)
       results = cb.Experiment(variant, replications=20, window=window, seed=7).run()
       samples.append(results[variant].avg_time_small.values[0])

   print(cb.analysis.compare(np.vstack(samples), a=0, b=1))

.. code-block:: text

   Comparison(a=0, b=1, n=20, difference=-0.00016, lower=-0.00025, upper=-0.00008)

Same seed plus same model structure means the same random stream for every
input. Ship *k* of replication *r* arrives at the same moment with the same
cargo in both variants, so the comparison is **paired**. The effect of two
extra tugs is tiny: less than a second per ship. Yet the interval excludes
zero, because pairing strips out almost all the noise. Common random numbers
let you detect small effects; whether they *matter* is your call.

``tutorial/tut_4_2.py`` uses the same technique to compare one versus two
large berths in a harbor that handles only large ships. With one berth, that
harbor can't keep up at all.

What you learned
----------------

* Build models from parts: child models for ownership, ``Ref`` for
  everything else, and ``__init__`` to turn domain arguments into fields.
* ``Condition.wait_until(predicate)`` + ``signal()`` waits for arbitrary
  combinations of state; ``@cb.event`` + ``cb.schedule`` runs one-off
  timed actions.
* ``@cb.process(priority=...)`` orders simultaneous events.
* ``Store[M]`` passes typed model handles between processes.
* Sweep ``Param`` values in one experiment; compare structural variants with
  separate experiments that share a seed. Both are paired comparisons.

Next, in :doc:`assembly`, we push parts through a production line and learn
to look inside a run with captured histories and diagrams.
