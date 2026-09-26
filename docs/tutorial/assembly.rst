.. _tut_5:

5. An assembly line
===================

A production line is a chain of stations. Parts arrive, wait, get processed,
and move on. Throughput is limited by the slowest station, the
**bottleneck**, and finding it is the first question anyone asks. This chapter
builds a three-station line where individual parts flow between stations,
then uses captured data, diagrams and a crossed sweep to find the bottleneck
and fix it.

The full script is ``tutorial/tut_5_1.py``.

.. contents:: In this chapter
   :local:
   :depth: 1

Parts that carry their own history
----------------------------------

Each part is a small dynamic model with just enough state to measure its
journey:

.. literalinclude:: ../../tutorial/tut_5_1.py
   :pyobject: Part

It has no processes. Stations push it around. A model without behavior is
perfectly fine: it's a record that lives in the simulation and can be
stored, passed and released.

Stations hand parts downstream
------------------------------

.. literalinclude:: ../../tutorial/tut_5_1.py
   :pyobject: Station
   :linenos:

Things to notice:

* **Typed inboxes.** ``inbox: cb.Store[Part]`` holds part handles.
  ``self.inbox.get()`` returns a ``Part`` view, so ``part.station_entry`` is
  a normal field access, and ``put`` refuses anything that isn't a ``Part``
  when the class is compiled.
* **Optional references.** ``downstream: cb.Ref["Station"] | None`` may be
  ``None``: the last station has no downstream station and hands parts to the
  ``FinishedParts`` sink instead. The ``if self.downstream is not None`` test
  works in compiled code exactly as it reads.
* **Host-only attributes.** ``self.name = name`` (line 13) is *not* declared
  as a field. It's an ordinary Python attribute that the compiled code never
  sees. That's handy for labels you use when printing results.
* **A resource per station.** The server acquires the station's resource while
  processing, so ``mean_in_use()`` becomes the station's utilization.

The line itself creates the stations back to front, so each one can be given
its downstream neighbour, and spawns parts into station 1:

.. literalinclude:: ../../tutorial/tut_5_1.py
   :pyobject: AssemblyLine.__init__
   :dedent: 4

.. literalinclude:: ../../tutorial/tut_5_1.py
   :pyobject: AssemblyLine.arrivals
   :dedent: 4

The ``system`` container counts parts in the system: up by one on arrival,
down by one when ``FinishedParts`` releases a part. Its time-weighted mean
level is the average work in progress.

Capturing what happened
-----------------------

The constructor also **captures** three kinds of history: the WIP container,
the cycle-time dataset and each station's wait-time dataset. After the run,
each captured entity is a :class:`~cimba.Signal` indexed by design point and
replication:

.. literalinclude:: ../../tutorial/tut_5_1.py
   :pyobject: plot_results

For a captured dataset, ``trial(p, r)`` returns ``(times, values)`` where
``times`` is empty and ``values`` holds every recorded sample: here, every
part's cycle time. For a captured container or resource you get the full step
history of its level.

Seeing the structure
--------------------

When a model has more than a couple of parts, a picture helps.
``cimba.diagrams`` draws two, straight from the configured model, without
compiling or running it:

.. code-block:: python

   from cimba import diagrams

   line = AssemblyLine()
   print(diagrams.structure(line).to_mermaid())       # who owns and references whom
   print(diagrams.process_graph(line).to_mermaid())   # how processes interact

The **structure** shows ownership (solid) and references (dotted). The
labels, ``assemblyline.station_2`` and so on, are the **canonical paths**
Cimba uses in error messages and provenance:

.. cimba-diagram:: tutorial.tut_5_1:AssemblyLine
   :kind: structure
   :direction: LR

The **process graph** follows a part through the line. Arrivals spawn it and
put it in station 1's inbox. Each server gets it, acquires its resource,
reads a processing time and puts the part downstream, until the finishing
process records the cycle time:

.. cimba-diagram:: tutorial.tut_5_1:AssemblyLine
   :direction: LR

Both return a :class:`cimba.diagrams.Graph` that renders to Mermaid or
Graphviz DOT. The script saves both as ``.mmd`` files next to its plots. See
:doc:`../guides/diagrams` for everything they can show.

Reading the results
-------------------

.. code-block:: console

   $ uv run python tutorial/tut_5_1.py
   --- Simulation Results Analysis (Cimba) ---
   Total parts produced: 1389
   Average cycle time: 3023.54 minutes
   Maximum cycle time: 5948.39 minutes
   Throughput rate: 0.14 parts/minute
   Station 1: wait 2135.80 min, utilization 99.77%
   Station 2: wait 1487.98 min, utilization 99.97%
   Station 3: wait 4.63 min, utilization 55.85%
   Average parts in system: 1031.00
   Maximum parts in system: 2064
   Parts still in system: 2062

Something is badly wrong, and the model tells us exactly what. Parts arrive
every 3 minutes on average (0.33 per minute), but only 0.14 per minute leave.
Work in progress climbs to over 2,000 parts and is still climbing when the
window closes. Stations 1 and 2 are busy essentially all the time.

The mean processing times are 5, 7 and 4 minutes. Station 2 can finish at
most 1/7 ≈ 0.143 parts per minute, which matches the measured throughput:
**station 2 is the bottleneck**. Station 1 (1/5 per minute) is also slower
than the arrivals, and station 3 is starved.

Fixing the bottleneck with a crossed sweep
------------------------------------------

Suppose we can speed up stations 1 and 2 to a mean of 2.5 minutes each. Which
upgrades do we need? Each station's processing time is an input, so we can
sweep its **source**, and two independent sweeps **cross**:

.. code-block:: python

   line = AssemblyLine()
   line.station_1.processing_time = cb.sweep(inputs.dist.exponential(mean=5.0),
                                             inputs.dist.exponential(mean=2.5))
   line.station_2.processing_time = cb.sweep(inputs.dist.exponential(mean=7.0),
                                             inputs.dist.exponential(mean=2.5))

   results = cb.Experiment(line, replications=10,
                           window=cb.Window(duration=10_000.0), seed=45).run()
   cb.analysis.summary(results[line].throughput_rate).mean

.. list-table::
   :header-rows: 1

   * - Point
     - Station 1 mean
     - Station 2 mean
     - Throughput (parts/min)
   * - 0
     - 5.0
     - 7.0
     - 0.140
   * - 1
     - 5.0
     - 2.5
     - 0.199
   * - 2
     - 2.5
     - 7.0
     - 0.140
   * - 3
     - 2.5
     - 2.5
     - 0.249

This is the classic **shifting bottleneck**. Speeding up station 1 alone
(point 2) changes nothing, because station 2 still limits the line. Speeding up
station 2 alone (point 1) lifts throughput to station 1's limit of 0.2.
Upgrading both (point 3) helps most, but throughput stops at 0.25, not 0.33:
now station 3, at 4 minutes per part, is the bottleneck.

Two sweeps give 2 × 2 = 4 design points. Design points are ordered like
nested loops, with the first sweep found in the model tree as the outer loop.
``results.levels(sweep)`` returns each point's value on any axis, so you never
need to work the order out by hand. If you want two sweeps to move
*together* instead of crossing, create them with ``cb.sweeps`` (see
:doc:`../concepts/experiments`).

What you learned
----------------

* Dynamic models can be pure data records passed between typed stores.
* ``Ref[M] | None`` expresses optional links; undeclared attributes stay in
  Python.
* Captured datasets and containers give per-trial raw data for plots.
* ``cimba.diagrams.structure`` and ``process_graph`` draw the configured
  structure and the process interactions.
* Independent sweeps cross into a full factorial design, and sweeping input
  *sources* is as easy as sweeping parameters.

The last chapter, :doc:`inventory`, turns to what makes Cimba Python
different: driving a model with real data.
