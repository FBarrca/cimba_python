Assembly Line
=============

``tutorial/assembly_line.py`` is a complete manufacturing-line model. It is
larger than the tiny queue examples because it combines dynamic parts, station
inboxes, exclusive processing resources, per-station measurements, whole-system
measurements, and graph output in one runnable script.

Run it from the repository root with plotting dependencies installed:

.. code-block:: bash

   uv run --extra plot python tutorial/assembly_line.py

The simulated system
--------------------

The model represents a three-station assembly line:

* parts arrive after exponentially distributed interarrival times,
* each part enters Station 1, Station 2, and Station 3 in order,
* each station has an inbox and one processing resource,
* a station takes a part from its inbox, records its waiting time, acquires the
  resource, holds for an exponentially distributed processing time, releases
  the resource, and forwards the part downstream,
* the finished-parts sink records total cycle time and removes the part from
  the active system count.

The station processing means are 5, 7, and 4 minutes. The interarrival mean is
3 minutes, so the model is intentionally busy: the middle station is the
bottleneck, queues can form, and utilization is an important output.

Model structure
---------------

``Part`` is a :class:`~cimba.sim.Struct` that stores the part id, arrival time,
and the time when the part entered its current station. Those fields travel
with the spawned part process.

``Station`` is a reusable :class:`~cimba.sim.Component`. Each station owns:

* ``inbox``, a :class:`~cimba.sim.Store` of waiting part handles,
* ``resource``, a :class:`~cimba.sim.Resource` representing the processor,
* ``wait_time``, a :class:`~cimba.sim.Dataset` for station waiting times,
* ``downstream``, a reference to the next station or finished-parts sink.

The three stations are declared as fields on ``AssemblyLine`` and chained
together through their ``downstream`` references. That keeps the station logic
generic: Station 1, Station 2, and Station 3 all run the same process body with
different processing-time parameters.

Process flow
------------

The model has three main process patterns.

``arrivals`` waits for the next arrival, spawns a part lifecycle process, and
records the part's id and system arrival time.

``part_lifecycle`` increments the system population, stamps the current station
entry time, and puts the current process handle into Station 1's inbox. After
that, station processes move the part by passing the same handle through
stores.

``Station.server`` loops forever. It takes a part handle from the station
inbox, computes how long the part waited since its previous station-entry
stamp, processes the part while holding the station resource, updates the
station-entry stamp, and puts the handle into the next inbox.

``FinishedParts.finish`` is the downstream endpoint. It records the total cycle
time, decrements the system population, and hands the part to ``reclaim`` so
the spawned process can be despawned.

Measurements
------------

The example records both local station metrics and whole-system metrics.

Each station collects:

* average wait time from its ``wait_time`` dataset,
* utilization from the time average of its processing resource.

The model-level collector records:

* total parts produced,
* average and maximum cycle time,
* throughput rate,
* average, maximum, and final number of parts in the system.

The collector also writes raw cycle-time, wait-time, and system-population
series to temporary files. The plotting helper reads those files back into
NumPy arrays to produce histograms, utilization bars, and a step plot of parts
in the system. ``main`` currently writes the process graph by default; the
``plot_results`` call is present but commented out.

Process graph output
--------------------

After the experiment runs, ``plot_process_dag`` writes Mermaid and Graphviz
representations under ``tutorial/assembly_line_plots/``. If Graphviz ``dot`` is
available, it also writes PNG and SVG renderings. This is useful for checking
that the arrival process, part lifecycle, station servers, and finished-parts
sink are connected the way the model intends.

Full source
-----------

.. literalinclude:: ../../tutorial/assembly_line.py
   :language: python
