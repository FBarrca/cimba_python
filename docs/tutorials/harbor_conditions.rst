.. _py_tut_4:

A Harbor with Complex Resources and Conditions
==============================================

The fourth C tutorial models a harbor with ships, tugs, berths, wind, tide, and
conditions for safe docking. The central feature is the same in Python:
:class:`cimba.Condition` lets a process wait until an arbitrary Python predicate
becomes true.

An Empty Simulation Template
----------------------------

The C tutorial begins with an empty shell. Python's equivalent is simply a
simulation context with a stop event:

.. literalinclude:: ../../tutorial/tut_4_0.py
   :language: python

Processes, Resources, and Conditions
------------------------------------

The simulated harbor state can be an ordinary dictionary. Tugs and berths are
resource pools, the communications channel is a binary resource, and the
harbormaster is a condition:

.. code-block:: python

   ctx = {
       "env": {"wind_magnitude": 20.0, "water_depth": 5.0},
       "tugs": cimba.ResourcePool("Tugs", capacity=2),
       "berths": [
           cimba.ResourcePool("Small berth", 1),
           cimba.ResourcePool("Large berth", 1),
       ],
       "comms": cimba.Resource("Comms"),
       "harbormaster": cimba.Condition("Harbormaster"),
       "departed": cimba.ObjectQueue("Departed ships"),
       "time_in_system": [cimba.Dataset(), cimba.Dataset()],
       "ship_by_process": {},
   }
   ctx["harbormaster"].subscribe(ctx["tugs"], *ctx["berths"])

Building Our Ships
------------------

The C tutorial derives ``struct ship`` from ``cmb_process``. Python keeps ship
characteristics in normal objects or dictionaries and maps a process object to
its ship data:

.. code-block:: python

   ship = {
       "size": SMALL,
       "tugs_needed": 1,
       "max_wind": 10.0,
       "min_depth": 8.0,
       "unloading_time": 2.0,
   }
   proc = cimba.Process("Ship_000001_small", ship_proc, ctx)
   ctx["ship_by_process"][proc] = ship
   proc.start()

Weather and Tides
-----------------

In the full C tutorial, weather and tide are separate processes that update the
environment and signal the harbormaster. The checked-in compact Python tutorial
uses fixed values so the tests stay deterministic, but the C-like stochastic
version uses the current Python distribution functions directly:

.. code-block:: python

   def weather_and_tide(me, ctx):
       while True:
           old_wind = ctx["env"]["wind_magnitude"]
           ctx["env"]["wind_magnitude"] = 0.5 * cimba.rayleigh(5.0) + 0.5 * old_wind
           ctx["env"]["wind_direction"] = cimba.pert(0.0, 225.0, 360.0)

           astronomical_tide = ctx["tide_model"].depth_at(cimba.time())
           weather_tide = 0.1 * ctx["env"]["wind_magnitude"]
           ctx["env"]["water_depth"] = astronomical_tide + weather_tide

           ctx["harbormaster"].signal()
           cimba.hold(1.0)

``cimba.rayleigh(s)`` takes the Rayleigh scale parameter. ``cimba.pert(min,
mode, max)`` is the lowercase Python wrapper for Cimba's PERT distribution.

The general rule is the same as in C: whenever model state changes in a way that
could make a waiting condition true, signal the condition.

Resources and Condition Variables
---------------------------------

A condition predicate receives the waiting process and the context object. It
must inspect state and return ``True`` or ``False``:

.. code-block:: python

   def is_ready_to_dock(process, ctx):
       ship = ctx["ship_by_process"][process]
       return (
           ctx["env"]["water_depth"] >= ship["min_depth"]
           and ctx["env"]["wind_magnitude"] <= ship["max_wind"]
           and ctx["tugs"].available >= ship["tugs_needed"]
           and ctx["berths"][ship["size"]].available >= 1
       )

As in the C tutorial, a waiting process should re-check the predicate after it
wakes. Another process may have consumed the relevant resources first:

.. code-block:: python

   while not is_ready_to_dock(me, ctx):
       assert ctx["harbormaster"].wait(is_ready_to_dock, ctx) == cimba.SUCCESS

The C tutorial uses native resource-guard observer registration so releases from
tugs and berths automatically forward a signal to the harbormaster. Python
exposes this as :meth:`cimba.Condition.subscribe`:

.. code-block:: python

   ctx["harbormaster"].subscribe(ctx["tugs"], *ctx["berths"])

Use explicit ``condition.signal()`` for model state that is not represented by a
Cimba resource guard, such as weather and tide changes.

The Life of a Ship
------------------

Once a ship is cleared, it acquires berth and tug resources, uses the
communications resource, docks, unloads, acquires tugs again, leaves, and puts a
departure record into an object queue:

.. code-block:: python

   def ship_proc(me, ctx):
       ship = ctx["ship_by_process"][me]
       t_arrival = cimba.time()

       while not is_ready_to_dock(me, ctx):
           assert ctx["harbormaster"].wait(is_ready_to_dock, ctx) == cimba.SUCCESS

       berth = ctx["berths"][ship["size"]]
       assert berth.acquire(1) == cimba.SUCCESS
       assert ctx["tugs"].acquire(ship["tugs_needed"]) == cimba.SUCCESS

       assert ctx["comms"].acquire() == cimba.SUCCESS
       cimba.hold(cimba.gamma(5.0, 0.01))
       ctx["comms"].release()

       cimba.hold(cimba.pert(0.4, 0.5, 0.8))
       ctx["tugs"].release(ship["tugs_needed"])

       avg = ship["unloading_time"]
       cimba.hold(cimba.pert(0.75 * avg, avg, 2.0 * avg))

       assert ctx["tugs"].acquire(ship["tugs_needed"]) == cimba.SUCCESS
       assert ctx["comms"].acquire() == cimba.SUCCESS
       cimba.hold(cimba.gamma(5.0, 0.01))
       ctx["comms"].release()

       cimba.hold(cimba.pert(0.4, 0.5, 0.8))
       berth.release(1)
       ctx["tugs"].release(ship["tugs_needed"])

       system_time = cimba.time() - t_arrival
       ctx["departed"].put((me.name, ship["size"], system_time))
       return system_time

Because the harbormaster condition subscribed to the tug and berth resource
guards during setup, these releases are forwarded in C; the ship process does
not need a Python-level ``harbormaster.signal()`` call.

A separate departure process consumes those records and collects statistics in
:class:`cimba.Dataset` objects. This is the Python equivalent of the C tutorial's
departure process reclaiming ship objects and reading process exit values.

Running a Trial
---------------

The complete compact trial is:

.. literalinclude:: ../../tutorial/tut_4_1.py
   :language: python

Turning Up the Power
--------------------

The C tutorial turns the harbor into a 600-trial experiment over dredging depth,
tugs, berth counts, traffic levels, and replications. The Python tutorial keeps
a small scenario comparison:

.. literalinclude:: ../../tutorial/tut_4_2.py
   :language: python

A larger Python version would put each scenario/replication in a trial grid and
call :func:`cimba.run_experiment`, exactly like the M/M/1 tutorial.
