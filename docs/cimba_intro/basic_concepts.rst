Basic Concepts
==============

Simulations
-----------

A :class:`cimba.Simulation` owns Cimba's thread-local event queue, simulation
clock, and pseudo-random generator. Use it as a context manager:

.. code-block:: python

   import cimba

   with cimba.Simulation(seed=123) as sim:
       sim.stop_at(10.0)
       sim.execute()

The event queue runs until it is empty. ``stop_at()`` schedules a stop event at
an absolute simulation time.

Processes
---------

A :class:`cimba.Process` is an active simulated entity. It runs a Python
callable with the positional and keyword arguments you pass to
:class:`cimba.Process`.

.. code-block:: python

   def worker(context):
       cimba.hold(5.0)
       context.append(cimba.time())

   with cimba.Simulation(seed=123) as sim:
       done = []
       cimba.Process("Worker", worker, done).start()
       sim.execute()

``cimba.hold(duration)`` suspends the current Cimba process for simulated time,
not wall-clock time.

Passive objects
---------------

Buffers, queues, resources, conditions, datasets, and time series are passive
objects used by processes. If they are created while a simulation is active, the
simulation keeps them alive and closes them in reverse creation order.

.. code-block:: python

   with cimba.Simulation(seed=123) as sim:
       queue = cimba.Buffer("Queue")
       cimba.Process("Arrival", arrival, queue).start()
       cimba.Process("Service", service, queue).start()
       sim.stop_at(100.0)
       sim.execute()

Random values
-------------

Random functions use the active thread's Cimba random generator. Passing an
explicit seed to :class:`cimba.Simulation` makes a run reproducible.

.. code-block:: python

   cimba.exponential(2.0)
   cimba.normal(mu=10.0, sigma=1.5)
   cimba.dice(1, 6)

For the native random generator details, see the `Cimba C API reference`_.
