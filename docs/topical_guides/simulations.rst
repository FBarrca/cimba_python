Simulations and Event Queues
============================

The simulation context
----------------------

:class:`cimba.Simulation` initializes the native event queue and the
thread-local random generator. Only one active simulation is supported per
Python thread.

.. code-block:: python

   with cimba.Simulation(start_time=0.0, seed=123) as sim:
       ...

When the context exits, Cimba closes owned objects and releases native state.
Objects are closed in reverse creation order, which matches the usual
dependency order in a model.

Running events
--------------

``execute()`` runs scheduled events until the queue is empty. ``execute_next()``
runs one event and returns ``False`` if there is nothing left to execute.

.. code-block:: python

   while sim.execute_next():
       if sim.now > 100.0:
           sim.clear()

Most models use ``execute()`` and an explicit stop condition.

Stopping
--------

``stop_at(when)`` schedules an event at absolute simulation time ``when`` that
stops active processes and clears the event queue.

``clear()`` immediately removes scheduled events. It is useful inside a process
when the model has reached a condition that should end the current run.

Random seeds
------------

Passing ``seed=`` makes runs reproducible. If no seed is supplied, Cimba uses a
hardware-derived seed and exposes it as ``sim.seed_used``.

.. code-block:: python

   with cimba.Simulation(seed=123) as sim:
       print(sim.seed_used)
