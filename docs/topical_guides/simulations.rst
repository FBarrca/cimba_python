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

Direct events
-------------

``schedule(callback, when, subject=None, obj=None, priority=0)`` schedules a
Python callback at an absolute simulation time and returns the native event
handle. The callback is called as ``callback(subject, obj)``.

The returned handle can be used with ``cancel_event()``, ``reschedule_event()``,
``reprioritize_event()``, ``is_event_scheduled()``, ``event_time()``, and
``event_priority()``. These handle operations map directly to Cimba's native
event queue.

``schedule_native()`` is the advanced path for Cython/native extensions that
already have a ``cmb_event_func`` pointer in a ``cimba.event_func`` capsule. It
keeps callback dispatch entirely native; callers are responsible for native
pointer lifetimes.

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
