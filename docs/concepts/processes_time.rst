Processes and Simulated Time
============================

A process is an active entity in the simulated world. In ``cimba.sim``, a
process is an ordinary Python function registered with ``@model.process``.
Inside that function, ``sim`` calls such as ``sim.hold()``, ``sim.get()``, and
``sim.acquire()`` can pause the process and let another scheduled activity run.

.. code-block:: python

   import cimba.random as random

   @model.process
   def doctor(env: Clinic):
       while True:
           sim.get(env.waiting_room, 1)
           service_time = random.exponential(env.mean_service)
           sim.hold(service_time)
           env.served += 1

There is no ``yield`` in the process body. If the waiting room is empty,
``sim.get()`` blocks the doctor process until an arrival puts a patient in the
queue. When the process resumes, execution continues immediately after the
blocking call.

Simulated time, not wall-clock time
-----------------------------------

``sim.hold(duration)`` advances the process in simulated time. It does not
sleep the operating-system thread for that many seconds.

.. code-block:: python

   started = sim.now()
   sim.hold(15.0)
   elapsed = sim.now() - started

The dispatcher always runs the next scheduled event in time order. A trial may
simulate hours, days, or years while the real program takes far less wall-clock
time.

Blocking is the modeling language
---------------------------------

Most process interactions are expressed by blocking on a model entity:

* ``sim.hold()`` waits for simulated time.
* ``sim.get()`` waits for enough content in a ``sim.Queue``.
* ``sim.acquire()`` waits for a ``sim.Resource``.
* ``sim.pool_acquire()`` waits for capacity in a ``sim.Pool``.
* ``sim.wait_for()`` waits until a ``sim.Condition`` predicate is true.

When the operation can complete, the process resumes. Blocking calls return a
signal value, so a process can react to interruption, timeout, cancellation, or
normal success when the model needs that detail.

Process copies
--------------

Use ``copies=`` when the model has several identical active entities:

.. code-block:: python

   @model.process(copies=3)
   def clerk(env: Clinic, idx: int):
       while True:
           sim.get(env.waiting_room, 1)
           sim.hold(random.exponential(env.mean_service))
           env.served += 1

The second argument receives the copy index. Use it when each copy needs a
stable number for routing, logging, or separate state. If the copies are truly
interchangeable, a ``sim.Pool`` may be a better fit than multiple process
copies.

Process code should stay focused
--------------------------------

Process bodies are compiled for the hot simulation loop. Keep them mostly to
numeric control flow, ``env`` fields, and ``sim`` calls. Do setup, plotting,
dataframe work, and rich Python object manipulation outside the trial run.

For a complete model that starts simple and then adds stopping, logging,
resources, and richer behavior, see :ref:`the tutorial <tutorial>`.
