.. _py_tut_2:

Acquiring, Preempting, Holding, and Releasing Resources
=======================================================

This tutorial mirrors the second C tutorial: first the resource concepts, then
process signals, then a small mouse/rat/cat stress test showing preemption and
interruption.

Resources and Resource Pools
----------------------------

Cimba Python provides the same two resource concepts:

* :class:`cimba.Resource` is a binary semaphore. One process may hold it.
* :class:`cimba.ResourcePool` is a counting semaphore. Several processes may
  hold units from a fixed capacity.

The ordinary pattern is identical to C:

.. code-block:: python

   assert server.acquire() == cimba.SUCCESS
   try:
       cimba.hold(service_time)
   finally:
       server.release()

For a resource pool:

.. code-block:: python

   assert tugs.acquire(3) == cimba.SUCCESS
   cimba.hold(0.5)
   tugs.release(3)

If a process requests more units than the pool capacity, the binding raises an
error from the native assertion path. If the request is valid but unavailable,
the process waits in priority order until the units can be acquired or until it
is interrupted, stopped, or preempted.

Preemptions and Interruptions
-----------------------------

Calls that may yield return a signal. The most important values are:

* ``cimba.SUCCESS``: the call completed normally.
* ``cimba.PREEMPTED``: a higher-priority process took a held resource.
* ``cimba.INTERRUPTED``: a generic interruption signal.
* ``cimba.STOPPED``, ``cimba.CANCELLED``, and ``cimba.TIMEOUT`` for other
  native process outcomes.

Positive integer signals are available to the application:

.. code-block:: python

   def cat(target):
       cimba.hold(0.5)
       target.interrupt(77)


   def waiting_mouse(cheese):
       sig = cheese.acquire(1)
       if sig == 77:
           print("the cat interrupted this wait")

Preemption uses process priorities. A higher-priority process can call
``preempt()`` to acquire a resource by taking it from lower-priority holders:

.. code-block:: python

   def mouse(me, resource):
       assert resource.acquire() == cimba.SUCCESS
       sig = cimba.hold(10.0)
       if sig == cimba.PREEMPTED:
           assert resource.held_by(me) == 0


   def rat(me, resource):
       cimba.hold(1.0)
       me.priority = 10
       assert resource.preempt() == cimba.SUCCESS
       resource.release()

Targets like ``mouse`` and ``rat`` need ``pass_process=True`` when started
because they inspect or update the :class:`cimba.Process` object.

Buffers and Object Queues, Interrupted
--------------------------------------

Buffers and queues are not preempted in the same sense as resources, but a
waiting put or get can be interrupted. The return values carry the partial
state:

.. code-block:: python

   sig, obtained = buffer.get(10)
   if sig != cimba.SUCCESS:
       print(f"only got {obtained} before interruption")

For object queues and priority queues, interrupted gets return ``(signal,
None)`` and interrupted puts return a non-success signal.

While the Cat Is Away...
------------------------

The C tutorial uses a lively model of mice acquiring cheese, rats preempting it,
and a cat interrupting waiting rodents. The Python tutorial keeps the same
semantics in compact, deterministic functions that are easy to test:

.. literalinclude:: ../../tutorial/tut_2_1.py
   :language: python

The important lesson is the same as in C: any call that may yield can return
with a signal other than ``SUCCESS``. Model code should decide what those
signals mean and update its own Python state accordingly.

Real World Uses
---------------

Preemption and interruption are useful in models where priority changes matter:
emergency-room triage, machine breakdowns, operating-system scheduling,
transportation disruptions, and manufacturing job shops. In Python these
interactions use the same native Cimba scheduler and resource guards as the C
API, with Python objects carrying the model-specific state.
