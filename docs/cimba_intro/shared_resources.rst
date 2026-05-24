Shared Resources
================

Buffers
-------

A :class:`cimba.Buffer` stores numeric amounts and supports blocking ``put`` and
``get`` operations.

.. code-block:: python

   def producer(queue):
       while True:
           cimba.hold(1.0)
           queue.put(1)

   def consumer(queue):
       while True:
           queue.get(1)
           cimba.hold(0.8)

If a process tries to get from an empty buffer, it waits until enough content is
available. If a finite-capacity buffer is full, a producer waits for space.

Object queues
-------------

:class:`cimba.ObjectQueue` stores Python objects in FIFO order.
:class:`cimba.PriorityQueue` stores Python objects ordered by priority and
returns a handle that can be used to cancel or reprioritize queued items.

Resources
---------

:class:`cimba.Resource` is a binary semaphore. :class:`cimba.ResourcePool` is a
counting semaphore.

.. code-block:: python

   def job(server):
       server.acquire()
       try:
           cimba.hold(3.0)
       finally:
           server.release()

   with cimba.Simulation(seed=123) as sim:
       server = cimba.Resource("Server")
       cimba.Process("Job 1", job, server).start()
       cimba.Process("Job 2", job, server).start()
       sim.execute()

Conditions
----------

:class:`cimba.Condition` lets a process wait on a Python predicate. When another
process calls ``signal()``, Cimba evaluates waiting predicates and resumes the
processes whose predicates are true.

.. code-block:: python

   def enough_items(process, ctx):
       return ctx["items"] >= 3

   def waiter(ctx):
       ctx["condition"].wait(enough_items, ctx)

   def supplier(ctx):
       while ctx["items"] < 3:
           cimba.hold(1.0)
           ctx["items"] += 1
           ctx["condition"].signal()
