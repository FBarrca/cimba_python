Queues and Resources
====================

Numeric buffers
---------------

Use :class:`cimba.Buffer` when the state is a numeric amount: inventory level,
queue length, fluid quantity, or tokens.

``put(amount)`` returns ``(signal, remaining)``. ``get(amount)`` returns
``(signal, obtained)``. On ``SUCCESS``, the remaining amount is zero and the
obtained amount equals the requested amount.

Object queues
-------------

Use :class:`cimba.ObjectQueue` when each queued item has identity or data:
customers, jobs, packets, orders, or requests.

``get()`` returns ``(signal, object_or_none)``. On interruption, the object may
be ``None``.

Priority queues
---------------

Use :class:`cimba.PriorityQueue` when queued objects need priority ordering.
``put()`` returns a handle:

.. code-block:: python

   signal, handle = queue.put(order, priority=order.priority)
   queue.reprioritize(handle, priority=10)

The handle remains meaningful only while the item is still queued.

Resources and resource pools
----------------------------

Use :class:`cimba.Resource` for one server and :class:`cimba.ResourcePool` for a
capacity of more than one.

Both support acquisition, release, preemption, and time-series recording.
Release from the same process that acquired the resource.

Conditions
----------

Use :class:`cimba.Condition` when the waiting condition is model-specific and
cannot be expressed as a queue or resource operation.

Because the predicate is Python code, keep it side-effect free. A predicate
should inspect state and return ``True`` or ``False``.
