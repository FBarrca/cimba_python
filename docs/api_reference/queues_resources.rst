Queues and Resources
====================

.. py:data:: cimba.UNLIMITED

   Capacity sentinel for buffers and queues with no practical size limit.

.. py:class:: cimba.Buffer(name, capacity=None)

   Numeric fixed-capacity buffer with blocking put/get semantics.

   .. py:attribute:: name
   .. py:attribute:: capacity
   .. py:attribute:: level
   .. py:attribute:: space

   .. py:method:: put(amount=1)

      Put an amount into the buffer, waiting for space if needed. Returns
      ``(signal, remaining)``.

   .. py:method:: get(amount=1)

      Get an amount from the buffer, waiting for content if needed. Returns
      ``(signal, obtained)``.

   .. py:method:: start_recording()
   .. py:method:: stop_recording()
   .. py:method:: history()
   .. py:method:: close()

.. py:class:: cimba.ObjectQueue(name, capacity=None)

   FIFO queue for Python objects.

   .. py:attribute:: name
   .. py:attribute:: capacity
   .. py:attribute:: length
   .. py:attribute:: space

   .. py:method:: put(obj)
   .. py:method:: get()
   .. py:method:: position(obj)
   .. py:method:: start_recording()
   .. py:method:: stop_recording()
   .. py:method:: history()
   .. py:method:: close()

.. py:class:: cimba.PriorityQueue(name, capacity=None)

   Priority queue for Python objects.

   .. py:attribute:: name
   .. py:attribute:: capacity
   .. py:attribute:: length
   .. py:attribute:: space

   .. py:method:: put(obj, priority=0)
   .. py:method:: get()
   .. py:method:: position(handle)
   .. py:method:: cancel(handle)
   .. py:method:: reprioritize(handle, priority)
   .. py:method:: start_recording()
   .. py:method:: stop_recording()
   .. py:method:: history()
   .. py:method:: close()

.. py:class:: cimba.Resource(name)

   Binary semaphore resource.

   .. py:attribute:: name
   .. py:attribute:: in_use
   .. py:attribute:: available

   .. py:method:: acquire()
   .. py:method:: preempt()
   .. py:method:: release()
   .. py:method:: held_by(process)
   .. py:method:: start_recording()
   .. py:method:: stop_recording()
   .. py:method:: history()
   .. py:method:: close()

.. py:class:: cimba.ResourcePool(name, capacity)

   Counting semaphore resource pool.

   .. py:attribute:: name
   .. py:attribute:: capacity
   .. py:attribute:: in_use
   .. py:attribute:: available

   .. py:method:: acquire(amount=1)
   .. py:method:: preempt(amount=1)
   .. py:method:: release(amount=1)
   .. py:method:: held_by(process)
   .. py:method:: start_recording()
   .. py:method:: stop_recording()
   .. py:method:: history()
   .. py:method:: close()

.. py:class:: cimba.Condition(name)

   Condition variable for arbitrary Python predicates.

   .. py:method:: wait(predicate, context=None)

      Wait until ``predicate(process, context)`` returns true.

   .. py:method:: signal()

      Evaluate waiting predicates and reactivate those that are true.

   .. py:method:: subscribe(*sources, on=None)

      Forward native resource-guard signals from resources, buffers, or queues
      to this condition. ``on=None`` observes the single guard for
      :class:`Resource` and :class:`ResourcePool`, and both ``"front"`` and
      ``"rear"`` guards for :class:`Buffer`, :class:`ObjectQueue`, and
      :class:`PriorityQueue`.

   .. py:method:: unsubscribe(*sources, on=None)

      Stop forwarding native signals from the given sources. Returns the number
      of subscriptions removed.

   .. py:method:: cancel(process)

      Remove ``process`` from this condition and wake it with
      :data:`cimba.CANCELLED`.

   .. py:method:: remove(process)

      Remove ``process`` from this condition without waking it.

   .. py:method:: close()
