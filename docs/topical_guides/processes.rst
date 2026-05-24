Processes
=========

Process functions
-----------------

A process target is a normal Python callable. Positional and keyword arguments
passed to :class:`cimba.Process` are forwarded to the target.

.. code-block:: python

   def customer(model):
       model.server.acquire()
       cimba.hold(model.service_time())
       model.server.release()

   cimba.Process("Customer", customer, model).start()

Bound methods are often the cleanest way to keep model state close to model
behavior:

.. code-block:: python

   cimba.Process("Customer", model.customer).start()

Use ``pass_process=True`` only when the target needs the
:class:`cimba.Process` object itself:

.. code-block:: python

   def sleeper(me, log):
       me.timer_set(5.0)
       log.append(cimba.yield_process())

   cimba.Process("Sleeper", sleeper, log, pass_process=True).start()

The process can return a value. Another process can retrieve it with
``exit_value()`` after waiting for the process.

Scheduling and lifecycle
------------------------

``Process(...).start()`` schedules the process at the current simulation time
and returns the process object.

The lifecycle constants are:

* ``PROCESS_CREATED``
* ``PROCESS_RUNNING``
* ``PROCESS_FINISHED``

Process priorities
------------------

Priorities are integer values used by waiting and resource queues. A higher
priority process is considered before a lower priority process where the native
object supports priority ordering.

.. code-block:: python

   urgent = cimba.Process("Urgent", job, model, priority=10).start()
   urgent.priority = 20

Timers and yielding
-------------------

``hold(duration)`` is the most common way to yield. ``yield_process()`` yields
without scheduling a time wakeup; another process, event, or timer must resume
the yielded process.

``timer_add()`` schedules an independent wakeup. ``timer_set()`` clears existing
timers before adding one.
