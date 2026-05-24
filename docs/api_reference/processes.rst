Processes
=========

.. py:function:: cimba.hold(duration)

   Suspend the current process for ``duration`` simulated time units.

.. py:function:: cimba.yield_process()

   Yield the current process until another process, timer, or event resumes it.

.. py:function:: cimba.wait_event(handle)

   Yield the current process until the scheduled event fires or is canceled.
   Returns ``SUCCESS`` when the event fires and ``CANCELLED`` when it is
   canceled.

.. py:function:: cimba.process_exit(value=None)

   Exit the current process with an optional Python exit value while unwinding
   Python ``finally`` blocks normally.

.. py:function:: cimba.current_process()

   Return the currently running :class:`cimba.Process`, or ``None`` outside
   process execution.

.. py:class:: cimba.Process(name, target, /, *args, priority=0, pass_process=False, **kwargs)

   Named stackful Cimba process executing a Python callable. The callable is
   called as ``target(*args, **kwargs)``. When ``pass_process`` is true, it is
   called as ``target(process, *args, **kwargs)``.

   .. py:attribute:: name

      Process name used by Cimba logging and diagnostics.

   .. py:attribute:: priority

      Current process priority used in waiting and resource queues.

   .. py:attribute:: status

      Current process lifecycle state.

   .. py:method:: start()

      Schedule the process to start at the current simulation time and return
      the process object.

   .. py:method:: stop()

      Request cooperative cancellation of a running Python-backed process.

   .. py:method:: interrupt(signal=INTERRUPTED, priority=0)

      Interrupt a waiting process with a non-success signal.

   .. py:method:: resume(signal=SUCCESS)

      Schedule a yielded process to resume with the given signal.

   .. py:method:: wait()

      Wait until this process finishes, returning the wakeup signal.

   .. py:method:: timer_add(duration, signal=TIMEOUT)

      Add an independent timer that resumes this process after ``duration``.

   .. py:method:: timer_set(duration, signal=TIMEOUT)

      Clear existing timers and set one timer for this process.

   .. py:method:: timer_cancel(handle)

      Cancel a timer by handle.

   .. py:method:: timers_clear()

      Cancel all timers currently scheduled for this process.

   .. py:method:: exit_value()

      Return the Python value produced by a finished process, if any.

   .. py:method:: close()

      Release the native process and any owned Python exit value.

Signals
-------

.. py:data:: cimba.SUCCESS
.. py:data:: cimba.PREEMPTED
.. py:data:: cimba.INTERRUPTED
.. py:data:: cimba.STOPPED
.. py:data:: cimba.CANCELLED
.. py:data:: cimba.TIMEOUT

Lifecycle states
----------------

.. py:data:: cimba.PROCESS_CREATED
.. py:data:: cimba.PROCESS_RUNNING
.. py:data:: cimba.PROCESS_FINISHED
