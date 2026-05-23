Core
====

.. py:function:: cimba.native_version()

   Return the version string reported by the bundled Cimba C library.

.. py:function:: cimba.time()

   Return the current simulation clock time for the active event queue.

.. py:class:: cimba.Simulation(start_time=0.0, seed=None, log_info=False)

   Own the thread-local Cimba event queue, simulation clock, and random
   generator. Use it as a context manager.

   .. py:attribute:: seed_used

      Seed used to initialize the PRNG stream.

   .. py:attribute:: closed

      Whether native simulation state has been released.

   .. py:attribute:: now

      Current simulation time.

   .. py:attribute:: event_count

      Number of scheduled future events.

   .. py:method:: stop_at(when, priority=0)

      Schedule the run to stop at absolute simulation time ``when``.

   .. py:method:: clear()

      Clear scheduled events, ending the current run.

   .. py:method:: execute_next()

      Execute one event and return ``False`` if the event queue is empty.

   .. py:method:: execute()

      Run until the event queue is empty.

   .. py:method:: close()

      Stop owned processes and release Cimba's thread-local state.
