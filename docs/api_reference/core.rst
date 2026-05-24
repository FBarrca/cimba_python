Core
====

.. py:function:: cimba.native_version()

   Return the version string reported by the bundled Cimba C library.

.. py:function:: cimba.run_experiment(trial_fn, n=None, *, seed=None, seeds=None, backend="process", processes=None)

   Run independent replications of ``trial_fn(index, seed)`` and return results
   indexed by replication.

   The default ``backend="process"`` uses forked worker processes and is
   recommended for Python-defined simulations. Return pickleable values such as
   floats, tuples, or dictionaries. ``processes`` controls the worker count and
   defaults to :class:`multiprocessing.Pool` behavior.

   ``backend="thread"`` uses Cimba's native pthread worker pool inside one
   Python process. It can return in-process native Cimba objects and only
   parallelizes on free-threaded Python builds; GIL-enabled interpreters run the
   trials serially with a warning.

.. py:function:: cimba.run_native_experiment(experiment_buffer, trial_struct_size, trial_func_capsule)

   Run native Cimba pthread replications over a writable C-contiguous trial
   buffer using a ``cimba.trial_func`` capsule. Python callables are rejected.

.. py:function:: cimba.set_native_thread_hooks(init_capsule=None, user_arg_capsule=None, exit_capsule=None)

   Set native pthread init/user-context/exit hook capsules, or clear them by
   calling with no arguments.

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

      Schedule cooperative cancellation of owned Python processes, then stop
      the run at absolute simulation time ``when``.

   .. py:method:: clear()

      Clear scheduled events, ending the current run.

   .. py:method:: execute_next()

      Execute one event and return ``False`` if the event queue is empty.

   .. py:method:: execute()

      Run until the event queue is empty.

   .. py:method:: close()

      Cooperatively cancel owned running processes and release Cimba's
      thread-local state.
