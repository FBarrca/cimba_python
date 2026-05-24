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

   .. py:attribute:: current_event

      Handle of the currently or most recently executed event.

   .. py:method:: stop_at(when, priority=0)

      Schedule cooperative cancellation of owned Python processes, then stop
      the run at absolute simulation time ``when``.

   .. py:method:: schedule(callback, when, subject=None, obj=None, priority=0)

      Schedule ``callback(subject, obj)`` at absolute simulation time ``when``.
      This is a Python-dispatch event; native queue ordering and handle lookup
      still stay in C.

   .. py:method:: schedule_native(action_capsule, when, subject_capsule=None, object_capsule=None, priority=0)

      Schedule a native ``cmb_event_func`` from a ``cimba.event_func`` capsule.
      The caller owns any native subject/object pointer lifetimes.

   .. py:method:: cancel_event(handle)

      Cancel a scheduled event by handle and return whether it was found.

   .. py:method:: reschedule_event(handle, when)

      Move a scheduled event to another absolute simulation time.

   .. py:method:: reprioritize_event(handle, priority)

      Change a scheduled event's priority.

   .. py:method:: is_event_scheduled(handle)

      Return whether the event handle is still scheduled.

   .. py:method:: event_time(handle)

      Return the scheduled absolute time for an event.

   .. py:method:: event_priority(handle)

      Return the scheduled priority for an event.

   .. py:method:: clear()

      Clear scheduled events, ending the current run.

   .. py:method:: execute_next()

      Execute one event and return ``False`` if the event queue is empty.

   .. py:method:: execute()

      Run until the event queue is empty.

   .. py:method:: close()

      Cooperatively cancel owned running processes and release Cimba's
      thread-local state.
