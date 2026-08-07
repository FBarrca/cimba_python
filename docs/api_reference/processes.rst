Processes and Events
====================

A process is a Python method declared with ``@sim.process`` on a model or
component class. Only one process in a trial runs at a time;
when it calls a blocking ``sim`` operation, control returns to the dispatcher
until the process is ready to resume.

Process verbs
-------------

``hold()``, ``now()``, ``current()``, ``interrupt()``, ``stop()``,
``wait_process()``, ``wait_event()``, ``resume()``, ``suspend()``,
``status()``, ``set_priority()``, ``timer_set()``, ``timer_add()``,
``timer_cancel()``, ``timers_clear()``, ``spawn()``, ``despawn()``.

Blocking calls return a signal. ``sim.SUCCESS`` means the operation completed;
signals such as ``sim.PREEMPTED``, ``sim.INTERRUPTED``, ``sim.STOPPED``,
``sim.CANCELLED``, and ``sim.TIMEOUT`` let a process decide how to clean up and
what to do next.

Dynamic processes
-----------------

A process decorated with ``spawnable=True`` is created at runtime with
``sim.spawn(env.<name>, env, priority=0)``. The returned handle can be used to
initialize its ``sim.Struct`` fields before it first runs. Finished spawned
processes can be reclaimed with ``sim.despawn(handle)``. Component-owned
spawnables use the same call through the component namespace, for example
``sim.spawn(env.flow.visitor, env)``.

Low-level events
----------------

Callbacks declared with ``@sim.event`` are exposed in ``sim.Event`` fields.
Use ``field="..."`` when the callback and field have different names.
``env.<event>.schedule(delay, data=0, priority=0)`` and
``.schedule_at(at, ...)`` return a scheduled-instance handle with its own
``.cancel()``, ``.reschedule(at)``, ``.reprioritize(priority)``,
``.scheduled()``, ``.time()``, ``.priority()``, and ``.wait_event()``
methods. ``sim.current_event()``, ``sim.event_count()``, and
``sim.clear_events()`` remain free functions (they have no single event to
act as a receiver).

When several events share the same time, the higher-priority event runs first;
ties at the same priority run in arrival order. See :doc:`../advanced/priority`
for how this compares to process, resource/pool, and priority-queue priority.
