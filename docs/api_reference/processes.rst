Processes and Events
====================

A process is a Python function registered with ``@model.process`` (or a
component ``@sim.process`` method). Only one process in a trial runs at a time;
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

A process named in a ``sim.Spawnable`` field is created at runtime with
``sim.spawn(env.<name>, env, priority=0)``. The returned handle can be used to
initialize its ``sim.Struct`` fields before it first runs. Finished spawned
processes can be reclaimed with ``sim.despawn(handle)``. Component-owned
spawnables use the same call through the component namespace, for example
``sim.spawn(env.flow.visitor, env)``.

Low-level events
----------------

Callbacks registered with ``@model.event`` are exposed in ``sim.Event`` fields.
Use ``schedule()``, ``schedule_at()``, ``event_cancel()``,
``event_reschedule()``, ``event_reprioritize()``, ``event_scheduled()``,
``event_time()``, ``event_priority()``, ``current_event()``,
``event_count()``, and ``clear_events()``.

When several events share the same time, the higher-priority event runs first;
ties at the same priority run in arrival order.
