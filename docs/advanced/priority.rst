Priority: Processes, Resources, Events, and Queues
====================================================

The word "priority" means four *different*, independent things in
``cimba.sim``, one per mechanism. They share a sign convention -- a higher
signed 64-bit integer always means higher priority, and ties always fall
back to arrival order (FIFO) -- but setting one does not set the others.
This page collects all four in one place so the distinction (and the
similarity) is easy to check.

.. list-table::
   :header-rows: 1

   * - Mechanism
     - Set with
     - Compared against
     - Tie-break
   * - Process priority
     - ``@sim.process(priority=n)``, ``sim.set_priority(process, n)``
     - other processes waiting on the same resource/pool
     - arrival order
   * - Event priority
     - ``env.<event>.schedule(delay, priority=n)``, ``.reprioritize(n)``
     - other events due at the same simulated time
     - scheduling order
   * - Priority-queue priority
     - ``env.<pqueues>[i].put(obj, priority)``, ``.reprioritize(entry, n)``
     - other objects in the same priority queue
     - put order
   * - (no separate resource/pool priority)
     - --
     - uses the *process's* priority, see below
     - --

All four default to ``0`` and accept any ``int``, including negative
values, so "push to the very back" is just as expressible as "push to the
front."

Process priority: resource and pool queueing
---------------------------------------------

A process's own priority governs how it queues for a ``sim.Resource`` or
``sim.Pool``. Set it once at registration or change it at runtime:

.. code-block:: python

   class Clinic(sim.Model):
       @sim.process(priority=5)
       def vip_patient(self: "Clinic"):
           ...

       @sim.process
       def regular_patient(self: "Clinic"):
           me = sim.current()
           sim.set_priority(me, -2)
           ...

When several processes are waiting on the same ``.acquire()``, the
highest-priority waiter goes first; among waiters with equal priority,
whoever asked first is served first. Changing a process's priority while it
is already waiting moves it to its new place in that order.

``.preempt()`` uses the same numbers to decide who can be *displaced*, not
just who waits longest:

* ``sim.Resource.preempt()`` takes the resource from the current holder only
  if the caller's priority is strictly higher than the holder's; otherwise it
  falls back to waiting politely in the normal queue.
* ``sim.Pool.preempt(amount)`` starts by taking capacity from the
  *lowest*-priority holder, working forward, and stops as soon as it would
  have to take from a holder with equal or higher priority than the caller
  -- any shortfall waits normally.

Cimba avoids priority inversion in the basic case (a run of lower-priority
waiters cannot starve a higher-priority one indefinitely), but it does not
reorder the queue to let a request further back jump ahead just because it
could be satisfied sooner (e.g. three units free, the front waiter wants
five, and three waiters behind it each want one). If that kind of
skip-ahead matters for your model, adjust priorities explicitly with
``sim.set_priority()`` to bring the right process to the front.

See :doc:`components` for the acquire/release/preempt walkthrough and
:doc:`../tutorial` (:ref:`tut_2`) for a stress-tested example with both
polite (``acquire()``) and aggressive (``preempt()``) agents competing for
one pool.

Event priority: same-time ordering
-----------------------------------

``env.<event>.schedule()``/``.schedule_at()`` take a ``priority=`` keyword,
and the handle they return has its own ``.reprioritize(priority)``:

.. code-block:: python

   class Clinic(sim.Model):
       log_snapshot: sim.Event
       close_shift: sim.Event

       @sim.event(field="log_snapshot")
       def record_snapshot(self: "Clinic"):
           ...

       @sim.event(field="close_shift")
       def handle_close(self: "Clinic"):
           ...

       @sim.process
       def driver(self: "Clinic"):
           self.log_snapshot.schedule(0.0, priority=-100)  # run last today
           self.close_shift.schedule(480.0, priority=10)   # run first at t=480

Event priority only matters when two or more events are due at *exactly* the
same simulated time: the higher-priority one fires first, and among equal
priorities, whichever was scheduled first fires first. A common use is
making sure a "close the day" event runs before other things scheduled for
the same instant, or pushing a diagnostic/logging event to run last (a large
negative priority) so it observes everyone else's final state.

See :doc:`events_timers_signals` for the full scheduling API.

Priority-queue priority: explicit ordering, independent of the caller
------------------------------------------------------------------------

A ``sim.PQueues`` element's priority is a value *you* choose per object, with
no link to the priority of whichever process happened to put it there:

.. code-block:: python

   GOLD_CARD_PRIORITY = 5

   class Attraction(sim.Component):
       @sim.process
       def visitor(self, env, vip: Visitor):
           priority = GOLD_CARD_PRIORITY if vip.gold_card else 0
           entry = env.ride_queue[0].put(sim.current(), priority)

Objects come out highest-priority first; among equal priorities, whoever
was put in earliest comes out first. ``.reprioritize(entry, priority)``
moves an existing entry (used for jockeying: reconsidering a wait and
asking for a better spot), and ``.position(entry)`` reports an entry's
current 1-based rank so a process can decide whether it is worth waiting
or switching lines.

See :doc:`../tutorial` (:ref:`tut_3`) for the full balking/jockeying/reneging
park example, which uses priority queues exactly this way for a
"gold card" fast lane.

Keeping the four straight
--------------------------

Because all four mechanisms share the same "bigger number wins, ties go to
whoever arrived first" rule, it is tempting to assume they are the same
knob. They are not: a process's priority does not automatically become the
priority of an event it schedules or an object it puts in a priority queue
-- if you want that coupling (e.g. a VIP process's own priority should also
jump its requests to the front of a ride queue), pass it through
explicitly, as in the gold-card example above.
