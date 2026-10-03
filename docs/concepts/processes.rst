Time, processes and hooks
=========================

Simulated time
--------------

A trial has a clock that starts at 0 and an **event queue** of things
scheduled to happen at future times. The engine repeatedly takes the earliest
event and runs it: a process waking up, a scheduled event method, the window
opening or closing. Time jumps from event to event. Nothing happens
"between" events, so an hour of quiet costs nothing.

``cb.now()`` returns the current simulated time. Time units are whatever you
decide (minutes, hours, days), as long as you're consistent.

When several events fall at the same time, higher **priority** goes first;
equal priorities run in the order they were scheduled.

Processes
---------

A method decorated with ``@cb.process`` runs as a **process**: a coroutine
with its own stack, living in simulated time.

.. code-block:: python

   @cb.process                        # one copy, priority 0
   def arrivals(self): ...

   @cb.process(copies=3, priority=1)  # three identical servers, run first on ties
   def server(self): ...

* Static processes start at time 0, after every ``on_start`` hook.
* ``copies=n`` starts *n* independent processes over the same model.
* A process may run forever (``while True:``) or return. Returning ends that
  process, not the trial.
* Processes are stackful. A blocking call can appear anywhere, including
  inside a loop, a branch or a helper function the process calls. The
  process resumes exactly where it stopped, with its local variables intact.

What blocking looks like
------------------------

Here is how the first events of the M/M/1 model from :doc:`../tutorial/mm1`
unfold. Only one process runs at a time. A blocking call hands control back to
the event loop, which wakes whichever process is due next:

.. mermaid::

   sequenceDiagram
     participant L as event loop
     participant A as arrival
     participant S as service
     participant Q as queue
     L->>A: t = 0: start
     A->>L: hold(gap₁)
     L->>S: t = 0: start
     S->>Q: get(1)
     Note over S,Q: queue empty, so service blocks
     L->>A: t = t₁: wake
     A->>Q: put(1)
     Q-->>S: a customer is available: service resumes
     A->>L: hold(gap₂)
     S->>L: hold(service₁)
     L->>S: t = t₁ + service₁: wake
     S->>Q: get(1)
     Note over S,Q: blocks again unless another customer arrived

Blocking calls and signals
--------------------------

These calls may suspend the calling process:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Call
     - Suspends until…
   * - ``cb.hold(duration)``
     - ``duration`` time units have passed.
   * - ``container.get(n)`` / ``put(n)``
     - the container can satisfy the request.
   * - ``store.get()`` / ``store.put(value)``
     - an item is available / there is room.
   * - ``resource.acquire(n)`` / ``preempt(n)``
     - ``n`` units are granted.
   * - ``condition.wait_until(predicate, timeout=math.inf)``
     - the predicate holds, the timeout expires, or the process is interrupted.
   * - ``cb.suspend()``
     - something resumes, interrupts or times out the process.

``hold``, ``acquire``, ``preempt`` and ``suspend`` return an integer
**signal** that explains why the process resumed:

.. list-table::
   :header-rows: 1
   :widths: 12 20 68

   * - Signal
     - Name
     - Cause
   * - ``0``
     - success
     - Normal completion.
   * - ``-1``
     - preempted
     - A higher-priority process took this process's resource units.
   * - ``-2``
     - interrupted
     - ``Process.interrupt()`` was called on this process.
   * - ``-3``
     - stopped
     - The process was stopped.
   * - ``-4``
     - cancelled
     - The wait was cancelled.
   * - ``-5``
     - timeout
     - A timeout expired.
   * - ``> 0``
     - user-defined
     - A custom ``interrupt`` code, a ``resume(signal)`` or a
       ``timer_set(delay, signal)``.

Defining named module constants (``PREEMPTED = -1``) keeps models readable.

Hooks
-----

``@cb.on_start``
   Runs once per model instance before any process starts. Use it to set
   initial state that depends on parameters
   (``self.on_hand = self.initial_inventory``) or to put initial stock into
   entities.

``@cb.on_end``
   Runs once per model instance after the event loop finishes. Use it to read
   entity statistics into outputs.

For static models, hooks run **bottom-up**: children before their parents.
So a parent's ``on_end`` can aggregate outputs its children have just
computed. ``on_end`` hooks of dynamic models that are still alive run after
those of the static tree.

The trial lifecycle
-------------------

.. code-block:: text

   create entities
   on_start hooks (bottom-up)
   start processes at t = 0
   ─────────── event loop ───────────
     t = warmup                    open the window: reset entity statistics, clear datasets
     t = warmup + duration         close the window: stop time-weighted recording
     t = warmup + duration + cooldown   stop all processes (static and spawned)
     … remaining events drain …
   ──────────────────────────────────
   on_end hooks (bottom-up, then live dynamic models)
   copy captures and outputs, release everything

With ``cb.Window.until_idle()`` the window opens at 0 and the loop runs until
no events remain. :doc:`experiments` covers the window in detail.

Stopping early
--------------

``cb.end_trial()`` clears the event queue and ends the calling process. The
``on_end`` hooks still run and the trial counts as **successful**. Use it
for domain stop conditions such as "after the 1,000th customer".

Conditions and predicates
-------------------------

A :class:`~cimba.Condition` lets a process wait for any combination of
state. Two pieces are involved:

.. code-block:: python

   class Gate(cb.Model):
       ready: cb.State[bool] = False
       gate: cb.Condition

       @cb.predicate
       def is_ready(self):
           return self.ready

       @cb.process
       def waiter(self):
           self.gate.wait_until(self.is_ready)    # returns whether ready holds
           ...

       @cb.process
       def opener(self):
           cb.hold(2.0)
           self.ready = True
           self.gate.signal()                      # re-evaluate waiters now

* A **predicate** is a method decorated ``@cb.predicate`` that returns a
  bool. Pass it *without calling it*: ``wait_until(self.is_ready)``. A
  predicate on another model works too: ``wait_until(self.plant.is_open)``.
* ``signal()`` makes the condition re-evaluate its waiters' predicates and
  wake those that are true. Conditions don't watch state by themselves:
  whoever changes relevant state must signal.
* ``wait_until`` checks the predicate before blocking and returns a bool.
  Use ``if self.gate.wait_until(self.is_ready, timeout=5.0): ...`` to wait at
  most five time units.

Events
------

An **event** is a method decorated ``@cb.event`` that runs once at a
scheduled time, outside any process:

.. code-block:: python

   @cb.event
   def shift_change(self):
       self.shift += 1
       self.gate.signal()

   @cb.process
   def planner(self):
       handle = cb.schedule(self.shift_change, 8.0)   # 8 time units from now
       if handle.pending():
           handle.cancel()

``cb.schedule(event, delay, priority=0)`` returns a
:class:`~cimba.Scheduled` handle with ``pending()`` and ``cancel()``. Event
methods must not block.

Process handles and timers
--------------------------

``cb.this_process()`` returns a :class:`~cimba.Process` handle for the running
process. ``handle.pointer`` is an integer you can store in a ``State[int]``,
and ``cb.Process(pointer)`` rebuilds the handle elsewhere, for example in the
process that needs to wake it.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method
     - Effect
   * - ``status()``
     - ``0`` initialized, ``1`` running, ``2`` finished.
   * - ``priority_set(p)``
     - Change the process's priority.
   * - ``interrupt(signal, priority=0)``
     - Wake the process from its current blocking call, which returns
       ``signal``. Resources held are kept.
   * - ``resume(signal=0)``
     - Wake a process blocked in ``cb.suspend()``.
   * - ``timer_set(delay, signal)``
     - Wake this process with ``signal`` after ``delay``, unless it wakes
       earlier. Several timers may be pending.
   * - ``timers_clear()``
     - Cancel all of the process's pending timers.

The *enqueue, set timers, suspend, and see who wins* pattern, shown in
:doc:`../tutorial/park`, is how you model patience, deadlines and timeouts.

Dynamic models: spawn and release
---------------------------------

``cb.spawn(ModelClass, field=value, ...)`` creates a model during a trial:

1. memory for its record is allocated from the trial;
2. fields are set from keyword arguments, falling back to class defaults
   (every field without a default must be given);
3. input fields must be passed an input of the spawning model; the new model
   gets an independent cursor and a new random stream derived from the trial
   seed and the spawn order;
4. its ``on_start`` hooks run immediately;
5. its processes are scheduled to start at the current time;
6. a handle to the new model is returned.

``cb.release(model)`` stops the model's processes (ending the caller if it
releases its own model) and marks the model released. Its memory is
reclaimed when the trial ends.

``cb.is_dynamic(model)`` returns whether an instance was created by
``cb.spawn``. It returns ``False`` for static tree models and stays ``True``
for spawned models after release. Like ``spawn`` and ``release``, it is
available only in compiled model code.

Use it when the same class supports static and dynamic instances, without
maintaining a constructor flag. Track domain state separately to prevent
repeated release:

.. code-block:: python

   class Document(cb.Model):
       closed: cb.State[bool] = False

       @cb.function
       def close(self) -> None:
           if self.closed:
               return
           self.closed = True
           if cb.is_dynamic(self):
               cb.release(self)

Here, closing a static document only sets its ``closed`` state; its
processes must check that state themselves. Closing a spawned document
also stops its model processes.

.. note::

   At the end of the cooldown the window stops every process, spawned ones
   included. Under ``Window.until_idle()`` nothing is stopped, so spawned
   processes must finish or block for the trial to end.

Logging from compiled code
--------------------------

``cb.log(flags, "literal message", value=None)`` writes a line to the
engine's log, prefixed with the trial index, simulated time and process name.
``flags`` is a 32-bit category mask. Messages print only when their category
is enabled with ``cb.set_engine_log_level(mask)`` on the host (off by
default, so ``log`` calls can stay in your model). The message must be a
string literal.
