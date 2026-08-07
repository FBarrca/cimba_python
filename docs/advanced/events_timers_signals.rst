Events, Timers, and Signals
===========================

Most models can be written with process blocking calls. Explicit events,
process timers, and signal handling are useful when a model needs direct
wakeups, cancellation, deadlines, or scheduled callbacks.

Explicit events
---------------

Declare a ``sim.Event`` field and bind a class-declared ``@sim.event``
callback:

.. code-block:: python

   class Clinic(sim.Model):
       close_shift: sim.Event
       arrivals: sim.Processes
       closed: sim.State

       @sim.event(field="close_shift")
       def on_close_shift(self: "Clinic"):
           self.closed = 1
           sim.stop(self.arrivals[0], 0)

       @sim.process(field="arrivals")
       def arrival_loop(self: "Clinic"):
           while True:
               sim.hold(1.0)

       @sim.process
       def supervisor(self: "Clinic"):
           self.close_shift.schedule(480.0)
           sim.suspend()


   model = Clinic("clinic")

``env.<event>.schedule()`` uses a delay from the current time.
``.schedule_at()`` uses an absolute simulation time. Both return a
scheduled-instance handle with its own methods: ``.cancel()``,
``.reschedule()``, ``.reprioritize()``, ``.scheduled()``, ``.time()``,
``.priority()``, and ``.wait_event()``. When two events are due at the exact
same simulated time, ``priority=`` (default 0) decides which fires first,
ties going to whichever was scheduled first; see :doc:`priority` for how
this compares to process, resource/pool, and priority-queue priority.

Waiting on scheduled events
---------------------------

An event can be used as a deadline that another process waits for:

.. code-block:: python

   class Clinic(sim.Model):
       close_shift: sim.Event
       closed: sim.State

       @sim.event(field="close_shift")
       def on_close_shift(self: "Clinic"):
           self.closed = 1

       @sim.process
       def reminder(self: "Clinic"):
           handle = self.close_shift.schedule(480.0)
           sig = handle.wait_event()
           if sig == sim.SUCCESS:
               self.closed = 1

If the event is cancelled before it fires, ``.wait_event()`` returns a
non-success signal. Check the signal when cancellation changes the model path.

Process timers
--------------

Timers wake a specific process. They are a natural fit for impatience,
timeouts, appointment no-shows, and retry deadlines:

.. code-block:: python

   TIMER_PATIENCE = 17


   class Clinic(sim.Model):
       @sim.process(spawnable=True)
       def patient(self: "Clinic", p: Patient):
           me = sim.current()
           sim.timer_set(me, p.patience, TIMER_PATIENCE)
           sig = sim.suspend()
           if sig == TIMER_PATIENCE:
               # The patient waited too long.
               return
           sim.timers_clear(me)
           # The patient was resumed by service before the timer fired.

``sim.timer_set()`` clears existing timers before adding one. ``sim.timer_add()``
adds another independent timer. ``sim.timer_cancel()`` cancels one timer handle,
and ``sim.timers_clear()`` clears all timers for a process.

Signals and cleanup
-------------------

Blocking calls return signals. ``sim.SUCCESS`` means the operation completed
normally. Other values can indicate timeout, interruption, stop, cancellation,
or preemption:

.. code-block:: python

   import cimba.random as random

   sig = env.doctor.acquire()
   if sig != sim.SUCCESS:
       return

   try:
       sig = sim.hold(random.exponential(env.mean_service))
       if sig == sim.SUCCESS:
           env.completed += 1
   finally:
       if env.doctor.held(sim.current()):
           env.doctor.release()

Treat every blocking call as a possible handoff point. Another process may
interrupt this process, stop it, preempt held capacity, or resume it with a
domain-specific signal before it runs again.

Use explicit events and timers when they make the model rule clearer. If a
normal ``sim.hold()``, queue operation, resource acquire, or condition wait
expresses the rule directly, prefer the simpler blocking operation.

For process fundamentals, see :doc:`../concepts/processes_time`.
