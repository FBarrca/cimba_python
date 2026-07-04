Events, Timers, and Signals
===========================

Most models can be written with process blocking calls. Explicit events,
process timers, and signal handling are useful when a model needs direct
wakeups, cancellation, deadlines, or scheduled callbacks.

Explicit events
---------------

Declare a ``sim.Event`` field and register a matching ``@model.event``
callback:

.. code-block:: python

   class Clinic(sim.Model):
       close_shift: sim.Event
       arrivals: sim.Processes
       closed: sim.State


   model = Clinic("clinic")


   @model.event
   def close_shift(env: Clinic):
       env.closed = 1
       sim.stop(env.arrivals[0], 0)


   @model.process
   def supervisor(env: Clinic):
       sim.schedule(env.close_shift, env, 480.0)
       sim.suspend()

``sim.schedule()`` uses a delay from the current time. ``sim.schedule_at()``
uses an absolute simulation time. Scheduled event handles can be cancelled,
rescheduled, reprioritized, inspected, and waited on.

Waiting on scheduled events
---------------------------

An event can be used as a deadline that another process waits for:

.. code-block:: python

   @model.process
   def reminder(env: Clinic):
       handle = sim.schedule(env.close_shift, env, 480.0)
       sig = sim.wait_event(handle)
       if sig == sim.SUCCESS:
           env.closed = 1

If the event is cancelled before it fires, ``sim.wait_event()`` returns a
non-success signal. Check the signal when cancellation changes the model path.

Process timers
--------------

Timers wake a specific process. They are a natural fit for impatience,
timeouts, appointment no-shows, and retry deadlines:

.. code-block:: python

   TIMER_PATIENCE = 17


   @model.process
   def patient(env: Clinic, p: Patient):
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

   sig = sim.acquire(env.doctor)
   if sig != sim.SUCCESS:
       return

   try:
       sig = sim.hold(sim.exponential(env.mean_service))
       if sig == sim.SUCCESS:
           env.completed += 1
   finally:
       if sim.held(env.doctor, sim.current()):
           sim.release(env.doctor)

Treat every blocking call as a possible handoff point. Another process may
interrupt this process, stop it, preempt held capacity, or resume it with a
domain-specific signal before it runs again.

Use explicit events and timers when they make the model rule clearer. If a
normal ``sim.hold()``, queue operation, resource acquire, or condition wait
expresses the rule directly, prefer the simpler blocking operation.

For process fundamentals, see :doc:`../concepts/processes_time`.
