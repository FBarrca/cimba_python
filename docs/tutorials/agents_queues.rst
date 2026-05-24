.. _py_tut_3:

Agents Balking, Reneging, and Jockeying in Queues
=================================================

The third C tutorial moves from a queue length to active customers. Customers
can choose not to join a long queue, leave after losing patience, or switch to a
shorter queue. In Python, the same behavior is modeled with ordinary Python
objects plus :class:`cimba.Process` instances.

Process-Like Model Objects
--------------------------

The C version derives ``struct visitor`` from ``cmb_process``. Python does not
need that inheritance pattern. A visitor can be a normal class that stores a
reference to the process running it:

.. code-block:: python

   class Visitor:
       def __init__(self, name: str, patience: float = 1.0):
           self.name = name
           self.patience = patience
           self.entry_time_queue = 0.0
           self.riding_time = 0.0
           self.waiting_time = 0.0
           self.num_attractions_visited = 0
           self.status = "new"


   def visitor_proc(me, ctx):
       visitor = ctx["visitor"]
       visitor.process = me
       ...

This gives the same modeling freedom: the process is an active coroutine while
running, and the visitor object can also be placed into a queue and handled as a
passive object by a server. Start this target with ``pass_process=True`` because
it stores the process object on the visitor.

Servers and Priority Queues
---------------------------

The C server gets visitors from a priority queue, clears their timers, runs the
ride, and resumes them. The Python version is nearly the same:

.. code-block:: python

   def server(ctx):
       while True:
           sig, visitor = ctx["queue"].get()
           assert sig == cimba.SUCCESS
           visitor.process.timers_clear()
           visitor.waiting_time += cimba.time() - visitor.entry_time_queue
           cimba.hold(ctx["ride_duration"])
           visitor.riding_time += ctx["ride_duration"]
           visitor.process.resume(cimba.SUCCESS)

The resume does not directly run the target process inline. It schedules a
wakeup through the dispatcher, preserving Cimba's asymmetric coroutine model.

Setting and Clearing Timers
---------------------------

Jockeying and reneging are modeled with process timers. ``timer_set`` clears
existing timers and sets one new timer; ``timer_add`` adds another independent
timer:

.. code-block:: python

   TIMER_JOCKEYING = 17
   TIMER_RENEGING = 42

   me.timer_set(visitor.patience, TIMER_JOCKEYING)
   me.timer_add(10.0 * visitor.patience, TIMER_RENEGING)

   while True:
       sig = cimba.yield_process()
       if sig == TIMER_JOCKEYING:
           ...
       elif sig == TIMER_RENEGING:
           ...
       else:
           assert sig == cimba.SUCCESS
           ...

When the visitor joins a :class:`cimba.PriorityQueue`, ``put()`` returns a
handle. The handle is used to check the visitor's current position, cancel the
queue entry when reneging, or move the visitor to another queue:

.. code-block:: python

   sig, handle = queue.put(visitor, priority=me.priority)
   assert sig == cimba.SUCCESS

   if new_queue.length < queue.position(handle):
       assert queue.cancel(handle)
       queue = new_queue
       sig, handle = queue.put(visitor, priority=me.priority + 1)
       assert sig == cimba.SUCCESS

The complete compact demonstration is in ``tutorial/tut_3_1.py``:

.. literalinclude:: ../../tutorial/tut_3_1.py
   :language: python

Alias Sampling Probabilities
----------------------------

The full C amusement-park tutorial uses Vose alias sampling to choose a next
attraction from transition probabilities. Python exposes the same native idea as
:class:`cimba.AliasSampler`:

.. code-block:: python

   with cimba.AliasSampler([0.1, 0.7, 0.2]) as quo_vadis:
       next_attraction = quo_vadis.sample()

For occasional one-shot draws, use :func:`cimba.loaded_dice`:

.. code-block:: python

   next_attraction = cimba.loaded_dice([0.1, 0.7, 0.2])

A Day in the Park
-----------------

The C tutorial builds a complete amusement park with multiple attractions,
queues, servers, walking times, balking thresholds, and detailed statistics.
The Python API has the pieces needed for the same model: processes, priority
queues, timers, object queues, datasets, :func:`cimba.pert` random variates,
and alias sampling. The checked-in Python tutorial intentionally keeps the
model small so the queue/timer/resume mechanics remain visible.

For the full model, collect visitor metrics in :class:`cimba.DataSummary`
objects and queue histories with :func:`cimba.reporting.resource_report`, matching
the C tutorial's summary lines and detailed queue reports:

.. code-block:: python

   num_rides = cimba.DataSummary()
   time_in_park = cimba.DataSummary()

   # In the departure process:
   num_rides.add(visitor.num_attractions_visited)
   time_in_park.add(cimba.time() - visitor.arrival_time)

   print(cimba.reporting.summarize(num_rides))
   print(cimba.reporting.format_report(cimba.reporting.resource_report(queue)))

Parallelizing the full park model would follow the same pattern as
:ref:`py_tut_1`: write one function that builds and runs a complete park day,
return plain Python metrics, and call :func:`cimba.run_experiment`.
