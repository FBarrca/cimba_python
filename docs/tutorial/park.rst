.. _tut_3:

3. An amusement park
====================

So far the "customers" in our models were anonymous: a number in a container.
Many questions need individuals. How long did *this* visitor spend in the
park? Did gold-card holders wait less? How many people gave up and left a
line? For that, each visitor needs its own state and its own behavior. And
visitors come and go during the day, so they can't be built up front like the
rest of the model.

This chapter introduces **dynamic models**: model instances created with
``cb.spawn`` while a trial runs and removed with ``cb.release``. Along the way
we meet priority queues you can leave, process **timers**, and
``suspend``/``resume``.

The scenario, from the C library's tutorial 3:

* Visitors arrive at the park entrance and wander between **nine
  attractions**, choosing the next one from a transition matrix and walking a
  PERT-distributed time between them.
* Each attraction has one or more **priority lines**, served by batch-loading
  ride servers. A quarter of visitors hold **gold cards** and queue at a higher
  priority.
* Each visitor has a **patience** factor that drives three behaviors:

  - **balking**: refusing to join if the shortest line is too long;
  - **jockeying**: switching to a shorter line after waiting a while;
  - **reneging**: giving up and walking away after waiting too long.

The full script is ``tutorial/tut_3_1.py``. Here is its process graph. The
park spawns visitors; each visitor may join, and leave, any of the eleven ride
lines and records its day in the park's datasets. Each line feeds its ride
servers.

.. cimba-diagram:: tutorial.tut_3_1:Park
   :direction: LR

.. contents:: In this chapter
   :local:
   :depth: 1

A visitor is a model too
------------------------

There is exactly one concept for "a thing with data and behavior" in Cimba
Python, and that is :class:`~cimba.Model`. A visitor is declared the same way
as the park:

.. literalinclude:: ../../tutorial/tut_3_1.py
   :pyobject: Visitor
   :lines: 1-11

What makes it *dynamic* is how it is created. The park's arrival process
spawns one per arriving guest:

.. literalinclude:: ../../tutorial/tut_3_1.py
   :pyobject: Park.arrivals
   :dedent: 4

``cb.spawn(Visitor, field=value, ...)`` allocates a new ``Visitor`` for this
trial, sets its fields from the keyword arguments, runs its ``on_start``
hooks, and schedules its processes to start at the current time. It returns a
handle to the new model, so you can put it in a store or set more fields.

The keyword arguments must cover every field without a default. Here that's
the ``park`` reference and the ``patience``, ``priority`` and ``entry_park``
states. Fields with a class default, like ``rides = 0``, take it.

.. admonition:: What a dynamic model may contain
   :class: note

   Spawned models hold values and behavior: ``Param``, ``State``,
   ``Output``, constants, ``Ref`` fields, inputs, processes and hooks. They
   can't own entities or child models. Put shared queues and resources on a
   static model and give the spawned model a ``Ref`` to it, as ``Visitor``
   does with ``park``.

   To give a spawned model an input, pass one of the spawner's inputs:
   ``cb.spawn(Order, lead_time=self.lead_time)``. The new model gets its own
   cursor and its own random stream, derived deterministically from the trial
   seed and the spawn order.

When the visitor reaches the exit, it records its statistics and releases
itself:

.. literalinclude:: ../../tutorial/tut_3_1.py
   :pyobject: Visitor.visit
   :lines: 70-75
   :dedent: 4

``cb.release(model)`` stops the model's processes. When a process releases its
own model, as here, that process ends at the call. Memory is reclaimed when
the trial ends.

.. note::

   At the end of the cooldown the window stops every process, including
   those of spawned models, so a spawned process that loops forever can't
   keep a trial alive. With ``Window.until_idle()`` there is no such stop:
   make sure spawned processes finish or block.

Lines you can leave: priority stores
------------------------------------

Each ride line is a :class:`~cimba.PriorityStore` of visitors:

.. literalinclude:: ../../tutorial/tut_3_1.py
   :pyobject: RideQueue

``cb.PriorityStore[Visitor]`` is typed: it holds ``Visitor`` handles, and
putting anything else in it is a *compile-time* error. ``get()`` returns the
highest-priority item that has waited longest, blocking while the store is
empty. So in the ride server, ``visitor`` is a full view of the visitor model,
and the server can update its ``waiting`` and ``riding`` state directly.

Joining a line with ``put(item, priority)`` is fire-and-forget. A visitor who
may *leave* the line needs a ticket, so model-valued priority stores offer
three more calls:

* ``ticket = line.enqueue(item, priority)`` joins and returns a ticket;
* ``line.position(ticket)`` tells how many are ahead;
* ``line.cancel(ticket)`` leaves the line. It returns ``False`` if the item
  was already taken.

Waiting with timers
-------------------

Here is the clever part of the visitor. After joining a line, it sets two
alarms on its own process and goes to sleep:

.. literalinclude:: ../../tutorial/tut_3_1.py
   :pyobject: Visitor.visit
   :lines: 36-68
   :dedent: 4

``me.timer_set(delay, signal)`` arranges for the process to be woken with
``signal`` after ``delay`` time units, unless something wakes it first.
``cb.suspend()`` blocks until *something* does: a timer, an interrupt or a
``resume``. It returns the signal, so the visitor knows why it woke:

* ``TIMER_JOCKEYING`` (17): look for a shorter line, and switch if it is
  shorter than our position in this one;
* ``TIMER_RENEGING`` (42): give up, leave the line, clear the remaining
  timers and walk on;
* anything else: the ride server resumed us after the ride.

On the other side, the ride server boards the visitor, cancels the visitor's
pending alarms and resumes it when the ride is over:

.. code-block:: python

   visitor_process = cb.Process(visitor.process_pointer)
   visitor_process.timers_clear()
   ...
   visitor_process.resume(0)

This pattern of *enqueue, arm timers, suspend, and let whoever finishes first
decide* is how you model patience, timeouts and service-level deadlines in
Cimba.

Several servers per line
------------------------

Some attractions run two or three ride vehicles off the same line. A process
can run in several **copies**, each an independent coroutine over the same
model:

.. literalinclude:: ../../tutorial/tut_3_1.py
   :pyobject: RideQueueTwo

``@cb.process(copies=2)`` starts two server processes for every
``RideQueueTwo``. Because ``RideQueueTwo`` subclasses ``RideQueue``, it
inherits all the fields and only overrides the process declaration. The park
builds a *heterogeneous* list, with some lines of each class:

.. literalinclude:: ../../tutorial/tut_3_1.py
   :pyobject: Park.__init__
   :dedent: 4

``ride_queues: list[RideQueue]`` accepts any subclass. In compiled code,
``self.park.ride_queues[index]`` is seen through the base class, so the
visitor can read ``attraction`` and ``line`` on every item regardless of its
subclass. Each class is compiled once, however many instances there are.

A park that closes
------------------

The park stops admitting visitors at closing time, but people already inside
should finish their day. That's what the **cooldown** is for:

.. literalinclude:: ../../tutorial/tut_3_1.py
   :pyobject: main
   :lines: 1-8

Arrivals stop at ``PARK_OPEN``. For the next 2,000 minutes of cooldown the
remaining visitors ride, walk and leave. Datasets keep collecting during the
cooldown, so every visitor's time in the park is recorded. Time-weighted
statistics, such as a resource's utilization, stop at the end of
``duration``.

.. code-block:: console

   $ uv run python tutorial/tut_3_1.py
   cimba 3.0.0; 9 attractions, 11 queues, 14 ride servers
   20 trials in 1.88 s
   n_visitors: 476.350
   avg_rides: 2.493
   avg_time_in_park: 58.854
   avg_riding: 16.821
   avg_waiting: 4.737
   avg_walking: 33.450
   n_balks: 0.000
   n_jockeys: 0.000
   n_reneges: 195.500

About 476 visitors a day, each riding two or three attractions and spending
most of their hour walking. Nobody balks or jockeys at these patience
settings, but reneging is common. That's a hint that some ride lines move
slowly relative to visitors' patience. Try raising ``ARRIVAL_RATE`` or
lowering ``RENEGING_THRESHOLD`` and watch the behaviors change.

What you learned
----------------

* ``cb.spawn(ModelClass, field=value, ...)`` creates a model during a trial;
  ``cb.release(model)`` removes it. Spawned models hold values, refs, inputs
  and behavior, but no entities or children.
* ``PriorityStore[M]`` holds typed model handles; ``enqueue``, ``position``
  and ``cancel`` let items leave a line.
* ``timer_set``, ``suspend``, ``resume`` and ``timers_clear`` express
  patience and deadlines.
* ``@cb.process(copies=n)`` runs several identical processes; subclasses can
  override processes; ``list[Base]`` holds mixed subclasses.
* The cooldown lets in-flight work finish after the measurement window.

In :doc:`harbor` we assemble a model from several cooperating sub-models and
learn to wait on arbitrary conditions.
