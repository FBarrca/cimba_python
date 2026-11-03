Entities
========

Entities are the native objects processes synchronize through. Declare them
as fields; each trial gets its own fresh copies. All entity methods are called
from compiled code.

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Entity
     - Think of it as…
     - Typical use
   * - :class:`~cimba.Container`
     - a counter of identical units
     - customers waiting (when identity doesn't matter), work in progress,
       stock levels
   * - :class:`~cimba.FractionalContainer`
     - a level of a divisible quantity
     - fluid in a tank, material in a silo, energy in a battery
   * - :class:`Store[T] <cimba.Store>`
     - a FIFO queue of values
     - jobs, parts, orders, messages
   * - :class:`PriorityStore[T] <cimba.PriorityStore>`
     - a queue ordered by priority
     - priority customers, lines people can leave
   * - :class:`~cimba.Resource`
     - a pool of *n* units to borrow
     - servers, machines, berths, tugs, licences
   * - :class:`~cimba.Condition`
     - a place to wait for "something changed"
     - complex readiness rules
   * - :class:`~cimba.Dataset`
     - a list of samples
     - per-customer times, per-order sizes

Container
---------

A counted level. ``put(n)`` adds ``n`` units, ``get(n)`` removes ``n``.
Amounts are integers. ``get`` takes what is there and blocks until the rest
arrives.

.. list-table::
   :widths: 35 65

   * - ``put(n)`` / ``get(n)``
     - Add / remove ``n`` units (``get`` blocks while the level is too low).
   * - ``level()``
     - Current level.
   * - ``mean_level()``
     - Time-weighted mean level over the measurement window.
   * - ``max_level()``
     - Maximum level over the measurement window.

A container starts empty, or with ``Container(initial=n)`` units in every
trial. For a level that depends on parameters, ``put`` it in an
``@cb.on_start`` hook instead.

Passing a ``float`` amount is a compile-time error rather than a silent
truncation; use a :class:`~cimba.FractionalContainer` for fractional amounts.

FractionalContainer
-------------------

A level of a divisible quantity, with the same methods as ``Container`` but
``float`` amounts: ``put(0.75)``, ``get(1.2)``, and a ``float`` ``level()``.
It starts empty or with ``FractionalContainer(initial=x)``, and holds at most
``capacity`` (unlimited by default):

.. code-block:: python

   silo: cb.FractionalContainer = cb.FractionalContainer(initial=12_000.0,
                                                         capacity=30_000.0)

``get(x)`` behaves as it does on a ``Container``: it takes what is there and
waits for the rest, so ``get(5.0)`` with 3.0 present takes the 3.0 and blocks
until 2.0 more has been put. ``put(x)`` is the mirror image: it fills the
container to ``capacity`` and waits for room for the rest.

Waiting processes of both container kinds are served in priority order, then
first come, first served, with one difference from a ``Resource``: a ``get``
or ``put`` that has moved part of its amount and waits again rejoins the line
behind the waiting requests of the same priority. Two equal-priority ``get``
calls waiting for more than arrives at once therefore take turns at the
arriving units rather than the first finishing before the second starts.

The level is kept in fixed point, with ``decimals`` decimal places (6 by
default, so milligrams when the unit is a kilogram). Each amount is rounded
to that many places once, half away from zero, as it goes in or out; the level
itself is never rounded. So the books always balance exactly: the level is the initial level
plus everything put in minus everything taken out. ``put(0.3)`` followed by
three ``get(0.1)`` calls leaves exactly ``0.0``, and ten thousand ``get(0.1)``
calls drain a ``put(1000.0)`` to exactly ``0.0``, where plain ``float``
arithmetic would leave a residue or wait for one that is missing.

``put`` and ``get`` return the amount actually added or obtained. This is the
amount rounded to ``decimals`` places, or less if the wait was interrupted:
an interrupted ``put`` or ``get`` keeps what it already moved. Add up these
return values to keep running totals that match the level.

Amounts must be finite and nonnegative; anything else fails the trial, with
a reason such as ``plant.silo: put(-1) is not a finite nonnegative amount``.
The level can count up to 2\ :sup:`64` quanta, about 1.8e13 units with 6
decimals; reaching that fails the trial. Use fewer ``decimals`` if you need
larger levels, or more (up to 12) for finer amounts.

Store and PriorityStore
-----------------------

A store holds **values**, typed by its annotation: ``Store[int]``,
``Store[float]`` or ``Store[SomeModel]``. Model-valued stores hold handles to
model instances, so ``get()`` returns a full view of the stored model.
Putting the wrong type in is a compile-time error.

.. list-table::
   :widths: 40 60

   * - ``put(value)``
     - Append (blocks while the store is at capacity).
   * - ``get()``
     - Remove and return the oldest item (blocks while empty).
   * - ``length()``
     - Items waiting (stores of models).
   * - ``Store(capacity=n)``
     - Configure a finite capacity (default: effectively unlimited).

A ``PriorityStore`` adds a priority (an integer, higher first; ties are
FIFO):

.. list-table::
   :widths: 40 60

   * - ``put(value, priority=0)``
     - Insert with a priority.
   * - ``get()``
     - Remove and return the highest-priority, longest-waiting item.
   * - ``enqueue(model, priority)``
     - Insert and return a **ticket** (stores of models).
   * - ``position(ticket)``
     - Items ahead of this ticket.
   * - ``cancel(ticket)``
     - Remove the ticketed item. Returns ``False`` if it was already taken.

Resource
--------

A pool of identical units, ``Resource(capacity=n)`` (default 1). The
capacity can be zero and can change during a trial.

.. list-table::
   :widths: 35 65

   * - ``acquire(n=1, priority=None)``
     - Take ``n`` units, waiting in priority order. Returns a signal.
   * - ``preempt(n=1, priority=None)``
     - Take ``n`` units, taking them from lower-priority holders if needed.
       A preempted holder loses *all* its units and sees signal ``-1``.
   * - ``release(n=1)``
     - Return ``n`` units.
   * - ``held(process)``
     - Units currently held by a process handle.
   * - ``available()``
     - Free units, ``0`` while the units in use are at or above the capacity.
   * - ``capacity()``
     - The current capacity.
   * - ``set_capacity(n, *, mode="finish")``
     - Change the capacity (see :ref:`concepts-resource-capacity`).
   * - ``mean_in_use()``
     - Time-weighted mean of units in use over the measurement window.
   * - ``mean_capacity()``
     - Time-weighted mean capacity over the measurement window.
   * - ``utilization()``
     - ``mean_in_use() / mean_capacity()``; ``nan`` if the window offered no
       capacity.

.. _concepts-resource-order:

The ordering rules:

* **Waiting.** Requests are served highest priority first, then first come,
  first served. A request that can't be fully served yet takes what is free
  and keeps waiting at the front, blocking the requests behind it even if
  they would fit. Nothing bypasses it: each time it waits again it keeps its
  original place in line, so a later request of equal priority is never
  served before it.
* **Preempting.** ``preempt`` takes units only from holders with *strictly*
  lower priority than the request: the lowest first, and among equals the
  one whose holding started most recently. Then it waits like ``acquire``
  for whatever is still missing.
* **Holding.** A process's units in a resource are one holding, ranked at
  the priority of its latest ``acquire`` or ``preempt``. An interrupted call
  keeps the holding's previous amount and priority.

By default (``priority=None``) a request and its holding are ranked at the
process priority, and :meth:`Process.priority_set <cimba.Process.priority_set>`
re-ranks both. ``priority=p`` ranks them at ``p`` instead, and
``priority_set`` leaves them alone. This separates the two jobs of process
priority: the request priority decides who is served and who is preempted,
while the process priority still decides which of several events at the same
instant runs first, including the wakeup that grants the request.

.. code-block:: python

   RESUME = 10

   @cb.process
   def work(self):
       ...
       if self.machine.acquire(priority=RESUME) == 0:  # ahead of fresh work
           ...

.. _concepts-resource-capacity:

Changing the capacity
~~~~~~~~~~~~~~~~~~~~~

``set_capacity(n, mode=...)`` models shifts, breaks and staffing calendars.
Raising the capacity grants waiting requests at once, in queue order.
Lowering it takes one of two string-literal modes:

* ``"finish"`` (the default): holders keep their units and finish their
  work. The units in use may exceed ``n`` until enough are released; nothing
  new is granted until they drop below ``n``.
* ``"preempt"``: holders are preempted in the ``preempt`` victim order
  (lowest holding priority, most recent first) until the units in use are at
  most ``n``. Victims lose all their units and see ``-1``, as with
  ``preempt``; the calling process is never its own victim.

Capacity statistics follow the window like the in-use history, so
``utilization()`` measures use against the capacity that was actually
offered, not against the peak. A calendar is an ordinary process; see
:ref:`policies-shift-calendar`.

.. _concepts-acquire-any:

One unit from any of several resources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``cb.acquire_any((self.clerks, self.seniors))`` takes one unit from
whichever resource grants first and returns its index in the tuple. A free
unit is taken at once, from the first listed resource that has one.
Otherwise the request waits in *every* resource's queue, at the place an
``acquire`` made now would get there, so it competes with direct acquirers
by the same ordering rules. The first resource to grant serves it, and the
request leaves the other queues at that moment.

.. code-block:: python

   role = cb.acquire_any((self.clerks, self.seniors))
   if role >= 0:
       cb.hold(self.service.next())
       (self.clerks, self.seniors)[role].release()

The unit is an ordinary holding of that resource: release it with that
resource's ``release()``; ``preempt`` can take it. ``priority=`` ranks the
request as for ``acquire``. A negative return value is the signal that ended
the wait (``-1`` preempted elsewhere, ``-2`` interrupted, a timer's signal),
and leaves no request behind in any queue.

Use ``acquire_any`` when a task needs *one unit of any of several pools*.
For any other rule ("two clerks, or one senior", "the least busy pool"),
wait on a :ref:`Condition that observes the resources
<concepts-condition-observe>` and acquire once it holds.

Condition
---------

``wait_until(predicate, timeout=math.inf)`` waits for a ``@cb.predicate``
method to hold, a timeout, or an interruption, and returns whether the
predicate holds. ``signal()`` re-evaluates all waiters. See :doc:`processes`.

.. _concepts-condition-observe:

Observing entities
~~~~~~~~~~~~~~~~~~

``Condition(observe=...)`` makes a condition signal itself whenever an
observed ``Resource``, ``Store``, ``PriorityStore``, ``Container`` or
``FractionalContainer`` changes, as if ``signal()`` were called right after
every release, grant, capacity change, put and get:

.. code-block:: python

   class Office(cb.Model):
       clerks: cb.Resource = cb.Resource(capacity=2)
       seniors: cb.Resource = cb.Resource(capacity=1)
       staffed: cb.Condition = cb.Condition(observe=(clerks, seniors))

       @cb.predicate
       def team_free(self) -> bool:     # two clerks, or one senior
           return self.clerks.available() >= 2 or self.seniors.available() >= 1

Observation is structure: it is set when the model is built and fixed for
the experiment. Pass entities of the same model tree, which the experiment
checks. In ``__init__``, observe your own or your children's entities
(``cb.Condition(observe=(self.north.counter, self.south.counter))``); for
an owner's entities, call ``condition.observe(owner.pool)`` in
:meth:`~cimba.Model.attached`. Each model instance gets its own copies of
class-default entities, so the class-level form above observes each
instance's own resources.

A waiter woken this way still competes for what it saw: re-check, or
acquire, after ``wait_until`` returns. Changes to plain fields still need a
``signal()``.

Dataset
-------

A tally of samples, recorded explicitly.

.. list-table::
   :widths: 35 65

   * - ``record(x)``
     - Add a sample.
   * - ``sample_mean()``, ``sample_count()``, ``sample_max()``
     - Summaries of the samples recorded so far.

Statistics and the measurement window
-------------------------------------

Entities measure only what happens in the **measurement window**:

* **Time-weighted** statistics (container level, resource use, store
  lengths) are recorded from ``warmup`` to ``warmup + duration``.
* **Datasets** are cleared when the window opens, at ``warmup``. Samples
  recorded later, *including during the cooldown*, are kept. That lets
  in-flight work that finishes during the cooldown be counted, as in the
  park tutorial.

With ``Window.until_idle()`` recording starts at time 0 and continues to the
end.

Capturing raw histories
-----------------------

Summaries are computed inside the trial and are cheap. When you need the raw
data, call ``capture()`` on the entity object before the run:

.. code-block:: python

   model.queue = cb.Container()
   model.queue.capture()
   ...
   times, levels = results[model].queue.trial(point, replication)

.. list-table::
   :header-rows: 1

   * - Captured entity
     - ``times``
     - ``values``
   * - ``Container``
     - change times
     - level after each change
   * - ``Resource``
     - change times
     - units in use
   * - ``Store`` / ``PriorityStore``
     - change times
     - queue length
   * - ``Dataset``
     - empty
     - every recorded sample

Histories cover the measurement window. Captures keep every sample of every
trial in memory, so capture only what you need.
