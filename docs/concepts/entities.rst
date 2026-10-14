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

A counted level. ``put(n)`` adds ``n`` units, ``get(n)`` removes ``n``,
blocking until that many are there. Amounts are integers.

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
container to ``capacity`` and waits for room for the rest. Waiting processes
are served in priority order, then first come, first served.

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

A pool of ``capacity`` identical units, ``Resource(capacity=n)`` (default 1).

.. list-table::
   :widths: 35 65

   * - ``acquire(n=1)``
     - Take ``n`` units, waiting in priority order. Returns a signal.
   * - ``preempt(n=1)``
     - Take ``n`` units, taking them from lower-priority holders if needed.
       A preempted holder loses *all* its units and sees signal ``-1``.
   * - ``release(n=1)``
     - Return ``n`` units.
   * - ``held(process)``
     - Units currently held by a process handle.
   * - ``available()``
     - Free units.
   * - ``mean_in_use()``
     - Time-weighted mean of units in use over the measurement window
       (divide by capacity for utilization).

Condition
---------

``wait_until(predicate, timeout=math.inf)`` waits for a ``@cb.predicate``
method to hold, a timeout, or an interruption, and returns whether the
predicate holds. ``signal()`` re-evaluates all waiters. See :doc:`processes`.

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
