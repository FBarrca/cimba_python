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

``wait_until(predicate)`` blocks until the condition is signalled *and*
``predicate`` (a ``@cb.predicate`` method) returns true. ``signal()``
re-evaluates all waiters. See :doc:`processes`.

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
