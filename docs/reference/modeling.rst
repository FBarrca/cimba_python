Modeling API
============

.. currentmodule:: cimba

Everything on this page is imported from ``cimba``, conventionally as
``import cimba as cb``. Calls marked *compiled* work only inside compiled
model methods; on the host they raise ``cimba.modeling.NotInCompiledCode``.

Model
-----

.. class:: Model

   Base class of every model. Subclass it and declare fields with
   annotations and behavior with decorated methods. Constructing an instance
   has no side effects: nothing is compiled or run until an experiment runs.

   Assigning attributes is validated: a :func:`sweep` is accepted only by
   ``Param``, ``Input``, ``Series`` and child-model fields. ``Input``/``Series``
   fields accept only sources or sweeps of sources; a child sweep must
   contain models of the field's type.

   .. method:: describe()

      Return a tuple with one dict per model instance in the tree:
      ``{"label": path, "class": name, "fields": ((name, kind), ...)}``.

Field types
-----------

.. class:: Param[T]

   A trial-constant value of type ``float``, ``int`` or ``bool``. It can be
   set on the class, the instance, or as a sweep.

.. class:: State[T]

   A mutable per-trial value. Static models need an initial value.

.. class:: Output[T]

   A per-trial result, initially ``nan`` (``float``) or ``0``. Collected as
   :class:`Samples`.

.. class:: Ref[M]

   A reference to another model instance of class ``M`` (or a subclass).
   Write ``cb.Ref[M] | None`` for an optional reference. Forward references
   by string are allowed: ``cb.Ref["Harbor"]``.

.. class:: Input[T]

   A sequence input of ``float``, ``int`` or ``bool``. Bind a source on the
   class (default) or the instance.

   .. method:: next()

      *Compiled.* Return the next value and advance the cursor. If the source
      is exhausted, the exhaustion policy applies.

   .. method:: remaining()

      *Compiled.* Values left in a finite source; ``-1`` for a distribution.

.. class:: Series(*, step, origin=0.0)

   A time-indexed input. Declare it as
   ``x: cb.Series[float] = cb.Series(step=1.0, origin=0.0)``; ``step`` must be
   positive. Time *t* maps to bucket ``floor((t - origin) / step)``.

   .. method:: now()

      *Compiled.* The value of the bucket containing the current time.

   .. method:: at(time)

      *Compiled.* The value of the bucket containing ``time`` (not before
      ``origin``).

Other annotations: a model class (child; it may be assigned a
:func:`sweep` of model objects, giving one model tree per option),
``list[M]`` (owned children),
``list[cb.Ref[M]]`` (references, items may be ``None``), and a plain
``float``/``int``/``bool`` (constant). Unannotated attributes are host-only.

Entities
--------

Entity methods are *compiled*, except ``capture()`` and the constructors.

.. class:: Container(initial=0)

   A counted level with integer amounts. Each trial starts with ``initial``
   units (a nonnegative integer).

   .. method:: put(amount)
   .. method:: get(amount)

      Add or remove ``amount`` units; ``get`` blocks until enough are present.

   .. method:: level()
   .. method:: mean_level()
   .. method:: max_level()

      Current level; time-weighted mean and maximum over the measurement
      window.

   .. method:: capture()

      *Host.* Keep the level history; read as a :class:`Signal`.

.. class:: Store(*, capacity=1_000_000_000)

   A FIFO store of values, declared ``Store[int]``, ``Store[float]`` or
   ``Store[M]`` for a model class ``M``.

   .. method:: put(value)

      Append ``value``; blocks while the store is full. For ``Store[M]``,
      ``value`` must be a model of class ``M`` (checked at compile time).

   .. method:: get()

      Remove and return the oldest item; blocks while empty.

   .. method:: length()

      Number of items (model-valued stores).

   .. method:: capture()

      *Host.* Keep the length history.

.. class:: PriorityStore(*, capacity=1_000_000_000)

   A store ordered by integer priority, higher first, FIFO among equals.

   .. method:: put(value, priority=0)
   .. method:: get()
   .. method:: length()
   .. method:: enqueue(value, priority=0)

      Insert and return a ticket (model-valued stores).

   .. method:: position(ticket)

      Number of items ahead of the ticket.

   .. method:: cancel(ticket)

      Remove the ticketed item; ``False`` if it was already taken.

.. class:: Resource(*, capacity=1)

   A pool of ``capacity`` identical units.

   .. method:: acquire(amount=1)

      Take units, blocking in priority order. Returns a signal.

   .. method:: preempt(amount=1)

      Take units from lower-priority holders if necessary. Preempted holders
      lose all their units and see signal ``-1``. Returns a signal.

   .. method:: release(amount=1)
   .. method:: held(process)

      Units held by a :class:`Process` handle.

   .. method:: available()
   .. method:: mean_in_use()

      Time-weighted mean units in use over the window.

   .. method:: capture()

      *Host.* Keep the in-use history.

.. class:: Condition()

   .. method:: wait_until(predicate, timeout=math.inf) -> bool

      Return immediately if ``predicate`` (a ``@predicate`` method, passed
      uncalled) holds; otherwise wait until it holds after a signal, a timeout
      expires, or the process is interrupted. Return whether it holds when
      the wait ends. ``timeout`` is nonnegative simulation time: infinity
      means no deadline and zero polls without blocking. The wait cancels
      only its own timer before returning.

   .. method:: signal()

      Re-evaluate waiting processes' predicates and wake those that hold.

.. class:: Dataset()

   .. method:: record(value)
   .. method:: sample_mean()
   .. method:: sample_count()
   .. method:: sample_max()

      Samples recorded since the window opened (including the cooldown).

   .. method:: capture()

      *Host.* Keep every sample.

Decorators
----------

.. decorator:: process
               process(*, copies=1, priority=0)

   Run the method as a process of every instance: ``copies`` independent
   coroutines, with scheduling ``priority``. Static processes start at time
   0; processes of spawned models start at the spawn time.

.. decorator:: on_start

   Run once per instance before processes start (static models: bottom-up;
   spawned models: at spawn).

.. decorator:: on_end

   Run once per instance after the event loop (static models bottom-up, then
   live spawned models).

.. decorator:: predicate

   A method returning ``bool``, for :meth:`Condition.wait_until`.

.. decorator:: event

   A method run at a time set with :func:`schedule`. It must not block.

.. decorator:: function

   A method callable from compiled code as ``model.name(args)``. Every
   parameter is annotated ``float``, ``int``, ``bool`` or a model class, and
   the return type is one of those or ``None``. Scalar parameters may have
   defaults; arguments are positional. Subclasses may override it with the
   same signature (and the decorator). Calls dispatch on the model's actual
   class, so a base-typed reference or list item runs its subclass's
   override. Functions may block. On the host it is an ordinary method. The
   name must not be a field name or a name reserved by :class:`Model`.

Verbs
-----

All verbs are *compiled*.

.. function:: hold(duration)

   Suspend the calling process for ``duration`` time units. Returns a signal.

.. function:: now()

   Current simulated time.

.. function:: suspend()

   Suspend until resumed, interrupted or woken by a timer. Returns the signal.

.. function:: spawn(ModelClass, **fields)

   Create a dynamic model instance, set its fields, run its ``on_start``
   hooks, schedule its processes, and return a handle to it. Fields without
   a default must be supplied; input fields take an input of the caller.

.. function:: release(model)

   Stop a spawned model's processes and retire it.

.. function:: is_dynamic(model)

   Return ``True`` if this instance was created by :func:`spawn`, or
   ``False`` for a static model in the configured tree. Remains ``True``
   after :func:`release`: this reports creation mode, not whether the model
   is still active. Accepts model instances only.

.. function:: schedule(event, delay, priority=0)

   Run the ``@event`` method ``event`` after ``delay`` (≥ 0). Returns a
   :class:`Scheduled`.

.. function:: this_process()

   A :class:`Process` handle for the calling process.

.. function:: end_trial()

   Stop the trial now: clear pending events and end the caller. ``on_end``
   hooks run and the trial succeeds.

.. function:: log(flags, message, value=None)

   Write ``message`` (a string literal) and an optional number to the engine
   log if a category in ``flags`` is enabled.

Handles
-------

.. class:: Process(pointer)

   A handle to a process. ``cb.Process(pointer)`` rebuilds a handle from its
   integer ``pointer`` attribute.

   .. attribute:: pointer
   .. method:: status()

      ``0`` initialized, ``1`` running, ``2`` finished.

   .. method:: priority_set(priority)
   .. method:: interrupt(signal=-1, priority=0)

      Wake the process from its current blocking call, which returns
      ``signal``. Use ``-2`` for "interrupted" or a positive custom code.

   .. method:: resume(signal=0)
   .. method:: timer_set(delay, signal)

      Wake the process with ``signal`` after ``delay`` unless it wakes first.

   .. method:: timers_clear()

.. class:: Scheduled

   Returned by :func:`schedule`.

   .. method:: pending()
   .. method:: cancel()

      Cancel if pending; returns whether it was cancelled.

Engine utilities (host)
-----------------------

.. function:: engine_version()

   Version string of the bundled Cimba engine.

.. function:: set_engine_log_level(flags)

   Enable the log categories in the 32-bit mask ``flags`` (``0`` disables
   all) for subsequent runs.

.. function:: cache_info()
              clear_cache()

   Inspect or clear the in-process cache of compiled classes.

.. data:: __version__

   The Cimba Python version.
