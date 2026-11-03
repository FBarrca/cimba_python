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
   ``Param``, ``Input``, ``Series`` and child-model fields. A field typed with
   an ``enum.IntEnum`` accepts members, or an ``int`` that is one of the
   enum's values (stored as that member); anything else raises
   ``ModelDefinitionError``. ``Input``/``Series``
   fields accept sources or sweeps of sources; optional inputs also accept
   ``None``. A child sweep must contain models of the field's type.

   .. method:: describe()

      Return a tuple with one dict per model instance in the tree:
      ``{"label": path, "class": name, "fields": ((name, kind), ...)}``.

   .. method:: attached(owner, field)

      Host-only hook, called after this model is assigned to an owned child
      field or list. Every child sweep option receives the hook. ``owner``
      is the parent model, ``field`` its field name. The default does nothing;
      override it to bind a ``Ref`` to the owner. Reference assignments do
      not call it. Class-default children are copied and attached once per
      owner during assembly. Reassign a list to attach newly added children.

Field types
-----------

A plain annotated ``float``, ``int``, ``bool`` or ``enum.IntEnum``
(``on_hand: float = 0.0``) is a per-trial value: it starts at its value in every trial and compiled code
may change it. A static model needs a value, on the class or the instance.
The wrappers below give a field a role on top of that.

.. class:: Param[T]

   A trial-constant value of type ``float``, ``int``, ``bool`` or an
   ``enum.IntEnum``. It can be set on the class, the instance, or as a sweep.
   An enum value is stored natively as the member's integer value and reads
   as the member in compiled code and on the host. Decisions require
   ``float``, ``int`` or ``bool``.

.. class:: Output[T]

   A per-trial value that is also a result: its value at the end of the
   trial is collected as :class:`Samples`. It starts each trial at its
   default, if it has one
   (``revenue: cb.Output[float] = 0.0``), otherwise at ``nan`` (``float``) or
   ``0`` (``int``, ``bool``, ``enum.IntEnum``), so processes can count into
   it directly. An enum output is collected as its integer values.

.. class:: Ref[M]

   A reference to another model instance of class ``M`` (or a subclass).
   Write ``cb.Ref[M] | None`` for an optional reference. Forward references
   by string are allowed: ``cb.Ref["Harbor"]``.

   *Compiled:* ``model.field = other`` points it at another model of class
   ``M`` or a subclass, static or spawned; another class is a compile error.
   An optional reference may be assigned ``None``. An optional value assigned
   to a required reference fails the trial if it is ``None``, as does using a
   ``None`` reference as a model.

   ``M`` may also be an entity class: ``cb.Ref[cb.Resource]``,
   ``cb.Ref[cb.Store[int]]`` and so on for every entity kind, and
   ``list[cb.Ref[cb.Resource]]`` for a list. Assign it an entity that a model
   of the same tree holds (``pool.resources = [role.res for role in
   roles]``). When the experiment is built, a value that is not such an
   entity, an entity of another tree (or of a swept option the tree doesn't
   contain), or one held by a field of another type (a ``Store[float]`` for a
   ``cb.Ref[cb.Store[int]]``) raises :class:`ModelDefinitionError` naming the
   field; the items of an entity list can't be ``None``.

   *Compiled:* a referenced entity is the entity itself: all its methods,
   indexing and ``len()`` of a list, with bounds checks as for any list. It
   can be assigned another entity of its type, and passed to
   :func:`spawn`.

.. class:: Array(contents=0)

   A per-instance table of ``int``, ``float`` or ``bool``, declared as
   ``x: cb.Array[int]``. ``contents`` is a nonnegative length (that many
   zeros, or ``False``) or a list of initial values. Assign it as a class
   default or on the host; it may be a :func:`sweep` of arrays. A spawned
   model takes an ``int`` length as its spawn argument, or uses the class
   default.

   *Compiled:* ``a[i]``, ``a[i] = v``, ``a[i] += v``, ``len(a)`` and
   ``for x in a``. An index outside ``0 .. len(a) - 1`` fails the trial. A
   released spawned model's arrays are freed: their length is ``0``. Arrays
   are not results.

.. class:: Input[T]

   A sequence input of ``float``, ``int`` or ``bool``. Bind a source on the
   class (default) or the instance.
   ``Input[T] | None`` allows no binding: ``remaining()`` is zero, while
   ``next()`` fails the trial.

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
and ``list[cb.Ref[M]]`` (references, items may be ``None``; for an entity
class ``M``, entity references, items may not be ``None``). Unannotated
attributes are host-only.

Entities
--------

Entity methods are *compiled*, except ``capture()`` and the constructors.

.. class:: Container(initial=0)

   A counted level with integer amounts. Each trial starts with ``initial``
   units (a nonnegative integer).

   .. method:: put(amount)
   .. method:: get(amount)

      Add or remove ``amount`` units. ``get`` takes what is present and blocks
      until the rest arrives. Waiting calls are served by process priority,
      then first come, first served, but a ``get`` that has taken part of its
      amount rejoins the line behind waiting calls of equal priority each
      time it waits again.

   .. method:: level()
   .. method:: mean_level()
   .. method:: max_level()

      Current level; time-weighted mean and maximum over the measurement
      window.

   .. method:: capture()

      *Host.* Keep the level history; read as a :class:`Signal`.

   Amounts must be ``int``; a ``float`` amount is a compile-time error.

.. class:: FractionalContainer(initial=0.0, *, capacity=inf, decimals=6)

   A level with ``float`` amounts, kept in fixed point with ``decimals``
   decimal places (an ``int`` from 0 to 12). Each trial starts with
   ``initial``, a finite nonnegative number no greater than ``capacity``.

   .. method:: put(amount) -> float
   .. method:: get(amount) -> float

      Add or remove ``amount``, rounded half away from zero to ``decimals``
      places. ``put`` fills
      to ``capacity`` and blocks until there is room for the rest; ``get``
      takes what is present and blocks until the rest arrives. Both return
      the amount moved: the rounded ``amount``, or less if the wait was
      interrupted. Amounts must be finite and nonnegative, and an unlimited
      level must stay below 2\ :sup:`64` quanta.

   .. method:: round(amount) -> float

      ``amount``, of either sign, rounded half away from zero to ``decimals``
      places as ``put`` and ``get`` round it. Round an amount computed
      outside the level -- a sum, a ratio, a share -- so it compares exactly
      with ``level()``: ``round(0.1 + 0.2) == 0.3``.

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

   A pool of identical units. ``capacity`` is the number each trial starts
   with, a nonnegative ``int`` (zero is a closed pool). On the host it is the
   ``initial_capacity`` attribute; model code reads the current capacity with
   :meth:`capacity` and changes it with :meth:`set_capacity`.

   .. method:: acquire(amount=1, priority=None)

      Take ``amount`` units, waiting for any that aren't free. Returns a
      signal: ``0`` once all are held. If the wait is interrupted, the call
      returns the signal and the process holds what it held before the call.

      Waiting requests are served by **request priority**, higher first, and
      first come, first served among equal priorities. Only the request at
      the front is served: one that can't be fully served yet keeps the units
      it has taken so far and blocks the requests behind it, even if they
      would fit. It keeps its place in line each time it waits again, so a
      later request of equal priority is never served first.

      ``priority=None`` ranks the request at the process priority and follows
      it: :meth:`Process.priority_set` moves the waiting request and the
      units it holds. An ``int`` ranks the request at that value instead, and
      ``priority_set`` doesn't move it. Either way, the process priority still
      orders the process's events, including its wakeup when the request is
      granted, so a high request priority gets a process to the front of the
      queue without making it run before other processes at the same instant.

      A process's units in one resource are one **holding**, ranked at the
      priority of its latest ``acquire`` or ``preempt`` call. A call that is
      interrupted gives the holding its previous priority back.

   .. method:: preempt(amount=1, priority=None)

      As :meth:`acquire`, but first takes units from holders whose holding
      priority is *strictly* lower than the request priority: the lowest
      first and, among equals, the one whose holding started most recently.
      A preempted holder loses all its units in this resource and sees signal
      ``-1``. If that is still not enough, the request waits in line for the
      rest. Returns a signal.

   .. method:: release(amount=1)
   .. method:: held(process)

      Units held by a :class:`Process` handle.

   .. method:: available()

      Free units: the capacity minus the units in use, or ``0`` while the
      units in use are at or above the capacity.

   .. method:: capacity()

      The current capacity.

   .. method:: set_capacity(n, *, mode="finish")

      Change the capacity to ``n``, a nonnegative ``int``; a negative ``n``
      fails the trial. Setting the current capacity again does nothing. It
      never waits, so it works in processes, functions and hooks.

      Raising it grants waiting requests at once, in queue order. Lowering
      it depends on ``mode``, a string literal resolved when the model
      compiles:

      * ``"finish"``: holders keep their units, so the units in use may stay
        above ``n`` for a while. Nothing is granted until they drop below
        ``n``, and :meth:`available` is ``0`` meanwhile.
      * ``"preempt"``: holdings are taken in the :meth:`preempt` victim
        order, lowest holding priority first, the most recent among equals,
        until the units in use are at most ``n``. Each victim loses all its
        units in this resource and sees signal ``-1``, waking at its own
        process priority. Room left below ``n`` goes to waiting requests. The
        calling process is never its own victim.

      A request larger than the current capacity waits until the capacity
      rises enough.

   .. method:: mean_in_use()

      Time-weighted mean units in use over the window.

   .. method:: mean_capacity()

      Time-weighted mean capacity over the window, sampled with the in-use
      history.

   .. method:: utilization()

      ``mean_in_use() / mean_capacity()``: the time-weighted units in use
      over the time-weighted capacity, both over the window. It can exceed
      ``1`` while ``"finish"`` lets holders run past a lowered capacity, and
      it is ``nan`` when the window offered no capacity.

   .. method:: capture()

      *Host.* Keep the in-use history.

.. class:: Condition(*, observe=())

   A wait for an arbitrary predicate over model state. ``observe`` is a
   :class:`Resource`, :class:`Store`, :class:`PriorityStore`,
   :class:`Container` or :class:`FractionalContainer`, or a tuple of them.
   Every change to an observed entity signals the condition, as if
   :meth:`signal` were called right after it: a release, grant or capacity
   change of a resource, a put or get of a store or container. Model code
   then needs no ``signal()`` of its own for those changes.

   Observation is structure, fixed for an experiment. Observed entities must
   belong to the same model tree; the experiment checks this when it is set
   up and raises :class:`~cimba.schema.ModelDefinitionError` otherwise. A
   condition declared as a class default can observe entities declared
   before it in the same class body. Every model instance gets its own copy
   of the class-default entities, observing each other's copies.

   .. method:: observe(*entities) -> Condition

      *Host.* Also observe ``entities``. Use it for entities that don't exist
      when the condition is created, such as an owner's in
      :meth:`Model.attached`. Returns the condition.

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
   parameter is annotated ``float``, ``int``, ``bool``, an ``enum.IntEnum``
   or a model class, and the return type is one of those or ``None``. An
   enum parameter takes members of that enum only. Scalar parameters may have
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

.. function:: acquire_any(resources, priority=None) -> int

   Take one unit from whichever of ``resources`` grants first and return its
   index in ``resources``. ``resources`` is either

   * a nonempty tuple of :class:`Resource` entities written in model code,
     such as ``(self.clerks, self.seniors)`` or
     ``(self.desk.clerks, self.back.clerks)``, or
   * a ``list[cb.Ref[cb.Resource]]`` field, such as ``self.pool.resources``,
     whose members and length are configuration (see :class:`Ref`). Its
     length is read when the call runs; an empty list fails the trial.

   Anything else is a compile error. Both forms behave the same way. Listing
   a resource twice is allowed and returns its first index.

   * If any of them has a free unit, the call takes it at once from the first
     listed one that has.
   * Otherwise the request waits in **every** resource's queue at the place a
     :meth:`Resource.acquire` made now would get there: same request
     priority, same arrival time. Each resource serves it in turn with its
     direct acquirers, by priority and then first come, first served.
   * The first resource to grant serves it. The request leaves every other
     queue at that moment, so two resources never both grant it.

   ``priority`` works as for :meth:`Resource.acquire`: ``None`` ranks the
   request at the process priority in every queue and follows
   :meth:`Process.priority_set`; an ``int`` fixes it. The unit is an
   ordinary holding of the resource that granted it: release it with that
   resource's :meth:`~Resource.release`, read it with
   :meth:`~Resource.held`, and a :meth:`~Resource.preempt` can take it.

   If the wait ends otherwise, the call returns the negative signal and
   leaves no request behind in any queue: ``-1`` when the process is
   preempted from another resource, ``-2`` (or another code) when
   interrupted. A process timer set with :meth:`Process.timer_set` bounds the
   wait; the call then returns the timer's signal. Raising a resource's
   capacity with :meth:`~Resource.set_capacity` can grant a waiting
   ``acquire_any``; a resource without room never does.

   It takes one unit, never more: grabbing parts of an amount from several
   pools has no single fair answer. For anything other than "one unit of any
   of these", wait on a :class:`Condition` that observes the resources.

.. function:: now()

   Current simulated time.

.. function:: suspend()

   Suspend until resumed, interrupted or woken by a timer. Returns the signal.

.. function:: spawn(ModelClass, **fields)

   Create a dynamic model instance, set its fields, run its ``on_start``
   hooks, schedule its processes, and return a handle to it. Fields without
   a default must be supplied; input fields take an input of the caller.
   Enum fields take members of their enum. An array field takes an ``int``
   length (a zeroed array; a negative length fails the trial) or uses the
   class default.
   Optional inputs may be omitted or passed as ``None`` to bind an empty source.

.. function:: release(model)

   Stop this model's processes, including copies and pending starts. Releasing
   the caller's own model ends the calling process. Static tree models keep
   their record, entities, children and ``on_end`` hooks; spawned models are
   retired and skip their ``on_end`` hooks, and their arrays are freed.
   Records are reclaimed at trial end, so references to them stay valid.
   Children continue independently. Releasing the same instance twice fails
   the trial.

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
