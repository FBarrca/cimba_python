.. _tut_2:

2. Mice, rats and a cat
=======================

Chapter 1's processes were polite: they queued up and took turns. Real systems
are messier. Urgent jobs jump the line, machines break down in the middle of a
task, and a supervisor pulls a worker off one job to do another. This chapter
is about processes that **compete** and **interfere**.

The scenario is deliberately playful (it is the C library's tutorial 2), but
every mechanism in it shows up in serious models:

* Five **mice** and two **rats** compete for a pile of 20 cheese cubes.
* Each animal repeatedly picks a random amount and a random priority, and
  tries to grab that much cheese.
* Mice **acquire** politely: if there isn't enough, they wait.
* Rats **preempt**: they snatch cheese from lower-priority holders, who lose
  *everything* they were holding.
* A **cat** naps, wakes at random and chases a random rodent, **interrupting**
  whatever it was doing.

The complete script is ``tutorial/tut_2_1.py``.

.. contents:: In this chapter
   :local:
   :depth: 1

A resource with capacity
------------------------

The cheese pile is a :class:`~cimba.Resource`, a pool of identical units
that processes acquire and release:

.. code-block:: python

   class CheeseGame(cb.Model):
       rodents: list[Rodent]
       cheese: cb.Resource = cb.Resource(capacity=CHEESE_AMOUNT)
       ...

Giving the field a *value* on the class sets its configuration. Here the pool
has 20 units. (Each model instance still gets its own resource, and each trial
its own native copy.)

The resource verbs, all called from compiled code, are:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Call
     - Meaning
   * - ``acquire(n=1)``
     - Take ``n`` units, waiting until they are free. Returns a signal.
   * - ``preempt(n=1)``
     - Take ``n`` units, grabbing them from lower-priority holders if
       necessary. Returns a signal.
   * - ``release(n=1)``
     - Give ``n`` units back.
   * - ``held(process)``
     - How many units a process holds right now.
   * - ``available()``
     - Free units right now.
   * - ``mean_in_use()``
     - Time-weighted mean of units in use during the measurement window.

A plain ``Resource(capacity=1)`` is a single server or a mutex. A larger
capacity models a pool of workers, berths, tugboats or licenses.

A population of rodents
-----------------------

Each rodent is its own model, so it can have its own process and its own
settings:

.. literalinclude:: ../../tutorial/tut_2_1.py
   :pyobject: Rodent
   :lines: 1-8

Three new kinds of field appear:

``game: cb.Ref["CheeseGame"]``
   A **reference** to another model instance, here back to the game that owns
   the rodent. The string form lets you refer to a class defined later in the
   file. In compiled code, ``self.game.cheese`` follows the reference to the
   game's resource.

``is_rat: cb.Param[int] = 0``
   A **parameter**: a value fixed for the whole trial. Each rodent sets its own
   in ``__init__``. Parameters can also be *swept* in an experiment.

``process_pointer: cb.State[int] = 0``
   **State**: a per-trial value that processes may change. It starts at the
   given initial value in every trial.

The game creates the rodents in its constructor and keeps them in a
**collection**, a plain ``list`` annotated as ``list[Rodent]``:

.. literalinclude:: ../../tutorial/tut_2_1.py
   :pyobject: CheeseGame.__init__

A ``list[Model]`` field holds **children**: they belong to the game and appear
in the model tree as ``cheesegame.rodents[0]``, ``cheesegame.rodents[1]`` and
so on. Every rodent's ``forage`` process starts automatically. Seven
instances, one class, compiled once.

.. tip::

   If a list should *point to* models that live elsewhere in the tree rather
   than own them, annotate it as ``list[cb.Ref[Rodent]]``.

Priorities, preemption and signals
----------------------------------

Here's the heart of a rodent's life:

.. literalinclude:: ../../tutorial/tut_2_1.py
   :pyobject: Rodent.forage
   :lines: 1-14

``cb.this_process()`` returns a :class:`~cimba.Process` handle for the running
process. With it, the rodent sets its own **priority** before every grab.
Priority decides who is served first among waiting processes, and who may be
robbed by ``preempt``: a rat can only take cheese from holders with *lower*
priority.

Notice that ``acquire`` and ``preempt`` return a value. Every blocking call in
Cimba returns a **signal** that tells you *why* the process resumed:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Value
     - Meaning
     - Typical cause
   * - ``0``
     - success
     - The call completed normally.
   * - ``-1``
     - preempted
     - A higher-priority process took this process's resource units.
   * - ``-2``
     - interrupted
     - Another process called ``interrupt()`` on this one.
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
     - user signal
     - Your own code: an interrupt with a custom code, or a process timer
       (chapter 3).

It's good practice to name the ones you use as module constants, for example
``PREEMPTED = -1``. The rodent inspects the signal and keeps score:

.. literalinclude:: ../../tutorial/tut_2_1.py
   :pyobject: Rodent.forage
   :lines: 16-37

``cb.hold()`` returns a signal too. A rodent napping with its cheese can be
preempted *while holding*, and it finds out when ``hold`` returns ``-1``
instead of ``0``. A preempted process loses all its units of that resource,
which is why the model compares ``held`` before and after.

Interrupting another process
----------------------------

The cat has to reach a *specific* rodent's process. Each rodent stores its
process handle's pointer in its state as soon as it starts
(``self.process_pointer = me.pointer``), and the cat rebuilds a handle from
it:

.. literalinclude:: ../../tutorial/tut_2_1.py
   :pyobject: CheeseGame.cat

``cb.Process(pointer).interrupt(signal, priority)`` wakes the target from
whatever blocking call it is in. The call returns early with ``signal``, here
either ``-2`` (the generic "interrupted") or a random user code between 10 and
100. Unlike preemption, an interrupt doesn't take resources away. The rodent
keeps its cheese, but its wait is cut short.

Randomness that isn't an input
------------------------------

This model has no ``Input`` fields. The amounts, priorities, nap lengths and
coin flips all use :mod:`cimba.random`:

.. code-block:: python

   amount = cb.random.dice(3, 10)
   me.priority_set(cb.random.dice(-5, 15))
   signal = cb.hold(cb.random.exponential(1.0))

How do you choose between the two?

* Use an **input** (``cb.Input``) for variability that represents the *real
  world*: arrivals, service times, demand, lead times, failures. Those are the
  things you will want to fit to data, replay from records, resample or sweep.
  Each input gets its own random stream, so its values don't shift when other
  parts of the model draw more or fewer numbers.
* Use **cimba.random** for *model-internal* randomness: routing coin flips,
  tie-breaking, synthetic behavior that no dataset will ever describe.
  These draws come from the trial's default stream.

``cimba.random`` functions only work inside compiled model code. On the
host, before or after a run, use ``numpy.random.default_rng(seed)``.

Running the game
----------------

.. literalinclude:: ../../tutorial/tut_2_1.py
   :pyobject: CheeseGame.game_stats

The ``on_end`` hook copies the counters into outputs. Integer outputs work the
same way as float ones.

.. literalinclude:: ../../tutorial/tut_2_1.py
   :pyobject: main
   :lines: 1-8

.. code-block:: console

   $ uv run python tutorial/tut_2_1.py
   cimba 3.0.0; 5 mice, 2 rats, 20 cheese cubes
   10 trials in 1.69 s
   mice 370053.5 277871.6 67893.0 14389.5
   rats 357443.1 183293.2 16095.2 5555.9
   cat chases: 28496.0

Each row reads *grabbed, stolen, preempted, interrupted*, averaged over ten
replications of 100,000 time units. The rats, only two of them, grab nearly as
much as five mice and are preempted far less often. The final assertion checks
that every rodent's own bookkeeping matches the resource's record of what it
holds: ``accounting_errors`` is zero in every trial.

What you learned
----------------

* ``Resource(capacity=n)`` models a pool; ``acquire`` waits, ``preempt``
  takes from lower-priority holders.
* ``Ref[M]`` points to another model; ``list[M]`` holds child models;
  ``Param`` is fixed per trial and ``State`` changes during it.
* Every blocking call returns a **signal** that says why the process woke up.
* ``cb.this_process()`` and ``cb.Process(pointer)`` give you handles to set
  priorities and interrupt other processes.
* Real-world variability belongs in inputs; model-internal randomness can use
  ``cimba.random``.

In :doc:`park`, customers stop being counters and become individuals that
come and go during the trial.
