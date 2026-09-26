Writing compiled code
=====================

Methods decorated with ``@cb.process``, ``@cb.on_start``, ``@cb.on_end``,
``@cb.predicate`` and ``@cb.event`` are compiled to machine code with
`Numba <https://numba.readthedocs.io/>`_ in *nopython* mode. That's what makes
trials fast and lets them run in parallel without the GIL. It also means
their bodies use a subset of Python. This page lists what works, what
doesn't, and the idioms that make the difference painless.

What ``self`` is
----------------

Inside a compiled method, ``self`` is a **typed view** of one trial's record
for that model instance, not the Python object. Through it you can:

* read and write ``Param``, ``State`` and ``Output`` fields and constants
  (``self.on_hand -= shipped``);
* call entity methods (``self.queue.get(1)``);
* read inputs (``self.demand.now()``);
* follow children, references and lists
  (``self.network.facilities[i].on_hand``);
* pass predicates and events (``wait_until(self.is_ready)``,
  ``cb.schedule(self.fire, 2.0)``).

Host-only attributes (unannotated ones like ``self.name``) and undecorated
methods of the class are **not** visible; ``@cb.function`` methods are.

What you can use
----------------

* Numbers and booleans, arithmetic, ``math`` functions, ``min``/``max``/
  ``abs``, ``int()``/``float()``/``bool()``.
* ``if``/``elif``/``else``, ``while``, ``for i in range(...)``, ``break``,
  ``continue``, local variables and tuples.
* Most of NumPy's scalar and array math (``np.sin``, ``np.sqrt``,
  ``np.int64``, small local arrays).
* **Module-level constants**: numbers, tuples and NumPy arrays defined at
  the top of your module (``TRANSITION_PROBS[at, j]``). They are frozen into
  the compiled code when the class is first compiled.
* The Cimba verbs (``cb.hold``, ``cb.now``, ``cb.spawn``, …),
  :mod:`cimba.random`, entity methods and input reads.
* **Model functions**: methods decorated with ``@cb.function``, called
  like methods (``self.reorder(level)``, ``self.policy.quantity(x)``). They
  are compiled with their class, can be overridden by subclasses (the
  model's actual class decides which one runs), and can block. Parameters and
  the return value must be annotated:

.. code-block:: python

   class Facility(cb.Model):
       base_stock: cb.Param[float] = 100.0
       on_hand: cb.State[float] = 0.0

       @cb.function
       def reorder_quantity(self) -> float:
           return max(0.0, self.base_stock - self.on_hand)

       @cb.process
       def replenish(self):
           while True:
               cb.hold(1.0)
               quantity = self.reorder_quantity()

* **Plain helper functions** compiled with ``numba.njit`` at module level,
  for logic that doesn't belong to a model (they may take a model view as an
  argument).

Functions and helpers may call blocking verbs too. Processes are stackful, so
a ``cb.hold`` inside one suspends the calling process.

What you can't use
------------------

* Arbitrary Python objects, dicts of objects, classes, ``try``/``except``,
  generators, f-strings or string building. ``cb.log`` messages must be
  string literals.
* Undecorated methods of your model (``self.helper()``). Decorate them with
  ``@cb.function``; the compile error tells you so.
* Host-side libraries (pandas, SciPy, your database client).
* Changing a global after the first run and expecting the model to see it.
  Globals are compiled in; pass changing values as ``Param`` fields.

Keep variable types stable
--------------------------

Numba infers one type per local variable. If a variable holds an ``int`` in
one branch and a ``float`` in another, or a Python ``int`` and later a NumPy
``int64``, compilation can fail with a unification error. Initialize
variables with the type they'll keep:

.. code-block:: python

   chosen = np.int64(0)         # later assigned from np.int64(index)
   shortest = 1_000_000
   total = 0.0                  # not 0, if you'll add floats to it

When compilation fails
----------------------

A compile error is raised from ``run()`` as ``ModelCompileError``, naming the
file, line, class and method:

.. code-block:: text

   cimba.compiler.classes.ModelCompileError: models.py:42: Box.fill: 'helper'

The underlying Numba message follows. Common causes are calling a method
that isn't a ``@cb.function``, passing the wrong number or type of arguments
to one, using an unsupported Python feature, and type unification. Fix the method and run again. Nothing is cached for a class that
failed to compile.

Calling verbs outside a trial
-----------------------------

Verbs, entity methods and input reads only mean something inside a trial.
Called from ordinary Python, they raise ``NotInCompiledCode`` (or
``RuntimeError`` for :mod:`cimba.random`) with an explanation. For random
numbers on the host, before or after a run, use
``numpy.random.default_rng(seed)``.

Compilation cost
----------------

Each class is compiled the first time a run needs it, typically in a second
or so per class, and cached for the rest of the Python process. The cache key
is the class alone: rewiring the tree, resizing a list, changing parameters,
sweeping, or binding a different input source never recompiles.
``results.meta.compile`` reports what a run compiled; ``cb.cache_info()``
and ``cb.clear_cache()`` manage the cache.
