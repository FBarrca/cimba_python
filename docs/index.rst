Cimba Python
============

.. image:: static/cimba_logo_large.jpg
   :alt: Cimba logo
   :width: 360px

**Write discrete-event simulation models as Python classes. Run thousands of
trials at native speed on every core. Drive the same model with fitted
distributions, recorded data, or bootstrap resamples without changing a line of
model code.**

Cimba Python compiles your model classes with Numba and runs them on the
`Cimba C engine <https://github.com/ambonvik/cimba>`_, a stackful-coroutine
simulation kernel. You write ordinary-looking Python with blocking calls such
as ``cb.hold(5.0)`` or ``queue.get()``. No ``yield``, no callbacks, no
interpreter in the hot loop.

.. code-block:: python

   import cimba as cb
   from cimba import inputs


   class Shop(cb.Model):
       arrival_gap: cb.Input[float] = inputs.dist.exponential(mean=1.25)
       service_time: cb.Input[float] = inputs.dist.exponential(mean=1.0)
       line: cb.Container
       mean_line: cb.Output[float]

       @cb.process
       def customers(self):
           while True:
               cb.hold(self.arrival_gap.next())
               self.line.put(1)

       @cb.process
       def clerk(self):
           while True:
               self.line.get(1)
               cb.hold(self.service_time.next())

       @cb.on_end
       def measure(self):
           self.mean_line = self.line.mean_level()


   shop = Shop()
   results = cb.Experiment(shop, replications=100,
                           window=cb.Window(warmup=100, duration=1_000),
                           seed=1).run()
   print(cb.analysis.summary(results[shop].mean_line).mean)

   # Tomorrow: replay what really happened. Same class, no recompilation.
   shop.arrival_gap = inputs.trace(observed_gaps, on_exhausted="wrap")

Where to start
--------------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :doc:`quickstart`
     - Install, run your first model, and read its results in five minutes.
   * - :doc:`tutorial/index`
     - A guided tour. Six chapters build from a single queue to a data-driven
       supply chain, one idea at a time.
   * - :doc:`concepts/index`
     - The mental model: models, processes, entities, inputs, experiments and
       results, each explained on its own page.
   * - :doc:`guides/index`
     - Short recipes for specific jobs: replay recorded data, validate an
       input model, compare designs, debug a failed trial.
   * - :doc:`reference/index`
     - Every class, method and distribution, with signatures and semantics.
   * - :doc:`architecture/index`
     - How the library is built: layers, the input-modeling pillar, what
       happens inside ``run()``, and the native runtime.

The ideas in one paragraph
--------------------------

A **model** is a Python class. Its annotated fields declare parameters, state,
outputs, simulation **entities** (queues, resources, stores) and **inputs**.
Methods marked ``@cb.process`` run as concurrent simulated processes. An input
says *what* the model consumes, such as inter-arrival times or daily demand, but
never *how* those values are produced. You decide that when you configure an
experiment, by binding a **source**: a distribution, a recorded trace, a
bootstrap resample or a fitted time-series model. An **experiment** runs every
combination of your **sweeps** for a number of **replications**, one
independent **trial** each, in parallel. You read **results** back through the
same objects you configured: ``results[shop].mean_line``.

.. toctree::
   :hidden:
   :caption: Get started

   quickstart
   installation

.. toctree::
   :hidden:
   :caption: Learn

   tutorial/index
   concepts/index

.. toctree::
   :hidden:
   :caption: Use

   guides/index
   reference/index

.. toctree::
   :hidden:
   :caption: Understand the internals

   architecture/index

.. toctree::
   :hidden:
   :caption: Project

   migration
