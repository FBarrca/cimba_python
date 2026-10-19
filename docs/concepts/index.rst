Concepts
========

The tutorial shows the ideas in action. These pages explain them one at a
time, precisely, including the edge cases. Start with the big picture; after
that, read in any order.

.. list-table::
   :widths: 30 70

   * - :doc:`big_picture`
     - The whole library on one page: two worlds, five steps, one
       vocabulary.
   * - :doc:`models`
     - Model classes, the kinds of field, composition, references,
       collections, inheritance, static and dynamic instances.
   * - :doc:`processes`
     - Simulated time, processes, blocking calls and signals, hooks,
       conditions, events, timers and process handles.
   * - :doc:`entities`
     - Containers, stores, priority stores, resources, conditions and
       datasets, and what they measure.
   * - :doc:`inputs`
     - Inputs and series, the four kinds of source, random streams,
       exhaustion, extension and provenance.
   * - :doc:`experiments`
     - Trials, replications, sweeps and design points, the measurement
       window, seeding and common random numbers, parallel execution.
   * - :doc:`results`
     - Reading results through objects, samples, input records, captured
       signals, failures, and the analysis helpers.
   * - :doc:`optimization`
     - How decisions and objectives define a search, why candidates share
       seeds, and how fresh trials select and estimate a solution.
   * - :doc:`compiled_code`
     - What you can write inside process and hook methods, and why.
   * - :doc:`glossary`
     - Every term in one alphabetical list.

.. toctree::
   :hidden:

   big_picture
   models
   processes
   entities
   inputs
   experiments
   results
   optimization
   compiled_code
   glossary
