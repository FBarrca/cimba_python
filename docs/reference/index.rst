Reference
=========

Precise signatures and semantics for the public API. The stable public
packages are ``cimba``, ``cimba.inputs``, ``cimba.random``, ``cimba.analysis``
and ``cimba.optimize`` (plus the small ``cimba.diagrams`` helper). Everything
else is internal and may change between releases.

.. list-table::
   :widths: 30 70

   * - :doc:`modeling`
     - ``cimba``: ``Model``, field types, entities, decorators, verbs and
       handles.
   * - :doc:`inputs`
     - ``cimba.inputs``: every distribution, traces, bootstraps, fitted
       models, fitting and custom sources.
   * - :doc:`experiments`
     - ``Experiment``, ``Window``, sweeps, ``Results``, ``Samples``,
       ``InputRecord``, ``Signal`` and errors.
   * - :doc:`analysis`
     - ``cimba.analysis``, ``cimba.random`` and ``cimba.diagrams``.
   * - :doc:`optimize`
     - Decisions, study methods, differential-evolution settings, estimates
       and search reports.
   * - :doc:`api`
     - Auto-generated listing from the source docstrings.

.. toctree::
   :hidden:

   modeling
   inputs
   experiments
   analysis
   optimize
   api
