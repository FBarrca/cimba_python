Advanced Topics
===============

This section is the next layer after :doc:`../concepts/index`. It covers the
``cimba.sim`` features that help larger models stay organized, observable, and
reproducible: components, dynamic processes, per-process fields, explicit
events, timers, trace replay and bootstrap resampling, process graphs,
logging, and reporting.

The snippets continue the clinic and service-desk examples from Core Concepts.
They are partial by design. Use them to understand the modeling pattern, then
turn to the tutorial and API reference for complete programs and exact helper
lists.

.. toctree::
   :maxdepth: 2

   components
   structs_spawnables
   events_timers_signals
   traces
   bootstrapping
   process_graphs
   reporting_logging

Where this fits
---------------

Read Advanced Topics when the basic model shape is already clear and the
question has become "how do I keep this model maintainable as it grows?"

For a full worked example of an agent-heavy, resource-heavy model, see
:doc:`../tutorial`.
