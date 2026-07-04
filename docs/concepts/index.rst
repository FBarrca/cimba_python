Core Concepts
=============

This section explains the mental model behind the public ``cimba.sim`` API.
It is for model builders who want to understand what a Cimba Python model is,
how simulated time moves, and how an experiment turns many trial runs into
data.

The examples use a small clinic or service desk model: people arrive, wait for
service, use a limited service capacity, and produce outputs for analysis. The
snippets are intentionally partial. For complete runnable programs, continue
with :ref:`the tutorial <tutorial>`.

.. toctree::
   :maxdepth: 2

   models_trials_env
   processes_time
   shared_entities
   experiments_results

Where this fits
---------------

Read these pages before the tutorial if you want the vocabulary first. Read
the tutorial first if you prefer to learn by building a complete model and then
come back here when a concept needs a firmer shape.

Use the :doc:`API reference <../api_reference/index>` for the full API
overview once the core modeling picture is clear.
