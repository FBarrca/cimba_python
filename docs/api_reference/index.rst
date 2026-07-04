.. _api_reference:

API Reference
=============

Model code imports the modeling API from :mod:`cimba.sim` and a few top-level
helpers from :mod:`cimba`. The pages below group the public surface by area.

A model is declared as a :class:`~cimba.sim.Model` subclass whose annotated
fields describe the shape of one trial. Process behaviour is written as ordinary
Python functions registered with ``@model.process`` and ``@model.collect``, and
experiments are built with ``model.experiment(...)`` and run with
``Experiment.run()``.

.. toctree::
   :maxdepth: 2

   models
   processes
   entities
   data
   random
   logging
   traces

Top-level package
-----------------

.. automodule:: cimba
   :members:
