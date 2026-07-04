M/M/1 Queue
===========

This model has an arrival process, a service process, and one
:class:`~cimba.sim.Queue` representing the line. Arrivals put work into the
queue; the server takes work out one item at a time and holds for a random
service time. The collect function records the time-average queue length.

.. literalinclude:: ../../tutorial/tut_1_1.py
   :language: python

The examples in ``tutorial/tut_1_1.py`` through ``tutorial/tut_1_7.py`` build
this model up in smaller steps, ending with a parallel utilization sweep whose
results are compared against the analytical ``rho^2 / (1 - rho)`` queue length.
