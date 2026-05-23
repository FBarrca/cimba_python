Cimba Python
============

.. image:: ../subprojects/cimba/images/logo_large.jpg

Cimba Python is a process-oriented discrete event simulation package backed by
the Cimba C engine. It gives Python models native Cimba processes, queues,
resources, random distributions, and time-weighted statistics while keeping the
model code in Python.

If you are new to Cimba, start with :doc:`cimba_intro/index`. If you already
know the basics, the :doc:`topical_guides/index`, :doc:`examples/index`, and
:doc:`api_reference/index` sections are usually the shortest route to an
answer.

A small example
---------------

.. code-block:: python

    import cimba


    def clock(me, name):
        while True:
            cimba.hold(1.0)
            print(f"{cimba.time():.0f}: {name}")


    with cimba.Simulation(seed=123) as sim:
        cimba.Process("Clock", clock, "tick").start()
        sim.stop_at(3.0)
        sim.execute()

This creates a simulation, starts a process, advances simulated time with
:func:`cimba.hold`, and stops at an absolute simulation time.

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    cimba_intro/index
    topical_guides/index
    examples/index
    api_reference/index
    about/index
