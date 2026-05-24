.. _py_tut_5:

Adding CUDA GPU Power for Simulation Physics
============================================

The final section of the C tutorial is explicitly work in progress. It sketches
an AWACS scenario where Cimba processes model aircraft and targets while CUDA
handles heavy physics calculations.

The Python tutorial keeps the same simulation shape but not the CUDA machinery:
targets are active processes, a sensor process samples their state, and a
summary object collects detections.

The AWACS Scenario on a Single CPU
----------------------------------

The C tutorial visualizes the full scenario in ParaView:

.. figure:: ../../subprojects/cimba/images/tut_5_1.png
   :alt: AWACS aircraft, terrain, and detected ground targets.

   A snapshot of the AWACS scenario used to motivate GPU-backed simulation
   physics.

.. literalinclude:: ../../tutorial/tut_5_1.py
   :language: python

The important Cimba idea is still visible: many active targets have their own
process loops, and another process observes them at regular simulated times.
The heavy physics function could be ordinary Python, NumPy, a C extension, CUDA
through a third-party library, or any other thread-safe callable.

CUDA Integration Status
-----------------------

There is no high-level Python tutorial API today for assigning Cimba worker
threads to CUDA devices or streams. The low-level native thread hook capsule API
exists for advanced integrations, but the tutorial-level CUDA stream assignment
described in the C docs is not yet exposed as a friendly Python feature.
