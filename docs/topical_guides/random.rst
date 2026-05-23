Random Numbers
==============

Simulation-owned streams
------------------------

The active :class:`cimba.Simulation` initializes Cimba's thread-local random
generator. Random functions draw from that stream:

.. code-block:: python

   with cimba.Simulation(seed=123):
       value = cimba.exponential(4.0)

Use explicit seeds for reproducible examples and tests.

Available distributions
-----------------------

Cimba Python exposes:

* ``random()`` and ``random_u64()``
* ``uniform(min, max)``
* ``triangular(min, mode, max)``
* ``normal(mu=0.0, sigma=1.0)``
* ``exponential(mean)``
* ``gamma(shape, scale=1.0)``
* ``beta(a, b, min=0.0, max=1.0)``
* ``pert(min, mode, max)``
* ``pert_mod(min, mode, max, lambda_)``
* ``dice(min, max)``
* ``flip()``
* ``bernoulli(p)``

Deterministic seed mixing
-------------------------

``fmix64(seed, nonce)`` mixes a master seed with a deterministic nonce. It is
useful when you want reproducible but distinct seeds for repeated trials.

.. code-block:: python

   master = 12345
   trial_seed = cimba.fmix64(master, rep)

For generator details and distribution algorithms, see the
`Cimba C API reference`_.
