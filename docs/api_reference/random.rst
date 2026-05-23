Random Numbers
==============

.. py:function:: cimba.hwseed()

   Return a hardware-derived 64-bit seed.

.. py:function:: cimba.seed(value=None)

   Initialize the thread-local PRNG and return the seed used.

.. py:function:: cimba.current_seed()

   Return the seed used for the current thread's PRNG stream.

.. py:function:: cimba.random()

   Draw a continuous uniform random variate on ``[0.0, 1.0]``.

.. py:function:: cimba.random_u64()

   Draw a raw uniformly distributed 64-bit pseudo-random bit pattern.

.. py:function:: cimba.fmix64(seed, nonce)

   Mix a master seed and deterministic nonce into a reproducible 64-bit seed.

.. py:function:: cimba.uniform(min, max)
.. py:function:: cimba.triangular(min, mode, max)
.. py:function:: cimba.normal(mu=0.0, sigma=1.0)
.. py:function:: cimba.exponential(mean)
.. py:function:: cimba.gamma(shape, scale=1.0)
.. py:function:: cimba.beta(a, b, min=0.0, max=1.0)
.. py:function:: cimba.pert(min, mode, max)
.. py:function:: cimba.pert_mod(min, mode, max, lambda_)
.. py:function:: cimba.dice(min, max)
.. py:function:: cimba.flip()
.. py:function:: cimba.bernoulli(p)

   Draw from the named distribution using the current thread's Cimba PRNG.
