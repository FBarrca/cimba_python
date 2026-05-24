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
.. py:function:: cimba.lognormal(m, s)
.. py:function:: cimba.logistic(m, s)
.. py:function:: cimba.cauchy(mode, scale)
.. py:function:: cimba.exponential(mean)
.. py:function:: cimba.erlang(k, mean)
.. py:function:: cimba.hypoexponential(means)
.. py:function:: cimba.hyperexponential(means, probabilities)
.. py:function:: cimba.gamma(shape, scale=1.0)
.. py:function:: cimba.beta(a, b, min=0.0, max=1.0)
.. py:function:: cimba.pert(min, mode, max)
.. py:function:: cimba.pert_mod(min, mode, max, lambda_)
.. py:function:: cimba.weibull(shape, scale)
.. py:function:: cimba.pareto(shape, mode)
.. py:function:: cimba.chi_squared(k)
.. py:function:: cimba.f_dist(a, b)
.. py:function:: cimba.student_t(v, m=0.0, s=1.0)
.. py:function:: cimba.rayleigh(s)
.. py:function:: cimba.dice(min, max)
.. py:function:: cimba.flip()
.. py:function:: cimba.bernoulli(p)
.. py:function:: cimba.geometric(p)
.. py:function:: cimba.binomial(n, p)
.. py:function:: cimba.negative_binomial(m, p)
.. py:function:: cimba.pascal(m, p)
.. py:function:: cimba.poisson(r)
.. py:function:: cimba.loaded_dice(probabilities)

   Draw from the named distribution using the current thread's Cimba PRNG.

.. py:class:: cimba.AliasSampler(probabilities)

   Reusable Vose alias sampler for non-uniform discrete probabilities.
   ``probabilities`` must be non-empty, finite, non-negative, and sum to
   ``1.0``. Use :func:`cimba.loaded_dice` for occasional one-shot draws and
   ``AliasSampler`` when the same probability table is sampled many times.

   .. py:method:: sample()

      Draw one index from the configured probability table.

   .. py:method:: close()

      Destroy the native alias table. Sampling after close raises
      :class:`RuntimeError`.
