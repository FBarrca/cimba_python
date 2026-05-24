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
* ``lognormal(m, s)``
* ``logistic(m, s)``
* ``cauchy(mode, scale)``
* ``exponential(mean)``
* ``erlang(k, mean)``
* ``hypoexponential(means)``
* ``hyperexponential(means, probabilities)``
* ``gamma(shape, scale=1.0)``
* ``beta(a, b, min=0.0, max=1.0)``
* ``pert(min, mode, max)``
* ``pert_mod(min, mode, max, lambda_)``
* ``weibull(shape, scale)``
* ``pareto(shape, mode)``
* ``chi_squared(k)``
* ``f_dist(a, b)``
* ``student_t(v, m=0.0, s=1.0)``
* ``rayleigh(s)``
* ``dice(min, max)``
* ``flip()``
* ``bernoulli(p)``
* ``geometric(p)``
* ``binomial(n, p)``
* ``negative_binomial(m, p)``
* ``pascal(m, p)``
* ``poisson(r)``
* ``loaded_dice(probabilities)``

Discrete empirical probabilities
--------------------------------

``loaded_dice(probabilities)`` draws one index from a non-uniform discrete
distribution. It is simple and useful for occasional draws:

.. code-block:: python

   outcome = cimba.loaded_dice([0.2, 0.3, 0.5])

For repeated draws from the same probability table, use ``AliasSampler``. It
precomputes a native Vose alias table once and then samples it in constant time:

.. code-block:: python

   with cimba.AliasSampler([0.2, 0.3, 0.5]) as sampler:
       outcomes = [sampler.sample() for _ in range(1000)]

Probability sequences must be non-empty, finite, non-negative, and sum to
``1.0``. ``AliasSampler`` owns a native table, so close it when finished or use
it as a context manager.

Deterministic seed mixing
-------------------------

``fmix64(seed, nonce)`` mixes a master seed with a deterministic nonce. It is
useful when you want reproducible but distinct seeds for repeated trials.

.. code-block:: python

   master = 12345
   trial_seed = cimba.fmix64(master, rep)

For generator details and distribution algorithms, see the
`Cimba C API reference`_.
