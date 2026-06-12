Cimba Python API Reference
==========================

Top-level package
-----------------

.. automodule:: cimba
   :members:

Simulation API
--------------

Use :mod:`cimba.sim` for the SimPy-flavored modeling API. The module depends on
Numba and the compiled native extension, so this reference lists the public
surface without importing it during documentation builds.

Model declarations:

``Model``, ``Experiment``, ``Env``, ``Handle``, ``Param``, ``Output``,
``State``, ``FloatState``, ``Queue``, ``Resource``, ``Pool``, ``Store``,
``Dataset``, ``Condition``, ``Predicate``, ``Processes``, ``PQueues``,
``capacity()``, ``count()``.

Process verbs:

``hold()``, ``now()``, ``current()``, ``interrupt()``, ``stop()``,
``wait_process()``, ``wait_event()``, ``resume()``, ``suspend()``,
``status()``, ``set_priority()``, ``timer_set()``, ``timer_add()``,
``timer_cancel()``, ``timers_clear()``.

Queues and resources:

``put()``, ``get()``, ``level()``, ``space()``, ``mean_level()``,
``acquire()``, ``release()``, ``preempt()``, ``available()``, ``in_use()``,
``held()``, ``mean_in_use()``, ``pool_acquire()``, ``pool_release()``,
``pool_preempt()``, ``pool_available()``, ``pool_held()``, ``pool_in_use()``,
``pool_mean_in_use()``.

Stores, priority queues, datasets, and conditions:

``store_put()``, ``store_get()``, ``store_take()``, ``store_length()``,
``store_space()``, ``store_position()``, ``store_mean_length()``,
``pq_put()``, ``pq_get()``, ``pq_take()``, ``pq_length()``, ``pq_space()``,
``pq_position()``, ``pq_reprioritize()``, ``pq_cancel()``,
``pq_mean_length()``, ``tally()``, ``dataset_mean()``, ``dataset_count()``,
``dataset_min()``, ``dataset_max()``, ``dataset_std()``, ``wait_for()``,
``signal()``.

Random draws:

``exponential()``, ``gamma()``, ``uniform()``, ``normal()``, ``random01()``,
``rayleigh()``, ``pert()``, ``pert_mod()``, ``bernoulli()``, ``flip()``,
``triangular()``, ``weibull()``, ``lognormal()``, ``erlang()``, ``beta()``,
``poisson()``, ``dice()``, ``std_normal()``, ``std_exponential()``,
``std_gamma()``, ``std_beta()``, ``logistic()``, ``cauchy()``, ``pareto()``,
``chisquared()``, ``f_dist()``, ``std_t()``, ``t_dist()``, ``geometric()``,
``binomial()``, ``negative_binomial()``, ``pascal()``, ``hypoexponential()``,
``hyperexponential()``, ``categorical()``, ``loaded_dice()``.

Signal constants and casts:

``SUCCESS``, ``PREEMPTED``, ``INTERRUPTED``, ``STOPPED``, ``CANCELLED``,
``TIMEOUT``, ``f2i()``, ``i2f()``.
