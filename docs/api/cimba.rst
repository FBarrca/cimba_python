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
``Dataset``, ``Condition``, ``Predicate``, ``Event``, ``Processes``,
``PQueues``, ``Spawnable``, ``Struct``, ``capacity()``, ``count()``.

Per-process fields (``sim.Struct``): declare the fields as ``float``/``int``
annotations on a ``sim.Struct`` subclass, and let the process function ask
for its own view with a final annotated parameter --
``def visitor(env, vip: Visitor)`` (with the copy index:
``def visitor(env, idx, vip: Visitor)``). Each copy carries its own fields,
zeroed at creation, and ``Visitor(handle)`` returns a read/write view of any
such process's fields inside model code -- the Python counterpart of the C
tutorial's ``struct visitor { struct cmb_process core; ... }`` pattern.
``@model.process(struct=Visitor)`` attaches the fields without the view
parameter.

Process verbs:

``hold()``, ``now()``, ``current()``, ``interrupt()``, ``stop()``,
``wait_process()``, ``wait_event()``, ``resume()``, ``suspend()``,
``status()``, ``set_priority()``, ``timer_set()``, ``timer_add()``,
``timer_cancel()``, ``timers_clear()``, ``spawn()``, ``despawn()``.

Dynamic processes: a process named in a ``sim.Spawnable`` field is not
started at setup; ``sim.spawn(env.<name>, env, priority=0)`` creates and
starts a copy at runtime and returns its handle. The new process begins
running once the caller blocks, so its ``sim.Struct`` fields (zeroed at
creation) can be initialized through the handle first. Finished processes
can be reclaimed early with ``sim.despawn(handle)`` to recycle memory
during long trials; any spawned processes still alive at the end of the
trial are stopped and reclaimed automatically, like the static ones.

Low-level events (callbacks registered with ``@model.event`` and published in
``sim.Event`` fields):

``schedule()``, ``schedule_at()``, ``event_cancel()``, ``event_reschedule()``,
``event_reprioritize()``, ``event_scheduled()``, ``event_time()``,
``event_priority()``, ``current_event()``, ``event_count()``,
``clear_events()``.

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

Logging:

``log_text()``, ``log_user()``, ``log_user_i64()``, ``log_user_f64()``.
Top-level ``cimba`` also exposes ``logger_flags_on()`` and
``logger_flags_off()``.

Signal constants and casts:

``SUCCESS``, ``PREEMPTED``, ``INTERRUPTED``, ``STOPPED``, ``CANCELLED``,
``TIMEOUT``, ``LOGGER_FATAL``, ``LOGGER_ERROR``, ``LOGGER_WARNING``,
``LOGGER_INFO``, ``f2i()``, ``i2f()``.
