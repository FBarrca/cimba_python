Logging and Signals
===================

Logging
-------

Static text is registered once with ``sim.log_text()``. Process bodies then log
static messages, integer values, or floating-point values without allocating
formatted Python strings inside the simulation loop:

``log_text()``, ``log_user()``, ``log_user_i64()``, ``log_user_f64()``.

Top-level ``cimba`` also exposes ``logger_flags_on()`` and
``logger_flags_off()`` to enable or disable logger categories.

Signal constants and casts
--------------------------

Blocking operations return one of these signals, and the logger flags select
severities:

``SUCCESS``, ``PREEMPTED``, ``INTERRUPTED``, ``STOPPED``, ``CANCELLED``,
``TIMEOUT``, ``LOGGER_FATAL``, ``LOGGER_ERROR``, ``LOGGER_WARNING``,
``LOGGER_INFO``, ``f2i()``, ``i2f()``.
