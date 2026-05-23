Logging
=======

.. py:data:: cimba.LOGGER_FATAL
.. py:data:: cimba.LOGGER_ERROR
.. py:data:: cimba.LOGGER_WARNING
.. py:data:: cimba.LOGGER_INFO

   Logger flags exported from the native Cimba logger.

.. py:function:: cimba.logger_flags_on(flags)

   Enable one or more Cimba logger flags in the current thread.

.. py:function:: cimba.logger_flags_off(flags)

   Disable one or more Cimba logger flags in the current thread.
