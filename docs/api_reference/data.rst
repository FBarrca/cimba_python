Data Collection
===============

.. py:class:: cimba.DataSummary()

   Single-pass unweighted summary of sample moments.

   .. py:method:: add(value)
   .. py:method:: reset()
   .. py:method:: merge(other)
   .. py:attribute:: count
   .. py:attribute:: min
   .. py:attribute:: max
   .. py:attribute:: mean
   .. py:attribute:: variance
   .. py:attribute:: stddev
   .. py:attribute:: skewness
   .. py:attribute:: kurtosis
   .. py:method:: close()

.. py:class:: cimba.WeightedSummary()

   Single-pass weighted summary of sample moments.

   .. py:method:: add(value, weight=1.0)
   .. py:method:: reset()
   .. py:method:: merge(other)
   .. py:attribute:: count
   .. py:attribute:: weight_sum
   .. py:attribute:: min
   .. py:attribute:: max
   .. py:attribute:: mean
   .. py:attribute:: variance
   .. py:attribute:: stddev
   .. py:attribute:: skewness
   .. py:attribute:: kurtosis
   .. py:method:: close()

.. py:class:: cimba.Dataset()

   Resizable collection of unweighted float samples.

   .. py:method:: add(value)
   .. py:method:: values()
   .. py:method:: summary()
   .. py:method:: reset()
   .. py:method:: copy()
   .. py:method:: merge(other)
   .. py:method:: sort()
   .. py:method:: acf(lags)
   .. py:method:: pacf(lags)
   .. py:attribute:: count
   .. py:attribute:: min
   .. py:attribute:: max
   .. py:attribute:: median
   .. py:method:: close()

.. py:class:: cimba.TimeSeries()

   Sequence of ``(time, value, weight)`` rows.

   .. py:method:: add(value, time)
   .. py:method:: finalize(time)
   .. py:method:: values()
   .. py:method:: summary()
   .. py:method:: reset()
   .. py:method:: copy()
   .. py:method:: sort_by_value()
   .. py:method:: sort_by_time()
   .. py:method:: acf(lags)
   .. py:method:: pacf(lags)
   .. py:attribute:: count
   .. py:attribute:: min
   .. py:attribute:: max
   .. py:attribute:: median
   .. py:method:: close()
