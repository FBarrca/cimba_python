Reporting
=========

.. py:module:: cimba.reporting

Structured reporting helpers build on the Python data surfaces already exposed
by :class:`cimba.Dataset` and :class:`cimba.TimeSeries`. They return plain Python
objects that can be formatted, inspected, or plotted.

The C API exposes this area as printer-oriented functions such as
``cmb_dataset_fivenum_print()``, ``cmb_dataset_histogram_print()``,
``cmb_dataset_correlogram_print()``, and resource-specific report printers.
Python keeps the same reporting coverage but returns structured values first.
Use ``format_report()`` for debug text, or the plotting helpers for figures.
The C functions are not exposed under their ``cmb_*`` names.

Install plotting support with the optional extra:

.. code-block:: console

   pip install "cimba[plot]"

.. py:class:: SummaryStats

   Availability-aware summary statistics. Fields that do not make sense for the
   available sample count are ``None``.

.. py:class:: FiveNumberSummary

   Five-number summary with ``min``, ``q1``, ``median``, ``q3``, ``max``, and a
   ``weighted`` flag. This is the Python equivalent of the C five-number
   printers.

.. py:class:: HistogramBin

   One histogram bucket with ``lower``, ``upper``, ``mass``, ``underflow``, and
   ``overflow`` fields.

.. py:class:: Histogram

   Histogram data with explicit underflow and overflow buckets.

.. py:class:: Correlogram

   Autocorrelation or partial-autocorrelation coefficients.

.. py:class:: HistoryReport

   Structured report containing a title, summary, histogram, and optional
   correlogram.

.. py:function:: summarize(source)

   Return :class:`SummaryStats` for a
   :class:`cimba.DataSummary`, :class:`cimba.WeightedSummary`,
   :class:`cimba.Dataset`, or :class:`cimba.TimeSeries`.

.. py:function:: histogram(source, bins=20, range=None, weighted="auto")

   Return a :class:`Histogram` for a :class:`cimba.Dataset` or
   :class:`cimba.TimeSeries`. Time series histograms are duration-weighted by
   default.

.. py:function:: five_number(source)

   Return a :class:`FiveNumberSummary` for a :class:`cimba.Dataset` or
   :class:`cimba.TimeSeries`. Time series summaries are duration-weighted.

.. py:function:: correlogram(source, lags, kind="acf")

   Return a :class:`Correlogram` using ``source.acf(lags)`` or
   ``source.pacf(lags)``.

.. py:function:: history_report(source, *, title=None, bins=20, lags=None, correlation=None)

   Return a :class:`HistoryReport` for a dataset or time series.

.. py:function:: resource_report(resource, *, bins=None, lags=None, correlation=None)

   Return a :class:`HistoryReport` for a recorded buffer, queue, resource, or
   resource pool.

.. py:function:: format_report(report)

   Return text for a :class:`HistoryReport` matching the native Cimba report:
   the ``cmb_buffer_print_report()`` summary and character histogram followed by
   the ``cmb_dataset_correlogram_print()`` correlogram.

.. py:function:: format_summary(summary_or_source)

   Return compact debug text for a :class:`SummaryStats`,
   :class:`cimba.DataSummary`, :class:`cimba.WeightedSummary`,
   :class:`cimba.Dataset`, or :class:`cimba.TimeSeries`.

.. py:function:: plot_history(history, ax=None, **kwargs)
.. py:function:: plot_histogram(histogram_or_source, ax=None, **kwargs)
.. py:function:: plot_correlogram(correlogram_or_source, ax=None, **kwargs)
.. py:function:: plot_report(report, axes=None, **kwargs)

   Plot with Matplotlib. The dependency is imported only when one of these
   helpers is called.

C API Equivalents
-----------------

.. list-table::
   :header-rows: 1

   * - C reporting helper
     - Python reporting helper
   * - ``cmb_datasummary_print()``, ``cmb_wtdsummary_print()``
     - :func:`summarize`, :func:`format_summary`
   * - ``cmb_dataset_fivenum_print()``, ``cmb_timeseries_fivenum_print()``
     - :func:`five_number`
   * - ``cmb_dataset_histogram_print()``, ``cmb_timeseries_histogram_print()``
     - :func:`histogram`, :func:`plot_histogram`
   * - ``cmb_dataset_ACF()``, ``cmb_dataset_PACF()``, correlogram printers
     - :func:`correlogram`, :func:`plot_correlogram`
   * - ``cmb_buffer_print_report()`` and the queue/resource report printers
     - :func:`resource_report`, :func:`format_report`, :func:`plot_report`
   * - ``cmb_dataset_print()``, ``cmb_timeseries_print()``
     - :meth:`cimba.Dataset.values`, :meth:`cimba.TimeSeries.values`
