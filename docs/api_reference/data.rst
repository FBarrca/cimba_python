Datasets, Summaries and Reporting
=================================

``sim.Dataset`` collects untimed samples with ``sim.tally()``. Time-weighted
histories are attached to simulation entities and read back through the
``*_history()`` accessors as time-series handles.

Datasets
--------

``tally()``, ``dataset_count()``, ``dataset_mean()``, ``dataset_min()``,
``dataset_max()``, ``dataset_std()``, ``dataset_median()``,
``dataset_quantile()``, ``dataset_print()``, ``dataset_print_file()``,
``dataset_fivenum()``, ``dataset_fivenum_file()``, ``dataset_histogram()``,
``dataset_histogram_file()``, ``dataset_correlogram()``,
``dataset_correlogram_file()``, ``dataset_pacf_correlogram()``,
``dataset_pacf_correlogram_file()``.

Time series
-----------

Entity histories come from ``queue_history()``, ``resource_history()``,
``pool_history()``, ``store_history()``, and ``pq_history()``. The returned
handles are summarized and reported with:

``timeseries_count()``, ``timeseries_min()``, ``timeseries_max()``,
``timeseries_mean()``, ``timeseries_std()``, ``timeseries_median()``,
``timeseries_print()``, ``timeseries_print_file()``, ``timeseries_fivenum()``,
``timeseries_fivenum_file()``, ``timeseries_histogram()``,
``timeseries_histogram_file()``, ``timeseries_correlogram()``,
``timeseries_correlogram_file()``, ``timeseries_pacf_correlogram()``,
``timeseries_pacf_correlogram_file()``.

For text reports and text-mode plots, the no-suffix helpers print to stdout for
console and notebook use; the ``*_file()`` variants write to a path handle
created with ``sim.log_text()``. These are most useful in single-trial
debugging; scalar ``sim.Output`` fields are usually better for large parallel
experiments.

``Model.experiment(..., warmup=..., duration=...)`` controls the measurement
window: warmup lets the model reach a representative state before summaries are
collected.
