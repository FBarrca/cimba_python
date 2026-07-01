import os

import cimba.sim as sim


def capture_native_stdout(fn):
    read_fd, write_fd = os.pipe()
    saved_fd = os.dup(1)
    try:
        os.dup2(write_fd, 1)
        os.close(write_fd)
        fn()
        os.dup2(saved_fd, 1)
        chunks = []
        while True:
            chunk = os.read(read_fd, 8192)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks).decode()
    finally:
        try:
            os.dup2(saved_fd, 1)
        except OSError:
            pass
        os.close(saved_fd)
        try:
            os.close(read_fd)
        except OSError:
            pass


class ReportingModel(sim.Model):
    ok: sim.Output
    n: sim.Output
    q: sim.Queue = sim.capacity(5)
    d: sim.Dataset
    resource: sim.Resource
    pool: sim.Pool = 2
    store: sim.Store = sim.capacity(4)
    pqs: sim.PQueues = sim.count(1)


def build_reporting_model() -> ReportingModel:
    model = ReportingModel()

    @model.process
    def driver(env: ReportingModel):
        for i in range(12):
            sim.tally(env.d, float(i % 4))

        sim.put(env.q, 2)
        sim.acquire(env.resource)
        sim.pool_acquire(env.pool, 1)
        sim.store_put(env.store, 101)
        sim.pq_put(env.pqs[0], 202, 5)
        sim.hold(1.0)

        sim.get(env.q, 1)
        sim.release(env.resource)
        sim.pool_release(env.pool, 1)
        sim.store_take(env.store)
        sim.pq_take(env.pqs[0])
        sim.hold(1.0)

        sim.get(env.q, 1)
        sim.suspend()

    return model


def test_native_text_report_file_variants_cover_public_helpers(tmp_path):
    report = tmp_path / "native_reports.txt"
    report_handle = sim.log_text(str(report))
    model = build_reporting_model()

    @model.collect
    def collect(env: ReportingModel):
        ts = sim.queue_history(env.q)
        env.n = float(sim.timeseries_count(ts))
        ok = sim.queue_report_file(env.q, report_handle, 0)
        ok += sim.resource_report_file(env.resource, report_handle, 1)
        ok += sim.pool_report_file(env.pool, report_handle, 1)
        ok += sim.store_report_file(env.store, report_handle, 1)
        ok += sim.pq_report_file(env.pqs[0], report_handle, 1)
        ok += sim.timeseries_print_file(ts, report_handle, 1)
        ok += sim.timeseries_fivenum_file(ts, report_handle, 1)
        ok += sim.timeseries_histogram_file(ts, report_handle, 1,
                                            4, 0.0, 4.0)
        ok += sim.timeseries_correlogram_file(ts, report_handle, 1, 2)
        ok += sim.timeseries_pacf_correlogram_file(ts, report_handle, 1, 2)
        ok += sim.dataset_print_file(env.d, report_handle, 1)
        ok += sim.dataset_fivenum_file(env.d, report_handle, 1)
        ok += sim.dataset_histogram_file(env.d, report_handle, 1,
                                         4, 0.0, 0.0)
        ok += sim.dataset_correlogram_file(env.d, report_handle, 1, 2)
        ok += sim.dataset_pacf_correlogram_file(env.d, report_handle, 1, 2)
        env.ok = float(ok)

    exp = model.experiment(replications=1, duration=5.0, warmup=0.0, seed=17)
    assert exp.run() == 0
    assert exp["ok"][0] == 15.0
    assert exp["n"][0] >= 3.0

    text = report.read_text()
    assert "Buffer levels for q" in text
    assert "Resource utilization for resource:" in text
    assert "Pool resource utilization for pool:" in text
    assert "Queue lengths for store:" in text
    assert "Queue lengths for pqs_0:" in text
    assert "#" in text
    assert "-1.0" in text and "1.0" in text


def test_native_text_report_stdout_variants_print_to_console():
    model = build_reporting_model()

    @model.collect
    def collect(env: ReportingModel):
        ts = sim.queue_history(env.q)
        ok = sim.queue_report(env.q)
        ok += sim.resource_report(env.resource)
        ok += sim.pool_report(env.pool)
        ok += sim.store_report(env.store)
        ok += sim.pq_report(env.pqs[0])
        ok += sim.timeseries_histogram(ts, 4, 0.0, 4.0)
        ok += sim.dataset_histogram(env.d, 4, 0.0, 0.0)
        env.ok = float(ok)

    exp = model.experiment(replications=1, duration=5.0, warmup=0.0, seed=23)
    text = capture_native_stdout(exp.run)
    assert exp["ok"][0] == 7.0
    assert "Buffer levels for q" in text
    assert "Resource utilization for resource:" in text
    assert "Pool resource utilization for pool:" in text
    assert "Queue lengths for store:" in text
    assert "Queue lengths for pqs_0:" in text
    assert "#" in text
