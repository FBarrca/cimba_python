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


class DatasetMethodModel(sim.Model):
    ok: sim.Output
    n: sim.Output
    avg: sim.Output
    sd: sim.Output
    q25: sim.Output
    lo: sim.Output
    hi: sim.Output
    med: sim.Output
    d: sim.Dataset


def build_reporting_model() -> ReportingModel:
    model = ReportingModel()

    @model.process
    def driver(env: ReportingModel):
        for i in range(12):
            env.d.add(float(i % 4))

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


def test_sim_dataset_methods_compile_in_model_callbacks(tmp_path):
    report = tmp_path / "dataset_methods.txt"
    report_handle = sim.log_text(str(report))
    model = DatasetMethodModel()

    @model.process
    def driver(env: DatasetMethodModel):
        for value in range(1, 5):
            env.d.add(float(value))

    @model.collect
    def collect(env: DatasetMethodModel):
        env.n = float(env.d.count())
        env.avg = env.d.mean()
        env.sd = env.d.std()
        env.q25 = env.d.quantile(0.25)
        env.lo = env.d.min()
        env.hi = env.d.max()
        env.med = env.d.median()
        ok = env.d.print_file(report_handle, append=0)
        ok += env.d.fivenum_file(report_handle, append=1)
        ok += env.d.histogram_file(
            report_handle, append=1, bins=4, low=0.0, high=0.0)
        env.ok = float(ok)

    exp = model.experiment(replications=1, duration=1.0, warmup=0.0, seed=29)
    assert exp.run() == 0
    assert exp["ok"][0] == 3.0
    assert exp["n"][0] == 4.0
    assert exp["avg"][0] == 2.5
    assert exp["sd"][0] > 1.29
    assert exp["q25"][0] == 1.75
    assert exp["lo"][0] == 1.0
    assert exp["hi"][0] == 4.0
    assert exp["med"][0] == 2.5
    assert "#" in report.read_text()


def test_native_text_report_file_variants_cover_dataset_methods(tmp_path):
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
        ok += env.d.print_file(report_handle, 1)
        ok += env.d.fivenum_file(report_handle, 1)
        ok += env.d.histogram_file(report_handle, 1, 4, 0.0, 0.0)
        ok += env.d.correlogram_file(report_handle, 1, 2)
        ok += env.d.pacf_correlogram_file(report_handle, 1, 2)
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
        ok += env.d.histogram(4, 0.0, 0.0)
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
