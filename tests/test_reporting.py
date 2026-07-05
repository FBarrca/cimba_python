import os

import numpy as np
import pytest

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

        env.q.put(2)
        env.resource.acquire()
        env.pool.acquire(1)
        env.store.put(101)
        env.pqs[0].put(202, 5)
        sim.hold(1.0)

        env.q.get(1)
        env.resource.release()
        env.pool.release(1)
        env.store.take()
        env.pqs[0].take()
        sim.hold(1.0)

        env.q.get(1)
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


def test_dataset_capture_returns_per_trial_arrays():
    class Samples(sim.Model):
        avg: sim.Output
        n: sim.Output
        d: sim.Dataset

    model = Samples()

    @model.process
    def driver(env: Samples):
        for i in range(4):
            env.d.add(float(i + 1))

    @model.collect
    def collect(env: Samples):
        env.avg = env.d.mean()
        env.n = float(env.d.count())
        env.d.capture()

    exp = model.experiment(replications=3, duration=1.0, warmup=0.0,
                           seed=41)
    with pytest.raises(RuntimeError, match="run\\(\\) the experiment"):
        exp.dataset("d")
    assert exp.run() == 0

    datasets = exp.datasets("d")
    assert len(datasets) == len(exp)
    for i, values in enumerate(datasets):
        assert values.dtype == np.float64
        assert values.ndim == 1
        assert values.tolist() == [1.0, 2.0, 3.0, 4.0]
        assert values.size == int(exp["n"][i])
        assert float(values.mean()) == pytest.approx(exp["avg"][i])

    first = exp.dataset("d", trial=0)
    first[:] = -1.0
    assert exp.run() == 0
    assert exp.dataset("d", trial=0).tolist() == [1.0, 2.0, 3.0, 4.0]


def test_dataset_capture_model_collect_uses_component_path():
    class Station(sim.Component):
        samples: sim.Dataset

        @sim.process
        def driver(self, env):
            self.samples.add(2.0)
            self.samples.add(5.0)

    class Clinic(sim.Model):
        station: Station = Station()

    model = Clinic()

    @model.collect
    def collect(env: Clinic):
        env.station.samples.capture()

    exp = model.experiment(replications=1, duration=1.0, warmup=0.0,
                           seed=19)
    assert exp.run() == 0
    assert exp.dataset("station__samples").tolist() == [2.0, 5.0]


def test_dataset_capture_rejects_invalid_targets():
    class Samples(sim.Model):
        d: sim.Dataset

    model = Samples()

    @model.process
    def driver(env: Samples):
        env.d.add(1.0)

    with pytest.raises(ValueError, match="dataset capture\\(\\) takes no"):
        @model.collect
        def collect_with_args(env: Samples):
            env.d.capture(1)

    class ComponentSamples(sim.Component):
        d: sim.Dataset

        @sim.process
        def driver(self, env):
            self.d.add(1.0)

        @sim.collect
        def collect(self, env):
            self.d.capture()

    with pytest.raises(ValueError, match="unsupported dataset method"):
        class Clinic(sim.Model):
            station: ComponentSamples = ComponentSamples()

        Clinic()


def test_native_text_report_file_variants_cover_dataset_methods(tmp_path):
    report = tmp_path / "native_reports.txt"
    report_handle = sim.log_text(str(report))
    model = build_reporting_model()

    @model.collect
    def collect(env: ReportingModel):
        env.n = float(env.q.history().count())
        ok = env.q.report_file(report_handle, 0)
        ok += env.resource.report_file(report_handle, 1)
        ok += env.pool.report_file(report_handle, 1)
        ok += env.store.report_file(report_handle, 1)
        ok += env.pqs[0].report_file(report_handle, 1)
        ok += env.q.history().print_file(report_handle, 1)
        ok += env.q.history().fivenum_file(report_handle, 1)
        ok += env.q.history().histogram_file(report_handle, 1,
                                             4, 0.0, 4.0)
        ok += env.q.history().correlogram_file(report_handle, 1, 2)
        ok += env.q.history().pacf_correlogram_file(report_handle, 1, 2)
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


def test_timeseries_history_method_compiles_in_model_callbacks():
    model = build_reporting_model()

    @model.collect
    def collect(env: ReportingModel):
        env.n = float(env.q.history().count())
        env.ok = env.q.history().mean() + env.pqs[0].history().mean()

    exp = model.experiment(replications=1, duration=5.0, warmup=0.0, seed=17)
    assert exp.run() == 0
    assert exp["n"][0] >= 3.0
    assert exp["ok"][0] > 0.0


def test_timeseries_history_capture_returns_per_trial_arrays():
    class CaptureModel(sim.Model):
        mean: sim.Output
        q: sim.Queue = sim.capacity(5)

    model = CaptureModel()

    @model.process
    def driver(env: CaptureModel):
        for _ in range(3):
            env.q.put(1)
            sim.hold(1.0)
            env.q.get(1)
            sim.hold(1.0)
        sim.suspend()

    @model.collect
    def collect(env: CaptureModel):
        env.mean = env.q.history().mean()
        env.q.history().capture()

    exp = model.experiment(replications=2, duration=10.0, warmup=0.0,
                           seed=17)

    with pytest.raises(RuntimeError, match="run"):
        exp.history("q")
    with pytest.raises(KeyError, match="unknown captured history"):
        exp.history("missing")

    assert exp.run() == 0

    histories = exp.histories("q")
    assert len(histories) == len(exp)
    for i, rows in enumerate(histories):
        assert rows.dtype == np.float64
        assert rows.ndim == 2
        assert rows.shape[1] == 3
        assert rows.shape[0] > 0
        duration = rows[:, 2].sum()
        assert duration > 0.0
        mean = float(np.sum(rows[:, 1] * rows[:, 2]) / duration)
        assert mean == pytest.approx(exp["mean"][i])

    first = exp.history("q", trial=0)
    first[:] = -1.0
    assert exp.run() == 0
    assert not np.all(exp.history("q", trial=0) == -1.0)


def test_timeseries_history_capture_rejects_invalid_targets():
    class DatasetCapture(sim.Model):
        d: sim.Dataset

    model = DatasetCapture()

    @model.process
    def dataset_driver(env: DatasetCapture):
        env.d.add(1.0)

    with pytest.raises(ValueError, match="unknown history field"):
        @model.collect
        def dataset_collect(env: DatasetCapture):
            env.d.history().capture()

    class IndexedCapture(sim.Model):
        pqs: sim.PQueues = sim.count(1)

    indexed = IndexedCapture()

    @indexed.process
    def indexed_driver(env: IndexedCapture):
        env.pqs[0].put(1, 0)

    with pytest.raises(ValueError, match="indexed entity"):
        @indexed.collect
        def indexed_collect(env: IndexedCapture):
            env.pqs[0].history().capture()


def test_timeseries_history_capture_rejects_component_collect():
    class Station(sim.Component):
        q: sim.Queue = sim.capacity(5)

        @sim.process
        def driver(self, env):
            self.q.put(1)

        @sim.collect
        def collect(self, env):
            self.q.history().capture()

    with pytest.raises(ValueError, match="unsupported timeseries method"):
        class Clinic(sim.Model):
            station: Station = Station()

        Clinic()


def test_timeseries_history_capture_model_collect_uses_component_path():
    class Station(sim.Component):
        q: sim.Queue = sim.capacity(5)

        @sim.process
        def driver(self, env):
            self.q.put(1)
            sim.hold(1.0)

    class Clinic(sim.Model):
        station: Station = Station()

    model = Clinic()

    @model.collect
    def collect(env: Clinic):
        env.station.q.history().capture()

    exp = model.experiment(replications=1, duration=2.0, warmup=0.0,
                           seed=18)
    assert exp.run() == 0
    rows = exp.history("station__q")
    assert rows.shape[1] == 3
    assert rows[:, 2].sum() > 0.0


def test_timeseries_history_method_compiles_in_components():
    class Station(sim.Component):
        q: sim.Queue = sim.capacity(5)
        resource: sim.Resource
        mean_qlen: sim.Output
        qcount: sim.Output
        mean_in_use: sim.Output

        @sim.process
        def driver(self, env):
            self.q.put(2)
            self.resource.acquire()
            sim.hold(1.0)
            self.q.get(1)
            self.resource.release()
            sim.hold(1.0)
            self.q.get(1)
            sim.suspend()

        @sim.collect
        def collect(self, env):
            self.mean_qlen = self.q.history().mean()
            self.qcount = float(self.q.history().count())
            self.mean_in_use = self.resource.history().mean()

    class Clinic(sim.Model):
        station: Station = Station()

    model = Clinic()
    exp = model.experiment(replications=1, duration=5.0, warmup=0.0, seed=17)
    assert exp.run() == 0
    assert exp["station__qcount"][0] == 5.0
    assert exp["station__mean_qlen"][0] > 0.0
    assert exp["station__mean_in_use"][0] > 0.0


def test_native_text_report_stdout_variants_print_to_console():
    model = build_reporting_model()

    @model.collect
    def collect(env: ReportingModel):
        ok = env.q.report()
        ok += env.resource.report()
        ok += env.pool.report()
        ok += env.store.report()
        ok += env.pqs[0].report()
        ok += env.q.history().histogram(4, 0.0, 4.0)
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
