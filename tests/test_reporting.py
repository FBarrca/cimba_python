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

    @sim.process
    def driver(self):
        for i in range(12):
            self.d.add(float(i % 4))

        self.q.put(2)
        self.resource.acquire()
        self.pool.acquire(1)
        self.store.put(101)
        self.pqs[0].put(202, 5)
        sim.hold(1.0)

        self.q.get(1)
        self.resource.release()
        self.pool.release(1)
        self.store.take()
        self.pqs[0].take()
        sim.hold(1.0)

        self.q.get(1)
        sim.suspend()


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

    @sim.process
    def driver(self):
        for value in range(1, 5):
            self.d.add(float(value))

def build_reporting_model() -> ReportingModel:
    model = ReportingModel()

    return model


def test_sim_dataset_methods_compile_in_model_callbacks(tmp_path):
    report = tmp_path / "dataset_methods.txt"
    report_handle = sim.log_text(str(report))

    class ConfiguredDatasetMethodModel(DatasetMethodModel):
        @sim.collect
        def collect(self):
            self.n = float(self.d.count())
            self.avg = self.d.mean()
            self.sd = self.d.std()
            self.q25 = self.d.quantile(0.25)
            self.lo = self.d.min()
            self.hi = self.d.max()
            self.med = self.d.median()
            ok = self.d.print_file(report_handle, append=0)
            ok += self.d.fivenum_file(report_handle, append=1)
            ok += self.d.histogram_file(
                report_handle, append=1, bins=4, low=0.0, high=0.0)
            self.ok = float(ok)

    model = ConfiguredDatasetMethodModel()



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

        @sim.process
        def driver(self):
            for i in range(4):
                self.d.add(float(i + 1))

        @sim.collect
        def collect(self):
            self.avg = self.d.mean()
            self.n = float(self.d.count())
            self.d.capture()

    model = Samples()



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

        @sim.collect
        def collect(self):
            self.station.samples.capture()

    model = Clinic()


    exp = model.experiment(replications=1, duration=1.0, warmup=0.0,
                           seed=19)
    assert exp.run() == 0
    assert exp.dataset("station__samples").tolist() == [2.0, 5.0]


def test_dataset_capture_rejects_invalid_targets():
    class Samples(sim.Model):
        d: sim.Dataset

        @sim.process
        def driver(self):
            self.d.add(1.0)

        @sim.collect
        def collect_with_args(self):
            self.d.capture(1)

    with pytest.raises(ValueError, match="dataset capture\\(\\) takes no"):
        Samples()

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

    class FileReportingModel(ReportingModel):
        @sim.collect
        def collect(self):
            self.n = float(self.q.history().count())
            ok = self.q.report_file(report_handle, 0)
            ok += self.resource.report_file(report_handle, 1)
            ok += self.pool.report_file(report_handle, 1)
            ok += self.store.report_file(report_handle, 1)
            ok += self.pqs[0].report_file(report_handle, 1)
            ok += self.q.history().print_file(report_handle, 1)
            ok += self.q.history().fivenum_file(report_handle, 1)
            ok += self.q.history().histogram_file(report_handle, 1,
                                                 4, 0.0, 4.0)
            ok += self.q.history().correlogram_file(report_handle, 1, 2)
            ok += self.q.history().pacf_correlogram_file(report_handle, 1, 2)
            ok += self.d.print_file(report_handle, 1)
            ok += self.d.fivenum_file(report_handle, 1)
            ok += self.d.histogram_file(report_handle, 1, 4, 0.0, 0.0)
            ok += self.d.correlogram_file(report_handle, 1, 2)
            ok += self.d.pacf_correlogram_file(report_handle, 1, 2)
            self.ok = float(ok)

    model = FileReportingModel()

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
    class TimeseriesReportingModel(ReportingModel):
        @sim.collect
        def collect(self):
            self.n = float(self.q.history().count())
            self.ok = self.q.history().mean() + self.pqs[0].history().mean()

    model = TimeseriesReportingModel()

    exp = model.experiment(replications=1, duration=5.0, warmup=0.0, seed=17)
    assert exp.run() == 0
    assert exp["n"][0] >= 3.0
    assert exp["ok"][0] > 0.0


def test_timeseries_history_capture_returns_per_trial_arrays():
    class CaptureModel(sim.Model):
        mean: sim.Output
        q: sim.Queue = sim.capacity(5)

        @sim.process
        def driver(self):
            for _ in range(3):
                self.q.put(1)
                sim.hold(1.0)
                self.q.get(1)
                sim.hold(1.0)
            sim.suspend()

        @sim.collect
        def collect(self):
            self.mean = self.q.history().mean()
            self.q.history().capture()

    model = CaptureModel()



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

def test_indexed_component_history_capture_returns_item_per_trial():
    class Counter(sim.Component):
        line: sim.Queue = sim.capacity(10)
        resource: sim.Resource
        store: sim.Store

    class QueueModel(sim.Model):
        counters: list[Counter] = [Counter(), Counter(), Counter()]
        q: sim.Queue = sim.capacity(5)

        @sim.process
        def arrivals(self):
            self.q.put(1)
            for index in range(len(self.counters)):
                for _ in range(index + 1):
                    self.counters[index].line.put(1)
                self.counters[index].resource.acquire()
                self.counters[index].store.put(index)
            sim.suspend()

        @sim.collect
        def collect(self):
            self.q.history().capture()
            for index in range(len(self.counters)):
                self.counters[index].line.history().capture()
                self.counters[index].resource.history().capture()
                self.counters[index].store.history().capture()

    model = QueueModel()



    exp = model.experiment(replications=2, duration=1.0, warmup=0.0,
                           seed=23)
    with pytest.raises(RuntimeError, match="run"):
        exp.histories("counters__line")
    assert exp.run() == 0

    all_lines = exp.histories("counters__line")
    assert len(all_lines) == len(exp)
    assert all(len(trial_rows) == 3 for trial_rows in all_lines)
    assert all(rows.dtype == np.float64 for trial_rows in all_lines
               for rows in trial_rows)
    assert all(rows.ndim == 2 and rows.shape[1] == 3
               for trial_rows in all_lines for rows in trial_rows)
    assert [float(rows[:, 1].max()) for rows in all_lines[0]] == [1, 2, 3]

    assert len(exp.histories("counters__resource")[0]) == 3
    assert len(exp.histories("counters__store")[0]) == 3
    assert np.array_equal(
        exp.history("counters__line", trial=0, index=2),
        all_lines[0][2],
    )
    with pytest.raises(TypeError, match="requires an index"):
        exp.history("counters__line")
    with pytest.raises(IndexError, match="collection index"):
        exp.history("counters__line", index=3)
    with pytest.raises(TypeError, match="not indexed"):
        exp.history("q", index=0)


def test_indexed_component_history_capture_rejects_unbounded_index():
    class Counter(sim.Component):
        line: sim.Queue = sim.capacity(5)

    class QueueModel(sim.Model):
        counters: list[Counter] = [Counter(), Counter()]

        @sim.collect
        def collect(self):
            index = 1
            self.counters[index].line.history().capture()

    with pytest.raises(ValueError, match="unbounded index"):
        QueueModel()

    class ConstantModel(sim.Model):
        counters: list[Counter] = [Counter(), Counter()]

        @sim.collect
        def constant_collect(self):
            self.counters[2].line.history().capture()

    with pytest.raises(ValueError, match="out of range"):
        ConstantModel()


def test_one_item_component_history_capture_keeps_collection_dimension():
    class Counter(sim.Component):
        line: sim.Queue = sim.capacity(5)

    class QueueModel(sim.Model):
        counters: list[Counter] = [Counter()]

        @sim.process
        def driver(self):
            self.counters[0].line.put(1)
            sim.suspend()

        @sim.collect
        def collect(self):
            self.counters[0].line.history().capture()

    model = QueueModel()



    exp = model.experiment(duration=1.0, warmup=0.0, seed=24)
    assert exp.run() == 0
    rows = exp.histories("counters__line")[0]
    assert len(rows) == 1
    assert rows[0].shape[1] == 3


def test_constant_indexed_history_capture_leaves_other_items_empty():
    class Counter(sim.Component):
        line: sim.Queue = sim.capacity(5)

    class QueueModel(sim.Model):
        counters: list[Counter] = [Counter(), Counter(), Counter()]

        @sim.process
        def driver(self):
            self.counters[1].line.put(1)
            sim.suspend()

        @sim.collect
        def collect(self):
            self.counters[1].line.history().capture()

    model = QueueModel()



    exp = model.experiment(duration=1.0, warmup=0.0, seed=25)
    assert exp.run() == 0
    rows = exp.histories("counters__line")[0]
    assert rows[0].shape == (0, 3)
    assert rows[1].shape[1] == 3 and rows[1].shape[0] > 0
    assert rows[2].shape == (0, 3)

def test_timeseries_history_capture_rejects_invalid_targets():
    class DatasetCapture(sim.Model):
        d: sim.Dataset

        @sim.process
        def dataset_driver(self):
            self.d.add(1.0)

        @sim.collect
        def dataset_collect(self):
            self.d.history().capture()

    with pytest.raises(ValueError, match="unknown history field"):
        DatasetCapture()

    class IndexedCapture(sim.Model):
        pqs: sim.PQueues = sim.count(1)

        @sim.process
        def indexed_driver(self):
            self.pqs[0].put(1, 0)

        @sim.collect
        def indexed_collect(self):
            self.pqs[0].history().capture()

    with pytest.raises(ValueError, match="indexed entity"):
        IndexedCapture()


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

        @sim.collect
        def collect(self):
            self.station.q.history().capture()

    model = Clinic()


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
    class StdoutReportingModel(ReportingModel):
        @sim.collect
        def collect(self):
            ok = self.q.report()
            ok += self.resource.report()
            ok += self.pool.report()
            ok += self.store.report()
            ok += self.pqs[0].report()
            ok += self.q.history().histogram(4, 0.0, 4.0)
            ok += self.d.histogram(4, 0.0, 0.0)
            self.ok = float(ok)

    model = StdoutReportingModel()

    exp = model.experiment(replications=1, duration=5.0, warmup=0.0, seed=23)
    text = capture_native_stdout(exp.run)
    assert exp["ok"][0] == 7.0
    assert "Buffer levels for q" in text
    assert "Resource utilization for resource:" in text
    assert "Pool resource utilization for pool:" in text
    assert "Queue lengths for store:" in text
    assert "Queue lengths for pqs_0:" in text
    assert "#" in text
