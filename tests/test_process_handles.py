"""Process handles and stackful suspension work in the new compiler."""

import cimba as cb


class SelfTimer(cb.Model):
    status: cb.Output[int]
    signal: cb.Output[int]
    woke_at: cb.Output[float]

    @cb.process
    def wait(self):
        process = cb.this_process()
        self.status = process.status()
        process.timer_set(1.5, 42)
        self.signal = cb.suspend()
        self.woke_at = cb.now()


def test_process_handle_and_suspend():
    model = SelfTimer()
    result = cb.Experiment(model, replications=2).run(workers=1)
    assert result[model].status.values.tolist() == [[1.0, 1.0]]
    assert result[model].signal.values.tolist() == [[42.0, 42.0]]
    assert result[model].woke_at.values.tolist() == [[1.5, 1.5]]


class EndEarly(cb.Model):
    ended_at: cb.Output[float]

    @cb.process
    def stop(self):
        cb.hold(1.0)
        cb.end_trial()

    @cb.on_end
    def finish(self):
        self.ended_at = cb.now()


def test_end_trial_completes_successfully_before_window_end():
    model = EndEarly()
    result = cb.Experiment(model, window=cb.Window(duration=10)).run(workers=1)
    assert not result.failed.any()
    assert result[model].ended_at.values.tolist() == [[1.0]]
