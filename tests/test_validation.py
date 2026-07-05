import pytest

import cimba


def test_amount_capacity_handle_seed_and_priority_validation():
    with pytest.raises(ValueError):
        cimba.Buffer("Bad", capacity=0)
    with pytest.raises(OverflowError):
        cimba.Buffer("Bad", capacity=1 << 64)
    with pytest.raises(TypeError):
        cimba.Buffer("Bad", capacity=1.5)
    with pytest.raises(ValueError):
        cimba.random.seed(-1)
    with pytest.raises(OverflowError):
        cimba.random.seed(1 << 64)
    with pytest.raises(OverflowError):
        cimba.Process("Bad", lambda: None, priority=1 << 63)

    with cimba.Simulation(seed=1):
        buf = cimba.Buffer("Buf", capacity=1)
        with pytest.raises(ValueError):
            buf.put(0)
        with pytest.raises(ValueError):
            buf.get(-1)

        queue = cimba.PriorityQueue("Priority")
        with pytest.raises(ValueError):
            queue.position(0)
        with pytest.raises(OverflowError):
            queue.reprioritize(1 << 64, 0)


def test_duration_and_signal_validation():
    with cimba.Simulation(seed=1):
        proc = cimba.Process("Proc", lambda: None).start()
        with pytest.raises(ValueError):
            proc.timer_add(-1.0)
        with pytest.raises(ValueError):
            proc.interrupt(cimba.SUCCESS)
        with pytest.raises(OverflowError):
            proc.interrupt(1 << 63)

    with pytest.raises(ValueError):
        with cimba.Simulation(seed=1) as sim:
            sim.stop_at(-1.0)
