from array import array
import importlib

import pytest

import cimba


_native = importlib.import_module("cimba._cimba")


def test_run_native_experiment_mutates_writable_struct_buffer():
    trials = array("Q", [1, 41, 99])

    cimba.run_native_experiment(trials, trials.itemsize, _native._test_trial_func_capsule())

    assert trials.tolist() == [2, 42, 100]


def test_native_thread_hook_capsules_can_be_set_and_cleared():
    trials = array("Q", [0, 10])

    cimba.set_native_thread_hooks(
        _native._test_thread_init_capsule(),
        _native._test_user_context_capsule(),
        _native._test_thread_exit_capsule(),
    )
    try:
        cimba.run_native_experiment(trials, trials.itemsize, _native._test_trial_func_capsule())
    finally:
        cimba.set_native_thread_hooks()

    assert trials.tolist() == [1, 11]


def test_native_experiment_rejects_python_callables_and_bad_capsules():
    trials = bytearray(8)

    with pytest.raises(TypeError):
        cimba.run_native_experiment(trials, 8, lambda ptr: None)
    with pytest.raises(TypeError):
        cimba.run_native_experiment(trials, 8, _native._test_thread_init_capsule())
    with pytest.raises(TypeError):
        cimba.set_native_thread_hooks(lambda arg, tid: None)


def test_native_experiment_validates_buffer_shape_and_size():
    capsule = _native._test_trial_func_capsule()

    with pytest.raises(TypeError):
        cimba.run_native_experiment(bytes(8), 8, capsule)
    with pytest.raises(TypeError):
        cimba.run_native_experiment(memoryview(bytearray(16))[::2], 1, capsule)
    with pytest.raises(ValueError):
        cimba.run_native_experiment(bytearray(10), 8, capsule)
    with pytest.raises(ValueError):
        cimba.run_native_experiment(bytearray(8), 0, capsule)
