"""Only the compiled modeling API can execute simulations from Python."""

import importlib.util

import cimba
from cimba import _cimba_native


def test_native_module_has_no_python_operations():
    assert [name for name in dir(_cimba_native) if not name.startswith("__")] == []


def test_lower_level_entry_points_are_removed():
    for name in ("run_experiment", "run_native_experiment", "set_native_thread_hooks", "gil_enabled"):
        assert not hasattr(cimba, name)
    for name in ("AliasSampler", "seed", "current_seed", "hwseed", "random_u64", "fmix64"):
        assert not hasattr(cimba.random, name)
    for name in ("cimba.cimba", "cimba.cmb_logger"):
        assert importlib.util.find_spec(name) is None
