"""Experiment execution and its model-shaped results."""

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

import numpy as np

from ._capture import (
    HISTORY_CAPTURE_STORE_FIELD,
    HISTORY_CAPTURE_TRIAL_FIELD,
    HistoryCaptureSpec,
    copy_capture_store,
    create_capture_store,
    destroy_capture_store,
)
from ._cimba import ffi, lib
from ._runtime import _TRIAL_COMPLETED_FIELD

if TYPE_CHECKING:
    from ._model import ComponentFieldSchema, Model

_ExperimentResultT = TypeVar(
    "_ExperimentResultT", default="ExperimentResults")


@dataclass(frozen=True)
class _ResultLeaf:
    family: str
    flattened_name: str


class _ResultNamespace:
    """Read-only attribute access over the model's structured results."""

    __slots__ = ("_experiment", "_entries", "_path")

    def __init__(
        self,
        experiment: "Experiment[Any]",
        entries: Mapping[str, "_ResultLeaf | _ResultNamespace"],
        path: str,
    ) -> None:
        object.__setattr__(self, "_experiment", experiment)
        object.__setattr__(self, "_entries", MappingProxyType(dict(entries)))
        object.__setattr__(self, "_path", path)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"result namespace '{self._path}' is read-only")

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        entry = self._entries.get(name)
        if entry is None:
            raise AttributeError(
                f"unknown result '{self._path}.{name}'; available names: "
                f"{', '.join(sorted(self._entries)) or '<none>'}"
            )
        if isinstance(entry, _ResultNamespace):
            return entry
        return self._experiment._result_value(entry)

    def __dir__(self) -> list[str]:
        return sorted(set(object.__dir__(self)) | set(self._entries))

    def __repr__(self) -> str:
        names = ", ".join(sorted(self._entries))
        return f"<{type(self).__name__} {self._path}: {names}>"


class ExperimentResults(_ResultNamespace):
    """The model-shaped structured results retained by an experiment.

    Outputs, captured datasets, and captured histories share this one object
    tree. Users may parameterize ``Model`` with a Protocol describing the
    exact tree for IDE completion and static checking.
    """


def _insert_result_leaf(
    tree: dict[str, Any],
    parts: Sequence[str],
    leaf: _ResultLeaf,
) -> None:
    """Insert a path, retaining structural component branches on conflicts."""
    if not parts or any(not part for part in parts):
        return
    node = tree
    for part in parts[:-1]:
        existing = node.get(part)
        if isinstance(existing, _ResultLeaf):
            return
        if existing is None:
            existing = {}
            node[part] = existing
        node = existing
    final = parts[-1]
    if isinstance(node.get(final), dict):
        return
    node.setdefault(final, leaf)


def _component_result_path(schema: "ComponentFieldSchema") -> tuple[str, ...]:
    return tuple(part for part in schema.path.replace("[]", "").split(".")
                 if part)


def _result_tree(experiment: "Experiment[Any]") -> dict[str, Any]:
    """Merge outputs and captures into the model's nested object tree."""
    model = experiment.model
    tree: dict[str, Any] = {}
    schemas = model.component_schema()

    component_output_names: set[str] = set()
    for schema in schemas:
        if schema.kind != "output":
            continue
        component_output_names.add(schema.flattened_name)
        _insert_result_leaf(
            tree,
            _component_result_path(schema),
            _ResultLeaf("outputs", schema.flattened_name),
        )
    for name in model.outputs:
        if name not in component_output_names:
            _insert_result_leaf(
                tree, (name,), _ResultLeaf("outputs", name))

    def add_captures(
        family: str,
        names: Sequence[str],
        kinds: set[str],
    ) -> None:
        schema_paths: dict[str, list[tuple[str, ...]]] = {}
        for schema in schemas:
            if schema.kind in kinds:
                schema_paths.setdefault(schema.flattened_name, []).append(
                    _component_result_path(schema))
        for name in names:
            paths = schema_paths.get(name, [(name,)])
            for path in paths:
                _insert_result_leaf(tree, path, _ResultLeaf(family, name))

    add_captures(
        "datasets", experiment._dataset_capture_names, {"dataset"})
    add_captures(
        "histories", experiment._history_capture_names,
        {"queue", "resource", "pool", "store", "pqueues"})
    return tree


def _result_namespaces(
    experiment: "Experiment[Any]",
    tree: Mapping[str, Any],
    path: str,
) -> Mapping[str, _ResultLeaf | _ResultNamespace]:
    entries: dict[str, _ResultLeaf | _ResultNamespace] = {}
    for name, value in tree.items():
        if isinstance(value, dict):
            entries[name] = _ResultNamespace(
                experiment,
                _result_namespaces(experiment, value, f"{path}.{name}"),
                f"{path}.{name}",
            )
        else:
            entries[name] = value
    return entries


def _build_result_namespace(
    experiment: "Experiment[Any]",
) -> ExperimentResults:
    return ExperimentResults(
        experiment,
        _result_namespaces(experiment, _result_tree(experiment), "results"),
        "results",
    )


class Experiment(Generic[_ExperimentResultT]):
    model: "Model[_ExperimentResultT]"
    #: One structured record per trial; outputs are filled in by run().
    trials: np.ndarray
    #: Number of failed trials in the last run(), or None before it.
    failures: int | None
    #: Replications per design point (trial order is design-point-major
    #: with replications innermost).
    replications: int
    #: Names of the parameters swept over more than one value.
    swept: tuple[str, ...]
    #: Typed/dynamic access to retained outputs, datasets, and histories.
    results: _ExperimentResultT

    def __init__(self, model: "Model[_ExperimentResultT]",
                 trials: np.ndarray, trial_addr: int,
                 keepalive: Sequence[np.ndarray] = (),
                 replications: int = 1, swept: Sequence[str] = (),
                 history_captures: Sequence[HistoryCaptureSpec] = (),
                 dataset_captures: Sequence[HistoryCaptureSpec] = ()):
        self.model = model
        self.trials = trials
        self._trial_addr = trial_addr
        # Trace arrays whose data pointers live in the trial records
        self._keepalive = tuple(keepalive)
        self.failures = None
        self._run_lock = threading.Lock()
        self.replications = replications
        self.swept = tuple(swept)
        ordered_captures = sorted(history_captures, key=lambda spec: spec.slot)
        ordered_datasets = sorted(dataset_captures, key=lambda spec: spec.slot)
        self._history_capture_specs = tuple(ordered_captures)
        self._dataset_capture_specs = tuple(ordered_datasets)
        self._capture_slot_count = sum(
            spec.slot_count
            for spec in (*self._history_capture_specs,
                         *self._dataset_capture_specs)
        )
        self._history_capture_names = tuple(
            spec.name for spec in self._history_capture_specs)
        self._dataset_capture_names = tuple(
            spec.name for spec in self._dataset_capture_specs)
        self._history_capture_data: dict[str, Any] | None = None
        self._dataset_capture_data: dict[str, Any] | None = None
        # The concrete runtime object is the dynamic fallback. A parameterized
        # Model supplies a narrower static result schema to type checkers.
        self.results = cast(_ExperimentResultT,
                            _build_result_namespace(self))

    def _result_value(self, leaf: _ResultLeaf) -> Any:
        if leaf.family == "outputs":
            return self.trials[leaf.flattened_name]
        if leaf.family == "datasets":
            return self.datasets(leaf.flattened_name)
        if leaf.family == "histories":
            return self.histories(leaf.flattened_name)
        raise ValueError(f"unknown result family: {leaf.family}")

    @property
    def failed(self) -> np.ndarray:
        """One failure flag per trial, independent of its output values."""
        if self.failures is None:
            raise RuntimeError("run() the experiment before reading failed")
        return self.trials[_TRIAL_COMPLETED_FIELD] == 0

    def run(self) -> int:
        """Run fresh trials in place, retaining the inputs and seeds.

        State starts at zero and outputs at NaN on every run. Native failures
        and uncaught compiled exceptions invalidate every output of that trial.
        """
        with self._run_lock:
            return self._run()

    def _run(self) -> int:
        trials = self.trials
        if trials.dtype.names is None:
            raise TypeError("experiment must be a structured array")
        if not trials.flags["C_CONTIGUOUS"]:
            raise ValueError("experiment array must be C-contiguous")
        if trials.ndim != 1 or trials.size == 0:
            raise ValueError("experiment must be a non-empty 1-D array")

        self.failures = None
        trials[_TRIAL_COMPLETED_FIELD] = 0
        for field in (*self.model.state, *self.model.float_state):
            trials[field] = 0
        for field in self.model.outputs:
            trials[field] = np.nan

        fptr = ffi.cast("void(*)(void *)", self._trial_addr)
        buf = ffi.from_buffer(trials, require_writable=True)
        capture_store = ffi.NULL
        if self._capture_slot_count:
            self._history_capture_data = None
            self._dataset_capture_data = None
            capture_store = create_capture_store(
                trials.size, self._capture_slot_count)
            trials[HISTORY_CAPTURE_TRIAL_FIELD] = np.arange(
                trials.size, dtype=np.uint64)
            trials[HISTORY_CAPTURE_STORE_FIELD] = int(
                ffi.cast("intptr_t", capture_store))
        try:
            failures = int(lib.cimba_run(buf, trials.size, trials.itemsize, fptr))
            if self._capture_slot_count:
                self._history_capture_data = copy_capture_store(
                    capture_store,
                    num_trials=trials.size,
                    specs=self._history_capture_specs,
                )
                self._dataset_capture_data = copy_capture_store(
                    capture_store,
                    num_trials=trials.size,
                    specs=self._dataset_capture_specs,
                )
        finally:
            if capture_store != ffi.NULL:
                destroy_capture_store(capture_store)
                trials[HISTORY_CAPTURE_STORE_FIELD] = 0

        failed = trials[_TRIAL_COMPLETED_FIELD] == 0
        if int(failed.sum()) != failures:
            raise RuntimeError("native trial outcomes disagree with completion records")
        for field in self.model.outputs:
            trials[field][failed] = np.nan
        self.failures = failures
        return self.failures

    def summary(self, *outputs: str,
                confidence: float = 0.95) -> np.ndarray:
        """Summarize outputs across replications: a structured array with
        one record per design point, holding the swept parameter values
        and, for each output, its replication mean under its own name and
        the Student-t confidence-interval half-width under
        ``<name>_hw``. With no arguments every output is summarized.

        Failed trials are excluded from every output. Missing (NaN) values
        from successful trials are excluded per output; the mean is NaN with
        no observations and the half-width is NaN with fewer than two."""
        if self.failures is None:
            raise RuntimeError("run() the experiment before summary()")
        names = list(outputs) if outputs else list(self.model.outputs)
        unknown = set(names) - set(self.model.outputs)
        if unknown:
            raise ValueError(f"unknown outputs: {sorted(unknown)}")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be in (0, 1)")

        from scipy.stats import t as _student_t

        reps = self.replications
        n_points = self.trials.size // reps
        cols = [(p, self.trials.dtype[p]) for p in self.swept]
        for o in names:
            cols += [(o, self.trials.dtype[o]),
                     (f"{o}_hw", self.trials.dtype[o])]
        table = np.zeros(n_points, dtype=cols)
        for p in self.swept:
            table[p] = self.trials[p][::reps]
        for o in names:
            vals = self.trials[o].copy()
            vals[self.failed] = np.nan
            vals = vals.reshape((n_points, reps) + vals.shape[1:])
            with np.errstate(invalid="ignore", divide="ignore"):
                n = (~np.isnan(vals)).sum(axis=1).astype(np.float64)
                mean = np.nansum(vals, axis=1) / n
                dev = vals - np.expand_dims(mean, 1)
                var = np.nansum(dev * dev, axis=1) / (n - 1.0)
                tcrit = _student_t.ppf((1.0 + confidence) / 2.0, n - 1.0)
                table[o] = mean
                table[f"{o}_hw"] = tcrit * np.sqrt(var / n)
        return table

    def __getitem__(self, field: str) -> np.ndarray:
        return self.trials[field]

    def __len__(self) -> int:
        return self.trials.size

    def histories(
        self,
        name: str,
    ) -> (tuple[np.ndarray, ...]
          | tuple[tuple[np.ndarray, ...], ...]):
        """Captured raw history arrays for every trial.

        Indexed component captures add an inner tuple containing one array
        per collection item in deterministic collection order.
        """
        if name not in self._history_capture_names:
            raise KeyError(f"unknown captured history: {name}")
        if self._history_capture_data is None:
            raise RuntimeError("run() the experiment before reading histories")
        return self._history_capture_data[name]

    def history(
        self,
        name: str,
        *,
        trial: int = 0,
        index: int | None = None,
    ) -> np.ndarray:
        """Captured raw history array for one trial and collection item."""
        rows = self.histories(name)
        if trial < 0 or trial >= len(rows):
            raise IndexError("history trial index out of range")
        spec = next(
            spec for spec in self._history_capture_specs
            if spec.name == name
        )
        row = rows[trial]
        if spec.shape is None:
            if index is not None:
                raise TypeError(f"history '{name}' is not indexed")
            assert isinstance(row, np.ndarray)
            return row
        if index is None:
            raise TypeError(
                f"history '{name}' requires an index for indexed capture")
        if type(index) is not int:
            raise TypeError("history index must be an int")
        if index < 0 or index >= spec.shape[0]:
            raise IndexError("history collection index out of range")
        return row[index]

    def datasets(self, name: str) -> tuple[np.ndarray, ...]:
        """Captured raw dataset arrays for every trial."""
        if name not in self._dataset_capture_names:
            raise KeyError(f"unknown captured dataset: {name}")
        if self._dataset_capture_data is None:
            raise RuntimeError("run() the experiment before reading datasets")
        return self._dataset_capture_data[name]

    def dataset(self, name: str, *, trial: int = 0) -> np.ndarray:
        """Captured raw dataset array for one trial."""
        rows = self.datasets(name)
        if trial < 0 or trial >= len(rows):
            raise IndexError("dataset trial index out of range")
        return rows[trial]
