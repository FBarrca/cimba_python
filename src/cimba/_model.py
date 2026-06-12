"""Model declaration, compilation, and experiment execution.

A ``Model`` collects declared entities, parameters, outputs, and process
functions, then compiles everything on first ``experiment()``:

* each process body is njit-compiled and wrapped in a ``cfunc`` whose
  address cimba can start as a stackful fiber;
* a trial function is generated as Python source (see ``_trial_source``),
  compiled, and turned into a ``cfunc``: it creates the declared entities,
  schedules the recording window, starts the processes, runs the event
  queue, collects statistics, and tears everything down;
* an ``Experiment`` is a structured numpy array with one record per trial
  (the ``env`` seen by process bodies) handed to ``cimba_run_experiment``,
  which runs trials in parallel across all cores.
"""

import keyword
from collections.abc import Callable, Iterable, Mapping
from typing import Any, TypedDict, TypeVar, overload

import numpy as np
from numpy.typing import ArrayLike

from numba import carray, cfunc, from_dtype, njit, types

from . import _bindings as _b
from ._cimba import ffi, lib
from ._intrinsics import addressof, ptr_caster

#: Opaque native entity handle (process, queue, resource, ...) as stored
#: in env fields.
Handle = int

#: The per-trial record passed to process bodies: a numpy structured
#: scalar whose fields are accessed as attributes inside nopython code.
#: Opaque to static typing.
Env = Any

_F = TypeVar("_F", bound=Callable[..., Any])

#: A `pools`/`stores` declaration: list of names (unbounded capacity) or
#: name -> capacity mapping (an int, a param name, or None for unbounded).
_Capacities = Mapping[str, int | str | None] | Iterable[str] | None

_STANDARD_FIELDS = [
    ("start_time", "<f8"),
    ("warmup_s", "<f8"),
    ("duration_s", "<f8"),
    ("cooldown_s", "<f8"),
    ("seed", "<u8"),
]
_RESERVED = {name for name, _ in _STANDARD_FIELDS}

# All ExternalFunction bindings, by name, for the generated trial source.
_EXTERN_FUNCS = {name: obj for name, obj in vars(_b).items()
                 if isinstance(obj, types.ExternalFunction)}

_UNBOUNDED = np.uint64(0xFFFFFFFFFFFFFFFF)


class _Compiled(TypedDict):
    """Artifacts of Model._compile(), kept alive for the model's lifetime.
    The callables are Numba cfunc/dispatcher objects (untyped upstream)."""

    trial: Any
    events: tuple[Any, Any, Any]
    procs: dict[str, Any]
    preds: dict[str, Any]
    collect: Any
    dtype: np.dtype


def _check_name(name: str, kind: str) -> None:
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"{kind} name '{name}' is not a valid identifier")
    if name in _RESERVED or name.startswith("_"):
        raise ValueError(f"{kind} name '{name}' is reserved")


def _as_capacity_dict(value: _Capacities) -> dict[str, int | str | None]:
    """Normalize a `stores`/`pools` declaration: a list means unbounded
    capacity; dict values are an int capacity or a param name string."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return {name: None for name in value}


class Model:
    """A simulation model translating cimba's concepts: declared entities
    (queues, resources, pools, stores, datasets, conditions), parameters,
    outputs, and state fields, plus process functions registered with
    @model.process. Compilation happens once, on first experiment()."""

    _source: str

    def __init__(self, name: str, *,
                 params: Iterable[str] = (),
                 outputs: Iterable[str] = (),
                 queues: Iterable[str] = (),
                 resources: Iterable[str] = (),
                 pools: _Capacities = None,
                 stores: _Capacities = None,
                 datasets: Iterable[str] = (),
                 conditions: Iterable[str] = (),
                 state: Iterable[str] = ()):
        self.name = name
        self.params = list(params)
        self.outputs = list(outputs)
        self.queues = list(queues)
        self.resources = list(resources)
        self.pools = _as_capacity_dict(pools)
        self.stores = _as_capacity_dict(stores)
        self.datasets = list(datasets)
        self.conditions = list(conditions)
        self.state = list(state)

        seen: set[str] = set()
        for kind, names in (("param", self.params),
                            ("output", self.outputs),
                            ("queue", self.queues),
                            ("resource", self.resources),
                            ("pool", self.pools),
                            ("store", self.stores),
                            ("dataset", self.datasets),
                            ("condition", self.conditions),
                            ("state", self.state)):
            for n in names:
                _check_name(n, kind)
                if n in seen:
                    raise ValueError(f"duplicate field name '{n}'")
                seen.add(n)
        for cap in list(self.pools.values()) + list(self.stores.values()):
            if cap is not None and not isinstance(cap, int) \
                    and cap not in self.params:
                raise ValueError(f"capacity '{cap}' is neither an int nor "
                                 "a declared param")
        self._seen = seen
        # (name, fn, copies, priority, indexed)
        self._processes: list[
            tuple[str, Callable[..., Any], int, int, bool]] = []
        self._predicates: list[tuple[str, Callable[..., Any]]] = []
        self._collect: Callable[..., Any] | None = None
        self._compiled: _Compiled | None = None

    # --- Declaration decorators ------------------------------------------
    @overload
    def process(self, fn: _F) -> _F: ...

    @overload
    def process(self, fn: None = None, *, copies: int = 1,
                priority: int = 0) -> Callable[[_F], _F]: ...

    def process(self, fn=None, *, copies: int = 1, priority: int = 0):
        """Register a process function `def fn(env)` or `def fn(env, idx)`
        (the latter receives its copy index). copies=n starts n identical
        processes; priority sets the cimba process priority."""
        if fn is None:
            return lambda f: self.process(f, copies=copies,
                                          priority=priority)
        if copies < 1:
            raise ValueError("copies must be >= 1")
        self._register_name(fn.__name__, "process")
        nargs = fn.__code__.co_argcount
        if nargs not in (1, 2):
            raise ValueError("process functions take (env) or (env, idx)")
        self._processes.append((fn.__name__, fn, copies, priority,
                                nargs == 2))
        return fn

    def predicate(self, fn: _F) -> _F:
        """Register a condition predicate `def fn(env) -> bool`. Its
        compiled address is published as the env field `_pred_<name>`,
        for use with sim.wait_for(env.<cond>, env._pred_<name>, env)."""
        self._register_name(fn.__name__, "predicate")
        self._predicates.append((fn.__name__, fn))
        return fn

    def collect(self, fn: _F) -> _F:
        """Register the statistics-collection function, run once at the
        end of each trial."""
        if self._collect is not None:
            raise ValueError("collect() already registered")
        if self._compiled is not None:
            raise RuntimeError("model is already compiled")
        self._collect = fn
        return fn

    def _register_name(self, name: str, kind: str) -> None:
        _check_name(name, kind)
        if name in self._seen:
            raise ValueError(f"duplicate name '{name}'")
        if self._compiled is not None:
            raise RuntimeError("model is already compiled")
        self._seen.add(name)

    # --- Trial record layout ----------------------------------------------
    @property
    def _entities(self) -> list[str]:
        return (self.queues + self.resources + list(self.pools)
                + list(self.stores) + self.datasets + self.conditions)

    @property
    def _process_handles(self) -> list[str]:
        return [f"_p_{pname}_{i}"
                for pname, _fn, copies, _pri, _ix in self._processes
                for i in range(copies)]

    @property
    def dtype(self) -> np.dtype:
        fields = list(_STANDARD_FIELDS)
        fields += [(p, "<f8") for p in self.params]
        fields += [(o, "<f8") for o in self.outputs]
        fields += [(h, "<i8") for h in self._entities]
        fields += [(s, "<i8") for s in self.state]
        fields += [(f"_pred_{n}", "<i8") for n, _ in self._predicates]
        for pname, _fn, copies, _pri, indexed in self._processes:
            fields += [(f"_p_{pname}_{i}", "<i8") for i in range(copies)]
            if indexed:
                for i in range(copies):
                    fields += [(f"_cx_{pname}_{i}_env", "<i8"),
                               (f"_cx_{pname}_{i}_idx", "<i8")]
        return np.dtype(fields)

    # --- Compilation --------------------------------------------------------
    def _compile_callbacks(
        self, rec: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], Any]:
        """njit-compile the registered process/predicate/collect functions
        and wrap processes and predicates in cfuncs cimba can call."""
        trial_ptr = types.CPointer(rec)
        proc_sig = types.intp(types.intp, trial_ptr)
        proc_sig_ix = types.intp(types.intp, types.CPointer(types.int64))
        pred_sig = types.boolean(types.intp, types.intp, trial_ptr)
        rec_from_addr = ptr_caster(rec)

        def make_proc(inner):
            @cfunc(proc_sig)
            def proc(me, ctxp):
                inner(carray(ctxp, 1)[0])
                return 0
            return proc

        def make_proc_indexed(inner):
            @cfunc(proc_sig_ix)
            def proc(me, ctxp):
                pair = carray(ctxp, 2)
                env = carray(rec_from_addr(pair[0]), 1)[0]
                inner(env, pair[1])
                return 0
            return proc

        def make_pred(inner):
            @cfunc(pred_sig)
            def pred(cnd, prc, ctxp):
                return inner(carray(ctxp, 1)[0])
            return pred

        proc_cfuncs = {}
        for pname, fn, _copies, _pri, indexed in self._processes:
            inner = njit(fn)
            proc_cfuncs[pname] = (make_proc_indexed(inner) if indexed
                                  else make_proc(inner))
        pred_cfuncs = {n: make_pred(njit(fn)) for n, fn in self._predicates}
        collect_inner = njit(self._collect) if self._collect else None
        return proc_cfuncs, pred_cfuncs, collect_inner

    def _codegen_namespace(self, proc_cfuncs: dict[str, Any],
                           collect_inner: Any) -> dict[str, Any]:
        """Globals for the generated trial source: the extern bindings,
        interned entity/process name strings, and process cfunc addresses."""
        ns = dict(_EXTERN_FUNCS)
        ns.update(carray=carray, addressof=addressof, np=np,
                  COLLECT=collect_inner, CAP=_UNBOUNDED)
        for e in self._entities:
            ns[f"NAME_{e}"] = _b.cstring(e)
        for pname, proc in proc_cfuncs.items():
            ns[f"NAME_{pname}"] = _b.cstring(pname)
            ns[f"F_{pname}"] = proc.address
        return ns

    def _trial_source(self, dtype: np.dtype) -> str:
        """Generate the trial function and the three recording-window event
        callbacks as Python source (njit-compilable against the namespace
        from _codegen_namespace)."""

        def cap_expr(cap):
            if cap is None:
                return "CAP"
            if isinstance(cap, int):
                return f"np.uint64({cap})"
            return f"np.uint64(env['{cap}'])"

        recorded = [("buffer", q) for q in self.queues]
        recorded += [("resource", r) for r in self.resources]
        recorded += [("resourcepool", p) for p in self.pools]
        recorded += [("objectqueue", s) for s in self.stores]
        handles = self._process_handles

        src = ["def _start_rec(subject, obj):",
               "    env = carray(subject, 1)[0]"]
        src += [f"    {k}_recording_start(env['{n}'])" for k, n in recorded]
        src += ["def _stop_rec(subject, obj):",
                "    env = carray(subject, 1)[0]"]
        src += [f"    {k}_recording_stop(env['{n}'])" for k, n in recorded]
        # Stop only processes still running; some may have finished on
        # their own (e.g. a source that produced a fixed number of items)
        src += ["def _end_sim(subject, obj):",
                "    env = carray(subject, 1)[0]"]
        src += [f"    if process_status(env['{h}']) == 1:\n"
                f"        process_stop(env['{h}'], 0)" for h in handles]

        src += ["def _trial(vtrl):",
                "    arr = carray(vtrl, 1)",
                "    env = arr[0]",
                "    self_addr = addressof(vtrl)",
                "    event_queue_initialize(env['start_time'])",
                "    random_initialize(env['seed'])",
                "    t = env['start_time'] + env['warmup_s']",
                "    event_schedule(EV_START, self_addr, 0, t, 0)",
                "    t = t + env['duration_s']",
                "    event_schedule(EV_STOP, self_addr, 0, t, 0)",
                "    t = t + env['cooldown_s']",
                "    event_schedule(EV_END, self_addr, 0, t, 0)"]
        for q in self.queues:
            src += ["    h = buffer_create()",
                    f"    buffer_initialize(h, NAME_{q}, CAP)",
                    f"    env['{q}'] = h"]
        for r in self.resources:
            src += ["    h = resource_create()",
                    f"    resource_initialize(h, NAME_{r})",
                    f"    env['{r}'] = h"]
        for p, cap in self.pools.items():
            src += ["    h = resourcepool_create()",
                    f"    resourcepool_initialize(h, NAME_{p}, "
                    f"{cap_expr(cap)})",
                    f"    env['{p}'] = h"]
        for s, cap in self.stores.items():
            src += ["    h = objectqueue_create()",
                    f"    objectqueue_initialize(h, NAME_{s}, "
                    f"{cap_expr(cap)})",
                    f"    env['{s}'] = h"]
        for d in self.datasets:
            src += ["    h = dataset_create()",
                    "    dataset_initialize(h)",
                    f"    env['{d}'] = h"]
        for c in self.conditions:
            src += ["    h = condition_create()",
                    f"    condition_initialize(h, NAME_{c})",
                    f"    env['{c}'] = h"]
        offsets = {n: off for n, (_dt, off) in dtype.fields.items()}
        for pname, _fn, copies, pri, indexed in self._processes:
            for i in range(copies):
                if indexed:
                    src += [f"    env['_cx_{pname}_{i}_env'] = self_addr",
                            f"    env['_cx_{pname}_{i}_idx'] = {i}",
                            f"    ctx = self_addr + "
                            f"{offsets[f'_cx_{pname}_{i}_env']}"]
                else:
                    src += ["    ctx = self_addr"]
                src += ["    p = process_create()",
                        f"    process_initialize(p, NAME_{pname}, "
                        f"F_{pname}, ctx, {pri})",
                        "    process_start(p)",
                        f"    env['_p_{pname}_{i}'] = p"]
        src += ["    event_queue_execute()"]
        if self._collect is not None:
            src += ["    COLLECT(env)"]
        for h in handles:
            src += [f"    process_terminate(env['{h}'])",
                    f"    process_destroy(env['{h}'])"]
        for q in self.queues:
            src += [f"    buffer_destroy(env['{q}'])"]
        for r in self.resources:
            src += [f"    resource_destroy(env['{r}'])"]
        for p in self.pools:
            src += [f"    resourcepool_destroy(env['{p}'])"]
        for s in self.stores:
            src += [f"    objectqueue_destroy(env['{s}'])"]
        for d in self.datasets:
            src += [f"    dataset_destroy(env['{d}'])"]
        for c in self.conditions:
            src += [f"    condition_destroy(env['{c}'])"]
        src += ["    event_queue_terminate()",
                "    random_terminate()"]
        return "\n".join(src)

    def _compile(self) -> _Compiled:
        if self._compiled is not None:
            return self._compiled
        if not self._processes:
            raise ValueError("model has no processes")

        dtype = self.dtype
        rec = from_dtype(dtype)
        trial_ptr = types.CPointer(rec)
        evt_sig = types.void(trial_ptr, types.intp)

        proc_cfuncs, pred_cfuncs, collect_inner = \
            self._compile_callbacks(rec)
        ns = self._codegen_namespace(proc_cfuncs, collect_inner)
        self._source = self._trial_source(dtype)
        exec(compile(self._source, f"<cimba model '{self.name}'>", "exec"),
             ns)

        start_rec = cfunc(evt_sig)(ns["_start_rec"])
        stop_rec = cfunc(evt_sig)(ns["_stop_rec"])
        end_sim = cfunc(evt_sig)(ns["_end_sim"])
        ns["EV_START"] = start_rec.address
        ns["EV_STOP"] = stop_rec.address
        ns["EV_END"] = end_sim.address
        trial = cfunc(types.void(trial_ptr))(ns["_trial"])

        # Keep every compiled artifact alive for the model's lifetime
        self._compiled = {
            "trial": trial,
            "events": (start_rec, stop_rec, end_sim),
            "procs": proc_cfuncs,
            "preds": pred_cfuncs,
            "collect": collect_inner,
            "dtype": dtype,
        }
        return self._compiled

    # --- Experiments ----------------------------------------------------------
    def experiment(self,
                   *,
                   replications: int = 1,
                   duration: float = 1.0e6,
                   warmup: float = 1.0e3,
                   cooldown: float = 0.0,
                   start_time: float = 0.0,
                   seed: int | None = None,
                   **param_values: ArrayLike) -> "Experiment":
        """Build an experiment: the cross product of the swept parameter
        values (scalars are held fixed), replicated with distinct seeds."""
        compiled = self._compile()

        missing = set(self.params) - set(param_values)
        unknown = set(param_values) - set(self.params)
        if missing:
            raise ValueError(f"missing parameter values: {sorted(missing)}")
        if unknown:
            raise ValueError(f"unknown parameters: {sorted(unknown)}")
        if replications < 1:
            raise ValueError("replications must be >= 1")

        axes = [np.atleast_1d(np.asarray(param_values[p], dtype=np.float64))
                for p in self.params]
        mesh = np.meshgrid(*axes, indexing="ij") if axes else []
        n_points = mesh[0].size if mesh else 1
        n_trials = n_points * replications

        trials = np.zeros(n_trials, dtype=compiled["dtype"])
        trials["start_time"] = start_time
        trials["warmup_s"] = warmup
        trials["duration_s"] = duration
        trials["cooldown_s"] = cooldown
        for p, m in zip(self.params, mesh):
            trials[p] = np.repeat(m.ravel(), replications)
        for o in self.outputs:
            trials[o] = np.nan
        for pname, pred in compiled["preds"].items():
            trials[f"_pred_{pname}"] = pred.address

        rng = np.random.default_rng(
            seed if seed is not None else int(lib.cmb_random_hwseed())
        )
        trials["seed"] = rng.integers(1, np.iinfo(np.uint64).max,
                                      size=n_trials, dtype=np.uint64)
        return Experiment(self, trials, compiled["trial"].address)


class Experiment:
    model: Model
    #: One structured record per trial; outputs are filled in by run().
    trials: np.ndarray
    #: Number of failed trials in the last run(), or None before it.
    failures: int | None

    def __init__(self, model: Model, trials: np.ndarray, trial_addr: int):
        self.model = model
        self.trials = trials
        self._trial_addr = trial_addr
        self.failures = None

    def run(self) -> int:
        """Run all trials in parallel, in place. Returns the number of
        failed trials (their outputs stay NaN)."""
        trials = self.trials
        if trials.dtype.names is None:
            raise TypeError("experiment must be a structured array")
        if not trials.flags["C_CONTIGUOUS"]:
            raise ValueError("experiment array must be C-contiguous")
        if trials.ndim != 1 or trials.size == 0:
            raise ValueError("experiment must be a non-empty 1-D array")

        fptr = ffi.cast("void(*)(void *)", self._trial_addr)
        buf = ffi.from_buffer(trials, require_writable=True)
        lib.cimba_run_experiment(buf, trials.size, trials.itemsize, fptr)

        if not self.model.outputs:
            self.failures = 0
        else:
            self.failures = int(
                np.isnan(self.trials[self.model.outputs[0]]).sum()
            )
        return self.failures

    def __getitem__(self, field: str) -> np.ndarray:
        return self.trials[field]

    def __len__(self) -> int:
        return self.trials.size
