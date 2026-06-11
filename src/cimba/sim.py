"""
cimba.sim - cimba's modeling concepts behind a SimPy-flavored API.

Processes are plain Python functions that block, in the style of SimPy --
but with no `yield`: cimba's processes are stackful fibers, so sim.hold()
and the acquire/get/wait verbs simply suspend the process, from any depth
of the call stack. Each process is compiled with Numba into machine code,
so models run at native speed on all cores; process bodies must stay in
nopython-compilable Python (numbers, loops, and sim.* calls).

Concept translation (cimba -> sim API):

    cmb_process       @model.process (copies=, priority=), sim.hold(),
                      sim.current(), sim.interrupt(), sim.stop(),
                      sim.wait_process(), sim.resume()
    cmb_buffer        Model(queues=[...]), sim.put()/sim.get()/sim.level()
    cmb_resource      Model(resources=[...]), sim.acquire()/sim.release()/
                      sim.preempt()
    cmb_resourcepool  Model(pools={name: capacity}), sim.pool_acquire()/
                      sim.pool_release()/sim.pool_preempt()
    cmb_objectqueue   Model(stores={name: capacity}), sim.store_put()/
                      sim.store_take() (objects are opaque int64 values;
                      sim.f2i()/sim.i2f() bit-cast timestamps in and out)
    cmb_condition     Model(conditions=[...]), @model.predicate,
                      sim.wait_for()/sim.signal()
    cmb_dataset       Model(datasets=[...]), sim.tally(),
                      sim.dataset_mean()/sim.dataset_count()
    statistics        recorded automatically over the measurement window:
                      sim.mean_level(), sim.mean_in_use(),
                      sim.pool_mean_in_use(), sim.store_mean_length()

A minimal model:

    mg1 = sim.Model("mg1", params=["utilization"], queues=["queue"],
                    outputs=["avg_queue_length"])

    @mg1.process
    def arrivals(env):
        while True:
            sim.hold(sim.exponential(1.0 / env.utilization))
            sim.put(env.queue, 1)

The `env` argument is the trial record: params, outputs, entity handles,
and declared state fields as attributes. Multi-copy processes may take a
second argument to learn their index: `def machine(env, idx)`. The trial
function, recording lifecycle, and all create/start/stop/destroy plumbing
are generated and compiled by Model.
"""

import keyword

import llvmlite.binding as _llvm
import numpy as np
from numba import carray, cfunc, from_dtype, njit, types
from numba.extending import intrinsic

from . import _cimba
from ._cimba import ffi, ffi as _ffi, lib

# Make the extension's exported cimba symbols visible to the JIT linker
_llvm.load_library_permanently(_cimba.__file__)

_intp = types.intp
_void = types.void
_i64 = types.int64
_u64 = types.uint64
_f64 = types.float64

# --- Numba ExternalFunction bindings (internal) ---------------------------
_event_queue_initialize = types.ExternalFunction(
    "cmb_event_queue_initialize", _void(_f64))
_event_queue_execute = types.ExternalFunction(
    "cmb_event_queue_execute", _void())
_event_queue_terminate = types.ExternalFunction(
    "cmb_event_queue_terminate", _void())
_event_schedule = types.ExternalFunction(
    "cmb_event_schedule", _u64(_intp, _intp, _intp, _f64, _i64))
_time = types.ExternalFunction("cmb_time", _f64())

_random_initialize = types.ExternalFunction(
    "cmb_random_initialize", _void(_u64))
_random_terminate = types.ExternalFunction("cmb_random_terminate", _void())
_random_exponential = types.ExternalFunction(
    "cpy_random_exponential", _f64(_f64))
_random_gamma = types.ExternalFunction("cpy_random_gamma", _f64(_f64, _f64))
_random01 = types.ExternalFunction("cpy_random01", _f64())
_random_uniform = types.ExternalFunction(
    "cpy_random_uniform", _f64(_f64, _f64))
_random_normal = types.ExternalFunction(
    "cpy_random_normal", _f64(_f64, _f64))

_process_create = types.ExternalFunction("cmb_process_create", _intp())
_process_initialize = types.ExternalFunction(
    "cmb_process_initialize", _void(_intp, _intp, _intp, _intp, _i64))
_process_start = types.ExternalFunction("cmb_process_start", _void(_intp))
_process_stop = types.ExternalFunction("cmb_process_stop", _i64(_intp, _intp))
_process_terminate = types.ExternalFunction(
    "cmb_process_terminate", _void(_intp))
_process_destroy = types.ExternalFunction("cmb_process_destroy", _void(_intp))
_process_hold = types.ExternalFunction("cmb_process_hold", _i64(_f64))
_process_interrupt = types.ExternalFunction(
    "cmb_process_interrupt", _void(_intp, _i64, _i64))
_process_wait_process = types.ExternalFunction(
    "cmb_process_wait_process", _i64(_intp))
_process_resume = types.ExternalFunction(
    "cmb_process_resume", _void(_intp, _i64))
_process_current = types.ExternalFunction("cpy_process_current", _intp())
_process_status = types.ExternalFunction("cpy_process_status", _i64(_intp))

_buffer_create = types.ExternalFunction("cmb_buffer_create", _intp())
_buffer_initialize = types.ExternalFunction(
    "cmb_buffer_initialize", _void(_intp, _intp, _u64))
_buffer_destroy = types.ExternalFunction("cmb_buffer_destroy", _void(_intp))
_buffer_recording_start = types.ExternalFunction(
    "cmb_buffer_recording_start", _void(_intp))
_buffer_recording_stop = types.ExternalFunction(
    "cmb_buffer_recording_stop", _void(_intp))
_buffer_put = types.ExternalFunction("cpy_buffer_put", _i64(_intp, _u64))
_buffer_get = types.ExternalFunction("cpy_buffer_get", _i64(_intp, _u64))
_buffer_mean_level = types.ExternalFunction(
    "cpy_buffer_mean_level", _f64(_intp))
_buffer_level = types.ExternalFunction("cpy_buffer_level", _u64(_intp))

_resource_create = types.ExternalFunction("cmb_resource_create", _intp())
_resource_initialize = types.ExternalFunction(
    "cmb_resource_initialize", _void(_intp, _intp))
_resource_destroy = types.ExternalFunction(
    "cmb_resource_destroy", _void(_intp))
_resource_acquire = types.ExternalFunction(
    "cmb_resource_acquire", _i64(_intp))
_resource_release = types.ExternalFunction(
    "cmb_resource_release", _void(_intp))
_resource_preempt = types.ExternalFunction(
    "cmb_resource_preempt", _i64(_intp))
_resource_recording_start = types.ExternalFunction(
    "cmb_resource_start_recording", _void(_intp))
_resource_recording_stop = types.ExternalFunction(
    "cmb_resource_stop_recording", _void(_intp))
_resource_in_use = types.ExternalFunction("cpy_resource_in_use", _u64(_intp))
_resource_mean_in_use = types.ExternalFunction(
    "cpy_resource_mean_in_use", _f64(_intp))

_resourcepool_create = types.ExternalFunction(
    "cmb_resourcepool_create", _intp())
_resourcepool_initialize = types.ExternalFunction(
    "cmb_resourcepool_initialize", _void(_intp, _intp, _u64))
_resourcepool_destroy = types.ExternalFunction(
    "cmb_resourcepool_destroy", _void(_intp))
_resourcepool_acquire = types.ExternalFunction(
    "cmb_resourcepool_acquire", _i64(_intp, _u64))
_resourcepool_preempt = types.ExternalFunction(
    "cmb_resourcepool_preempt", _i64(_intp, _u64))
_resourcepool_release = types.ExternalFunction(
    "cmb_resourcepool_release", _void(_intp, _u64))
_resourcepool_recording_start = types.ExternalFunction(
    "cmb_resourcepool_start_recording", _void(_intp))
_resourcepool_recording_stop = types.ExternalFunction(
    "cmb_resourcepool_stop_recording", _void(_intp))
_resourcepool_in_use = types.ExternalFunction(
    "cpy_resourcepool_in_use", _u64(_intp))
_resourcepool_mean_in_use = types.ExternalFunction(
    "cpy_resourcepool_mean_in_use", _f64(_intp))

_objectqueue_create = types.ExternalFunction(
    "cmb_objectqueue_create", _intp())
_objectqueue_initialize = types.ExternalFunction(
    "cmb_objectqueue_initialize", _void(_intp, _intp, _u64))
_objectqueue_destroy = types.ExternalFunction(
    "cmb_objectqueue_destroy", _void(_intp))
_objectqueue_put = types.ExternalFunction(
    "cpy_objectqueue_put", _i64(_intp, _intp))
_objectqueue_get = types.ExternalFunction(
    "cpy_objectqueue_get", _i64(_intp, _intp))
_objectqueue_take = types.ExternalFunction(
    "cpy_objectqueue_take", _intp(_intp))
_objectqueue_length = types.ExternalFunction(
    "cpy_objectqueue_length", _u64(_intp))
_objectqueue_recording_start = types.ExternalFunction(
    "cmb_objectqueue_recording_start", _void(_intp))
_objectqueue_recording_stop = types.ExternalFunction(
    "cmb_objectqueue_recording_stop", _void(_intp))
_objectqueue_mean_length = types.ExternalFunction(
    "cpy_objectqueue_mean_length", _f64(_intp))

_dataset_create = types.ExternalFunction("cmb_dataset_create", _intp())
_dataset_initialize = types.ExternalFunction(
    "cmb_dataset_initialize", _void(_intp))
_dataset_destroy = types.ExternalFunction("cmb_dataset_destroy", _void(_intp))
_dataset_add = types.ExternalFunction("cmb_dataset_add", _u64(_intp, _f64))
_dataset_mean = types.ExternalFunction("cpy_dataset_mean", _f64(_intp))
_dataset_count = types.ExternalFunction("cpy_dataset_count", _u64(_intp))

_condition_create = types.ExternalFunction("cmb_condition_create", _intp())
_condition_initialize = types.ExternalFunction(
    "cmb_condition_initialize", _void(_intp, _intp))
_condition_destroy = types.ExternalFunction(
    "cmb_condition_destroy", _void(_intp))
_condition_wait = types.ExternalFunction(
    "cmb_condition_wait", _i64(_intp, _intp, _intp))
_condition_signal = types.ExternalFunction(
    "cmb_condition_signal", _u64(_intp))


@intrinsic
def _addressof(typingctx, ptr):
    if not isinstance(ptr, types.CPointer):
        raise TypeError("_addressof() expects a typed pointer")

    def codegen(context, builder, signature, args):
        return builder.ptrtoint(args[0], context.get_value_type(types.intp))

    return types.intp(ptr), codegen


def _ptr_caster(pointee):
    ptr_type = types.CPointer(pointee)

    @intrinsic
    def cast(typingctx, addr):
        if not isinstance(addr, types.Integer):
            raise TypeError("expected an integer address")

        def codegen(context, builder, signature, args):
            return builder.inttoptr(args[0], context.get_value_type(ptr_type))

        return ptr_type(addr), codegen

    return cast


@intrinsic
def _record_addr(typingctx, rec):
    if not isinstance(rec, types.Record):
        raise TypeError("_record_addr() expects a record")

    def codegen(context, builder, signature, args):
        return builder.ptrtoint(args[0], context.get_value_type(types.intp))

    return types.intp(rec), codegen


_keepalive: list = []


def _cstring(s: bytes) -> int:
    buf = _ffi.new("char[]", s)
    _keepalive.append(buf)
    return int(_ffi.cast("intptr_t", buf))

__all__ = [
    "Model",
    "hold", "now", "current", "interrupt", "stop", "wait_process", "resume",
    "put", "get", "level", "mean_level",
    "acquire", "release", "preempt", "in_use", "mean_in_use",
    "pool_acquire", "pool_release", "pool_preempt", "pool_in_use",
    "pool_mean_in_use",
    "store_put", "store_take", "store_length", "store_mean_length",
    "tally", "dataset_mean", "dataset_count",
    "wait_for", "signal",
    "exponential", "gamma", "uniform", "normal", "random01",
    "f2i", "i2f",
]

# --- Process verbs -----------------------------------------------------------
hold = _process_hold
now = _time
current = _process_current
interrupt = _process_interrupt        # (process, sig, pri)
stop = _process_stop                  # (process, retval=0)
wait_process = _process_wait_process  # join
resume = _process_resume              # (process, sig)

# --- Buffers (cmb_buffer): counted amounts -----------------------------------
put = _buffer_put
get = _buffer_get
level = _buffer_level
mean_level = _buffer_mean_level

# --- Resources (cmb_resource): single holder, priority-aware -----------------
acquire = _resource_acquire
release = _resource_release
preempt = _resource_preempt
in_use = _resource_in_use
mean_in_use = _resource_mean_in_use

# --- Resource pools (cmb_resourcepool): capacity > 1 --------------------------
pool_acquire = _resourcepool_acquire      # (pool, amount)
pool_release = _resourcepool_release      # (pool, amount)
pool_preempt = _resourcepool_preempt      # (pool, amount)
pool_in_use = _resourcepool_in_use
pool_mean_in_use = _resourcepool_mean_in_use

# --- Stores (cmb_objectqueue): FIFO of opaque int64 objects -------------------
store_put = _objectqueue_put              # (store, object != 0)
store_take = _objectqueue_take            # blocking, returns object
store_length = _objectqueue_length
store_mean_length = _objectqueue_mean_length

# --- Datasets (cmb_dataset): tally statistics ---------------------------------
tally = _dataset_add                      # (dataset, value)
dataset_mean = _dataset_mean
dataset_count = _dataset_count

# --- Random draws ---------------------------------------------------------------
exponential = _random_exponential
gamma = _random_gamma
uniform = _random_uniform
normal = _random_normal
random01 = _random01

# --- Bit-casts for store objects -----------------------------------------------
@intrinsic
def f2i(typingctx, x):
    """Bit-cast a float64 to int64."""
    if x != types.float64:
        raise TypeError("f2i() expects a float64")

    def codegen(context, builder, signature, args):
        return builder.bitcast(args[0], context.get_value_type(types.int64))

    return types.int64(x), codegen


@intrinsic
def i2f(typingctx, i):
    """Bit-cast an int64 back to float64."""
    if not isinstance(i, types.Integer):
        raise TypeError("i2f() expects an int64")

    def codegen(context, builder, signature, args):
        return builder.bitcast(args[0], context.get_value_type(types.float64))

    return types.float64(i), codegen


# --- Conditions (cmb_condition) ------------------------------------------------
signal = _condition_signal


@njit
def wait_for(cond, pred, env):
    """Block until the predicate is satisfied; re-evaluated on every sim.signal(cond)."""
    return _condition_wait(cond, pred, _record_addr(env))


_STANDARD_FIELDS = [
    ("start_time", "<f8"),
    ("warmup_s", "<f8"),
    ("duration_s", "<f8"),
    ("cooldown_s", "<f8"),
    ("seed", "<u8"),
]
_RESERVED = {name for name, _ in _STANDARD_FIELDS}


def _check_name(name: str, kind: str):
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"{kind} name '{name}' is not a valid identifier")
    if name in _RESERVED or name.startswith("_"):
        raise ValueError(f"{kind} name '{name}' is reserved")


def _as_capacity_dict(value, kind) -> dict:
    """Normalize `stores`/`pools` declarations: a list means unbounded
    capacity; dict values are an int capacity or a param name string."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    return {name: None for name in value}


class Model:
    """A simulation model translating cimba's concepts: declared entities
    (queues, resources, pools, stores, datasets, conditions), parameters,
    outputs, and state fields, plus process functions registered with
    @model.process. Compilation happens once, on first experiment()."""

    def __init__(self, name: str, *, params=(), outputs=(), queues=(),
                 resources=(), pools=None, stores=None, datasets=(),
                 conditions=(), state=()):
        self.name = name
        self.params = list(params)
        self.outputs = list(outputs)
        self.queues = list(queues)
        self.resources = list(resources)
        self.pools = _as_capacity_dict(pools, "pool")
        self.stores = _as_capacity_dict(stores, "store")
        self.datasets = list(datasets)
        self.conditions = list(conditions)
        self.state = list(state)

        seen: set = set()
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
        self._processes: list[tuple[str, object, int, int, bool]] = []
        self._predicates: list[tuple[str, object]] = []
        self._collect = None
        self._compiled = None

    # --- Declaration decorators ------------------------------------------
    def process(self, fn=None, *, copies: int = 1, priority: int = 0):
        """Register a process function `def fn(env)` or `def fn(env, idx)`
        (the latter receives its copy index). copies=n starts n identical
        processes; priority sets the cimba process priority."""
        if fn is None:
            return lambda f: self.process(f, copies=copies,
                                          priority=priority)
        if copies < 1:
            raise ValueError("copies must be >= 1")
        _check_name(fn.__name__, "process")
        if fn.__name__ in self._seen:
            raise ValueError(f"duplicate name '{fn.__name__}'")
        self._seen.add(fn.__name__)
        if self._compiled is not None:
            raise RuntimeError("model is already compiled")
        nargs = fn.__code__.co_argcount
        if nargs not in (1, 2):
            raise ValueError("process functions take (env) or (env, idx)")
        self._processes.append((fn.__name__, fn, copies, priority,
                                nargs == 2))
        return fn

    def predicate(self, fn):
        """Register a condition predicate `def fn(env) -> bool`. Its
        compiled address is published as the env field `_pred_<name>`,
        for use with sim.wait_for(env.<cond>, env._pred_<name>, env)."""
        _check_name(fn.__name__, "predicate")
        if fn.__name__ in self._seen:
            raise ValueError(f"duplicate name '{fn.__name__}'")
        self._seen.add(fn.__name__)
        if self._compiled is not None:
            raise RuntimeError("model is already compiled")
        self._predicates.append((fn.__name__, fn))
        return fn

    def collect(self, fn):
        """Register the statistics-collection function, run once at the
        end of each trial."""
        if self._collect is not None:
            raise ValueError("collect() already registered")
        if self._compiled is not None:
            raise RuntimeError("model is already compiled")
        self._collect = fn
        return fn

    # --- Trial record layout ------------------------------------------------
    @property
    def dtype(self) -> np.dtype:
        fields = list(_STANDARD_FIELDS)
        fields += [(p, "<f8") for p in self.params]
        fields += [(o, "<f8") for o in self.outputs]
        handle_names = (self.queues + self.resources + list(self.pools)
                        + list(self.stores) + self.datasets
                        + self.conditions)
        fields += [(h, "<i8") for h in handle_names]
        fields += [(s, "<i8") for s in self.state]
        fields += [(f"_pred_{n}", "<i8") for n, _ in self._predicates]
        for pname, _fn, copies, _pri, indexed in self._processes:
            fields += [(f"_p_{pname}_{i}", "<i8") for i in range(copies)]
            if indexed:
                for i in range(copies):
                    fields += [(f"_cx_{pname}_{i}_env", "<i8"),
                               (f"_cx_{pname}_{i}_idx", "<i8")]
        return np.dtype(fields)

    # --- Compilation -----------------------------------------------------------
    def _compile(self):
        if self._compiled is not None:
            return self._compiled
        if not self._processes:
            raise ValueError("model has no processes")

        dtype = self.dtype
        rec = from_dtype(dtype)
        trial_ptr = types.CPointer(rec)
        rec_from_addr = _ptr_caster(rec)
        proc_sig = types.intp(types.intp, trial_ptr)
        proc_sig_ix = types.intp(types.intp, types.CPointer(types.int64))
        evt_sig = types.void(trial_ptr, types.intp)
        pred_sig = types.boolean(types.intp, types.intp, trial_ptr)

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

        # Codegen namespace
        ns = {"carray": carray, "addressof": _addressof, "np": np,
              "COLLECT": collect_inner,
              "CAP": np.uint64(0xFFFFFFFFFFFFFFFF)}
        for f in ("event_queue_initialize", "event_queue_execute",
                  "event_queue_terminate", "event_schedule",
                  "process_status",
                  "random_initialize", "random_terminate",
                  "buffer_create", "buffer_initialize", "buffer_destroy",
                  "buffer_recording_start", "buffer_recording_stop",
                  "resource_create", "resource_initialize",
                  "resource_destroy", "resource_recording_start",
                  "resource_recording_stop",
                  "resourcepool_create", "resourcepool_initialize",
                  "resourcepool_destroy", "resourcepool_recording_start",
                  "resourcepool_recording_stop",
                  "objectqueue_create", "objectqueue_initialize",
                  "objectqueue_destroy", "objectqueue_recording_start",
                  "objectqueue_recording_stop",
                  "dataset_create", "dataset_initialize", "dataset_destroy",
                  "condition_create", "condition_initialize",
                  "condition_destroy",
                  "process_create", "process_initialize", "process_start",
                  "process_stop", "process_terminate", "process_destroy"):
            ns[f] = globals()[f"_{f}"]
        all_entities = (self.queues + self.resources + list(self.pools)
                        + list(self.stores) + self.datasets
                        + self.conditions)
        for e in all_entities:
            ns[f"NAME_{e}"] = _cstring(e.encode())
        for pname, proc in proc_cfuncs.items():
            ns[f"NAME_{pname}"] = _cstring(pname.encode())
            ns[f"F_{pname}"] = proc.address

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

        handles = [f"_p_{pname}_{i}"
                   for pname, _fn, copies, _pri, _ix in self._processes
                   for i in range(copies)]

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
            src += [f"    h = buffer_create()",
                    f"    buffer_initialize(h, NAME_{q}, CAP)",
                    f"    env['{q}'] = h"]
        for r in self.resources:
            src += [f"    h = resource_create()",
                    f"    resource_initialize(h, NAME_{r})",
                    f"    env['{r}'] = h"]
        for p, cap in self.pools.items():
            src += [f"    h = resourcepool_create()",
                    f"    resourcepool_initialize(h, NAME_{p}, "
                    f"{cap_expr(cap)})",
                    f"    env['{p}'] = h"]
        for s, cap in self.stores.items():
            src += [f"    h = objectqueue_create()",
                    f"    objectqueue_initialize(h, NAME_{s}, "
                    f"{cap_expr(cap)})",
                    f"    env['{s}'] = h"]
        for d in self.datasets:
            src += [f"    h = dataset_create()",
                    f"    dataset_initialize(h)",
                    f"    env['{d}'] = h"]
        for c in self.conditions:
            src += [f"    h = condition_create()",
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
                src += [f"    p = process_create()",
                        f"    process_initialize(p, NAME_{pname}, "
                        f"F_{pname}, ctx, {pri})",
                        f"    process_start(p)",
                        f"    env['_p_{pname}_{i}'] = p"]
        src += ["    event_queue_execute()"]
        if collect_inner is not None:
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

        self._source = "\n".join(src)
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

    # --- Experiments ---------------------------------------------------------
    def experiment(self,
                   *,
                   replications: int = 1,
                   duration: float = 1.0e6,
                   warmup: float = 1.0e3,
                   cooldown: float = 0.0,
                   start_time: float = 0.0,
                   seed: int | None = None,
                   **param_values) -> "Experiment":
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
    def __init__(self, model: Model, trials: np.ndarray, trial_addr: int):
        self.model = model
        self.trials = trials
        self._trial_addr = trial_addr
        self.failures: int | None = None

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
