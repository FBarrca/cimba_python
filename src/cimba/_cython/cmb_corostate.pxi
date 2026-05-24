# This file is included by ../_cimba.pyx.
#
# CPython per-thread state save/restore across Cimba's stackful-coroutine
# context switches, for Python 3.14+. Pure binding/build-layer fix: the vendored
# C library is NOT modified. We interpose the coroutine context switch with the
# linker (meson.build: -Wl,--wrap=cmi_coroutine_context_switch) and, on each
# switch, save/restore the interpreter's per-thread frame and exception state.
#
# Why this is needed: Cimba runs Python on its own 64 KB coroutine stacks. All
# coroutines on a worker thread share one PyThreadState, hence one Python frame
# datastack and one exception stack. Up to 3.13 this is benign (Cimba process
# bodies call only C functions, so each holds exactly one frame and they coexist
# fine). 3.14 reworked the frame/exception machinery and the tail-calling
# interpreter; the stop_at cancellation cascade then pops process frames out of
# datastack order and corrupts the chain -> segfault in _PyFrame_ClearExceptCode.
#
# The fix gives every coroutine its own isolated frame datastack and exception
# stack, mirroring what greenlet does, but at the single choke point of the
# context switch (symmetric save before leaving / restore after being resumed,
# with the saved state living in the wrapper frame on the coroutine's own,
# preserved C stack). CPython itself frees/recycles the datastack chunks when a
# coroutine's frames pop on exit, so there is nothing to free here and no leak.
#
# Active for every Cimba coroutine switch on Python 3.14+ (main thread and
# run_experiment workers alike — the same frame/exception corruption hits a
# multi-process main-thread simulation under stop_at cancellation). Compiled out
# on Python 3.13 and earlier, where the wrapper just calls the real switch.

cdef extern from *:
    """
    #include <Python.h>

    /* Interposed via -Wl,--wrap=cmi_coroutine_context_switch (meson.build).
     * Callers reach __wrap_...; the vendored implementation is __real_.... */
    extern void *__real_cmi_coroutine_context_switch(void **old, void **newc, void *ret);

    #if PY_VERSION_HEX >= 0x030E0000
    /* Per-coroutine interpreter state. We swap only the frame stack, exception
     * handling stack, recursion accounting, and contextvars. Deliberately NOT
     * swapped: current_exception (always NULL at Cimba's hold/yield switch
     * points) and delete_later (a thread-global trashcan queue -- swapping it
     * strands deferred deletions of deep tracebacks/frames, a leak). Pointer
     * copies of context are balanced (saved out and back), so no refcount work
     * is needed at the switch. */
    typedef struct {
        struct _PyInterpreterFrame *current_frame;
        _PyErr_StackItem *exc_info;
        int py_recursion_remaining;
        int py_recursion_limit;
        int recursion_headroom;
        _PyStackChunk *datastack_chunk;
        PyObject **datastack_top;
        PyObject **datastack_limit;
        PyObject *context;
        uint64_t context_ver;
    } _cimba_corostate;

    static inline void _cimba_corostate_save(_cimba_corostate *s, PyThreadState *ts) {
        s->current_frame = ts->current_frame;
        s->exc_info = ts->exc_info;
        s->py_recursion_remaining = ts->py_recursion_remaining;
        s->py_recursion_limit = ts->py_recursion_limit;
        s->recursion_headroom = ts->recursion_headroom;
        s->datastack_chunk = ts->datastack_chunk;
        s->datastack_top = ts->datastack_top;
        s->datastack_limit = ts->datastack_limit;
        s->context = ts->context;
        s->context_ver = ts->context_ver;
    }
    static inline void _cimba_corostate_restore(const _cimba_corostate *s, PyThreadState *ts) {
        ts->current_frame = s->current_frame;
        ts->exc_info = s->exc_info;
        ts->py_recursion_remaining = s->py_recursion_remaining;
        ts->py_recursion_limit = s->py_recursion_limit;
        ts->recursion_headroom = s->recursion_headroom;
        ts->datastack_chunk = s->datastack_chunk;
        ts->datastack_top = s->datastack_top;
        ts->datastack_limit = s->datastack_limit;
        ts->context = s->context;
        ts->context_ver = s->context_ver;
    }

    void *__wrap_cmi_coroutine_context_switch(void **old, void **newc, void *ret) {
        PyThreadState *ts = PyThreadState_GetUnchecked();
        if (ts == NULL) {
            /* No active Python thread state (e.g. switch outside the GIL):
             * nothing to save, just forward to the real switch. */
            return __real_cmi_coroutine_context_switch(old, newc, ret);
        }
        _cimba_corostate saved;
        _cimba_corostate_save(&saved, ts);
        void *r = __real_cmi_coroutine_context_switch(old, newc, ret);
        /* Resumed (possibly much later, same OS thread): restore my state. */
        ts = PyThreadState_GetUnchecked();
        if (ts != NULL) {
            _cimba_corostate_restore(&saved, ts);
        }
        return r;
    }

    /* Give a freshly started coroutine its own empty frame datastack so its
     * frames can't collide with other coroutines' on the shared thread state.
     * CPython allocates a datastack chunk on the first frame push and recycles
     * it (via its one-chunk cache) when the coroutine's frames pop. The
     * exception-handling stack (exc_info) is carried by save/restore. */
    static inline void _cimba_corostate_enter_fresh(void) {
        PyThreadState *ts = PyThreadState_GetUnchecked();
        if (ts == NULL) { return; }
        ts->current_frame = NULL;
        ts->datastack_chunk = NULL;
        ts->datastack_top = NULL;
        ts->datastack_limit = NULL;
    }
    #else
    void *__wrap_cmi_coroutine_context_switch(void **old, void **newc, void *ret) {
        return __real_cmi_coroutine_context_switch(old, newc, ret);
    }
    static inline void _cimba_corostate_enter_fresh(void) { }
    #endif
    """
    void _cimba_corostate_enter_fresh()
