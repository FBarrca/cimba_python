/*
 * resource.h - cb.Resource: a pool of identical units that processes acquire,
 * preempt and release in amounts, with a capacity that can change during a
 * trial.
 *
 * It lives here rather than in the vendored engine so the engine stays
 * unmodified. The holders and the time-weighted histories follow
 * cmb_resourcepool; the waiting room is the engine's own cmb_resourceguard,
 * whose hashheap this file enqueues into directly so that a request can carry
 * an explicit rank and keep its place across the waits of one call. The
 * engine's process machinery (interrupt, stop, priority_set, grant wakeups)
 * then treats a waiting request exactly like any other guarded wait. See
 * docs/architecture/native.rst.
 */

#ifndef CIMBA_PY_RESOURCE_H
#define CIMBA_PY_RESOURCE_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "cmb_resourceguard.h"
#include "cmb_timeseries.h"
#include "cmi_hashheap.h"
#include "cmi_holdable.h"
#include "cmi_memregistry.h"
#include "nbshim.h"

struct cpy_resource {
    struct cmi_holdable core;           /* First: the engine sees a holdable */
    struct cmb_resourceguard guard;     /* The waiting room, observable */
    struct cmi_hashheap holders;        /* Holdings, next preemption victim first */
    uint64_t capacity;
    uint64_t max_capacity;              /* Largest capacity so far, for reports */
    uint64_t in_use;                    /* May exceed capacity after "finish" */
    bool is_recording;
    struct cmb_timeseries history;          /* in_use */
    struct cmb_timeseries capacity_history; /* capacity, sampled with in_use */
    struct cmi_memregistry_item destroy;
};

struct cpy_resource *cpy_resource_create(void);
void cpy_resource_initialize(struct cpy_resource *rp, const char *name,
                             uint64_t capacity);
void cpy_resource_terminate(struct cpy_resource *rp);
void cpy_resource_destroy(struct cpy_resource *rp);
void cpy_resource_recording_start(struct cpy_resource *rp);
void cpy_resource_recording_stop(struct cpy_resource *rp);
struct cmb_timeseries *cpy_resource_usage_history(struct cpy_resource *rp);

/* Model verbs. A rank-less call ranks the request and the caller's holding
 * by the process priority, tracking it; a _ranked call fixes both. */
CIMBA_PY_EXPORT int64_t cpy_resource_acquire(void *rp, uint64_t amount);
CIMBA_PY_EXPORT int64_t cpy_resource_preempt(void *rp, uint64_t amount);
CIMBA_PY_EXPORT int64_t cpy_resource_acquire_ranked(void *rp, uint64_t amount,
                                                    int64_t rank);
CIMBA_PY_EXPORT int64_t cpy_resource_preempt_ranked(void *rp, uint64_t amount,
                                                    int64_t rank);
CIMBA_PY_EXPORT void cpy_resource_release(void *rp, uint64_t amount);
/* One unit from whichever of count resources grants first: returns its index,
 * or the negative signal that ended the wait. handles[i] is the address of
 * member i's handle, the record member holding its resource pointer. Ranked
 * like acquire when ranked is nonzero, else by the process priority. */
CIMBA_PY_EXPORT int64_t cpy_resource_acquire_any(void *const *const *handles,
                                                 uint64_t count, uint64_t ranked,
                                                 int64_t rank);
CIMBA_PY_EXPORT void cpy_resource_set_capacity(void *rp, int64_t capacity,
                                               uint64_t preempt);

CIMBA_PY_EXPORT uint64_t cpy_resourcepool_held(const void *rp, const void *pp);
CIMBA_PY_EXPORT uint64_t cpy_resourcepool_available(const void *rp);
CIMBA_PY_EXPORT uint64_t cpy_resourcepool_in_use(const void *rp);
CIMBA_PY_EXPORT uint64_t cpy_resourcepool_capacity(const void *rp);
CIMBA_PY_EXPORT double cpy_resourcepool_mean_in_use(void *rp);
CIMBA_PY_EXPORT double cpy_resourcepool_mean_capacity(void *rp);
CIMBA_PY_EXPORT double cpy_resourcepool_utilization(void *rp);
CIMBA_PY_EXPORT void *cpy_resourcepool_history(void *rp);
void cpy_resourcepool_print_report(void *rp, FILE *fp);

#endif
