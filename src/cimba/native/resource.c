/*
 * resource.c - cb.Resource, our own pool of identical units.
 *
 * The acquire, preempt and release algorithms, the holders hashheap and the
 * histories follow the engine's cmb_resourcepool, so a model gets the same
 * results as it did on it. What this file adds without touching the engine:
 *
 * - Explicit request ranks. A rank-less request ranks its queue entry and the
 *   caller's holding by the process priority, tracking priority_set; a ranked
 *   request fixes both. One process has one holding per resource, ranked by
 *   its latest call; an interrupted call restores the previous amount and
 *   rank. Wakeups and preemption interrupts still run at the process priority.
 * - First come, first served across the waits of one call. A request that
 *   takes what is free and waits for the rest returns to its original place
 *   (entry time and sequence) each time it waits again.
 * - A capacity that can change, possibly to zero, with "finish" or "preempt".
 *
 * Waiting room. The queue is the engine's cmb_resourceguard. Instead of
 * cmb_resourceguard_wait, wait_placed() enqueues into the guard's hashheap
 * itself, with its own entry time, rank and key, registers the RESOURCE
 * awaitable the engine expects and suspends. From there the engine's own code
 * handles the entry: cmb_resourceguard_signal grants it with a wakeup at the
 * process priority, an interrupt or stop removes it (cmi_process_cancel_
 * awaiteds), and priority_set re-ranks it. Keys carry the top bit, so they
 * never collide with the engine's own guard keys, and every wait gets a fresh
 * key, so a stale grant from an earlier wait is discarded by the engine's key
 * check. The key's middle bits are the request's sequence number, so ordering
 * among equal ranks and entry times is the order the requests were made.
 *
 * Holdable tag. A process carries this resource's cmi_holdable tag while it
 * holds units or waits here. The engine calls drop() when a holder exits or
 * is stopped, and reprio() on priority_set, after it has re-ranked the
 * waiting entry at the new priority: reprio() re-ranks a tracking holding and
 * puts an explicitly ranked waiting entry back at its rank.
 *
 * Acquire any. cpy_resource_acquire_any waits for one unit from any of several
 * resources with one place (rank, entry time, key) in every member's queue.
 * Its demand function is the grant hook: when one member grants, it withdraws
 * the request from the other queues and their awaitables before the engine
 * schedules the wakeup, so no second member can grant it. See the section
 * "Acquire any" below.
 *
 * Holder hashheap payload: item[0] holder process, item[1] amount, item[2]
 * HOLDING_FIXED if the holding rank is explicit. Waiting entry payload, as
 * cmb_resourceguard reads it: item[0] process, item[1] demand function,
 * item[2] the request (on the waiting process's stack), which starts with a
 * struct request for an acquire_any too.
 */

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "cimba.h"
#include "cmi_memutils.h"
#include "cmi_process.h"
#include "resource.h"

struct holding {
    struct cmb_process *holder;
    uint64_t amount;
    void *flags;
};

/* Holding flag in item[2]: ranked explicitly, not by the process priority */
#define HOLDING_FIXED ((void *)(uintptr_t)1u)

/* The rank of one acquire or preempt call */
struct request {
    bool fixed;         /* Explicit rank rather than the process priority */
    int64_t rank;       /* The explicit rank, if fixed */
};

/* Where a request stands in the queue, kept across the waits of one call */
struct place {
    uint64_t seq;       /* 0 until the first wait */
    double time;        /* Entry time of the first wait */
    uint64_t episode;   /* Waits so far, giving each its own key */
};

#define KEY_TAG (UINT64_C(1) << 63)
#define EPISODE_BITS 20u
#define EPISODE_MAX ((UINT64_C(1) << EPISODE_BITS) - 1u)
#define SEQ_MASK ((UINT64_C(1) << (63u - EPISODE_BITS)) - 1u)

/* Request sequence numbers, shared by every resource of the thread */
static CMB_THREAD_LOCAL uint64_t request_seq = 0u;

static inline int64_t request_rank(const struct request *rq,
                                   const struct cmb_process *pp)
{
    return rq->fixed ? rq->rank : cmb_process_priority(pp);
}

static inline uint64_t holder_key(const struct cmb_process *pp)
{
    return pp->handle;
}

static inline struct cmi_hashheap *queue_of(struct cpy_resource *rp)
{
    return &(rp->guard.priority_queue);
}

/* Fails the trial with "<name>: <verb>(<amount>) <reason>" */
static void fail(const struct cpy_resource *rp, const char *verb,
                 const uint64_t amount, const char *reason)
{
    char failure[160];
    snprintf(failure, sizeof failure, "%s: %s(%" PRIu64 ") %s",
             rp->core.base.name, verb, amount, reason);
    cpy_trial_fail(failure);
}

/* The amount available to grab, zero while in_use is at or above capacity */
static inline uint64_t room(const struct cpy_resource *rp)
{
    return (rp->in_use < rp->capacity) ? (rp->capacity - rp->in_use) : 0u;
}

static void record_sample(struct cpy_resource *rp)
{
    if (rp->is_recording) {
        const double t = cmb_time();
        cmb_timeseries_add(&(rp->history), (double)rp->in_use, t);
        /* The capacity is sampled with the usage, so both cover one span */
        cmb_timeseries_add(&(rp->capacity_history), (double)rp->capacity, t);
    }
}

/* ---------------------------------------------------------------------------
 * Holdable tags
 */

static bool has_tag(const struct cmb_process *pp, const struct cpy_resource *rp)
{
    const struct cmi_slist_node *node = &(pp->resources);
    while (node->next != NULL) {
        const struct cmi_process_holdable *ph =
            cmi_container_of(node->next, struct cmi_process_holdable, listhead);
        if (ph->res == &(rp->core)) return true;
        node = node->next;
    }
    return false;
}

static void add_tag(struct cmb_process *pp, struct cpy_resource *rp)
{
    if (has_tag(pp, rp)) return;
    struct cmi_process_holdable *ph = cmi_mempool_alloc(&cmi_process_holdabletags);
    ph->res = &(rp->core);
    cmi_slist_push(&(pp->resources), &(ph->listhead));
}

uint64_t cpy_resourcepool_held(const void *vp, const void *pp)
{
    struct cpy_resource *rp = (struct cpy_resource *)vp;
    struct cmi_hashheap *hhp = &(rp->holders);
    const uint64_t heapidx = cmi_hash_find_index(hhp, holder_key(pp));
    if (heapidx == 0u) return 0u;
    return ((const struct holding *)(hhp->heap[heapidx].item))->amount;
}

/* Leaving a call: keep the tag only if the caller still holds units */
static void settle_tag(struct cmb_process *pp, struct cpy_resource *rp)
{
    if (cpy_resourcepool_held(rp, pp) == 0u && has_tag(pp, rp))
        (void)cmi_process_remove_holdable(pp, &(rp->core));
}

/*
 * drop_holder - The engine ejects a holder that exits or is stopped, without
 * resuming it. Its tag is already off its list. A process that only waited
 * here has no holding; the engine has already removed its waiting entry.
 */
static void drop_holder(struct cmi_holdable *hrp, const struct cmb_process *pp)
{
    struct cpy_resource *rp = (struct cpy_resource *)hrp;
    struct cmi_hashheap *hhp = &(rp->holders);
    const uint64_t key = holder_key(pp);
    const uint64_t heapidx = cmi_hash_find_index(hhp, key);
    if (heapidx != 0u) {
        const struct holding *h = (struct holding *)(hhp->heap[heapidx].item);
        cmb_assert_debug(h->amount <= rp->in_use);
        rp->in_use -= h->amount;
        (void)cmi_hashheap_cancel(hhp, key);
        /* As cmb_resourcepool: no history sample here, so the drop shows
         * from the next sample on */
        cmb_resourceguard_signal(&(rp->guard));
    }
}

/*
 * reprioritize - priority_set changed the process priority. A tracking
 * holding follows it. The engine has just re-ranked the process's waiting
 * entry, if any, at the new priority; an explicitly ranked one goes back.
 */
static void reprioritize(struct cmi_holdable *hrp, const struct cmb_process *pp,
                         const int64_t priority)
{
    struct cpy_resource *rp = (struct cpy_resource *)hrp;
    struct cmi_hashheap *hhp = &(rp->holders);
    const uint64_t key = holder_key(pp);
    if (cmi_hashheap_is_enqueued(hhp, key)) {
        const struct holding *h = (struct holding *)cmi_hashheap_item(hhp, key);
        if (h->flags != HOLDING_FIXED)
            cmi_hashheap_reprioritize(hhp, key, cmi_hashheap_drank(hhp, key),
                                      priority);
    }

    struct cmi_hashheap *qp = queue_of(rp);
    const uint64_t wkey = cmi_process_guard_key(pp, &(rp->guard));
    if (wkey != 0u && cmi_hashheap_is_enqueued(qp, wkey)) {
        void **item = cmi_hashheap_item(qp, wkey);
        const struct request *rq = item[2];
        if (rq->fixed && cmi_hashheap_irank(qp, wkey) != rq->rank)
            cmi_hashheap_reprioritize(qp, wkey, cmi_hashheap_drank(qp, wkey),
                                      rq->rank);
    }
}

/* ---------------------------------------------------------------------------
 * Lifecycle
 */

/*
 * holder_order - Lower rank first, then the most recent holding (LIFO on the
 * grant time), then the higher handle: the next preemption victim first.
 */
static bool holder_order(const struct cmi_heap_tag *a, const struct cmi_heap_tag *b)
{
    if (a->rank_i64 != b->rank_i64) return a->rank_i64 < b->rank_i64;
    if (a->rank_d64 != b->rank_d64) return a->rank_d64 > b->rank_d64;
    return a->hash_key > b->hash_key;
}

struct cpy_resource *cpy_resource_create(void)
{
    struct cpy_resource *rp = cmi_malloc(sizeof *rp);
    cmi_memset(rp, 0, sizeof *rp);
    rp->core.base.cookie = CMI_UNINITIALIZED;

    /* Registered so an abandoned trial tears it down like Cimba's own */
    cmi_dlist_initialize(&(rp->destroy.node));
    rp->destroy.teardown = (cmi_teardown_func *)cpy_resource_destroy;
    rp->destroy.object = rp;
    cmi_memregistry_add(&(rp->destroy));
    return rp;
}

void cpy_resource_initialize(struct cpy_resource *rp, const char *name,
                             const uint64_t capacity)
{
    /* The base registers its own terminate item; point it at ours */
    cmi_holdable_initialize(&(rp->core), name);
    rp->core.base.terminate.teardown = (cmi_teardown_func *)cpy_resource_terminate;
    rp->core.drop = drop_holder;
    rp->core.reprio = reprioritize;

    cmi_hashheap_initialize(&(rp->holders), 3u, holder_order);
    cmb_resourceguard_initialize(&(rp->guard), &(rp->core.base));
    rp->capacity = capacity;
    rp->max_capacity = capacity;
    rp->in_use = 0u;
    rp->is_recording = false;
    cmb_timeseries_initialize(&(rp->history));
    cmb_timeseries_initialize(&(rp->capacity_history));
}

void cpy_resource_terminate(struct cpy_resource *rp)
{
    if (rp->core.base.cookie != CMI_INITIALIZED) return;
    cmb_timeseries_terminate(&(rp->history));
    cmb_timeseries_terminate(&(rp->capacity_history));
    cmi_hashheap_terminate(&(rp->holders));
    cmb_resourceguard_terminate(&(rp->guard));
    /* Clears the base's memregistry item, unless demolishing */
    cmi_holdable_terminate(&(rp->core));
}

void cpy_resource_destroy(struct cpy_resource *rp)
{
    if (!cmi_memregistry_is_demolishing)
        cmi_memregistry_remove(&(rp->destroy));
    cmi_free(rp);
}

void cpy_resource_recording_start(struct cpy_resource *rp)
{
    rp->is_recording = true;
    record_sample(rp);
}

void cpy_resource_recording_stop(struct cpy_resource *rp)
{
    record_sample(rp);
    rp->is_recording = false;
}

struct cmb_timeseries *cpy_resource_usage_history(struct cpy_resource *rp)
{
    return &(rp->history);
}

/* ---------------------------------------------------------------------------
 * Waiting
 */

/* The guard's demand function: something to grab */
static bool is_available(const struct cmi_resourcebase *rbp,
                         const struct cmb_process *pp, const void *ctx)
{
    (void)pp;
    (void)ctx;
    const struct cpy_resource *rp = (const struct cpy_resource *)rbp;
    return rp->in_use < rp->capacity;
}

/*
 * wait_placed - cmb_resourceguard_wait at the request's own place: its first
 * entry time and sequence, at its rank. The epilogue is the engine's, so a
 * resumed wait cleans up exactly as a guarded wait does.
 */
/* The key of the next wait at a request's place: a sequence number and entry
 * time on its first wait, a fresh episode on each later one */
static uint64_t next_wait_key(struct place *pl)
{
    if (pl->seq == 0u) {
        request_seq = (request_seq + 1u) & SEQ_MASK;
        if (request_seq == 0u) request_seq = 1u;
        pl->seq = request_seq;
        pl->time = cmb_time();
        pl->episode = 0u;
    }
    else if (pl->episode < EPISODE_MAX) {
        pl->episode++;
    }
    return KEY_TAG | (pl->seq << EPISODE_BITS) | pl->episode;
}

static int64_t wait_placed(struct cpy_resource *rp, struct cmb_process *pp,
                           const struct request *rq, struct place *pl)
{
    const uint64_t key = next_wait_key(pl);

    struct cmi_hashheap *qp = queue_of(rp);
    (void)cmi_hashheap_enqueue(qp, pp, (void *)is_available, (void *)rq, NULL,
                               key, pl->time, request_rank(rq, pp));
    struct cmi_process_awaitable *awp =
        cmi_process_add_awaitable(pp, CMI_PROCESS_AWAITABLE_RESOURCE, &(rp->guard));
    awp->guard_key = key;

    const int64_t sig = cmb_process_yield();

    (void)cmi_process_remove_awaitable(pp, CMI_PROCESS_AWAITABLE_RESOURCE,
                                       &(rp->guard));
    if (cmi_hashheap_is_enqueued(qp, key)) {
        /* Resumed by a timer or a bare resume while still queued */
        (void)cmi_hashheap_cancel(qp, key);
    }
    else if (sig != CMB_PROCESS_SUCCESS && sig != CMB_PROCESS_CANCELLED) {
        /* Removed by an interrupt or preemption, perhaps after a grant that
         * is now stale: pass the opportunity on, as the engine's wait does */
        (void)cmb_resourceguard_signal(&(rp->guard));
    }
    return sig;
}

/* ---------------------------------------------------------------------------
 * Acquire, preempt, release
 */

/* Give an existing holding the rank of the current request, keeping its
 * grant time (the LIFO key among equals) */
static void set_holding_rank(struct cmi_hashheap *hhp, const uint64_t key,
                             const int64_t rank, void *flags)
{
    struct holding *h = (struct holding *)cmi_hashheap_item(hhp, key);
    h->flags = flags;
    if (cmi_hashheap_irank(hhp, key) != rank)
        cmi_hashheap_reprioritize(hhp, key, cmi_hashheap_drank(hhp, key), rank);
}

static void update_record(struct cpy_resource *rp, struct cmb_process *pp,
                          const uint64_t amount, const struct request *rq)
{
    struct cmi_hashheap *hhp = &(rp->holders);
    const uint64_t key = holder_key(pp);
    const uint64_t heapidx = cmi_hash_find_index(hhp, key);
    if (heapidx != 0u) {
        ((struct holding *)(hhp->heap[heapidx].item))->amount += amount;
    }
    else {
        add_tag(pp, rp);
        (void)cmi_hashheap_enqueue(hhp, pp, (void *)(uintptr_t)amount,
                                   rq->fixed ? HOLDING_FIXED : NULL, NULL,
                                   key, cmb_time(), request_rank(rq, pp));
    }
}

/* Shrink the caller's holding back to `amount`; returns the surplus */
static uint64_t reset_holder(struct cmi_hashheap *hhp,
                             const struct cmb_process *pp, const uint64_t amount)
{
    struct holding *h = (struct holding *)cmi_hashheap_item(hhp, holder_key(pp));
    const uint64_t surplus = h->amount - amount;
    h->amount = amount;
    return surplus;
}

/*
 * preempt_front - Throw out the holder at the front of the holders hashheap.
 * It loses its whole holding and any wait, and gets a CMB_PROCESS_PREEMPTED
 * interrupt at its own priority, but does not run until the caller yields.
 * Returns the amount taken back, which is no longer in use.
 */
static uint64_t preempt_front(struct cpy_resource *rp)
{
    struct cmi_hashheap *hhp = &(rp->holders);
    const struct holding *h = (struct holding *)cmi_hashheap_dequeue(hhp);
    struct cmb_process *victim = h->holder;
    const uint64_t loot = h->amount;
    cmb_assert_debug(loot <= rp->in_use);
    rp->in_use -= loot;

    (void)cmi_process_remove_holdable(victim, &(rp->core));
    /* It may be waiting for more, here or elsewhere: cancel that */
    cmi_process_cancel_awaiteds(victim);
    cmb_process_interrupt(victim, CMB_PROCESS_PREEMPTED, cmb_process_priority(victim));
    cmb_logger_info(stdout, "Preempted %" PRIu64 " from %s", loot, victim->name);
    return loot;
}

/*
 * acquire_inner - Greedy: take what is free, then preempt strictly lower
 * holding ranks if allowed, then wait at the request's place for the rest.
 */
static int64_t acquire_inner(struct cpy_resource *rp, const uint64_t req_amount,
                             const bool preempt, const struct request *rq)
{
    struct cmb_process *caller = cmb_process_current();
    struct cmi_hashheap *hhp = &(rp->holders);
    const uint64_t key = holder_key(caller);

    /* Does the caller already hold some? Note it for a rollback */
    uint64_t initially_held = 0u;
    int64_t initial_rank = 0;
    void *initial_flags = NULL;
    const uint64_t heapidx = cmi_hash_find_index(hhp, key);
    if (heapidx != 0u) {
        initially_held = ((const struct holding *)(hhp->heap[heapidx].item))->amount;
        initial_rank = hhp->heap[heapidx].rank_i64;
        initial_flags = hhp->heap[heapidx].item[2];
        /* The whole holding now ranks as this request. This also keeps the
         * caller from ranking strictly below itself as a preemption victim. */
        set_holding_rank(hhp, key, request_rank(rq, caller),
                         rq->fixed ? HOLDING_FIXED : NULL);
    }
    cmb_logger_info(stdout, "Has %" PRIu64 ", requests %" PRIu64 " from %s",
                    initially_held, req_amount, rp->core.base.name);

    uint64_t rem_claim = req_amount;
    struct place place = { 0u, 0.0, 0u };
    while (true) {
        const uint64_t available = room(rp);
        if (available >= rem_claim) {
            rp->in_use += rem_claim;
            record_sample(rp);
            update_record(rp, caller, rem_claim, rq);
            /* In case someone else can use the leftovers */
            cmb_resourceguard_signal(&(rp->guard));
            return CMB_PROCESS_SUCCESS;
        }
        else if (available > 0u) {
            rp->in_use += available;
            record_sample(rp);
            rem_claim -= available;
            update_record(rp, caller, available, rq);
        }

        if (preempt) {
            while (!cmi_hashheap_is_empty(hhp)
                   && cmi_hashheap_peek_irank(hhp) < request_rank(rq, caller)) {
                const uint64_t before = rp->in_use;
                (void)preempt_front(rp);
                /* Take what we need of the room, which also covers any excess
                 * use left by lowering the capacity, and leave the rest */
                const uint64_t avail = room(rp);
                const uint64_t take = (avail < rem_claim) ? avail : rem_claim;
                if (take > 0u) {
                    rp->in_use += take;
                    update_record(rp, caller, take, rq);
                    rem_claim -= take;
                }
                /* As cmb_resourcepool, the claim completing is sampled even
                 * when it puts nothing back */
                if (rp->in_use != before || rem_claim == 0u) record_sample(rp);
                if (rem_claim == 0u) {
                    cmb_resourceguard_signal(&(rp->guard));
                    return CMB_PROCESS_SUCCESS;
                }
            }
        }

        add_tag(caller, rp);
        const int64_t sig = wait_placed(rp, caller, rq, &place);
        if (sig == CMB_PROCESS_PREEMPTED) {
            /* Thrown out instead (here or elsewhere), returning as is */
            settle_tag(caller, rp);
            return sig;
        }
        else if (sig != CMB_PROCESS_SUCCESS) {
            /* Interrupted: back to the holding the call started with */
            if (initially_held > 0u) {
                const uint64_t surplus = reset_holder(hhp, caller, initially_held);
                set_holding_rank(hhp, key, initial_rank, initial_flags);
                rp->in_use -= surplus;
                record_sample(rp);
                cmb_resourceguard_signal(&(rp->guard));
            }
            else {
                rp->in_use -= cpy_resourcepool_held(rp, caller);
                record_sample(rp);
                (void)cmi_hashheap_cancel(hhp, key);
            }
            settle_tag(caller, rp);
            return sig;
        }
    }
}

/* Amounts arrive as uint64; a negative int from model code is huge */
static void check_amount(struct cpy_resource *rp, const char *verb,
                         const uint64_t amount, const char *outside)
{
    cpy_require_process(rp, outside);
    if (amount == 0u || amount > (uint64_t)INT64_MAX)
        fail(rp, verb, amount, "needs a positive amount");
}

int64_t cpy_resource_acquire(void *vp, const uint64_t amount)
{
    struct cpy_resource *rp = vp;
    check_amount(rp, "acquire", amount, "acquire() cannot hold a resource");
    const struct request rq = { false, 0 };
    return acquire_inner(rp, amount, false, &rq);
}

int64_t cpy_resource_preempt(void *vp, const uint64_t amount)
{
    struct cpy_resource *rp = vp;
    check_amount(rp, "preempt", amount, "preempt() cannot hold a resource");
    const struct request rq = { false, 0 };
    return acquire_inner(rp, amount, true, &rq);
}

int64_t cpy_resource_acquire_ranked(void *vp, const uint64_t amount,
                                    const int64_t rank)
{
    struct cpy_resource *rp = vp;
    check_amount(rp, "acquire", amount, "acquire() cannot hold a resource");
    const struct request rq = { true, rank };
    return acquire_inner(rp, amount, false, &rq);
}

int64_t cpy_resource_preempt_ranked(void *vp, const uint64_t amount,
                                    const int64_t rank)
{
    struct cpy_resource *rp = vp;
    check_amount(rp, "preempt", amount, "preempt() cannot hold a resource");
    const struct request rq = { true, rank };
    return acquire_inner(rp, amount, true, &rq);
}

/* Release part or all of the caller's holding */
void cpy_resource_release(void *vp, const uint64_t amount)
{
    struct cpy_resource *rp = vp;
    check_amount(rp, "release", amount, "release() holds no resource");
    struct cmb_process *pp = cmb_process_current();
    struct cmi_hashheap *hhp = &(rp->holders);
    const uint64_t key = holder_key(pp);
    const uint64_t held = cpy_resourcepool_held(rp, pp);
    if (amount > held) fail(rp, "release", amount, "exceeds the units held");

    if (held == amount) {
        (void)cmi_hashheap_cancel(hhp, key);
        (void)cmi_process_remove_holdable(pp, &(rp->core));
    }
    else {
        ((struct holding *)cmi_hashheap_item(hhp, key))->amount -= amount;
    }
    rp->in_use -= amount;
    record_sample(rp);
    cmb_logger_info(stdout, "%s released %" PRIu64, rp->core.base.name, amount);
    cmb_resourceguard_signal(&(rp->guard));
}

/*
 * cpy_resource_set_capacity - Raising the capacity grants waiting requests.
 * Lowering it either lets the holders finish, leaving in_use above the
 * capacity for a while, or preempts holders from the front of the holders
 * hashheap until in_use fits, sparing the calling process. A negative
 * capacity fails the trial; the mode is a compile-time literal.
 */
void cpy_resource_set_capacity(void *vp, const int64_t value, const uint64_t mode)
{
    struct cpy_resource *rp = vp;
    if (value < 0) {
        char failure[160];
        snprintf(failure, sizeof failure,
                 "%s: set_capacity(%" PRId64 ") needs a nonnegative capacity",
                 rp->core.base.name, value);
        cpy_trial_fail(failure);
    }
    const uint64_t capacity = (uint64_t)value;
    if (capacity == rp->capacity) return;

    cmb_logger_info(stdout, "%s capacity %" PRIu64 " -> %" PRIu64 ", in use %" PRIu64,
                    rp->core.base.name, rp->capacity, capacity, rp->in_use);
    rp->capacity = capacity;
    if (capacity > rp->max_capacity) rp->max_capacity = capacity;

    if (mode != 0u && rp->in_use > capacity) {
        struct cmi_hashheap *hhp = &(rp->holders);
        const struct cmb_process *caller = cmb_process_current();

        /* The caller is not its own victim: set its holding aside, then put
         * it back with its rank and grant time, so its place is unchanged */
        bool spared = false;
        void *spared_item[3] = { NULL, NULL, NULL };
        double spared_drank = 0.0;
        int64_t spared_irank = 0;
        while (rp->in_use > capacity && !cmi_hashheap_is_empty(hhp)) {
            void **item = cmi_hashheap_peek_item(hhp);
            if (item[0] == (void *)caller) {
                spared_drank = cmi_hashheap_peek_drank(hhp);
                spared_irank = cmi_hashheap_peek_irank(hhp);
                spared_item[0] = item[0];
                spared_item[1] = item[1];
                spared_item[2] = item[2];
                (void)cmi_hashheap_dequeue(hhp);
                spared = true;
                continue;
            }
            (void)preempt_front(rp);
        }
        if (spared)
            (void)cmi_hashheap_enqueue(hhp, spared_item[0], spared_item[1],
                                       spared_item[2], NULL, holder_key(caller),
                                       spared_drank, spared_irank);
    }

    record_sample(rp);
    /* Any room goes to the waiting requests, in queue order. Each grant
     * signals again for the next one when it takes its amount. Signal even
     * without room, so observers see the capacity change. */
    cmb_resourceguard_signal(&(rp->guard));
}

/* ---------------------------------------------------------------------------
 * Acquire any
 *
 * One unit from whichever member grants first. A free unit is taken at once,
 * from the first member listed that has one. Otherwise the request waits in
 * every member's queue with the same rank, entry time and key, the place an
 * acquire made now would get, so it competes with direct acquirers in each.
 * The grant hook (is_available_any, the entries' demand function) runs when a
 * member's guard grants the entry: it notes the member and withdraws the
 * request from every other queue and its awaitable there, before the engine
 * schedules the wakeup. The engine then sees one ordinary granted guarded
 * wait. Interrupts, stops and preemption remove every remaining entry through
 * the engine's own cancel_awaiteds. The unit becomes an ordinary holding of
 * that member, ranked by the request as acquire ranks one.
 */

#define NO_WINNER UINT64_MAX

/* Members are given by the addresses of their handles, the record members
 * that hold the resource pointers: a tuple's on the caller's stack, a
 * list[Ref[Resource]]'s pointer table as the host wrote it. Handles are set
 * when the trial starts and stay put until it ends, so the members are read
 * through them, with no copy and no limit on their number. */
typedef void *const *const *members_t;

struct any_request {
    struct request rq;                  /* First: reprioritize() reads it */
    members_t members;
    uint64_t count;
    uint64_t key;                       /* The current wait's key, in every queue */
    uint64_t winner;                    /* Granting member, NO_WINNER until then */
};

static inline struct cpy_resource *member(const members_t members, const uint64_t i)
{
    return (struct cpy_resource *)*members[i];
}

/* Whether members[i] repeats an earlier member; it then has no own entry */
static bool repeats(const members_t members, const uint64_t i)
{
    for (uint64_t j = 0u; j < i; j++)
        if (member(members, j) == member(members, i)) return true;
    return false;
}

/* The demand function and grant hook of an acquire_any entry */
static bool is_available_any(const struct cmi_resourcebase *rbp,
                             const struct cmb_process *pp, const void *ctx)
{
    const struct cpy_resource *rp = (const struct cpy_resource *)rbp;
    if (room(rp) == 0u) return false;

    /* Granted: cmb_resourceguard_signal dequeues this entry and schedules the
     * wakeup as soon as this returns true */
    struct any_request *ar = (struct any_request *)ctx;
    struct cmb_process *waiter = (struct cmb_process *)pp;
    for (uint64_t i = 0u; i < ar->count; i++) {
        struct cpy_resource *mp = member(ar->members, i);
        if (mp == rp) {
            if (ar->winner == NO_WINNER) ar->winner = i;
            continue;
        }
        if (repeats(ar->members, i)) continue;
        struct cmi_hashheap *qp = queue_of(mp);
        if (cmi_hashheap_is_enqueued(qp, ar->key))
            (void)cmi_hashheap_cancel(qp, ar->key);
        (void)cmi_process_remove_awaitable(waiter, CMI_PROCESS_AWAITABLE_RESOURCE,
                                           &(mp->guard));
    }
    return true;
}

/* Take one unit of rp for the caller, as a holding ranked by the request */
static void take_one(struct cpy_resource *rp, struct cmb_process *caller,
                     const struct request *rq)
{
    struct cmi_hashheap *hhp = &(rp->holders);
    const uint64_t key = holder_key(caller);
    if (cmi_hash_find_index(hhp, key) != 0u)
        set_holding_rank(hhp, key, request_rank(rq, caller),
                         rq->fixed ? HOLDING_FIXED : NULL);
    rp->in_use += 1u;
    record_sample(rp);
    update_record(rp, caller, 1u, rq);
    cmb_logger_info(stdout, "Acquired 1 from %s", rp->core.base.name);
    /* In case someone else can use the leftovers */
    cmb_resourceguard_signal(&(rp->guard));
}

static void settle_tags(struct cmb_process *pp, const members_t members,
                        const uint64_t count)
{
    for (uint64_t i = 0u; i < count; i++)
        if (!repeats(members, i)) settle_tag(pp, member(members, i));
}

int64_t cpy_resource_acquire_any(void *const *const *handles, const uint64_t count,
                                 const uint64_t ranked, const int64_t rank)
{
    if (count == 0u)
        cpy_trial_fail("acquire_any() needs at least one resource, got an empty list");
    const members_t members = handles;
    cpy_require_process(member(members, 0u), "acquire_any() cannot hold a resource");
    struct cmb_process *caller = cmb_process_current();

    struct any_request ar = { { ranked != 0u, rank }, members, count, 0u, NO_WINNER };
    const struct request *rq = &(ar.rq);
    struct place place = { 0u, 0.0, 0u };
    while (true) {
        for (uint64_t i = 0u; i < count; i++) {
            if (room(member(members, i)) > 0u) {
                take_one(member(members, i), caller, rq);
                settle_tags(caller, members, count);
                return (int64_t)i;
            }
        }

        /* Wait at one place in every member's queue */
        ar.key = next_wait_key(&place);
        ar.winner = NO_WINNER;
        for (uint64_t i = 0u; i < count; i++) {
            if (repeats(members, i)) continue;
            struct cpy_resource *mp = member(members, i);
            add_tag(caller, mp);
            (void)cmi_hashheap_enqueue(queue_of(mp), caller, (void *)is_available_any,
                                       (void *)&ar, NULL, ar.key, place.time,
                                       request_rank(rq, caller));
            struct cmi_process_awaitable *awp = cmi_process_add_awaitable(
                caller, CMI_PROCESS_AWAITABLE_RESOURCE, &(mp->guard));
            awp->guard_key = ar.key;
        }
        cmb_logger_info(stdout, "Waits for any of %" PRIu64 " resources", count);

        const int64_t sig = cmb_process_yield();

        /* Clean up as a guarded wait does, in every queue still holding it */
        for (uint64_t i = 0u; i < count; i++) {
            if (repeats(members, i)) continue;
            struct cpy_resource *mp = member(members, i);
            (void)cmi_process_remove_awaitable(caller, CMI_PROCESS_AWAITABLE_RESOURCE,
                                               &(mp->guard));
            if (cmi_hashheap_is_enqueued(queue_of(mp), ar.key))
                (void)cmi_hashheap_cancel(queue_of(mp), ar.key);
        }
        if (sig != CMB_PROCESS_SUCCESS) {
            /* A grant may have been pending: pass the opportunity on */
            if (sig != CMB_PROCESS_CANCELLED)
                for (uint64_t i = 0u; i < count; i++)
                    if (!repeats(members, i))
                        (void)cmb_resourceguard_signal(&(member(members, i)->guard));
            settle_tags(caller, members, count);
            return sig;
        }

        /* Granted by members[winner]. Someone may have taken the unit since;
         * then look again, and wait again at the same place if none is free. */
        if (ar.winner != NO_WINNER && room(member(members, ar.winner)) > 0u) {
            take_one(member(members, ar.winner), caller, rq);
            settle_tags(caller, members, count);
            return (int64_t)ar.winner;
        }
    }
}

/* ---------------------------------------------------------------------------
 * Accessors and statistics
 */

uint64_t cpy_resourcepool_available(const void *vp)
{
    return room(vp);
}

uint64_t cpy_resourcepool_in_use(const void *vp)
{
    return ((const struct cpy_resource *)vp)->in_use;
}

uint64_t cpy_resourcepool_capacity(const void *vp)
{
    return ((const struct cpy_resource *)vp)->capacity;
}

static double mean_of(const struct cmb_timeseries *tsp)
{
    struct cmb_wtdsummary ws;
    cmb_wtdsummary_initialize(&ws);
    (void)cmb_timeseries_summarize(tsp, &ws);
    const double mean = cmb_wtdsummary_mean(&ws);
    cmb_wtdsummary_terminate(&ws);
    return mean;
}

double cpy_resourcepool_mean_in_use(void *vp)
{
    return mean_of(&(((struct cpy_resource *)vp)->history));
}

double cpy_resourcepool_mean_capacity(void *vp)
{
    return mean_of(&(((struct cpy_resource *)vp)->capacity_history));
}

/* Time-weighted in-use over time-weighted capacity. Both histories are
 * sampled together, so the ratio covers one span. NaN when the window offered
 * no capacity (or holds no samples yet). */
double cpy_resourcepool_utilization(void *vp)
{
    const double capacity = cpy_resourcepool_mean_capacity(vp);
    if (!(capacity > 0.0)) return NAN;
    return cpy_resourcepool_mean_in_use(vp) / capacity;
}

void *cpy_resourcepool_history(void *vp)
{
    return &(((struct cpy_resource *)vp)->history);
}

void cpy_resourcepool_print_report(void *vp, FILE *fp)
{
    const struct cpy_resource *rp = vp;
    fprintf(fp, "Pool resource utilization for %s:\n", rp->core.base.name);
    struct cmb_wtdsummary ws = { 0 };
    cmb_wtdsummary_initialize(&ws);
    (void)cmb_timeseries_summarize(&(rp->history), &ws);
    cmb_wtdsummary_print(&ws, fp, true);
    cmb_wtdsummary_terminate(&ws);

    /* The usage never exceeds the largest capacity the resource has had */
    const uint64_t top = rp->max_capacity;
    const unsigned nbin = (top > 20u) ? 20u : (unsigned)top + 1u;
    cmb_timeseries_histogram_print(&(rp->history), fp, nbin, 0.0, (double)(top + 1u));
}
