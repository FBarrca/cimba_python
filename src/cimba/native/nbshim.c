/*
 * nbshim.c - exported wrappers for the Layer-2 Numba binding.
 *
 * Numba-compiled model code binds cimba functions as external symbols at
 * JIT link time. That works for everything declared extern, but several
 * hot-path functions are static inline in the public headers and have no
 * linkable symbol. This file wraps the ones the models need, plus a few
 * struct-size/accessor helpers so Numba code never needs C struct layouts.
 */

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "cimba.h"
#include "cmb_priorityqueue.h"   /* not part of the cimba.h umbrella */
#include "nbshim.h"

double cpy_random_exponential(const double mean)
{
    return cmb_random_exponential(mean);
}

double cpy_random_gamma(const double shape, const double scale)
{
    return cmb_random_gamma(shape, scale);
}

double cpy_random01(void)
{
    return cmb_random();
}

double cpy_random_uniform(const double min, const double max)
{
    return cmb_random_uniform(min, max);
}

double cpy_random_normal(const double mu, const double sigma)
{
    return cmb_random_normal(mu, sigma);
}

double cpy_random_rayleigh(const double s)
{
    return cmb_random_rayleigh(s);
}

double cpy_random_PERT(const double min, const double mode, const double max)
{
    return cmb_random_PERT(min, mode, max);
}

double cpy_random_PERT_mod(const double min, const double mode,
                           const double max, const double lambda)
{
    return cmb_random_PERT_mod(min, mode, max, lambda);
}

uint64_t cpy_random_bernoulli(const double p)
{
    return cmb_random_bernoulli(p);
}

uint64_t cpy_random_flip(void)
{
    return (uint64_t)cmb_random_flip();
}

double cpy_random_triangular(const double min, const double mode,
                             const double max)
{
    return cmb_random_triangular(min, mode, max);
}

double cpy_random_weibull(const double shape, const double scale)
{
    return cmb_random_weibull(shape, scale);
}

double cpy_random_lognormal(const double m, const double s)
{
    return cmb_random_lognormal(m, s);
}

double cpy_random_erlang(const uint64_t k, const double m)
{
    return cmb_random_erlang((unsigned)k, m);
}

double cpy_random_beta(const double a, const double b,
                       const double min, const double max)
{
    return cmb_random_beta(a, b, min, max);
}

uint64_t cpy_random_poisson(const double r)
{
    return cmb_random_poisson(r);
}

int64_t cpy_random_dice(const int64_t a, const int64_t b)
{
    return cmb_random_dice((long)a, (long)b);
}

double cpy_random_std_normal(void)
{
    return cmb_random_std_normal();
}

double cpy_random_std_exponential(void)
{
    return cmb_random_std_exponential();
}

double cpy_random_std_gamma(const double shape)
{
    return cmb_random_std_gamma(shape);
}

double cpy_random_std_beta(const double a, const double b)
{
    return cmb_random_std_beta(a, b);
}

double cpy_random_logistic(const double m, const double s)
{
    return cmb_random_logistic(m, s);
}

double cpy_random_cauchy(const double mode, const double scale)
{
    return cmb_random_cauchy(mode, scale);
}

double cpy_random_pareto(const double shape, const double mode)
{
    return cmb_random_pareto(shape, mode);
}

double cpy_random_chisquared(const double k)
{
    return cmb_random_chisquared(k);
}

double cpy_random_F_dist(const double a, const double b)
{
    return cmb_random_F_dist(a, b);
}

double cpy_random_std_t_dist(const double v)
{
    return cmb_random_std_t_dist(v);
}

double cpy_random_t_dist(const double m, const double s, const double v)
{
    return cmb_random_t_dist(m, s, v);
}

uint64_t cpy_random_geometric(const double p)
{
    return cmb_random_geometric(p);
}

uint64_t cpy_random_binomial(const uint64_t n, const double p)
{
    return cmb_random_binomial((unsigned)n, p);
}

uint64_t cpy_random_negative_binomial(const uint64_t m, const double p)
{
    return cmb_random_negative_binomial((unsigned)m, p);
}

uint64_t cpy_random_pascal(const uint64_t m, const double p)
{
    return cmb_random_pascal((unsigned)m, p);
}

/* Low-level event queue: bool returns widened to uint64_t for Numba */
uint64_t cpy_event_cancel(const uint64_t hndl)
{
    return (uint64_t)cmb_event_cancel(hndl);
}

uint64_t cpy_event_reschedule(const uint64_t hndl, const double time)
{
    return (uint64_t)cmb_event_reschedule(hndl, time);
}

uint64_t cpy_event_reprioritize(const uint64_t hndl, const int64_t priority)
{
    return (uint64_t)cmb_event_reprioritize(hndl, priority);
}

uint64_t cpy_event_is_scheduled(const uint64_t hndl)
{
    return (uint64_t)cmb_event_is_scheduled(hndl);
}

uint64_t cpy_wtdsummary_sizeof(void)
{
    return sizeof(struct cmb_wtdsummary);
}

double cpy_wtdsummary_mean(const void *wsp)
{
    return cmb_wtdsummary_mean(wsp);
}

/*
 * Value-passing wrappers around the in/out-pointer buffer API, so Numba
 * model code does not need to materialize a uint64 in memory per call.
 * Returns the cimba status code; the amount actually moved is amnt minus
 * whatever the call left unprocessed, which for the blocking calls used
 * here is always the full amount on success.
 */
int64_t cpy_buffer_put(void *bp, const uint64_t amnt)
{
    uint64_t n = amnt;
    return cmb_buffer_put(bp, &n);
}

int64_t cpy_buffer_get(void *bp, const uint64_t amnt)
{
    uint64_t n = amnt;
    return cmb_buffer_get(bp, &n);
}

/* Time-weighted mean of a recorded level/usage history */
static double mean_of_history(struct cmb_timeseries *tsp)
{
    struct cmb_wtdsummary ws;
    cmb_wtdsummary_initialize(&ws);
    cmb_timeseries_summarize(tsp, &ws);
    return cmb_wtdsummary_mean(&ws);
}

double cpy_buffer_mean_level(void *bp)
{
    return mean_of_history(cmb_buffer_history(bp));
}

double cpy_resource_mean_in_use(void *rp)
{
    return mean_of_history(cmb_resource_history(rp));
}

double cpy_resourcepool_mean_in_use(void *rpp)
{
    return mean_of_history(cmb_resourcepool_get_history(rpp));
}

double cpy_objectqueue_mean_length(void *oqp)
{
    return mean_of_history(cmb_objectqueue_history(oqp));
}

double cpy_priorityqueue_mean_length(void *pqp)
{
    return mean_of_history(cmb_priorityqueue_history(pqp));
}

void *cpy_buffer_history(void *bp)
{
    return cmb_buffer_history(bp);
}

void *cpy_resource_history(void *rp)
{
    return cmb_resource_history(rp);
}

void *cpy_resourcepool_history(void *rpp)
{
    return cmb_resourcepool_get_history(rpp);
}

void *cpy_objectqueue_history(void *oqp)
{
    return cmb_objectqueue_history(oqp);
}

void *cpy_priorityqueue_history(void *pqp)
{
    return cmb_priorityqueue_history(pqp);
}

uint64_t cpy_timeseries_count(const void *tsp)
{
    return cmb_timeseries_count(tsp);
}

double cpy_timeseries_min(const void *tsp)
{
    return cmb_timeseries_min(tsp);
}

double cpy_timeseries_max(const void *tsp)
{
    return cmb_timeseries_max(tsp);
}

static void summarize_history(const void *tsp, struct cmb_wtdsummary *wsp)
{
    cmb_wtdsummary_initialize(wsp);
    cmb_timeseries_summarize(tsp, wsp);
}

double cpy_timeseries_mean(const void *tsp)
{
    struct cmb_wtdsummary ws;
    summarize_history(tsp, &ws);
    return cmb_wtdsummary_mean(&ws);
}

double cpy_timeseries_stddev(const void *tsp)
{
    struct cmb_wtdsummary ws;
    summarize_history(tsp, &ws);
    return cmb_wtdsummary_stddev(&ws);
}

double cpy_timeseries_median(const void *tsp)
{
    return cmb_timeseries_median(tsp);
}

typedef void (*file_writer_func)(FILE *fp, void *ctx);

static uint64_t write_file(const intptr_t path,
                           const uint64_t append,
                           file_writer_func writer,
                           void *ctx)
{
    if (path == 0) {
        writer(stdout, ctx);
        return fflush(stdout) == 0 ? 1u : 0u;
    }
    const char *name = (const char *)path;
    FILE *fp = fopen(name, append ? "a" : "w");
    if (fp == NULL) {
        return 0u;
    }
    writer(fp, ctx);
    return fclose(fp) == 0 ? 1u : 0u;
}

struct dataset_histogram_ctx {
    const void *dsp;
    uint64_t num_bins;
    double low_lim;
    double high_lim;
};

struct correlogram_ctx {
    const void *dsp;
    uint64_t n;
};

static void dataset_print_writer(FILE *fp, void *ctx)
{
    cmb_dataset_print(ctx, fp);
}

static void dataset_fivenum_writer(FILE *fp, void *ctx)
{
    cmb_dataset_fivenum_print(ctx, fp, true);
}

static void dataset_histogram_writer(FILE *fp, void *ctx)
{
    const struct dataset_histogram_ctx *hp = ctx;
    cmb_dataset_histogram_print(hp->dsp, fp, (unsigned)hp->num_bins,
                                hp->low_lim, hp->high_lim);
}

static void dataset_correlogram_writer(FILE *fp, void *ctx)
{
    const struct correlogram_ctx *cp = ctx;
    cmb_dataset_correlogram_print(cp->dsp, fp, (unsigned)cp->n, NULL);
}

static void dataset_pacf_correlogram_writer(FILE *fp, void *ctx)
{
    const struct correlogram_ctx *cp = ctx;
    const uint64_t n = cp->n;
    double *pacf = calloc(n + 1u, sizeof *pacf);
    if (pacf == NULL) {
        return;
    }
    cmb_dataset_PACF(cp->dsp, (unsigned)n, pacf, NULL);
    cmb_dataset_correlogram_print(cp->dsp, fp, (unsigned)n, pacf);
    free(pacf);
}

uint64_t cpy_dataset_print_file(const void *dsp, const intptr_t path,
                                const uint64_t append)
{
    return write_file(path, append, dataset_print_writer, (void *)dsp);
}

uint64_t cpy_dataset_fivenum_file(const void *dsp, const intptr_t path,
                                  const uint64_t append)
{
    return write_file(path, append, dataset_fivenum_writer, (void *)dsp);
}

uint64_t cpy_dataset_histogram_file(const void *dsp, const intptr_t path,
                                    const uint64_t append,
                                    const uint64_t num_bins,
                                    const double low_lim,
                                    const double high_lim)
{
    struct dataset_histogram_ctx ctx = {
        .dsp = dsp,
        .num_bins = num_bins,
        .low_lim = low_lim,
        .high_lim = high_lim,
    };
    return write_file(path, append, dataset_histogram_writer, &ctx);
}

uint64_t cpy_dataset_correlogram_file(const void *dsp, const intptr_t path,
                                      const uint64_t append,
                                      const uint64_t n)
{
    struct correlogram_ctx ctx = { .dsp = dsp, .n = n };
    return write_file(path, append, dataset_correlogram_writer, &ctx);
}

uint64_t cpy_dataset_pacf_correlogram_file(const void *dsp,
                                           const intptr_t path,
                                           const uint64_t append,
                                           const uint64_t n)
{
    struct correlogram_ctx ctx = { .dsp = dsp, .n = n };
    return write_file(path, append, dataset_pacf_correlogram_writer, &ctx);
}

struct timeseries_histogram_ctx {
    const void *tsp;
    uint64_t num_bins;
    double low_lim;
    double high_lim;
};

static void timeseries_print_writer(FILE *fp, void *ctx)
{
    cmb_timeseries_print(ctx, fp);
}

static void timeseries_fivenum_writer(FILE *fp, void *ctx)
{
    cmb_timeseries_fivenum_print(ctx, fp, true);
}

static void timeseries_histogram_writer(FILE *fp, void *ctx)
{
    const struct timeseries_histogram_ctx *hp = ctx;
    cmb_timeseries_histogram_print(hp->tsp, fp, (uint16_t)hp->num_bins,
                                   hp->low_lim, hp->high_lim);
}

static void timeseries_correlogram_writer(FILE *fp, void *ctx)
{
    const struct correlogram_ctx *cp = ctx;
    cmb_timeseries_correlogram_print(cp->dsp, fp, (uint16_t)cp->n, NULL);
}

static void timeseries_pacf_correlogram_writer(FILE *fp, void *ctx)
{
    const struct correlogram_ctx *cp = ctx;
    const uint64_t n = cp->n;
    double *pacf = calloc(n + 1u, sizeof *pacf);
    if (pacf == NULL) {
        return;
    }
    cmb_timeseries_PACF(cp->dsp, (uint16_t)n, pacf, NULL);
    cmb_timeseries_correlogram_print(cp->dsp, fp, (uint16_t)n, pacf);
    free(pacf);
}

uint64_t cpy_timeseries_print_file(const void *tsp, const intptr_t path,
                                   const uint64_t append)
{
    return write_file(path, append, timeseries_print_writer, (void *)tsp);
}

uint64_t cpy_timeseries_fivenum_file(const void *tsp, const intptr_t path,
                                     const uint64_t append)
{
    return write_file(path, append, timeseries_fivenum_writer, (void *)tsp);
}

uint64_t cpy_timeseries_histogram_file(const void *tsp, const intptr_t path,
                                       const uint64_t append,
                                       const uint64_t num_bins,
                                       const double low_lim,
                                       const double high_lim)
{
    struct timeseries_histogram_ctx ctx = {
        .tsp = tsp,
        .num_bins = num_bins,
        .low_lim = low_lim,
        .high_lim = high_lim,
    };
    return write_file(path, append, timeseries_histogram_writer, &ctx);
}

uint64_t cpy_timeseries_correlogram_file(const void *tsp,
                                         const intptr_t path,
                                         const uint64_t append,
                                         const uint64_t n)
{
    struct correlogram_ctx ctx = { .dsp = tsp, .n = n };
    return write_file(path, append, timeseries_correlogram_writer, &ctx);
}

uint64_t cpy_timeseries_pacf_correlogram_file(const void *tsp,
                                              const intptr_t path,
                                              const uint64_t append,
                                              const uint64_t n)
{
    struct correlogram_ctx ctx = { .dsp = tsp, .n = n };
    return write_file(path, append, timeseries_pacf_correlogram_writer, &ctx);
}

static void buffer_report_writer(FILE *fp, void *ctx)
{
    cmb_buffer_print_report(ctx, fp);
}

static void resource_report_writer(FILE *fp, void *ctx)
{
    cmb_resource_print_report(ctx, fp);
}

static void resourcepool_report_writer(FILE *fp, void *ctx)
{
    cmb_resourcepool_print_report(ctx, fp);
}

static void objectqueue_report_writer(FILE *fp, void *ctx)
{
    cmb_objectqueue_report_print(ctx, fp);
}

static void priorityqueue_report_writer(FILE *fp, void *ctx)
{
    cmb_priorityqueue_report_print(ctx, fp);
}

uint64_t cpy_buffer_report_file(void *bp, const intptr_t path,
                                const uint64_t append)
{
    return write_file(path, append, buffer_report_writer, bp);
}

uint64_t cpy_resource_report_file(void *rp, const intptr_t path,
                                  const uint64_t append)
{
    return write_file(path, append, resource_report_writer, rp);
}

uint64_t cpy_resourcepool_report_file(void *rpp, const intptr_t path,
                                      const uint64_t append)
{
    return write_file(path, append, resourcepool_report_writer, rpp);
}

uint64_t cpy_objectqueue_report_file(void *oqp, const intptr_t path,
                                     const uint64_t append)
{
    return write_file(path, append, objectqueue_report_writer, oqp);
}

uint64_t cpy_priorityqueue_report_file(void *pqp, const intptr_t path,
                                       const uint64_t append)
{
    return write_file(path, append, priorityqueue_report_writer, pqp);
}

/* Value-passing object queue access: objects are opaque intptr_t values */
int64_t cpy_objectqueue_put(void *oqp, const intptr_t object)
{
    return cmb_objectqueue_put(oqp, (void *)object);
}

/* Blocking get; the received object value is stored to *objloc */
int64_t cpy_objectqueue_get(void *oqp, intptr_t *objloc)
{
    void *obj = NULL;
    const int64_t r = cmb_objectqueue_get(oqp, &obj);
    *objloc = (intptr_t)obj;
    return r;
}

/* Inline accessors from the headers */
uint64_t cpy_resource_in_use(const void *rp)
{
    return cmb_resource_in_use(rp);
}

uint64_t cpy_resource_held_by_process(const void *rp, const void *pp)
{
    return cmb_resource_held_by_process(rp, pp);
}

uint64_t cpy_resourcepool_available(const void *rpp)
{
    return cmb_resourcepool_available((void *)rpp);
}

uint64_t cpy_resourcepool_in_use(const void *rpp)
{
    /* Header inline takes a non-const pointer */
    return cmb_resourcepool_in_use((void *)rpp);
}

uint64_t cpy_objectqueue_length(const void *oqp)
{
    return cmb_objectqueue_length((void *)oqp);
}

uint64_t cpy_buffer_level(const void *bp)
{
    return cmb_buffer_level((void *)bp);
}

uint64_t cpy_buffer_space(const void *bp)
{
    return cmb_buffer_space((void *)bp);
}

uint64_t cpy_objectqueue_space(const void *oqp)
{
    return cmb_objectqueue_space((void *)oqp);
}

uint64_t cpy_resource_available(const void *rp)
{
    return cmb_resource_available(rp);
}

/* Tally statistics over a dataset */
double cpy_dataset_mean(const void *dsp)
{
    struct cmb_datasummary ds;
    cmb_datasummary_initialize(&ds);
    cmb_dataset_summarize(dsp, &ds);
    return cmb_datasummary_mean(&ds);
}

uint64_t cpy_dataset_count(const void *dsp)
{
    return cmb_dataset_count(dsp);
}

double cpy_dataset_min(const void *dsp)
{
    return cmb_dataset_min(dsp);
}

double cpy_dataset_max(const void *dsp)
{
    return cmb_dataset_max(dsp);
}

double cpy_dataset_stddev(const void *dsp)
{
    if (cmb_dataset_count(dsp) < 2u) {
        return 0.0;
    }
    struct cmb_datasummary ds;
    cmb_datasummary_initialize(&ds);
    cmb_dataset_summarize(dsp, &ds);
    return cmb_datasummary_stddev(&ds);
}

/* Blocking take from an object queue, returning the object value
   directly. A take interrupted by a signal returns 0 (no object). */
intptr_t cpy_objectqueue_take(void *oqp)
{
    void *obj = NULL;
    (void)cmb_objectqueue_get(oqp, &obj);
    return (intptr_t)obj;
}

/* Priority queues: objects are opaque intptr_t values. Put returns the
   entry handle used for position queries and cancellation. */
uint64_t cpy_priorityqueue_put(void *pqp, const intptr_t object,
                               const int64_t priority)
{
    uint64_t hndl = 0u;
    (void)cmb_priorityqueue_put(pqp, (void *)object, priority, &hndl);
    return hndl;
}

/* Blocking get; the received object value is stored to *objloc */
int64_t cpy_priorityqueue_get(void *pqp, intptr_t *objloc)
{
    void *obj = NULL;
    const int64_t r = cmb_priorityqueue_get(pqp, &obj);
    *objloc = (intptr_t)obj;
    return r;
}

/* Blocking get returning the object value directly (0 if interrupted) */
intptr_t cpy_priorityqueue_take(void *pqp)
{
    void *obj = NULL;
    (void)cmb_priorityqueue_get(pqp, &obj);
    return (intptr_t)obj;
}

uint64_t cpy_priorityqueue_length(const void *pqp)
{
    return cmb_priorityqueue_length((void *)pqp);
}

uint64_t cpy_priorityqueue_space(const void *pqp)
{
    return cmb_priorityqueue_space((void *)pqp);
}

void cpy_priorityqueue_reprioritize(void *pqp, const uint64_t hndl,
                                    const int64_t priority)
{
    cmb_priorityqueue_reprioritize(pqp, hndl, priority);
}

uint64_t cpy_priorityqueue_cancel(void *pqp, const uint64_t hndl)
{
    return (uint64_t)cmb_priorityqueue_cancel(pqp, hndl);
}

/* Replace all pending timers of the process with this one */
uint64_t cpy_process_timer_set(void *pp, const double dur, const int64_t sig)
{
    return cmb_process_timer_set(pp, dur, sig);
}

uint64_t cpy_process_timer_cancel(void *pp, const uint64_t hndl)
{
    return (uint64_t)cmb_process_timer_cancel(pp, hndl);
}

/* Process status as an integer: 0 created, 1 running, 2 finished */
int64_t cpy_process_status(const void *pp)
{
    return (int64_t)cmb_process_status(pp);
}

uint64_t cpy_process_sizeof(void)
{
    return sizeof(struct cmb_process);
}

/* Like cmb_process_create, but with room for derived-struct fields after
 * the cmb_process header (zeroed). cmb_process_destroy frees the block. */
intptr_t cpy_process_create_sized(const uint64_t nbytes)
{
    cmb_assert_release(nbytes >= sizeof(struct cmb_process));

    void *pp = calloc(1u, nbytes);
    cmb_assert_release(pp != NULL);

    return (intptr_t)pp;
}

/*
 * Registry of live sim.spawn()ed processes, one per trial thread.
 * spawn registers, despawn unregisters; at the end of the trial the
 * generated code stops the leftovers (like the static processes) and
 * reclaims their memory.
 */
static CMB_THREAD_LOCAL struct {
    struct cmb_process **items;
    uint64_t len;
    uint64_t cap;
} cpy_spawned = { NULL, 0u, 0u };

void cpy_spawned_register(void *pp)
{
    cmb_assert_release(pp != NULL);

    if (cpy_spawned.len == cpy_spawned.cap) {
        const uint64_t cap = (cpy_spawned.cap == 0u) ? 16u
                                                     : 2u * cpy_spawned.cap;
        cpy_spawned.items = realloc(cpy_spawned.items,
                                    cap * sizeof(*cpy_spawned.items));
        cmb_assert_release(cpy_spawned.items != NULL);
        cpy_spawned.cap = cap;
    }
    cpy_spawned.items[cpy_spawned.len++] = pp;
}

uint64_t cpy_spawned_unregister(void *pp)
{
    for (uint64_t i = 0u; i < cpy_spawned.len; i++) {
        if (cpy_spawned.items[i] == pp) {
            cpy_spawned.items[i] = cpy_spawned.items[--cpy_spawned.len];
            return 1u;
        }
    }
    return 0u;
}

void cpy_spawned_stop_all(void)
{
    for (uint64_t i = 0u; i < cpy_spawned.len; i++) {
        struct cmb_process *pp = cpy_spawned.items[i];
        if (cmb_process_status(pp) == CMB_PROCESS_RUNNING) {
            cmb_process_stop(pp, NULL);
        }
    }
}

void cpy_spawned_reclaim(void)
{
    for (uint64_t i = 0u; i < cpy_spawned.len; i++) {
        cmb_process_terminate(cpy_spawned.items[i]);
        cmb_process_destroy(cpy_spawned.items[i]);
    }
    free(cpy_spawned.items);
    cpy_spawned.items = NULL;
    cpy_spawned.len = 0u;
    cpy_spawned.cap = 0u;
}

void *cpy_process_current(void)
{
    return cmb_process_current();
}

/* Reschedule the caller at the current time, letting same-time events run */
int64_t cpy_process_yield(void)
{
    return cmb_process_yield();
}

uint32_t cpy_cpu_cores(void)
{
    extern uint32_t cmi_cpu_cores(void);
    return cmi_cpu_cores();
}

static uint32_t logger_flags_on_mask = 0u;
static uint32_t logger_flags_off_mask = 0u;

void cpy_logger_flags_on(const uint32_t flags)
{
    cmb_logger_flags_on(flags);
    logger_flags_on_mask |= flags;
    logger_flags_off_mask &= ~flags;
}

void cpy_logger_flags_off(const uint32_t flags)
{
    cmb_logger_flags_off(flags);
    logger_flags_off_mask |= flags;
    logger_flags_on_mask &= ~flags;
}

void cpy_logger_apply_flags(void)
{
    if (logger_flags_on_mask != 0u) {
        cmb_logger_flags_on(logger_flags_on_mask);
    }
    if (logger_flags_off_mask != 0u) {
        cmb_logger_flags_off(logger_flags_off_mask);
    }
}

void cpy_logger_user_msg(const uint32_t flags, const intptr_t message)
{
    cmi_logger_user(stdout, flags, "python", 0, "%s", (const char *)message);
}

void cpy_logger_user_i64(const uint32_t flags, const intptr_t label,
                         const int64_t value)
{
    cmi_logger_user(stdout, flags, "python", 0, "%s %" PRIi64,
                    (const char *)label, value);
}

void cpy_logger_user_f64(const uint32_t flags, const intptr_t label,
                         const double value)
{
    cmi_logger_user(stdout, flags, "python", 0, "%s %f",
                    (const char *)label, value);
}
