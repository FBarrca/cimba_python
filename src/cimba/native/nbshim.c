/*
 * nbshim.c - exported wrappers for the Layer-2 Numba binding.
 *
 * Numba-compiled model code binds cimba functions as external symbols at
 * JIT link time. That works for everything declared extern, but several
 * hot-path functions are static inline in the public headers and have no
 * linkable symbol. This file wraps the ones the models need, plus a few
 * struct-size/accessor helpers so Numba code never needs C struct layouts.
 */

#include <stdint.h>

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
