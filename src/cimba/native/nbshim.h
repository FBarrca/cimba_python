/*
 * nbshim.h - prototypes for the Numba support shims in nbshim.c.
 */

#ifndef CIMBA_PY_NBSHIM_H
#define CIMBA_PY_NBSHIM_H

#include <stdint.h>

#ifndef CIMBA_PY_EXPORT
#  if defined(_WIN32) || defined(__CYGWIN__)
#    define CIMBA_PY_EXPORT __declspec(dllexport)
#  else
#    define CIMBA_PY_EXPORT __attribute__((visibility("default")))
#  endif
#endif

CIMBA_PY_EXPORT double cpy_random_exponential(double mean);
CIMBA_PY_EXPORT double cpy_random_gamma(double shape, double scale);
CIMBA_PY_EXPORT double cpy_random01(void);
CIMBA_PY_EXPORT double cpy_random_uniform(double min, double max);
CIMBA_PY_EXPORT double cpy_random_normal(double mu, double sigma);
CIMBA_PY_EXPORT double cpy_random_rayleigh(double s);
CIMBA_PY_EXPORT double cpy_random_PERT(double min, double mode, double max);
CIMBA_PY_EXPORT double cpy_random_PERT_mod(double min, double mode,
                                           double max, double lambda);
CIMBA_PY_EXPORT uint64_t cpy_random_bernoulli(double p);
CIMBA_PY_EXPORT uint64_t cpy_random_flip(void);
CIMBA_PY_EXPORT double cpy_random_triangular(double min, double mode,
                                             double max);
CIMBA_PY_EXPORT double cpy_random_weibull(double shape, double scale);
CIMBA_PY_EXPORT double cpy_random_lognormal(double m, double s);
CIMBA_PY_EXPORT double cpy_random_erlang(uint64_t k, double m);
CIMBA_PY_EXPORT double cpy_random_beta(double a, double b,
                                       double min, double max);
CIMBA_PY_EXPORT uint64_t cpy_random_poisson(double r);
CIMBA_PY_EXPORT int64_t cpy_random_dice(int64_t a, int64_t b);
CIMBA_PY_EXPORT double cpy_random_std_normal(void);
CIMBA_PY_EXPORT double cpy_random_std_exponential(void);
CIMBA_PY_EXPORT double cpy_random_std_gamma(double shape);
CIMBA_PY_EXPORT double cpy_random_std_beta(double a, double b);
CIMBA_PY_EXPORT double cpy_random_logistic(double m, double s);
CIMBA_PY_EXPORT double cpy_random_cauchy(double mode, double scale);
CIMBA_PY_EXPORT double cpy_random_pareto(double shape, double mode);
CIMBA_PY_EXPORT double cpy_random_chisquared(double k);
CIMBA_PY_EXPORT double cpy_random_F_dist(double a, double b);
CIMBA_PY_EXPORT double cpy_random_std_t_dist(double v);
CIMBA_PY_EXPORT double cpy_random_t_dist(double m, double s, double v);
CIMBA_PY_EXPORT uint64_t cpy_random_geometric(double p);
CIMBA_PY_EXPORT uint64_t cpy_random_binomial(uint64_t n, double p);
CIMBA_PY_EXPORT uint64_t cpy_random_negative_binomial(uint64_t m, double p);
CIMBA_PY_EXPORT uint64_t cpy_random_pascal(uint64_t m, double p);
CIMBA_PY_EXPORT uint64_t cpy_resourcepool_available(const void *rpp);
CIMBA_PY_EXPORT uint64_t cpy_buffer_space(const void *bp);
CIMBA_PY_EXPORT uint64_t cpy_objectqueue_space(const void *oqp);
CIMBA_PY_EXPORT uint64_t cpy_resource_available(const void *rp);
CIMBA_PY_EXPORT double cpy_dataset_min(const void *dsp);
CIMBA_PY_EXPORT double cpy_dataset_max(const void *dsp);
CIMBA_PY_EXPORT double cpy_dataset_stddev(const void *dsp);
CIMBA_PY_EXPORT int64_t cpy_process_yield(void);
CIMBA_PY_EXPORT uint64_t cpy_wtdsummary_sizeof(void);
CIMBA_PY_EXPORT double cpy_wtdsummary_mean(const void *wsp);
CIMBA_PY_EXPORT int64_t cpy_buffer_put(void *bp, uint64_t amnt);
CIMBA_PY_EXPORT int64_t cpy_buffer_get(void *bp, uint64_t amnt);
CIMBA_PY_EXPORT double cpy_buffer_mean_level(void *bp);
CIMBA_PY_EXPORT double cpy_resource_mean_in_use(void *rp);
CIMBA_PY_EXPORT double cpy_resourcepool_mean_in_use(void *rpp);
CIMBA_PY_EXPORT double cpy_objectqueue_mean_length(void *oqp);
CIMBA_PY_EXPORT int64_t cpy_objectqueue_put(void *oqp, intptr_t object);
CIMBA_PY_EXPORT int64_t cpy_objectqueue_get(void *oqp, intptr_t *objloc);
CIMBA_PY_EXPORT uint64_t cpy_resource_in_use(const void *rp);
CIMBA_PY_EXPORT uint64_t cpy_resourcepool_in_use(const void *rpp);
CIMBA_PY_EXPORT uint64_t cpy_objectqueue_length(const void *oqp);
CIMBA_PY_EXPORT uint64_t cpy_buffer_level(const void *bp);
CIMBA_PY_EXPORT double cpy_dataset_mean(const void *dsp);
CIMBA_PY_EXPORT uint64_t cpy_dataset_count(const void *dsp);

CIMBA_PY_EXPORT intptr_t cpy_objectqueue_take(void *oqp);
CIMBA_PY_EXPORT uint64_t cpy_priorityqueue_put(void *pqp, intptr_t object,
                                               int64_t priority);
CIMBA_PY_EXPORT intptr_t cpy_priorityqueue_take(void *pqp);
CIMBA_PY_EXPORT uint64_t cpy_priorityqueue_length(const void *pqp);
CIMBA_PY_EXPORT uint64_t cpy_priorityqueue_cancel(void *pqp, uint64_t hndl);
CIMBA_PY_EXPORT uint64_t cpy_process_timer_set(void *pp, double dur,
                                               int64_t sig);
CIMBA_PY_EXPORT int64_t cpy_process_status(const void *pp);
CIMBA_PY_EXPORT void *cpy_process_current(void);
CIMBA_PY_EXPORT uint32_t cpy_cpu_cores(void);

#endif /* CIMBA_PY_NBSHIM_H */
