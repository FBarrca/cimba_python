/*
 * fractional.h - a level with fractional amounts, kept in fixed point.
 *
 * The level is a cmb_buffer counting quanta of 1/scale units, scale being
 * 10**decimals. Amounts are rounded half away from zero to whole quanta once,
 * on the way in; after that every put and get is exact integer arithmetic, so
 * the level plus everything taken out always equals the initial level plus
 * everything put in. The get/put algorithms, the capacity and the waiting
 * rooms are cmb_buffer's own.
 */

#ifndef CIMBA_PY_FRACTIONAL_H
#define CIMBA_PY_FRACTIONAL_H

#include <stdint.h>

#include "cmb_buffer.h"
#include "nbshim.h"

struct cpy_fractional {
    struct cmb_buffer buffer;                /* First: the buffer is the object */
    double scale;                            /* Quanta per unit */
    struct cmi_memregistry_item destroy;
};

struct cpy_fractional *cpy_fractional_create(void);
void cpy_fractional_initialize(struct cpy_fractional *fp, const char *name,
                               uint64_t capacity, double scale);
void cpy_fractional_terminate(struct cpy_fractional *fp);
void cpy_fractional_destroy(struct cpy_fractional *fp);

/* Fails this thread's trial with reason as its error; defined in trial.c.
 * The reason lives in the trial, not in thread-local storage: the library's
 * static TLS must fit glibc's small surplus for dlopen'd modules. */
void cpy_trial_fail(const char *reason);

CIMBA_PY_EXPORT double cpy_fractional_put(void *fp, double amount);
CIMBA_PY_EXPORT double cpy_fractional_get(void *fp, double amount);
CIMBA_PY_EXPORT double cpy_fractional_round(const void *fp, double amount);
CIMBA_PY_EXPORT double cpy_fractional_level(const void *fp);
CIMBA_PY_EXPORT double cpy_fractional_mean_level(void *fp);
CIMBA_PY_EXPORT double cpy_fractional_max_level(void *fp);

#endif
