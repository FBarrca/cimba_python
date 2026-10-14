/*
 * fractional.h - an unbounded fixed-point decimal level.
 *
 * The algorithm is cmb_buffer's with no capacity: a get takes what is there
 * and waits at the front guard for the rest, a put adds its amount and never
 * waits. The level is an exact int64 count of 10**-decimals steps. Amounts
 * are rounded half away from zero to a step once, on entry, and all level
 * arithmetic after that is exact.
 */

#ifndef CIMBA_PY_FRACTIONAL_H
#define CIMBA_PY_FRACTIONAL_H

#include <stdbool.h>
#include <stdint.h>

#include "cmb_resourceguard.h"
#include "cmb_timeseries.h"
#include "cmi_resourcebase.h"
#include "nbshim.h"

struct cpy_fractional {
    struct cmi_resourcebase base;            /* Name and initialization trap */
    struct cmb_resourceguard front_guard;    /* Waiting room for getters */
    bool strict;                             /* Reject amounts needing rounding */
    uint8_t decimals;
    uint8_t digits;
    int64_t steps;                           /* Level in steps */
    int64_t max_steps;                       /* 10**digits - 1 */
    double scale;                            /* 10**decimals */
    bool is_recording;
    struct cmb_timeseries history;
    struct cmi_memregistry_item destroy;
};

struct cpy_fractional *cpy_fractional_create(void);
void cpy_fractional_initialize(struct cpy_fractional *fp, const char *name,
                               uint64_t precision, uint64_t steps);
void cpy_fractional_terminate(struct cpy_fractional *fp);
void cpy_fractional_destroy(struct cpy_fractional *fp);
void cpy_fractional_recording_start(struct cpy_fractional *fp);
void cpy_fractional_recording_stop(struct cpy_fractional *fp);

/* Why an amount failed this thread's trial, or NULL; reading clears it */
const char *cpy_fractional_failure(void);

CIMBA_PY_EXPORT int64_t cpy_fractional_put(void *fp, double amount);
CIMBA_PY_EXPORT int64_t cpy_fractional_get(void *fp, double *amntp);
CIMBA_PY_EXPORT double cpy_fractional_round(const void *fp, double amount);
CIMBA_PY_EXPORT double cpy_fractional_level(const void *fp);
CIMBA_PY_EXPORT void *cpy_fractional_history(void *fp);

#endif
