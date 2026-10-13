/*
 * fractional.h - an unbounded level with double-valued amounts.
 *
 * The algorithm is cmb_buffer's with a double level and no capacity: a get
 * takes what is there and waits at the front guard for the rest, a put adds
 * its amount and never waits.
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
    double level;
    bool is_recording;
    struct cmb_timeseries history;
    struct cmi_memregistry_item destroy;
};

struct cpy_fractional *cpy_fractional_create(void);
void cpy_fractional_initialize(struct cpy_fractional *fp, const char *name);
void cpy_fractional_terminate(struct cpy_fractional *fp);
void cpy_fractional_destroy(struct cpy_fractional *fp);
void cpy_fractional_recording_start(struct cpy_fractional *fp);
void cpy_fractional_recording_stop(struct cpy_fractional *fp);

CIMBA_PY_EXPORT int64_t cpy_fractional_put(void *fp, double amount);
CIMBA_PY_EXPORT int64_t cpy_fractional_get(void *fp, double amount);
CIMBA_PY_EXPORT double cpy_fractional_level(const void *fp);
CIMBA_PY_EXPORT void *cpy_fractional_history(void *fp);

#endif
