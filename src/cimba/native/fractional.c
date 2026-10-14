/* FractionalContainer: a cmb_buffer counting fixed-point quanta. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "cimba.h"
#include "cmi_memutils.h"
#include "fractional.h"

/* The largest level. An unlimited container treats reaching it as an error
 * rather than letting cmb_buffer_put wait for space that never comes. */
#define CPY_FRACTIONAL_UNLIMITED UINT64_MAX

/* Fails the trial with "<name>: <verb>(<amount>) <reason>" */
static void fail(const struct cpy_fractional *fp, const char *verb,
                 const double amount, const char *reason)
{
    char failure[160];
    snprintf(failure, sizeof failure, "%s: %s(%.17g) %s",
             fp->buffer.base.name, verb, amount, reason);
    cpy_trial_fail(failure);
}

struct cpy_fractional *cpy_fractional_create(void)
{
    struct cpy_fractional *fp = cmi_malloc(sizeof *fp);
    cmi_memset(fp, 0, sizeof *fp);
    fp->buffer.base.cookie = CMI_UNINITIALIZED;

    /* Registered so an abandoned trial tears it down like Cimba's own */
    cmi_dlist_initialize(&(fp->destroy.node));
    fp->destroy.teardown = (cmi_teardown_func *)cpy_fractional_destroy;
    fp->destroy.object = fp;
    cmi_memregistry_add(&(fp->destroy));
    return fp;
}

void cpy_fractional_initialize(struct cpy_fractional *fp, const char *name,
                               const uint64_t capacity, const double scale)
{
    cmb_buffer_initialize(&(fp->buffer), name, capacity);
    fp->scale = scale;
}

void cpy_fractional_terminate(struct cpy_fractional *fp)
{
    cmb_buffer_terminate(&(fp->buffer));
}

void cpy_fractional_destroy(struct cpy_fractional *fp)
{
    if (!cmi_memregistry_is_demolishing)
        cmi_memregistry_remove(&(fp->destroy));
    cmi_free(fp);
}

/* Rounds an amount half away from zero to whole quanta. Negative, NaN,
 * infinite or too large amounts fail the trial. */
static uint64_t to_quanta(const struct cpy_fractional *fp, const char *verb,
                          const double amount)
{
    if (!isfinite(amount) || amount < 0.0)
        fail(fp, verb, amount, "is not a finite nonnegative amount");
    const double quanta = round(amount * fp->scale);
    if (quanta >= 0x1p64) fail(fp, verb, amount, "is too large");
    return (uint64_t)quanta;
}

/* Puts an amount, waiting for space if the container is full, and returns
 * the amount added: the amount rounded to the container's decimals, or less
 * if the wait was interrupted. */
double cpy_fractional_put(void *vp, const double amount)
{
    struct cpy_fractional *fp = vp;
    struct cmb_buffer *bp = &(fp->buffer);
    const uint64_t quanta = to_quanta(fp, "put", amount);
    if (quanta == 0u) return 0.0;
    if (quanta > bp->capacity - bp->level) {
        if (bp->capacity == CPY_FRACTIONAL_UNLIMITED)
            fail(fp, "put", amount, "overflows the level");
        /* Hooks and events run outside any process and cannot wait */
        if (cmb_process_current() == NULL)
            fail(fp, "put", amount, "would wait for space outside a process");
    }
    uint64_t remaining = quanta;
    (void)cmb_buffer_put(bp, &remaining);
    return (double)(quanta - remaining) / fp->scale;
}

/* Gets an amount, taking what is there and waiting for the rest, and returns
 * the amount obtained: the amount rounded to the container's decimals, or
 * less if the wait was interrupted. */
double cpy_fractional_get(void *vp, const double amount)
{
    struct cpy_fractional *fp = vp;
    uint64_t obtained = to_quanta(fp, "get", amount);
    if (obtained == 0u) return 0.0;
    if (obtained > fp->buffer.level && cmb_process_current() == NULL)
        fail(fp, "get", amount, "would wait for content outside a process");
    (void)cmb_buffer_get(&(fp->buffer), &obtained);
    return (double)obtained / fp->scale;
}

/* Rounds an amount of either sign half away from zero to the container's
 * decimals, as every put and get rounds it. */
double cpy_fractional_round(const void *vp, const double amount)
{
    const struct cpy_fractional *fp = vp;
    return round(amount * fp->scale) / fp->scale;
}

double cpy_fractional_level(const void *vp)
{
    const struct cpy_fractional *fp = vp;
    return (double)fp->buffer.level / fp->scale;
}

double cpy_fractional_mean_level(void *vp)
{
    struct cpy_fractional *fp = vp;
    return cpy_buffer_mean_level(&(fp->buffer)) / fp->scale;
}

double cpy_fractional_max_level(void *vp)
{
    struct cpy_fractional *fp = vp;
    return cmb_timeseries_max(cmb_buffer_history(&(fp->buffer))) / fp->scale;
}
