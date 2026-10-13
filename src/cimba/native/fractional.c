/* FractionalContainer: cmb_buffer's get/put with a double level. */
#include <math.h>
#include <stdint.h>

#include "cimba.h"
#include "cmi_memutils.h"
#include "fractional.h"

struct cpy_fractional *cpy_fractional_create(void)
{
    struct cpy_fractional *fp = cmi_malloc(sizeof *fp);
    cmi_memset(fp, 0, sizeof *fp);
    fp->base.cookie = CMI_UNINITIALIZED;

    /* Registered so an abandoned trial tears it down like Cimba's own */
    cmi_dlist_initialize(&(fp->destroy.node));
    fp->destroy.teardown = (cmi_teardown_func *)cpy_fractional_destroy;
    fp->destroy.object = fp;
    cmi_memregistry_add(&(fp->destroy));
    return fp;
}

void cpy_fractional_initialize(struct cpy_fractional *fp, const char *name)
{
    struct cmi_resourcebase *rbp = &(fp->base);
    cmi_resourcebase_initialize(rbp, name);
    /* Redirect the base class teardown to ours, as cmb_buffer does */
    rbp->terminate.teardown = (cmi_teardown_func *)cpy_fractional_terminate;
    cmb_resourceguard_initialize(&(fp->front_guard), rbp);
    fp->level = 0.0;
    fp->is_recording = false;
    cmb_timeseries_initialize(&(fp->history));
}

void cpy_fractional_terminate(struct cpy_fractional *fp)
{
    if (fp->base.cookie == CMI_INITIALIZED) {
        fp->is_recording = false;
        cmb_timeseries_terminate(&(fp->history));
        cmb_resourceguard_terminate(&(fp->front_guard));
        cmi_resourcebase_terminate(&(fp->base));
    }
}

void cpy_fractional_destroy(struct cpy_fractional *fp)
{
    if (!cmi_memregistry_is_demolishing)
        cmi_memregistry_remove(&(fp->destroy));
    cmi_free(fp);
}

static bool fractional_has_content(const struct cmi_resourcebase *rbp,
                                   const struct cmb_process *pp,
                                   const void *ctx)
{
    (void)pp;
    (void)ctx;
    return ((const struct cpy_fractional *)rbp)->level > 0.0;
}

static void record_sample(struct cpy_fractional *fp)
{
    if (fp->is_recording)
        cmb_timeseries_add(&(fp->history), fp->level, cmb_time());
}

void cpy_fractional_recording_start(struct cpy_fractional *fp)
{
    fp->is_recording = true;
    record_sample(fp);
}

void cpy_fractional_recording_stop(struct cpy_fractional *fp)
{
    record_sample(fp);
    fp->is_recording = false;
}

/* Integers cannot be negative, NaN or infinite; doubles can. An infinite
 * amount would make the level infinite and its statistics meaningless. */
static void check_amount(const double amount)
{
    if (!isfinite(amount) || amount < 0.0) cimba_trial_abandon();
}

/* Same defaults as Python's math.isclose(), plus an absolute floor so values
 * near zero still compare sanely. A level that is "enough" only within this
 * tolerance is treated as enough: the gap is smaller than the rounding error
 * ordinary float64 arithmetic already made getting there, on either a tiny
 * level (0.3 - 0.1 - 0.1 - 0.1 is 1e-17 short of 0.1) or a huge one (summing
 * to 1e18 can be off by ~1e2). Blocking on a gap that size would be waiting
 * for a difference the model never actually created. */
#define CPY_FRACTIONAL_REL_TOL 1e-9
#define CPY_FRACTIONAL_ABS_TOL 1e-9

static bool close_enough(const double have, const double want)
{
    const double scale = fmax(fabs(have), fabs(want));
    const double tol = fmax(CPY_FRACTIONAL_ABS_TOL, CPY_FRACTIONAL_REL_TOL * scale);
    return fabs(have - want) <= tol;
}

int64_t cpy_fractional_put(void *vp, const double amount)
{
    struct cpy_fractional *fp = vp;
    check_amount(amount);
    if (amount == 0.0) return CMB_PROCESS_SUCCESS;
    fp->level += amount;
    record_sample(fp);
    cmb_resourceguard_signal(&(fp->front_guard));
    return CMB_PROCESS_SUCCESS;
}

/*
 * cpy_fractional_get - Request and, if necessary, wait for an amount of the
 * level.
 *
 * On entry, *amntp is the amount requested. On return, *amntp is the amount
 * actually obtained: equal to the request on success, but it may be less
 * when the level was only "enough" within tolerance (see close_enough), or
 * when the wait was interrupted before the rest arrived. Check *amntp
 * against the amount asked for if the model cares about exact conservation.
 *
 * The return value CMB_PROCESS_SUCCESS (0) indicates that all went well. If
 * the call was interrupted, the return value is the interrupt signal
 * received, some value other than CMB_PROCESS_SUCCESS.
 */
int64_t cpy_fractional_get(void *vp, double *amntp)
{
    struct cpy_fractional *fp = vp;
    const double amount = *amntp;
    check_amount(amount);
    double remaining = amount;
    double obtained = 0.0;
    while (true) {
        if (fp->level >= remaining || close_enough(fp->level, remaining)) {
            /* Grab what we need. If the level is only "enough" within
             * tolerance, it may be a hair under remaining: take all of it
             * rather than wait for a gap that is rounding error, not a
             * shortfall the model created. */
            const double grab = fmin(fp->level, remaining);
            fp->level -= grab;
            obtained += grab;
            record_sample(fp);
            if (fp->level > 0.0) {
                /* In case someone else can use any leftovers */
                cmb_resourceguard_signal(&(fp->front_guard));
            }
            *amntp = obtained;
            return CMB_PROCESS_SUCCESS;
        }
        if (fp->level > 0.0) {
            /* Grab what is there. remaining stays positive: for doubles,
             * a - b == 0 only when a == b. */
            obtained += fp->level;
            remaining -= fp->level;
            fp->level = 0.0;
            record_sample(fp);
        }
        /* An interrupted wait keeps what it already took, as cmb_buffer does */
        const int64_t sig = cmb_resourceguard_wait(&(fp->front_guard),
                                                   fractional_has_content, NULL);
        if (sig != CMB_PROCESS_SUCCESS) {
            *amntp = obtained;
            return sig;
        }
    }
}

double cpy_fractional_level(const void *vp)
{
    return ((const struct cpy_fractional *)vp)->level;
}

void *cpy_fractional_history(void *vp)
{
    return &(((struct cpy_fractional *)vp)->history);
}
