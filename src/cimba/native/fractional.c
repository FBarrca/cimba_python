/* FractionalContainer: cmb_buffer's get/put with a fixed-point level. */
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cimba.h"
#include "cmi_memutils.h"
#include "fractional.h"

/* The failure is reported by the trial, which runs on the same thread */
static _Thread_local char failure[200];
static _Thread_local bool has_failure;

const char *cpy_fractional_failure(void)
{
    const bool had = has_failure;
    has_failure = false;
    return had ? failure : NULL;
}

/* Writes the shortest "%e" that reads back as amount, which is the number the
 * model wrote (0.3, not 0.29999999999999998890), and returns its significant
 * digits. Python's repr() picks the same digits. */
static int shortest(const double amount, char *text, const size_t size)
{
    for (int precision = 1; ; ++precision) {
        snprintf(text, size, "%.*e", precision - 1, amount);
        if (precision == 17 || strtod(text, NULL) == amount) return precision;
    }
}

/* Fails the trial with "<name>: <amount> <reason>", reason formatted with n */
static void fail(const struct cpy_fractional *fp, const double amount,
                 const char *reason, const unsigned n)
{
    char text[40];
    const int digits = shortest(amount, text, sizeof text);
    const int exponent = atoi(strchr(text, 'e') + 1);
    /* At least exponent + 1 digits, so %g prints 10 rather than 1e+01 */
    const int precision = digits > exponent + 1 ? digits : exponent + 1;
    const int length = snprintf(failure, sizeof failure, "%s: %.*g ", fp->base.name,
                                precision, amount);
    snprintf(failure + length, sizeof failure - (size_t)length, reason, n);
    has_failure = true;
    cimba_trial_abandon();
}

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

/* precision packs decimals | digits << 8 | strict << 16, and steps is the
 * initial level in steps; both are validated on the host. */
void cpy_fractional_initialize(struct cpy_fractional *fp, const char *name,
                               const uint64_t precision, const uint64_t steps)
{
    struct cmi_resourcebase *rbp = &(fp->base);
    cmi_resourcebase_initialize(rbp, name);
    /* Redirect the base class teardown to ours, as cmb_buffer does */
    rbp->terminate.teardown = (cmi_teardown_func *)cpy_fractional_terminate;
    cmb_resourceguard_initialize(&(fp->front_guard), rbp);
    fp->decimals = (uint8_t)(precision & 0xff);
    fp->digits = (uint8_t)((precision >> 8) & 0xff);
    fp->strict = ((precision >> 16) & 1) != 0;
    fp->max_steps = 1;
    for (uint8_t i = 0; i < fp->digits; ++i) fp->max_steps *= 10;
    fp->max_steps -= 1;
    fp->scale = pow(10.0, fp->decimals);
    fp->steps = (int64_t)steps;
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

/* steps * 10**-decimals, rounded once. Past 2**53, (double)steps would
 * already round before the division, so let strtod round the exact decimal. */
static double to_double(const struct cpy_fractional *fp, const int64_t steps)
{
    if (steps < ((int64_t)1 << 53)) return (double)steps / fp->scale;
    char text[48];
    snprintf(text, sizeof text, "%" PRId64 "e-%u", steps, (unsigned)fp->decimals);
    return strtod(text, NULL);
}

static bool fractional_has_content(const struct cmi_resourcebase *rbp,
                                   const struct cmb_process *pp,
                                   const void *ctx)
{
    (void)pp;
    (void)ctx;
    return ((const struct cpy_fractional *)rbp)->steps > 0;
}

double cpy_fractional_level(const void *vp)
{
    const struct cpy_fractional *fp = vp;
    return to_double(fp, fp->steps);
}

static void record_sample(struct cpy_fractional *fp)
{
    if (fp->is_recording)
        cmb_timeseries_add(&(fp->history), cpy_fractional_level(fp), cmb_time());
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

void *cpy_fractional_history(void *vp)
{
    return &(((struct cpy_fractional *)vp)->history);
}

/*
 * Rounds the shortest decimal of amount half away from zero to a step, so
 * 2.675 rounds to 2.68 at 2 decimals, where rounding its binary value
 * 2.67499999... would give 2.67.
 */
enum { EXACT, ROUNDED, TOO_LARGE };

static int to_steps(const struct cpy_fractional *fp, const double amount, int64_t *stepsp)
{
    *stepsp = 0;
    if (amount == 0.0) return EXACT;
    char text[40], digits[20];
    shortest(amount, text, sizeof text);
    int count = 0;
    const char *c = text;
    for (; *c != 'e'; ++c)
        if (*c != '.') digits[count++] = *c;
    /* amount is 0.dddd * 10**(exponent + 1), so in steps it is 0.dddd *
     * 10**point: the first point digits are whole steps, the rest round. */
    const int point = atoi(c + 1) + 1 + fp->decimals;
    if (point > 18) return TOO_LARGE;    /* more digits than an int64 holds */
    int64_t steps = 0;
    for (int i = 0; i < point; ++i)
        steps = steps * 10 + (i < count ? digits[i] - '0' : 0);
    if (point >= 0 && point < count && digits[point] >= '5') ++steps;
    if (steps > fp->max_steps) return TOO_LARGE;
    *stepsp = steps;
    /* The shortest decimal ends in a nonzero digit, so any digit past the
     * last step means the amount was rounded. */
    return point < count ? ROUNDED : EXACT;
}

/* amount in steps. Fails the trial on a negative, NaN, infinite or too large
 * amount, and, if strict, on one that needs rounding. */
static int64_t steps_of(const struct cpy_fractional *fp, const double amount,
                        const bool strict)
{
    if (!isfinite(amount) || amount < 0.0) cimba_trial_abandon();
    int64_t steps = 0;
    const int status = to_steps(fp, amount, &steps);
    if (status == TOO_LARGE)
        fail(fp, amount, "exceeds the container's %u digit(s)", fp->digits);
    if (status == ROUNDED && strict)
        fail(fp, amount, "has too many decimals; only %u decimal(s) permitted",
             fp->decimals);
    return steps;
}

double cpy_fractional_round(const void *vp, const double amount)
{
    const struct cpy_fractional *fp = vp;
    return to_double(fp, steps_of(fp, amount, false));
}

int64_t cpy_fractional_put(void *vp, const double amount)
{
    struct cpy_fractional *fp = vp;
    const int64_t steps = steps_of(fp, amount, fp->strict);
    if (steps == 0) return CMB_PROCESS_SUCCESS;
    if (steps > fp->max_steps - fp->steps)
        fail(fp, amount, "overflows the level past the container's %u digit(s)",
             fp->digits);
    fp->steps += steps;
    record_sample(fp);
    cmb_resourceguard_signal(&(fp->front_guard));
    return CMB_PROCESS_SUCCESS;
}

/*
 * cpy_fractional_get - Request and, if necessary, wait for an amount of the
 * level.
 *
 * On entry, *amntp is the amount requested. On return, *amntp is the amount
 * actually obtained: the rounded request on success, or less when the wait
 * was interrupted before the rest arrived.
 *
 * The return value CMB_PROCESS_SUCCESS (0) indicates that all went well. If
 * the call was interrupted, the return value is the interrupt signal
 * received, some value other than CMB_PROCESS_SUCCESS.
 */
int64_t cpy_fractional_get(void *vp, double *amntp)
{
    struct cpy_fractional *fp = vp;
    int64_t remaining = steps_of(fp, *amntp, fp->strict);
    int64_t obtained = 0;
    while (true) {
        if (fp->steps >= remaining) {
            /* Grab what we need */
            fp->steps -= remaining;
            obtained += remaining;
            record_sample(fp);
            if (fp->steps > 0) {
                /* In case someone else can use any leftovers */
                cmb_resourceguard_signal(&(fp->front_guard));
            }
            *amntp = to_double(fp, obtained);
            return CMB_PROCESS_SUCCESS;
        }
        if (fp->steps > 0) {
            /* Grab what is there */
            obtained += fp->steps;
            remaining -= fp->steps;
            fp->steps = 0;
            record_sample(fp);
        }
        /* An interrupted wait keeps what it already took, as cmb_buffer does */
        const int64_t sig = cmb_resourceguard_wait(&(fp->front_guard),
                                                   fractional_has_content, NULL);
        if (sig != CMB_PROCESS_SUCCESS) {
            *amntp = to_double(fp, obtained);
            return sig;
        }
    }
}
