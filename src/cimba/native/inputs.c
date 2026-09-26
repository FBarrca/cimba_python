/* Bounds-checked input consumption and independent native random streams.
 * The Cimba engine remains untouched. These streams belong to this binding;
 * model-internal cimba.random calls retain the engine's separate trial stream.
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "cimba.h"
#include "abi.h"

uint32_t cpy_abi_version(void) { return CPY_ABI_VERSION; }
size_t cpy_input_slot_sizeof(void) { return sizeof(cpy_input_slot); }

static uint64_t splitmix64(uint64_t *state)
{
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static uint64_t next64(cpy_input_slot *slot)
{
    uint64_t *s = slot->stream_state;
    const uint64_t value = s[0] + s[1] + s[3]++;
    s[0] = s[1] ^ (s[1] >> 11);
    s[1] = s[2] + (s[2] << 3);
    s[2] = ((s[2] << 24) | (s[2] >> 40)) + value;
    return value;
}

static double uniform(cpy_input_slot *slot)
{
    return (double)(next64(slot) >> 11) * 0x1.0p-53;
}

static double open_uniform(cpy_input_slot *slot)
{
    return ((double)(next64(slot) >> 12) + 0.5) * 0x1.0p-52;
}

static double standard_normal(cpy_input_slot *slot)
{
    const double u = 1.0 - uniform(slot);
    const double v = uniform(slot);
    return sqrt(-2.0 * log(u)) * cos(6.2831853071795864769 * v);
}

static double gamma_draw(cpy_input_slot *slot, double shape)
{
    if (shape < 1.0)
        return gamma_draw(slot, shape + 1.0) * pow(1.0 - uniform(slot), 1.0 / shape);
    const double d = shape - 1.0 / 3.0;
    const double c = 1.0 / sqrt(9.0 * d);
    for (;;) {
        const double z = standard_normal(slot);
        const double base = 1.0 + c * z;
        if (base <= 0.0) continue;
        const double v = base * base * base;
        const double u = uniform(slot);
        if (u < 1.0 - 0.0331 * z * z * z * z ||
            log(u) < 0.5 * z * z + d * (1.0 - v + log(v)))
            return d * v;
    }
}

static double poisson_draw(cpy_input_slot *slot, double lambda)
{
    if (lambda < 30.0) {
        const double threshold = exp(-lambda);
        double product = 1.0;
        uint64_t count = 0;
        do {
            count++;
            product *= 1.0 - uniform(slot);
        } while (product > threshold);
        return (double)(count - 1);
    }
    const double b = 0.931 + 2.53 * sqrt(lambda);
    const double a = -0.059 + 0.02483 * b;
    const double inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
    const double vr = 0.9277 - 3.6224 / (b - 2.0);
    for (;;) {
        const double u = uniform(slot) - 0.5;
        const double v = uniform(slot);
        const double us = 0.5 - fabs(u);
        const double k = floor((2.0 * a / us + b) * u + lambda + 0.43);
        if (us >= 0.07 && v <= vr) return k;
        if (k < 0.0 || (us < 0.013 && v > us)) continue;
        if (log(v) + log(inv_alpha) - log(a / (us * us) + b)
            <= -lambda + k * log(lambda) - lgamma(k + 1.0)) return k;
    }
}

void cpy_input_seed(cpy_input_slot *slot, uint64_t seed)
{
    free(slot->history);
    slot->history = NULL;
    slot->history_capacity = 0;
    for (int i = 0; i < 4; ++i)
        slot->stream_state[i] = splitmix64(&seed);
    for (int i = 0; i < 20; ++i) (void)next64(slot);
    slot->cursor = 0;
    slot->status = CPY_INPUT_OK;
    slot->last_bucket = -1;
}

static double distribution_next(cpy_input_slot *slot)
{
    const double *p = slot->parameters;
    double value = NAN;
    switch (slot->distribution) {
    case CPY_DIST_EXPONENTIAL: value = -p[0] * log(1.0 - uniform(slot)); break;
    case CPY_DIST_NORMAL: value = p[0] + p[1] * standard_normal(slot); break;
    case CPY_DIST_GAMMA: value = p[1] * gamma_draw(slot, p[0]); break;
    case CPY_DIST_LOGNORMAL: value = exp(p[0] + p[1] * standard_normal(slot)); break;
    case CPY_DIST_WEIBULL:
        value = p[1] * pow(-log(1.0 - uniform(slot)), 1.0 / p[0]); break;
    case CPY_DIST_POISSON: value = poisson_draw(slot, p[0]); break;
    case CPY_DIST_TRIANGULAR:
        {
            const double u = uniform(slot);
            const double breakpoint = (p[1] - p[0]) / (p[2] - p[0]);
            value = u < breakpoint ?
                p[0] + sqrt(u * (p[2] - p[0]) * (p[1] - p[0])) :
                p[2] - sqrt((1.0 - u) * (p[2] - p[0]) * (p[2] - p[1]));
        }
        break;
    case CPY_DIST_PERT:
        {
            const double width = p[2] - p[0];
            const double alpha = 1.0 + 4.0 * (p[1] - p[0]) / width;
            const double beta = 1.0 + 4.0 * (p[2] - p[1]) / width;
            const double x = gamma_draw(slot, alpha);
            const double y = gamma_draw(slot, beta);
            value = p[0] + width * x / (x + y);
        }
        break;
    case CPY_DIST_CATEGORICAL:
        if (slot->data == NULL || slot->length == 0) {
            slot->status = CPY_INPUT_FAILED;
            break;
        }
        {
            const double u = uniform(slot);
            double cumulative = 0.0;
            uint64_t index = slot->length - 1;
            for (uint64_t i = 0; i < slot->length; ++i) {
                cumulative += slot->data[slot->length + i];
                if (u < cumulative) { index = i; break; }
            }
            value = slot->data[index];
        }
        break;
    case CPY_DIST_UNIFORM:
        value = p[0] + (p[1] - p[0]) * uniform(slot); break;
    case CPY_DIST_LOGISTIC:
        {
            const double u = open_uniform(slot);
            value = p[0] + p[1] * log(u / (1.0 - u));
        }
        break;
    case CPY_DIST_CAUCHY:
        value = p[0] + p[1] * tan(3.14159265358979323846 *
                                    (open_uniform(slot) - 0.5)); break;
    case CPY_DIST_ERLANG:
        value = (p[1] / p[0]) * gamma_draw(slot, p[0]); break;
    case CPY_DIST_BETA:
        {
            const double x = gamma_draw(slot, p[0]);
            const double y = gamma_draw(slot, p[1]);
            value = p[2] + (p[3] - p[2]) * x / (x + y);
        }
        break;
    case CPY_DIST_PERT_MOD:
        {
            const double width = p[2] - p[0];
            const double alpha = 1.0 + p[3] * (p[1] - p[0]) / width;
            const double beta = 1.0 + p[3] * (p[2] - p[1]) / width;
            const double x = gamma_draw(slot, alpha);
            const double y = gamma_draw(slot, beta);
            value = p[0] + width * x / (x + y);
        }
        break;
    case CPY_DIST_RAYLEIGH:
        value = p[0] * sqrt(-2.0 * log(open_uniform(slot))); break;
    case CPY_DIST_BERNOULLI:
        value = uniform(slot) < p[0] ? 1.0 : 0.0; break;
    case CPY_DIST_GEOMETRIC:
        value = floor(log(open_uniform(slot)) / log1p(-p[0])) + 1.0; break;
    case CPY_DIST_BINOMIAL:
        value = 0.0;
        for (uint64_t i = 0; i < (uint64_t)p[0]; ++i)
            value += uniform(slot) < p[1] ? 1.0 : 0.0;
        break;
    case CPY_DIST_NEGATIVE_BINOMIAL:
        value = 0.0;
        for (uint64_t i = 0; i < (uint64_t)p[0]; ++i)
            value += floor(log(open_uniform(slot)) / log1p(-p[1]));
        break;
    case CPY_DIST_PARETO:
        value = p[1] / pow(open_uniform(slot), 1.0 / p[0]); break;
    case CPY_DIST_CHI_SQUARED:
        value = 2.0 * gamma_draw(slot, p[0] / 2.0); break;
    case CPY_DIST_F:
        value = (gamma_draw(slot, p[0] / 2.0) * 2.0 / p[0]) /
                (gamma_draw(slot, p[1] / 2.0) * 2.0 / p[1]); break;
    case CPY_DIST_STUDENT_T:
        value = p[1] + p[2] * standard_normal(slot) /
                sqrt(2.0 * gamma_draw(slot, p[0] / 2.0) / p[0]); break;
    case CPY_DIST_DICE:
        value = p[0] + floor(uniform(slot) * (p[1] - p[0] + 1.0)); break;
    case CPY_DIST_HYPOEXPONENTIAL:
        if (slot->data == NULL || slot->length == 0) {
            slot->status = CPY_INPUT_FAILED; break;
        }
        value = 0.0;
        for (uint64_t i = 0; i < slot->length; ++i)
            value -= slot->data[i] * log(open_uniform(slot));
        break;
    case CPY_DIST_HYPEREXPONENTIAL:
        if (slot->data == NULL || slot->length == 0) {
            slot->status = CPY_INPUT_FAILED; break;
        }
        {
            const double u = uniform(slot);
            double cumulative = 0.0;
            uint64_t index = slot->length - 1;
            for (uint64_t i = 0; i < slot->length; ++i) {
                cumulative += slot->data[slot->length + i];
                if (u < cumulative) { index = i; break; }
            }
            value = -slot->data[index] * log(open_uniform(slot));
        }
        break;
    default: slot->status = CPY_INPUT_FAILED; break;
    }
    return value;
}

double cpy_input_next(cpy_input_slot *slot)
{
    if (slot->status != CPY_INPUT_OK) return NAN;
    if (slot->kind == CPY_INPUT_DISTRIBUTION) {
        slot->cursor++;
        return distribution_next(slot);
    }
    if (slot->kind != CPY_INPUT_ROWS || slot->data == NULL || slot->length == 0) {
        slot->status = CPY_INPUT_FAILED;
        return NAN;
    }
    if (slot->cursor < slot->length)
        return slot->data[slot->cursor++];
    if (slot->policy == CPY_POLICY_WRAP) {
        const double value = slot->data[slot->cursor % slot->length];
        slot->cursor++;
        return value;
    }
    if (slot->policy == CPY_POLICY_END_TRIAL) {
        /* Like cb.end_trial(): stop the event loop and end the caller; the
         * on_end hooks still run and the trial succeeds. */
        cpy_end_trial();
        return NAN;
    }
    slot->status = slot->policy == CPY_POLICY_EXTEND ? CPY_INPUT_EXHAUSTED :
                   CPY_INPUT_FAILED;
    return NAN;
}

double cpy_series_at(cpy_input_slot *slot, double time)
{
    if (slot->status != CPY_INPUT_OK || slot->step <= 0 || !isfinite(time)) {
        slot->status = CPY_INPUT_FAILED;
        return NAN;
    }
    const double position = floor((time - slot->origin) / slot->step);
    if (position < 0 || position > (double)INT64_MAX) {
        slot->status = CPY_INPUT_FAILED;
        return NAN;
    }
    const int64_t bucket = (int64_t)position;
    if (slot->kind == CPY_INPUT_ROWS) {
        if (slot->data == NULL || slot->length == 0) {
            slot->status = CPY_INPUT_FAILED;
            return NAN;
        }
        if (bucket < (int64_t)slot->length) {
            if ((uint64_t)(bucket + 1) > slot->cursor)
                slot->cursor = (uint64_t)bucket + 1;
            return slot->data[bucket];
        }
        if (slot->policy == CPY_POLICY_WRAP && slot->length > 0) {
            if ((uint64_t)(bucket + 1) > slot->cursor)
                slot->cursor = (uint64_t)bucket + 1;
            return slot->data[bucket % (int64_t)slot->length];
        }
        if (slot->policy == CPY_POLICY_END_TRIAL) {
            cpy_end_trial();
            return NAN;
        }
        slot->status = slot->policy == CPY_POLICY_EXTEND ? CPY_INPUT_EXHAUSTED :
                       CPY_INPUT_FAILED;
        return NAN;
    }
    while (slot->cursor <= (uint64_t)bucket) {
        if (slot->cursor == slot->history_capacity) {
            uint64_t capacity = slot->history_capacity ? slot->history_capacity * 2 : 16;
            if (capacity <= slot->history_capacity || capacity > SIZE_MAX / sizeof(double)) {
                slot->status = CPY_INPUT_FAILED;
                return NAN;
            }
            double *expanded = realloc(slot->history, capacity * sizeof(double));
            if (expanded == NULL) {
                slot->status = CPY_INPUT_FAILED;
                return NAN;
            }
            slot->history = expanded;
            slot->history_capacity = capacity;
        }
        const double value = cpy_input_next(slot);
        if (slot->status != CPY_INPUT_OK) return NAN;
        slot->history[slot->cursor - 1] = value;
    }
    slot->last_bucket = bucket;
    slot->last_value = slot->history[bucket];
    return slot->last_value;
}

double cpy_series_now(cpy_input_slot *slot)
{
    return cpy_series_at(slot, cmb_time());
}

void cpy_input_release(cpy_input_slot *slot)
{
    free(slot->history);
    slot->history = NULL;
    slot->history_capacity = 0;
}
