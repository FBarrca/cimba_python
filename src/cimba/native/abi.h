#ifndef CIMBA_PY_ABI_H
#define CIMBA_PY_ABI_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) || defined(__CYGWIN__)
#define CPY_EXPORT __declspec(dllexport)
#else
#define CPY_EXPORT __attribute__((visibility("default")))
#endif

enum { CPY_ABI_VERSION = 1 };
enum { CPY_INPUT_DISTRIBUTION = 1, CPY_INPUT_ROWS = 2 };
enum { CPY_POLICY_FAIL = 1, CPY_POLICY_WRAP = 2,
       CPY_POLICY_END_TRIAL = 3, CPY_POLICY_EXTEND = 4 };
enum { CPY_INPUT_OK = 0, CPY_INPUT_FAILED = 1,
       CPY_INPUT_ENDED = 2, CPY_INPUT_EXHAUSTED = 3 };
enum { CPY_DIST_EXPONENTIAL = 1, CPY_DIST_NORMAL = 2,
       CPY_DIST_GAMMA = 3, CPY_DIST_LOGNORMAL = 4,
       CPY_DIST_WEIBULL = 5, CPY_DIST_POISSON = 6,
       CPY_DIST_TRIANGULAR = 7, CPY_DIST_PERT = 8,
       CPY_DIST_CATEGORICAL = 9, CPY_DIST_UNIFORM = 10,
       CPY_DIST_LOGISTIC = 11, CPY_DIST_CAUCHY = 12,
       CPY_DIST_ERLANG = 13, CPY_DIST_BETA = 14,
       CPY_DIST_PERT_MOD = 15, CPY_DIST_RAYLEIGH = 16,
       CPY_DIST_BERNOULLI = 17, CPY_DIST_GEOMETRIC = 18,
       CPY_DIST_BINOMIAL = 19, CPY_DIST_NEGATIVE_BINOMIAL = 20,
       CPY_DIST_PARETO = 21, CPY_DIST_CHI_SQUARED = 22,
       CPY_DIST_F = 23, CPY_DIST_STUDENT_T = 24,
       CPY_DIST_DICE = 25, CPY_DIST_HYPEREXPONENTIAL = 26,
       CPY_DIST_HYPOEXPONENTIAL = 27 };

typedef struct cpy_input_slot {
    uint32_t kind;
    uint32_t policy;
    uint32_t distribution;
    uint32_t status;
    double parameters[4];
    uint64_t stream_state[4];
    const double *data;
    uint64_t length;
    uint64_t cursor;
    double step;
    double origin;
    int64_t last_bucket;
    double last_value;
    double *history;
    uint64_t history_capacity;
} cpy_input_slot;

typedef struct cpy_trial_header {
    uint64_t seed;
    uint64_t trial_index;
    double warmup;
    double duration;
    double cooldown;
    uint32_t status;
    char error[256];
    void *capture;
    const void *descriptor;
} cpy_trial_header;

typedef struct cpy_entity_descriptor {
    uint64_t record_offset;
    uint64_t field_offset;
    uint32_t kind;
    uint32_t reserved;
    uint64_t capacity;
    char name[32];
} cpy_entity_descriptor;

typedef struct cpy_capture_series {
    uint64_t count;
    double *time;
    double *value;
} cpy_capture_series;

typedef struct cpy_process_descriptor {
    uint64_t record_offset;
    void *callback;
    int64_t priority;
    uint64_t copies;
    char name[32];
} cpy_process_descriptor;

typedef struct cpy_hook_descriptor {
    uint64_t record_offset;
    void *callback;
} cpy_hook_descriptor;

typedef struct cpy_input_descriptor {
    uint64_t record_offset;
    uint64_t field_offset;
    char name[96];
} cpy_input_descriptor;

typedef struct cpy_trial_descriptor {
    const cpy_entity_descriptor *entities;
    uint64_t entity_count;
    const cpy_process_descriptor *processes;
    uint64_t process_count;
    const cpy_hook_descriptor *starts;
    uint64_t start_count;
    const cpy_hook_descriptor *ends;
    uint64_t end_count;
    const cpy_input_descriptor *inputs;
    uint64_t input_count;
} cpy_trial_descriptor;

enum { CPY_TRIAL_PENDING = 0, CPY_TRIAL_RUNNING = 1,
       CPY_TRIAL_OK = 2, CPY_TRIAL_FAILED = 3,
       CPY_TRIAL_EXHAUSTED = 4, CPY_TRIAL_ENDED = 5 };
enum { CPY_ENTITY_CONTAINER = 1, CPY_ENTITY_RESOURCE = 2,
       CPY_ENTITY_DATASET = 3, CPY_ENTITY_STORE = 4,
       CPY_ENTITY_PRIORITY_STORE = 5, CPY_ENTITY_CONDITION = 6 };

CPY_EXPORT uint32_t cpy_abi_version(void);
CPY_EXPORT size_t cpy_input_slot_sizeof(void);
CPY_EXPORT size_t cpy_trial_header_sizeof(void);
CPY_EXPORT void cpy_input_seed(cpy_input_slot *slot, uint64_t seed);
CPY_EXPORT double cpy_input_next(cpy_input_slot *slot);
CPY_EXPORT double cpy_series_at(cpy_input_slot *slot, double time);
CPY_EXPORT double cpy_series_now(cpy_input_slot *slot);
CPY_EXPORT void cpy_input_release(cpy_input_slot *slot);
CPY_EXPORT void cpy_capture_release(cpy_capture_series *capture,
                                    uint64_t entity_count);
CPY_EXPORT uint64_t cpy_run(void *blocks, uint64_t count,
                             size_t block_size, uint32_t workers);
CPY_EXPORT void *cpy_model_allocate(size_t size, const void *class_descriptor);
CPY_EXPORT void cpy_model_input_bind(cpy_input_slot *slot);
CPY_EXPORT void cpy_model_start(void *record,
                                const cpy_process_descriptor *processes,
                                uint64_t process_count,
                                const cpy_hook_descriptor *starts,
                                uint64_t start_count,
                                const cpy_hook_descriptor *ends,
                                uint64_t end_count,
                                const cpy_input_descriptor *inputs,
                                uint64_t input_count);
CPY_EXPORT void cpy_model_release(void *record);
CPY_EXPORT void cpy_end_trial(void);

#endif
