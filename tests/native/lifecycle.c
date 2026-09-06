/* Standalone sanitizer regression for the Python binding's native shims. */
#include "cimba.h"
#include "nbshim.h"

struct trial {
    unsigned index;
    unsigned completed;
    struct cmb_dataset *data;
    struct cmb_buffer *buffer;
};

struct extended_process {
    struct cmb_process base;
    unsigned index;
};

static void *process_body(struct cmb_process *process, void *context)
{
    struct trial *trial = context;
    struct extended_process *extended = (struct extended_process *)process;
    cmb_assert_always(cpy_dataset_mean(trial->data) == 2.0);
    cmb_assert_always(cpy_dataset_stddev(trial->data) == 1.0);
    cmb_assert_always(cpy_dataset_quantile(trial->data, 0.25) == 1.5);
    (void)cpy_buffer_mean_level(trial->buffer);
    (void)cpy_timeseries_mean(cpy_buffer_history(trial->buffer));
    (void)cpy_timeseries_stddev(cpy_buffer_history(trial->buffer));
    if (extended->index == 19 && trial->index % 2 == 0) {
        cmb_logger_error(stderr, "Intentional lifecycle recovery test");
    }
    cmb_process_yield();
    return NULL;
}

static void run_trial(void *context)
{
    struct trial *trial = context;
    cmb_event_queue_initialize(0.0);
    cmb_random_initialize(trial->index);
    trial->data = cmb_dataset_create();
    cmb_dataset_initialize(trial->data);
    cmb_dataset_add(trial->data, 1.0);
    cmb_dataset_add(trial->data, 2.0);
    cmb_dataset_add(trial->data, 3.0);
    trial->buffer = cmb_buffer_create();
    cmb_buffer_initialize(trial->buffer, "buffer", 4);
    cmb_buffer_recording_start(trial->buffer);

    for (unsigned i = 0; i < 20; ++i) {
        struct extended_process *process = (void *)cpy_process_create_sized(
            sizeof(struct extended_process));
        process->index = i;
        cmb_process_initialize(&process->base, "spawn", process_body, trial, 0);
        cpy_spawned_register(process);
        cmb_process_start(&process->base);
    }
    cmb_event_queue_execute();
    cpy_spawned_stop_all();
    cpy_spawned_reclaim();
    /* An empty registry must also tolerate repeated cleanup. */
    cpy_spawned_reclaim();
    cmb_buffer_terminate(trial->buffer);
    cmb_buffer_destroy(trial->buffer);
    cmb_dataset_terminate(trial->data);
    cmb_dataset_destroy(trial->data);
    cmb_event_queue_terminate();
    cmb_random_terminate();
    cmb_assert_always(cmi_dlist_is_empty(&cmi_memregistry));
    ++trial->completed;
}

int main(void)
{
    for (unsigned workers = 1; workers <= 4; workers *= 4) {
        cimba_threads_use(workers);
        for (unsigned repeat = 0; repeat < 3; ++repeat) {
            struct trial trials[64] = {0};
            for (unsigned i = 0; i < 64; ++i) {
                trials[i].index = i;
            }
            cmb_assert_always(cimba_run(trials, 64, sizeof trials[0], run_trial) == 32);
            for (unsigned i = 0; i < 64; ++i) {
                cmb_assert_always(trials[i].completed == i % 2);
            }
        }
    }
    return 0;
}
