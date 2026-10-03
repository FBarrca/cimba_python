/* Fixed, descriptor-driven lifecycle. Model classes provide callbacks only. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cimba.h"
#include "cmb_priorityqueue.h"
#include "abi.h"

typedef void *(*cpy_process_callback)(struct cmb_process *, void *);
typedef void (*cpy_hook_callback)(void *);
extern void cpy_logger_apply_flags(void);

typedef struct spawned_model {
    struct spawned_model *next;
    void *record;
    struct cmb_process **processes;
    uint64_t *start_events;
    uint64_t process_count;
    const cpy_hook_descriptor *ends;
    uint64_t end_count;
    const cpy_input_descriptor *inputs;
    uint64_t input_count;
    int released;
} spawned_model;

typedef struct released_static_model {
    struct released_static_model *next;
    void *record;
} released_static_model;

typedef struct trial_resources {
    cpy_trial_header *header;
    const cpy_trial_descriptor *descriptor;
    struct cmb_process **processes;
    void **entities;
    uint64_t process_count;
    uint64_t next_input_stream;
    spawned_model *spawned;
    released_static_model *released_static;
    int queue_initialized;
    int random_initialized;
    int recording_started;
    int recording_stopped;
} trial_resources;

static _Thread_local trial_resources *active_trial;

void cpy_end_trial(void)
{
    if (active_trial == NULL) cimba_trial_abandon();
    cmb_event_queue_clear();
    if (cmb_process_current() != NULL) cmb_process_exit(NULL);
}

static void deferred_process_start(void *subject, void *object)
{
    (void)object;
    cmb_process_start(subject);
}

void *cpy_model_allocate(size_t size, const void *class_descriptor)
{
    trial_resources *resources = active_trial;
    if (resources == NULL || size < sizeof(void *)) cimba_trial_abandon();
    spawned_model *node = calloc(1, sizeof(*node));
    if (node == NULL) cimba_trial_abandon();
    node->record = calloc(1, size);
    if (node->record == NULL) {
        free(node);
        cimba_trial_abandon();
    }
    *(const void **)node->record = class_descriptor;
    node->next = resources->spawned;
    resources->spawned = node;
    return node->record;
}

void cpy_model_input_bind(cpy_input_slot *slot)
{
    trial_resources *resources = active_trial;
    if (resources == NULL || slot == NULL) cimba_trial_abandon();
    /* The slot was copied from an existing input. Its history belongs to
     * that input; a spawned model starts with independent bookkeeping. */
    slot->history = NULL;
    slot->history_capacity = 0;
    const uint64_t stream = ++resources->next_input_stream;
    const uint64_t seed = resources->header->seed ^
        (stream * UINT64_C(0x9e3779b97f4a7c15)) ^
        UINT64_C(0xd1b54a32d192ed03);
    cpy_input_seed(slot, seed);
}

void cpy_model_start(void *record, const cpy_process_descriptor *processes,
                     uint64_t process_count, const cpy_hook_descriptor *starts,
                     uint64_t start_count, const cpy_hook_descriptor *ends,
                     uint64_t end_count, const cpy_input_descriptor *inputs,
                     uint64_t input_count)
{
    trial_resources *resources = active_trial;
    if (resources == NULL) cimba_trial_abandon();
    spawned_model *node = resources->spawned;
    while (node != NULL && node->record != record) node = node->next;
    if (node == NULL || node->processes != NULL) cimba_trial_abandon();
    node->ends = ends;
    node->end_count = end_count;
    node->inputs = inputs;
    node->input_count = input_count;
    for (uint64_t i = 0; i < start_count; ++i)
        ((cpy_hook_callback)starts[i].callback)(record);
    if (node->released) return;
    for (uint64_t i = 0; i < process_count; ++i)
        node->process_count += processes[i].copies;
    node->processes = calloc(node->process_count ? node->process_count : 1,
                             sizeof(*node->processes));
    node->start_events = calloc(node->process_count ? node->process_count : 1,
                                sizeof(*node->start_events));
    if (node->processes == NULL || node->start_events == NULL)
        cimba_trial_abandon();
    uint64_t index = 0;
    for (uint64_t i = 0; i < process_count; ++i) {
        const cpy_process_descriptor *item = &processes[i];
        for (uint64_t copy = 0; copy < item->copies; ++copy) {
            struct cmb_process *process = cmb_process_create();
            cmb_process_initialize(process, item->name,
                                   (cpy_process_callback)item->callback,
                                   record, item->priority);
            node->processes[index] = process;
            node->start_events[index] = cmb_event_schedule(
                deferred_process_start, process, NULL, cmb_time(),
                item->priority);
            index++;
        }
    }
}

int64_t cpy_model_is_dynamic(void *record)
{
    trial_resources *resources = active_trial;
    if (resources == NULL) cimba_trial_abandon();
    /* Released nodes remain registered until trial cleanup. Creation mode
     * is independent of whether the model's processes are still active. */
    for (spawned_model *node = resources->spawned; node != NULL;
         node = node->next)
        if (node->record == record) return 1;
    return 0;
}

static int static_model_released(const trial_resources *resources, const void *record)
{
    for (released_static_model *node = resources->released_static; node != NULL;
         node = node->next)
        if (node->record == record) return 1;
    return 0;
}

static void free_released_static_models(trial_resources *resources)
{
    while (resources->released_static != NULL) {
        released_static_model *node = resources->released_static;
        resources->released_static = node->next;
        free(node);
    }
}

static void stop_model_process(struct cmb_process *process,
                               const struct cmb_process *current)
{
    if (process == NULL || process == current) return;
    const enum cmb_process_state state = cmb_process_status(process);
    if (state == CMB_PROCESS_INITIALIZED)
        /* Also cancel a start already handed from our deferred event to Cimba. */
        cmb_event_pattern_cancel(CMB_ANY_ACTION, process, CMB_ANY_OBJECT);
    else if (state == CMB_PROCESS_RUNNING)
        cmb_process_stop(process, NULL);
}

void cpy_model_release(void *record)
{
    trial_resources *resources = active_trial;
    if (resources == NULL) cimba_trial_abandon();
    spawned_model *node = resources->spawned;
    while (node != NULL && node->record != record) node = node->next;
    struct cmb_process *current = cmb_process_current();
    if (node == NULL) {
        if (static_model_released(resources, record)) cimba_trial_abandon();
        released_static_model *released = calloc(1, sizeof(*released));
        if (released == NULL) cimba_trial_abandon();
        released->record = record;
        released->next = resources->released_static;
        resources->released_static = released;
        /* Hooks can release a model before its processes are even created. */
        for (uint64_t i = 0; i < resources->process_count; ++i) {
            struct cmb_process *process = resources->processes[i];
            if (cmb_process_context(process) == record)
                stop_model_process(process, current);
        }
        if (current != NULL && cmb_process_context(current) == record)
            cmb_process_exit(NULL);
        return;
    }
    if (node->released) cimba_trial_abandon();
    node->released = 1;
    for (uint64_t i = 0; i < node->process_count; ++i) {
        struct cmb_process *process = node->processes[i];
        if (node->start_events[i] &&
            cmb_event_is_scheduled(node->start_events[i]))
            cmb_event_cancel(node->start_events[i]);
        stop_model_process(process, current);
    }
    if (current != NULL)
        for (uint64_t i = 0; i < node->process_count; ++i)
            if (node->processes[i] == current) cmb_process_exit(NULL);
}

static void free_spawned_nodes(trial_resources *resources, int normal)
{
    spawned_model *node = resources->spawned;
    while (node != NULL) {
        spawned_model *next = node->next;
        if (normal) {
            for (uint64_t i = 0; i < node->process_count; ++i) {
                struct cmb_process *process = node->processes[i];
                if (process == NULL) continue;
                if (node->start_events[i] &&
                    cmb_event_is_scheduled(node->start_events[i]))
                    cmb_event_cancel(node->start_events[i]);
                if (cmb_process_status(process) == CMB_PROCESS_RUNNING)
                    cmb_process_stop(process, NULL);
                cmb_process_terminate(process);
                cmb_process_destroy(process);
            }
        }
        free(node->processes);
        free(node->start_events);
        for (uint64_t i = 0; i < node->input_count; ++i)
            cpy_input_release((cpy_input_slot *)((char *)node->record +
                              node->inputs[i].field_offset));
        free(node->record);
        free(node);
        node = next;
    }
    resources->spawned = NULL;
}

static void inspect_inputs(trial_resources *resources)
{
    cpy_trial_header *header = resources->header;
    const cpy_trial_descriptor *descriptor = resources->descriptor;
    for (uint64_t i = 0; i < descriptor->input_count; ++i) {
        const cpy_input_descriptor *item = &descriptor->inputs[i];
        cpy_input_slot *slot = (cpy_input_slot *)((char *)header +
                             item->record_offset + item->field_offset);
        if (slot->status == CPY_INPUT_OK) continue;
        if (slot->status == CPY_INPUT_EXHAUSTED)
            header->status = CPY_TRIAL_EXHAUSTED;
        else if (slot->status == CPY_INPUT_ENDED)
            header->status = CPY_TRIAL_ENDED;
        else
            header->status = CPY_TRIAL_FAILED;
        if (slot->kind == CPY_INPUT_EMPTY)
            snprintf(header->error, sizeof(header->error),
                     "%s: optional input is unbound", item->name);
        else
            snprintf(header->error, sizeof(header->error),
                     "%s exhausted after %llu values", item->name,
                     (unsigned long long)slot->cursor);
        return;
    }
    for (spawned_model *node = resources->spawned; node != NULL;
         node = node->next) {
        for (uint64_t i = 0; i < node->input_count; ++i) {
            cpy_input_slot *slot = (cpy_input_slot *)((char *)node->record +
                                    node->inputs[i].field_offset);
            if (slot->status == CPY_INPUT_OK) continue;
            header->status = slot->status == CPY_INPUT_EXHAUSTED ?
                CPY_TRIAL_EXHAUSTED : slot->status == CPY_INPUT_ENDED ?
                CPY_TRIAL_ENDED : CPY_TRIAL_FAILED;
            if (slot->kind == CPY_INPUT_EMPTY)
                snprintf(header->error, sizeof(header->error),
                         "%s: optional input is unbound", node->inputs[i].name);
            else
                snprintf(header->error, sizeof(header->error),
                         "%s exhausted after %llu values",
                         node->inputs[i].name,
                         (unsigned long long)slot->cursor);
            return;
        }
    }
    header->status = CPY_TRIAL_FAILED;
    snprintf(header->error, sizeof(header->error), "native trial aborted");
}

static void release_inputs(trial_resources *resources)
{
    const cpy_trial_descriptor *descriptor = resources->descriptor;
    for (uint64_t i = 0; i < descriptor->input_count; ++i) {
        const cpy_input_descriptor *item = &descriptor->inputs[i];
        cpy_input_slot *slot = (cpy_input_slot *)((char *)resources->header +
                             item->record_offset + item->field_offset);
        cpy_input_release(slot);
    }
}

static void abandoned_trial(void *argument)
{
    trial_resources *resources = argument;
    active_trial = NULL;
    inspect_inputs(resources);
    release_inputs(resources);
    free_spawned_nodes(resources, 0);
    free_released_static_models(resources);
    cpy_capture_release(resources->header->capture,
                        resources->descriptor->entity_count);
    resources->header->capture = NULL;
    free(resources->processes);
    free(resources->entities);
    if (resources->random_initialized) cmb_random_terminate();
    /* The engine's recovery path frees the event queue after its registry. */
    free(resources);
}

static void record_start(void *subject, void *object)
{
    (void)object;
    trial_resources *resources = subject;
    resources->recording_started = 1;
    for (uint64_t i = 0; i < resources->descriptor->entity_count; ++i) {
        const uint32_t kind = resources->descriptor->entities[i].kind;
        if (kind == CPY_ENTITY_CONTAINER)
            cmb_buffer_recording_start(resources->entities[i]);
        else if (kind == CPY_ENTITY_RESOURCE)
            cmb_resourcepool_start_recording(resources->entities[i]);
        else if (kind == CPY_ENTITY_DATASET)
            cmb_dataset_reset(resources->entities[i]);
        else if (kind == CPY_ENTITY_STORE)
            cmb_objectqueue_recording_start(resources->entities[i]);
        else if (kind == CPY_ENTITY_PRIORITY_STORE)
            cmb_priorityqueue_recording_start(resources->entities[i]);
    }
}

static void record_stop(void *subject, void *object)
{
    (void)object;
    trial_resources *resources = subject;
    if (!resources->recording_started || resources->recording_stopped) return;
    resources->recording_stopped = 1;
    for (uint64_t i = 0; i < resources->descriptor->entity_count; ++i) {
        const uint32_t kind = resources->descriptor->entities[i].kind;
        if (kind == CPY_ENTITY_CONTAINER)
            cmb_buffer_recording_stop(resources->entities[i]);
        else if (kind == CPY_ENTITY_RESOURCE)
            cmb_resourcepool_stop_recording(resources->entities[i]);
        else if (kind == CPY_ENTITY_STORE)
            cmb_objectqueue_recording_stop(resources->entities[i]);
        else if (kind == CPY_ENTITY_PRIORITY_STORE)
            cmb_priorityqueue_recording_stop(resources->entities[i]);
    }
}

static void stop_processes(void *subject, void *object)
{
    (void)object;
    trial_resources *resources = subject;
    for (uint64_t i = 0; i < resources->process_count; ++i)
        if (cmb_process_status(resources->processes[i]) == CMB_PROCESS_RUNNING)
            cmb_process_stop(resources->processes[i], NULL);
    /* Spawned models end with the window too: cancel starts that are still
     * pending and stop running processes, so a spawned process that never
     * finishes cannot keep the trial alive. */
    for (spawned_model *node = resources->spawned; node != NULL;
         node = node->next) {
        if (node->released) continue;
        for (uint64_t i = 0; i < node->process_count; ++i) {
            if (node->start_events[i] &&
                cmb_event_is_scheduled(node->start_events[i]))
                cmb_event_cancel(node->start_events[i]);
            struct cmb_process *process = node->processes[i];
            if (process != NULL &&
                cmb_process_status(process) == CMB_PROCESS_RUNNING)
                cmb_process_stop(process, NULL);
        }
    }
}

void cpy_capture_release(cpy_capture_series *capture, uint64_t entity_count)
{
    if (capture == NULL) return;
    for (uint64_t i = 0; i < entity_count; ++i) {
        free(capture[i].time);
        free(capture[i].value);
    }
    free(capture);
}

static void copy_captures(trial_resources *resources)
{
    const cpy_trial_descriptor *descriptor = resources->descriptor;
    int any = 0;
    for (uint64_t i = 0; i < descriptor->entity_count; ++i)
        any |= descriptor->entities[i].reserved != 0;
    if (!any) return;
    cpy_capture_series *capture = calloc(descriptor->entity_count,
                                         sizeof(*capture));
    if (capture == NULL) cimba_trial_abandon();
    resources->header->capture = capture;
    for (uint64_t i = 0; i < descriptor->entity_count; ++i) {
        const uint32_t kind = descriptor->entities[i].kind;
        if (!descriptor->entities[i].reserved) continue;
        const struct cmb_dataset *dataset = NULL;
        const struct cmb_timeseries *history = NULL;
        if (kind == CPY_ENTITY_CONTAINER)
            history = cmb_buffer_history(resources->entities[i]);
        else if (kind == CPY_ENTITY_RESOURCE)
            history = cmb_resourcepool_get_history(resources->entities[i]);
        else if (kind == CPY_ENTITY_STORE)
            history = cmb_objectqueue_history(resources->entities[i]);
        else if (kind == CPY_ENTITY_PRIORITY_STORE)
            history = cmb_priorityqueue_history(resources->entities[i]);
        else if (kind == CPY_ENTITY_DATASET)
            dataset = resources->entities[i];
        if (history != NULL) dataset = &history->ds;
        if (dataset == NULL) continue;
        capture[i].count = dataset->count;
        const size_t bytes = dataset->count * sizeof(double);
        if (!bytes) continue;
        capture[i].value = malloc(bytes);
        if (capture[i].value == NULL) cimba_trial_abandon();
        memcpy(capture[i].value, dataset->xa, bytes);
        if (history != NULL) {
            capture[i].time = malloc(bytes);
            if (capture[i].time == NULL) cimba_trial_abandon();
            memcpy(capture[i].time, history->ta, bytes);
        }
    }
}

static void run_trial(void *block)
{
    cpy_logger_apply_flags();
    cpy_trial_header *header = block;
    const cpy_trial_descriptor *descriptor = header->descriptor;
    trial_resources *resources = calloc(1, sizeof(*resources));
    if (resources == NULL) abort();
    resources->header = header;
    resources->descriptor = descriptor;
    active_trial = resources;
    header->status = CPY_TRIAL_RUNNING;
    header->error[0] = '\0';
    cimba_trial_cleanup_set(abandoned_trial, resources);

    cmb_event_queue_initialize(0.0);
    resources->queue_initialized = 1;
    cmb_random_initialize(header->seed);
    resources->random_initialized = 1;

    resources->entities = calloc(descriptor->entity_count ? descriptor->entity_count : 1,
                                 sizeof(*resources->entities));
    if (resources->entities == NULL) cimba_trial_abandon();
    for (uint64_t i = 0; i < descriptor->entity_count; ++i) {
        const cpy_entity_descriptor *item = &descriptor->entities[i];
        void *entity = NULL;
        if (item->kind == CPY_ENTITY_CONTAINER) {
            struct cmb_buffer *container = cmb_buffer_create();
            cmb_buffer_initialize(container, item->name, item->capacity);
            if (item->initial) {
                uint64_t amount = item->initial;
                cmb_buffer_put(container, &amount);
            }
            entity = container;
        } else if (item->kind == CPY_ENTITY_RESOURCE) {
            struct cmb_resourcepool *resource = cmb_resourcepool_create();
            cmb_resourcepool_initialize(resource, item->name, item->capacity);
            entity = resource;
        } else if (item->kind == CPY_ENTITY_DATASET) {
            struct cmb_dataset *dataset = cmb_dataset_create();
            cmb_dataset_initialize(dataset);
            entity = dataset;
        } else if (item->kind == CPY_ENTITY_STORE) {
            struct cmb_objectqueue *store = cmb_objectqueue_create();
            cmb_objectqueue_initialize(store, item->name, item->capacity);
            entity = store;
        } else if (item->kind == CPY_ENTITY_PRIORITY_STORE) {
            struct cmb_priorityqueue *store = cmb_priorityqueue_create();
            cmb_priorityqueue_initialize(store, item->name, item->capacity);
            entity = store;
        } else if (item->kind == CPY_ENTITY_CONDITION) {
            struct cmb_condition *condition = cmb_condition_create();
            cmb_condition_initialize(condition, item->name);
            entity = condition;
        } else {
            cimba_trial_abandon();
        }
        resources->entities[i] = entity;
        *(void **)((char *)block + item->record_offset + item->field_offset) = entity;
    }

    for (uint64_t i = 0; i < descriptor->start_count; ++i) {
        const cpy_hook_descriptor *item = &descriptor->starts[i];
        ((cpy_hook_callback)item->callback)((char *)block + item->record_offset);
    }

    for (uint64_t i = 0; i < descriptor->process_count; ++i)
        resources->process_count += descriptor->processes[i].copies;
    resources->processes = calloc(resources->process_count ? resources->process_count : 1,
                                  sizeof(*resources->processes));
    if (resources->processes == NULL) cimba_trial_abandon();
    uint64_t index = 0;
    for (uint64_t i = 0; i < descriptor->process_count; ++i) {
        const cpy_process_descriptor *item = &descriptor->processes[i];
        for (uint64_t copy = 0; copy < item->copies; ++copy) {
            struct cmb_process *process = cmb_process_create();
            cmb_process_initialize(process, item->name,
                                   (cpy_process_callback)item->callback,
                                   (char *)block + item->record_offset,
                                   item->priority);
            resources->processes[index++] = process;
            if (!static_model_released(resources, (char *)block + item->record_offset))
                cmb_process_start(process);
        }
    }

    /* Without a warmup the window opens now, as it does for until-idle runs:
     * an event at time 0 would run after the process starts queued above
     * and clear the samples they record at time 0. */
    if (!isfinite(header->duration) || header->warmup == 0.0)
        record_start(resources, NULL);
    else
        cmb_event_schedule(record_start, resources, NULL, header->warmup, 0);
    if (isfinite(header->duration)) {
        cmb_event_schedule(record_stop, resources, NULL,
                           header->warmup + header->duration, 0);
        cmb_event_schedule(stop_processes, resources, NULL,
                           header->warmup + header->duration + header->cooldown, 0);
    }
    cmb_event_queue_execute();
    record_stop(resources, NULL);

    for (uint64_t i = 0; i < descriptor->end_count; ++i) {
        const cpy_hook_descriptor *item = &descriptor->ends[i];
        ((cpy_hook_callback)item->callback)((char *)block + item->record_offset);
    }
    for (spawned_model *node = resources->spawned; node != NULL;
         node = node->next)
        if (!node->released)
            for (uint64_t i = 0; i < node->end_count; ++i)
                ((cpy_hook_callback)node->ends[i].callback)(node->record);

    copy_captures(resources);

    stop_processes(resources, NULL);
    for (uint64_t i = 0; i < resources->process_count; ++i) {
        cmb_process_terminate(resources->processes[i]);
        cmb_process_destroy(resources->processes[i]);
    }
    free_spawned_nodes(resources, 1);
    free_released_static_models(resources);
    for (uint64_t i = 0; i < descriptor->entity_count; ++i) {
        const uint32_t kind = descriptor->entities[i].kind;
        if (kind == CPY_ENTITY_CONTAINER) {
            cmb_buffer_terminate(resources->entities[i]);
            cmb_buffer_destroy(resources->entities[i]);
        } else if (kind == CPY_ENTITY_RESOURCE) {
            cmb_resourcepool_terminate(resources->entities[i]);
            cmb_resourcepool_destroy(resources->entities[i]);
        } else if (kind == CPY_ENTITY_DATASET) {
            cmb_dataset_terminate(resources->entities[i]);
            cmb_dataset_destroy(resources->entities[i]);
        } else if (kind == CPY_ENTITY_STORE) {
            cmb_objectqueue_terminate(resources->entities[i]);
            cmb_objectqueue_destroy(resources->entities[i]);
        } else if (kind == CPY_ENTITY_PRIORITY_STORE) {
            cmb_priorityqueue_terminate(resources->entities[i]);
            cmb_priorityqueue_destroy(resources->entities[i]);
        } else if (kind == CPY_ENTITY_CONDITION) {
            cmb_condition_terminate(resources->entities[i]);
            cmb_condition_destroy(resources->entities[i]);
        }
    }
    free(resources->processes);
    free(resources->entities);
    release_inputs(resources);
    cmb_random_terminate();
    cmb_event_queue_terminate();
    cimba_trial_cleanup_set(NULL, NULL);
    active_trial = NULL;
    free(resources);
    header->status = CPY_TRIAL_OK;
}

size_t cpy_trial_header_sizeof(void) { return sizeof(cpy_trial_header); }

uint64_t cpy_run(void *blocks, uint64_t count, size_t block_size, uint32_t workers)
{
    if (workers > 0) cimba_threads_use(workers);
    return cimba_run(blocks, count, block_size, run_trial);
}
