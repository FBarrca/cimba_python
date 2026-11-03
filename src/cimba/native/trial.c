/* Fixed, descriptor-driven lifecycle. Model classes provide callbacks only. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cimba.h"
#include "cmi_coroutine.h"
#include "cmb_priorityqueue.h"
#include "cmb_resourceguard.h"
#include "cmi_hashheap.h"
#include "cmi_memregistry.h"
#include "cmi_memutils.h"
#include "cmi_slist.h"
#include "abi.h"
#include "fractional.h"
#include "resource.h"

typedef void *(*cpy_process_callback)(struct cmb_process *, void *);
typedef void (*cpy_hook_callback)(void *);
extern void cpy_logger_apply_flags(void);

typedef struct spawned_process {
    cpy_process_callback callback;
    void *record;
    struct cmb_process **slot;
} spawned_process;

/* The element buffer of one cb.Array field of a spawned model. The record
 * holds its address and length at field_offset; released with the model. */
typedef struct spawned_array {
    struct spawned_array *next;
    uint64_t field_offset;
    void *elements;
} spawned_array;

typedef struct spawned_model {
    struct spawned_model *next;
    void *record;
    spawned_array *arrays;
    struct cmb_process **processes;
    spawned_process *contexts;
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

/*
 * Observing conditions. The engine forwards every signal of a resource guard
 * to the guards registered as its observers (cmb_resourceguard_register), as
 * one cmb_resourceguard_signal each, which looks at the front waiter only. A
 * condition must evaluate every waiter, as cmb_condition_signal does, so each
 * observation registers a relay guard instead: its queue holds one permanent
 * entry whose demand function signals the condition and declines the grant.
 */
typedef struct observer_relay {
    struct cmb_resourceguard guard;
    struct cmb_resourceguard *subjects[2];
    uint64_t subject_count;
} observer_relay;

/* A finished spawned process whose struct is freed once the clock has moved
 * on (see "Retiring spawned processes" below). */
typedef struct retired_process {
    struct cmb_process *process;
    double time;
} retired_process;

typedef struct trial_resources {
    cpy_trial_header *header;
    const cpy_trial_descriptor *descriptor;
    struct cmb_process **processes;
    void **entities;
    uint64_t process_count;
    uint64_t next_input_stream;
    spawned_model *spawned;
    spawned_model **spawned_index;       /* open addressing, keyed by record */
    uint64_t spawned_index_capacity;     /* a power of two, or 0 */
    uint64_t spawned_index_count;
    released_static_model *released_static;
    observer_relay *relays;
    uint64_t relay_count;
    retired_process *retired;            /* FIFO ring, in retirement order */
    uint64_t retired_head;
    uint64_t retired_count;
    uint64_t retired_capacity;
    double signalled_at;                 /* Last interrupt or resume scheduled */
    int queue_initialized;
    int random_initialized;
    int recording_started;
    int recording_stopped;
    int failed;
} trial_resources;

static _Thread_local trial_resources *active_trial;

void cpy_trial_fail(const char *reason)
{
    trial_resources *resources = active_trial;
    if (resources != NULL) {
        snprintf(resources->header->error, sizeof(resources->header->error),
                 "%s", reason);
        resources->failed = 1;
    }
    cimba_trial_abandon();
}

void cpy_reference_fail(const char *expected)
{
    char failure[160];
    snprintf(failure, sizeof failure, "a None reference was used as a %s model", expected);
    cpy_trial_fail(failure);
}

void cpy_array_index_fail(const char *label, int64_t index, int64_t length)
{
    char failure[192];
    if (length < 0)
        snprintf(failure, sizeof failure,
                 "%s[%lld]: the model was released, so its arrays are gone",
                 label, (long long)index);
    else
        snprintf(failure, sizeof failure, "%s[%lld]: index out of range for length %lld",
                 label, (long long)index, (long long)length);
    cpy_trial_fail(failure);
}

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

/*
 * Retiring spawned processes. The engine's process exit and destroy make sure
 * no pending event can still reach the process by sweeping the whole event
 * queue for it, nine times over its lifetime. With one process per entity in
 * flight that queue holds an event per live process, so each exit costs time
 * linear in the live processes and a trial quadratic time (#27). The engine
 * stays unmodified; spawned processes avoid both sweeps where they can be
 * shown to find nothing:
 *
 * - Every wakeup the sweeps look for (interrupt, resume, preempt, and the
 *   process, event, condition and resource grant wakeups) is scheduled at the
 *   time it is scheduled from, so it runs before the clock moves on. A struct
 *   that is only freed once the clock has passed its retirement can't be
 *   reached by any of them, so it is freed without the destroy sweeps. Until
 *   then such a wakeup finds a finished process, which every handler checks.
 * - The exit sweeps cancel pending interrupts, resumes and preempts. Model
 *   code schedules these only through cpy_process_interrupt and
 *   cpy_process_resume, which note the time. A process that finishes holding
 *   nothing, waiting for nothing and awaited by none, when none was
 *   scheduled at this time, has none pending: it ends its coroutine directly.
 *   Any other exit goes through cmb_process_exit as before.
 */

void cpy_process_interrupt(void *pp, const int64_t sig, const int64_t pri)
{
    if (active_trial != NULL) active_trial->signalled_at = cmb_time();
    cmb_process_interrupt(pp, sig, pri);
}

void cpy_process_resume(void *pp, const int64_t sig)
{
    if (active_trial != NULL) active_trial->signalled_at = cmb_time();
    cmb_process_resume(pp, sig);
}

/* cmb_process_destroy without its sweeps: only for a struct no event can reach */
static void free_retired(struct cmb_process *process)
{
    cmi_memregistry_remove(&(process->destroy));
    cmi_free(process);
}

/* Free the retired processes no event can reach any more: those of earlier times */
static void free_retired_before(trial_resources *resources, const double time)
{
    while (resources->retired_count > 0u) {
        retired_process *oldest = &resources->retired[resources->retired_head];
        if (!(oldest->time < time)) break;
        free_retired(oldest->process);
        resources->retired_head = (resources->retired_head + 1u) %
                                  resources->retired_capacity;
        resources->retired_count--;
    }
}

static void keep_retired(trial_resources *resources, struct cmb_process *process)
{
    if (resources->retired_count == resources->retired_capacity) {
        const uint64_t capacity = resources->retired_capacity ?
                                  2u * resources->retired_capacity : 64u;
        retired_process *ring = malloc(capacity * sizeof(*ring));
        if (ring == NULL) cimba_trial_abandon();
        for (uint64_t i = 0u; i < resources->retired_count; ++i)
            ring[i] = resources->retired[(resources->retired_head + i) %
                                         resources->retired_capacity];
        free(resources->retired);
        resources->retired = ring;
        resources->retired_head = 0u;
        resources->retired_capacity = capacity;
    }
    const uint64_t tail = (resources->retired_head + resources->retired_count) %
                          resources->retired_capacity;
    resources->retired[tail] = (retired_process){ process, cmb_time() };
    resources->retired_count++;
}

/* At trial end nothing runs any more; on abandonment the engine's registry
 * frees the structs, so only the ring goes. */
static void release_retired(trial_resources *resources, int normal)
{
    if (normal) free_retired_before(resources, INFINITY);
    free(resources->retired);
    resources->retired = NULL;
    resources->retired_count = 0u;
}

static void retire_spawned_process(void *subject, void *object)
{
    struct cmb_process *process = subject;
    struct cmb_process **slot = object;
    if (*slot == NULL) return;
    /* An event runs on the dispatcher stack, after the process has exited.
     * Model records remain alive for references and end hooks; their finished
     * process stacks can return to the engine's pool immediately. */
    if (cmb_process_status(process) == CMB_PROCESS_RUNNING) return;
    trial_resources *resources = active_trial;
    cmb_process_terminate(process);
    free_retired_before(resources, cmb_time());
    keep_retired(resources, process);
    *slot = NULL;
}

/* Whether the current process can end without cmb_process_exit: when it holds
 * no resources, awaits nothing, has no waiters and no signal is pending at this
 * instant, all cmb_process_exit would do is sweep the event queue for wakeups
 * there are none of. */
static int exits_without_sweep(const struct cmb_process *process)
{
    return cmi_slist_is_empty(&(process->resources)) &&
        cmi_slist_is_empty(&(process->awaits)) &&
        cmi_slist_is_empty(&(process->waiters)) &&
        active_trial->signalled_at < cmb_time();
}

static void *run_spawned_process(struct cmb_process *process, void *argument)
{
    spawned_process *context = argument;
    void *result = context->callback(process, context->record);
    cmb_event_schedule(retire_spawned_process, process, context->slot,
                       cmb_time(), 0);
    if (exits_without_sweep(process)) cmi_coroutine_exit(result);
    return result;
}

/* Spawned models are looked up by record on every release, array, start and
 * is_dynamic call. Released nodes stay registered until the trial ends, so a
 * linear walk of `spawned` costs O(models spawned so far) per call; this index
 * makes the lookup O(1). Nodes are only ever added, never removed. */
static uint64_t spawned_index_slot(const void *record, uint64_t capacity)
{
    return (((uint64_t)(uintptr_t)record >> 4) * UINT64_C(0x9e3779b97f4a7c15)) &
        (capacity - 1u);
}

static void spawned_index_insert(trial_resources *resources, spawned_model *node)
{
    if ((resources->spawned_index_count + 1u) * 2u > resources->spawned_index_capacity) {
        const uint64_t old_capacity = resources->spawned_index_capacity;
        spawned_model **old_index = resources->spawned_index;
        const uint64_t capacity = old_capacity == 0u ? 1024u : old_capacity * 2u;
        spawned_model **index = calloc((size_t)capacity, sizeof(*index));
        if (index == NULL) cimba_trial_abandon();
        for (uint64_t i = 0; i < old_capacity; ++i) {
            spawned_model *moved = old_index[i];
            if (moved == NULL) continue;
            uint64_t slot = spawned_index_slot(moved->record, capacity);
            while (index[slot] != NULL) slot = (slot + 1u) & (capacity - 1u);
            index[slot] = moved;
        }
        free(old_index);
        resources->spawned_index = index;
        resources->spawned_index_capacity = capacity;
    }
    const uint64_t capacity = resources->spawned_index_capacity;
    uint64_t slot = spawned_index_slot(node->record, capacity);
    while (resources->spawned_index[slot] != NULL) slot = (slot + 1u) & (capacity - 1u);
    resources->spawned_index[slot] = node;
    resources->spawned_index_count++;
}

static spawned_model *spawned_index_find(const trial_resources *resources, const void *record)
{
    const uint64_t capacity = resources->spawned_index_capacity;
    if (capacity == 0u) return NULL;
    uint64_t slot = spawned_index_slot(record, capacity);
    for (spawned_model *node = resources->spawned_index[slot]; node != NULL;
         node = resources->spawned_index[slot]) {
        if (node->record == record) return node;
        slot = (slot + 1u) & (capacity - 1u);
    }
    return NULL;
}

static void free_spawned_index(trial_resources *resources)
{
    free(resources->spawned_index);
    resources->spawned_index = NULL;
    resources->spawned_index_capacity = 0u;
    resources->spawned_index_count = 0u;
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
    spawned_index_insert(resources, node);
    return node->record;
}

/* Writes the array's (elements, length) pair at offset of the record */
static void array_slot_set(void *record, uint64_t offset, void *elements, int64_t length)
{
    void **slot = (void **)((char *)record + offset);
    slot[0] = elements;
    ((int64_t *)slot)[1] = length;
}

static void free_spawned_arrays(spawned_model *node, int64_t length)
{
    while (node->arrays != NULL) {
        spawned_array *array = node->arrays;
        node->arrays = array->next;
        array_slot_set(node->record, array->field_offset, NULL, length);
        free(array->elements);
        free(array);
    }
}

void cpy_model_array(void *record, uint64_t offset, int64_t length,
                     uint64_t element_size, const void *initial, const char *label)
{
    trial_resources *resources = active_trial;
    if (resources == NULL) cimba_trial_abandon();
    spawned_model *node = spawned_index_find(resources, record);
    if (node == NULL || element_size == 0u) cimba_trial_abandon();
    char failure[192];
    if (length < 0) {
        snprintf(failure, sizeof failure, "%s: array length %lld is negative",
                 label, (long long)length);
        cpy_trial_fail(failure);
    }
    if (length == 0) {
        array_slot_set(record, offset, NULL, 0);
        return;
    }
    if ((uint64_t)length > SIZE_MAX / element_size) {
        snprintf(failure, sizeof failure, "%s: array length %lld is too large",
                 label, (long long)length);
        cpy_trial_fail(failure);
    }
    spawned_array *array = calloc(1, sizeof(*array));
    if (array == NULL) cimba_trial_abandon();
    array->elements = calloc((size_t)length, element_size);
    if (array->elements == NULL) {
        free(array);
        snprintf(failure, sizeof failure, "%s: cannot allocate %lld elements",
                 label, (long long)length);
        cpy_trial_fail(failure);
    }
    if (initial != NULL)
        memcpy(array->elements, initial, (size_t)length * element_size);
    array->field_offset = offset;
    array->next = node->arrays;
    node->arrays = array;
    array_slot_set(record, offset, array->elements, length);
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
    spawned_model *node = spawned_index_find(resources, record);
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
    node->contexts = calloc(node->process_count ? node->process_count : 1,
                            sizeof(*node->contexts));
    if (node->processes == NULL || node->start_events == NULL || node->contexts == NULL)
        cimba_trial_abandon();
    uint64_t index = 0;
    for (uint64_t i = 0; i < process_count; ++i) {
        const cpy_process_descriptor *item = &processes[i];
        for (uint64_t copy = 0; copy < item->copies; ++copy) {
            struct cmb_process *process = cmb_process_create();
            spawned_process *context = &node->contexts[index];
            context->callback = (cpy_process_callback)item->callback;
            context->record = record;
            context->slot = &node->processes[index];
            cmb_process_initialize(process, item->name,
                                   run_spawned_process, context, item->priority);
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
    return spawned_index_find(resources, record) != NULL;
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
    spawned_model *node = spawned_index_find(resources, record);
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
    /* Nothing of a released model runs again, not even its end hooks, so its
     * arrays go now. The record stays until the trial ends, so references
     * remain valid; their arrays read as length -1 and indexing fails. */
    free_spawned_arrays(node, -1);
    for (uint64_t i = 0; i < node->process_count; ++i) {
        struct cmb_process *process = node->processes[i];
        if (process == NULL) continue;
        if (node->start_events[i] &&
            cmb_event_is_scheduled(node->start_events[i]))
            cmb_event_cancel(node->start_events[i]);
        stop_model_process(process, current);
        cmb_event_schedule(retire_spawned_process, process, &node->processes[i],
                           cmb_time(), 0);
    }
    if (current != NULL)
        for (uint64_t i = 0; i < node->process_count; ++i)
            if (node->processes[i] == current) {
                if (exits_without_sweep(current)) cmi_coroutine_exit(NULL);
                cmb_process_exit(NULL);
            }
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
        free(node->contexts);
        free(node->start_events);
        free_spawned_arrays(node, -1);
        for (uint64_t i = 0; i < node->input_count; ++i)
            cpy_input_release((cpy_input_slot *)((char *)node->record +
                              node->inputs[i].field_offset));
        free(node->record);
        free(node);
        node = next;
    }
    resources->spawned = NULL;
    free_spawned_index(resources);
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
    /* A failure with a reason already wrote it to header->error */
    if (!resources->failed)
        snprintf(header->error, sizeof(header->error), "%s",
                 "native trial aborted");
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

static bool relay_signal(const struct cmi_resourcebase *rbp,
                         const struct cmb_process *process, const void *context)
{
    (void)process;
    (void)context;
    (void)cmb_condition_signal((struct cmb_condition *)rbp);
    return false;
}

/* The guards whose signals report a change to the entity */
static uint64_t observed_guards(uint32_t kind, void *entity,
                                struct cmb_resourceguard **guards)
{
    if (kind == CPY_ENTITY_RESOURCE) {
        guards[0] = &(((struct cpy_resource *)entity)->guard);
        return 1;
    }
    if (kind == CPY_ENTITY_CONTAINER || kind == CPY_ENTITY_FRACTIONAL) {
        /* A FractionalContainer starts with its cmb_buffer */
        guards[0] = &(((struct cmb_buffer *)entity)->front_guard);
        guards[1] = &(((struct cmb_buffer *)entity)->rear_guard);
        return 2;
    }
    if (kind == CPY_ENTITY_STORE) {
        guards[0] = &(((struct cmb_objectqueue *)entity)->front_guard);
        guards[1] = &(((struct cmb_objectqueue *)entity)->rear_guard);
        return 2;
    }
    if (kind == CPY_ENTITY_PRIORITY_STORE) {
        guards[0] = &(((struct cmb_priorityqueue *)entity)->front_guard);
        guards[1] = &(((struct cmb_priorityqueue *)entity)->rear_guard);
        return 2;
    }
    return 0;
}

/* After every entity exists: one subscribed relay per observation */
static void subscribe_observers(trial_resources *resources)
{
    const cpy_trial_descriptor *descriptor = resources->descriptor;
    if (descriptor->observation_count == 0) return;
    resources->relays = calloc(descriptor->observation_count,
                               sizeof(*resources->relays));
    if (resources->relays == NULL) cimba_trial_abandon();
    for (uint64_t i = 0; i < descriptor->observation_count; ++i) {
        const cpy_observation_descriptor *item = &descriptor->observations[i];
        if (item->condition >= descriptor->entity_count ||
            item->subject >= descriptor->entity_count ||
            descriptor->entities[item->condition].kind != CPY_ENTITY_CONDITION)
            cimba_trial_abandon();
        struct cmb_condition *condition = resources->entities[item->condition];
        observer_relay *relay = &resources->relays[i];
        relay->subject_count = observed_guards(
            descriptor->entities[item->subject].kind,
            resources->entities[item->subject], relay->subjects);
        if (relay->subject_count == 0) cimba_trial_abandon();
        cmb_resourceguard_initialize(&(relay->guard), &(condition->base));
        (void)cmi_hashheap_enqueue(&(relay->guard.priority_queue), NULL,
                                   (void *)relay_signal, NULL, NULL, 1u, 0.0, 0);
        for (uint64_t j = 0; j < relay->subject_count; ++j)
            cmb_resourceguard_register(relay->subjects[j], &(relay->guard));
        resources->relay_count = i + 1;
    }
}

/* Before any entity is destroyed, also on abandonment */
static void unsubscribe_observers(trial_resources *resources)
{
    for (uint64_t i = 0; i < resources->relay_count; ++i) {
        observer_relay *relay = &resources->relays[i];
        for (uint64_t j = 0; j < relay->subject_count; ++j)
            (void)cmb_resourceguard_unregister(relay->subjects[j], &(relay->guard));
        cmb_resourceguard_terminate(&(relay->guard));
    }
    free(resources->relays);
    resources->relays = NULL;
    resources->relay_count = 0;
}

static void abandoned_trial(void *argument)
{
    trial_resources *resources = argument;
    active_trial = NULL;
    /* The entities are still alive: the engine tears them down after this */
    unsubscribe_observers(resources);
    inspect_inputs(resources);
    release_inputs(resources);
    free_spawned_nodes(resources, 0);
    release_retired(resources, 0);
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
        if (kind == CPY_ENTITY_CONTAINER || kind == CPY_ENTITY_FRACTIONAL)
            cmb_buffer_recording_start(resources->entities[i]);
        else if (kind == CPY_ENTITY_RESOURCE)
            cpy_resource_recording_start(resources->entities[i]);
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
        if (kind == CPY_ENTITY_CONTAINER || kind == CPY_ENTITY_FRACTIONAL)
            cmb_buffer_recording_stop(resources->entities[i]);
        else if (kind == CPY_ENTITY_RESOURCE)
            cpy_resource_recording_stop(resources->entities[i]);
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
        if (kind == CPY_ENTITY_CONTAINER || kind == CPY_ENTITY_FRACTIONAL)
            history = cmb_buffer_history(resources->entities[i]);
        else if (kind == CPY_ENTITY_RESOURCE)
            history = cpy_resource_usage_history(resources->entities[i]);
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
        if (kind == CPY_ENTITY_FRACTIONAL) {
            /* The history counts quanta; report it in units */
            for (uint64_t j = 0; j < dataset->count; ++j)
                capture[i].value[j] /= descriptor->entities[i].scale;
        }
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
    resources->signalled_at = -INFINITY;
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
        } else if (item->kind == CPY_ENTITY_FRACTIONAL) {
            struct cpy_fractional *container = cpy_fractional_create();
            cpy_fractional_initialize(container, item->name, item->capacity,
                                      item->scale);
            if (item->initial) {
                uint64_t amount = item->initial;
                cmb_buffer_put(&(container->buffer), &amount);
            }
            entity = container;
        } else if (item->kind == CPY_ENTITY_RESOURCE) {
            struct cpy_resource *resource = cpy_resource_create();
            cpy_resource_initialize(resource, item->name, item->capacity);
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
    subscribe_observers(resources);

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
    release_retired(resources, 1);
    free_released_static_models(resources);
    unsubscribe_observers(resources);
    for (uint64_t i = 0; i < descriptor->entity_count; ++i) {
        const uint32_t kind = descriptor->entities[i].kind;
        if (kind == CPY_ENTITY_CONTAINER) {
            cmb_buffer_terminate(resources->entities[i]);
            cmb_buffer_destroy(resources->entities[i]);
        } else if (kind == CPY_ENTITY_FRACTIONAL) {
            cpy_fractional_terminate(resources->entities[i]);
            cpy_fractional_destroy(resources->entities[i]);
        } else if (kind == CPY_ENTITY_RESOURCE) {
            cpy_resource_terminate(resources->entities[i]);
            cpy_resource_destroy(resources->entities[i]);
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
    /* The engine keeps the thread count process-wide, so set it on every run:
     * 0 restores one worker per core after an earlier explicit count. */
    cimba_threads_use(workers);
    return cimba_run(blocks, count, block_size, run_trial);
}
