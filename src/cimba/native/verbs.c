/* Typed scalar queue verbs, local to the Python runtime. */
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

#include "cimba.h"
#include "cmb_priorityqueue.h"
#include "abi.h"

static void *word_as_pointer(uint64_t bits)
{
    return (void *)(uintptr_t)bits;
}

static uint64_t pointer_as_word(void *value)
{
    return (uint64_t)(uintptr_t)value;
}

CPY_EXPORT int64_t cpy_store_put_int(void *store, int64_t value)
{
    return cmb_objectqueue_put(store, word_as_pointer((uint64_t)value));
}

CPY_EXPORT int64_t cpy_store_get_int(void *store)
{
    void *value = NULL;
    cmb_objectqueue_get(store, &value);
    return (int64_t)pointer_as_word(value);
}

CPY_EXPORT int64_t cpy_store_put_float(void *store, double value)
{
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return cmb_objectqueue_put(store, word_as_pointer(bits));
}

CPY_EXPORT double cpy_store_get_float(void *store)
{
    void *value = NULL;
    cmb_objectqueue_get(store, &value);
    uint64_t bits = pointer_as_word(value);
    double result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

CPY_EXPORT int64_t cpy_store_put_model(void *store, void *record)
{
    return cmb_objectqueue_put(store, record);
}

CPY_EXPORT void *cpy_store_get_model(void *store)
{
    void *record = NULL;
    cmb_objectqueue_get(store, &record);
    return record;
}

CPY_EXPORT int64_t cpy_priority_put_int(void *store, int64_t value,
                                        int64_t priority)
{
    return cmb_priorityqueue_put(store, word_as_pointer((uint64_t)value),
                                 priority, NULL);
}

CPY_EXPORT int64_t cpy_priority_get_int(void *store)
{
    void *value = NULL;
    cmb_priorityqueue_get(store, &value);
    return (int64_t)pointer_as_word(value);
}

CPY_EXPORT int64_t cpy_priority_put_float(void *store, double value,
                                          int64_t priority)
{
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return cmb_priorityqueue_put(store, word_as_pointer(bits), priority, NULL);
}

CPY_EXPORT double cpy_priority_get_float(void *store)
{
    void *value = NULL;
    cmb_priorityqueue_get(store, &value);
    uint64_t bits = pointer_as_word(value);
    double result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

CPY_EXPORT int64_t cpy_priority_put_model(void *store, void *record,
                                          int64_t priority)
{
    return cmb_priorityqueue_put(store, record, priority, NULL);
}

CPY_EXPORT void *cpy_priority_get_model(void *store)
{
    void *record = NULL;
    cmb_priorityqueue_get(store, &record);
    return record;
}

CPY_EXPORT uint64_t cpy_priority_enqueue_model(void *store, void *record,
                                                int64_t priority)
{
    uint64_t handle = 0;
    if (cmb_priorityqueue_put(store, record, priority, &handle) != 0)
        return 0;
    return handle;
}

CPY_EXPORT uint64_t cpy_priority_cancel(void *store, uint64_t handle)
{
    return cmb_priorityqueue_cancel(store, handle);
}

CPY_EXPORT uint64_t cpy_priority_position(void *store, uint64_t handle)
{
    return cmb_priorityqueue_position(store, handle);
}

CPY_EXPORT uint64_t cpy_priority_length(void *store)
{
    return cmb_priorityqueue_length(store);
}

CPY_EXPORT uint64_t cpy_store_length(void *store)
{
    return cmb_objectqueue_length(store);
}

typedef uint8_t (*cpy_predicate_callback)(void *);
typedef void (*cpy_event_callback)(void *);

typedef struct cpy_bound_predicate {
    cpy_predicate_callback callback;
    void *context;
} cpy_bound_predicate;

static bool predicate_demand(const struct cmb_condition *condition,
                             const struct cmb_process *process,
                             const void *argument)
{
    (void)condition;
    (void)process;
    const cpy_bound_predicate *predicate = argument;
    return predicate->callback(predicate->context) != 0;
}

CPY_EXPORT int64_t cpy_condition_wait(void *condition, void *callback,
                                      void *context)
{
    cpy_bound_predicate predicate = {
        (cpy_predicate_callback)callback, context,
    };
    return cmb_condition_wait(condition, predicate_demand, &predicate);
}

CPY_EXPORT int64_t cpy_condition_wait_until(void *condition, void *callback,
                                           void *context, double timeout)
{
    if (isnan(timeout) || timeout < 0.0) cimba_trial_abandon();
    cpy_bound_predicate predicate = {
        (cpy_predicate_callback)callback, context,
    };
    bool satisfied = predicate.callback(predicate.context) != 0;
    if (satisfied || timeout == 0.0) return satisfied;

    struct cmb_process *process = cmb_process_current();
    if (process == NULL) cimba_trial_abandon();
    const double deadline = cmb_time() + timeout;
    /* Own one timer, leaving any application timers intact. */
    const uint64_t timer = isfinite(timeout)
        ? cmb_process_timer_add(process, timeout, CMB_PROCESS_TIMEOUT) : 0;
    do {
        const int64_t signal = cmb_condition_wait(condition, predicate_demand,
                                                  &predicate);
        satisfied = predicate.callback(predicate.context) != 0;
        if (satisfied || signal != CMB_PROCESS_SUCCESS || cmb_time() >= deadline)
            break;
        /* Another waiter can change the predicate before we resume. */
    } while (true);
    if (timer != 0) cmb_process_timer_cancel(process, timer);
    return satisfied;
}

CPY_EXPORT uint64_t cpy_condition_signal(void *condition)
{
    return cmb_condition_signal(condition);
}

static void scheduled_event(void *context, void *callback)
{
    ((cpy_event_callback)callback)(context);
}

CPY_EXPORT uint64_t cpy_schedule_event(void *callback, void *context,
                                       double delay, int64_t priority)
{
    if (!isfinite(delay) || delay < 0.0) cimba_trial_abandon();
    return cmb_event_schedule(scheduled_event, context, callback,
                              cmb_time() + delay, priority);
}

CPY_EXPORT uint64_t cpy_schedule_cancel(uint64_t handle)
{
    return cmb_event_cancel(handle);
}

CPY_EXPORT uint64_t cpy_schedule_pending(uint64_t handle)
{
    return cmb_event_is_scheduled(handle);
}
