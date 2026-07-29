/*
 * cimba.c - the top level simulation execution.
 *
 * Encapsulates the details of setting up and executing pthreads worker threads
 * to execute the experiments specified by the user. We first create a number of
 * worker threads equal to the number of logical cores on the machine, then let
 * these pull and execute trials from the experiment array.
 *
 * Copyright (c) Asbjørn M. Bonvik 1994, 1995, 2025-2026.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdatomic.h>
#include <stdint.h>
#include <xmmintrin.h>

#include "cimba.h"

#include "cmi_mempool.h"
#include "cmi_memutils.h"
#include "cmi_thread.h"

/* Only used from here, no header file needed */
extern uint32_t cmi_cpu_cores(void);

/*
 * Global control variables shared by all threads but static to this file
 */
static void *cmg_experiment_arr;
static size_t cmg_trial_struct_sz;
static cimba_trial_func *cmg_trial_func = NULL;
static uint64_t cmg_total_trials;
static cimba_thread_init_func *cmg_thread_init_func = NULL;
static void *cmg_thread_init_usrarg = NULL;
static cimba_thread_exit_func *cmg_thread_exit_func = NULL;
static cmi_mutex_t cmg_experiment_mutex = CMI_MUTEX_INITIALIZER;

static _Atomic uint64_t cmg_next_trial_idx;

/* User-defined context per thread */
CMB_THREAD_LOCAL void *cmi_thread_context = NULL;

/*
 * cimba_version - Return the version string as const char *
 */
const char *cimba_version(void)
{
    return CIMBA_VERSION;
}

/* Set the initialization callback function */
void cimba_set_thread_hooks(cimba_thread_init_func *initfunc,
                            void *usrarg,
                            cimba_thread_exit_func *exitfunc)
{
    cmg_thread_init_func = initfunc;
    cmg_thread_init_usrarg = usrarg;
    cmg_thread_exit_func = exitfunc;
}

/* Return whatever context was created by the cmg_thread_init_func, if any */
void *cimba_thread_context(void)
{
    return cmi_thread_context;
}

/*
 * thread_exit_wrapper - Run an optional embedding cleanup hook.
 */
static void thread_exit_wrapper(void *context)
{
    if (cmg_thread_exit_func != NULL) {
        cmg_thread_exit_func(context);
    }
}

/*
 * worker_thread_func - The function passed to pthread_create. It finds the next
 * available trial from the experiment array, executes it, and repeats. If no
 * more trials are waiting, it exits. An atomic uint64_t is used to track the
 * number of remaining trials.
 */
static void *worker_thread_func(void *arg)
{
    const uint64_t tid = (uint64_t)arg;

    /* Any user-defined initialization needed? */
    if (cmg_thread_init_func != NULL) {
        cmi_thread_context = cmg_thread_init_func(cmg_thread_init_usrarg, tid);
    }

    while (true) {
        const uint64_t idx = atomic_fetch_add_explicit(
            &cmg_next_trial_idx, 1, memory_order_relaxed
        );
        if (idx >= cmg_total_trials) {
            break;
        }

        void *trial = ((char *)cmg_experiment_arr) + (idx * cmg_trial_struct_sz);
        cmi_logger_trial_idx = idx;

        if (cmg_trial_func != NULL) {
            /* Normal usage, a common function, multiple data */
            (*cmg_trial_func)(trial);
        }
        else {
            /* No common function, extracting function to use for this trial */
            cimba_trial_func *trial_func = (cimba_trial_func *)(((char *)trial)
                                                         + cmg_trial_struct_sz);
            (*trial_func)(trial);
        }
    }

    thread_exit_wrapper(cmi_thread_context);
    cmi_mempool_cleanup(NULL);

    return NULL;
}

/*
 * cimba_run_experiment - The main simulation executive function. Initiates the
 * worker threads and waits for them to finish. That's all.
 *
 * The intended use case is to have only one instance of this function running
 * at a time, while the individual trials are multithreaded below it. However,
 * we'll put in a mutex to protect it against hard-to-debug consequences of
 * unintentional misuse.
 */
void cimba_run_experiment(void *your_experiment_array,
                          const uint64_t num_trials,
                          const size_t trial_struct_size,
                          cimba_trial_func *your_trial_func)
{
    cmb_assert_release(your_experiment_array != NULL);
    cmb_assert_release(num_trials > 0u);
    cmb_assert_release(trial_struct_size > 0u);

    /* A mutex to make sure the Cimba globals are protected */
    cmi_mutex_lock(&cmg_experiment_mutex);

    /* Initialize globals for the threads */
    atomic_store_explicit(&cmg_next_trial_idx, 0u, memory_order_relaxed);
    cmg_experiment_arr = your_experiment_array;
    cmg_trial_struct_sz = trial_struct_size;
    cmg_trial_func = your_trial_func;
    cmg_total_trials = num_trials;

    /* Start the worker threads and let them help themselves to the trials */
    const uint32_t ncores = cmi_cpu_cores();
    cmi_thread_t *threads = cmi_calloc(ncores, sizeof(*threads));
    for (uint64_t ui = 0u; ui < ncores; ui++) {
        const int r = cmi_thread_create(
            &threads[ui], worker_thread_func, (void *)(uintptr_t)ui
        );
        cmb_assert_always(r == 0);
    }

    /* ...worker threads are executing your trials in the background here... */

    /* Wait for all worker threads to finish */
    for (uint64_t ui = 0u; ui < ncores; ui++) {
        const int r = cmi_thread_join(threads[ui]);
        cmb_assert_always(r == 0);
    }

    cmi_free(threads);

    /* Only unlock when all is said and done */
    cmi_mutex_unlock(&cmg_experiment_mutex);
}
