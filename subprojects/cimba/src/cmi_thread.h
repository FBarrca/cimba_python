/*
 * cmi_thread.h - small cross-platform thread and mutex abstraction
 *
 * Copyright (c) FBarrca 2026.
 * Licensed under the Apache License, Version 2.0.
 */

#ifndef CIMBA_CMI_THREAD_H
#define CIMBA_CMI_THREAD_H

#include "cmi_config.h"

#if CMI_OS == CMI_WINDOWS
  #include <windows.h>
  typedef SRWLOCK cmi_mutex_t;
  typedef HANDLE cmi_thread_t;
  #define CMI_MUTEX_INITIALIZER SRWLOCK_INIT
#else
  #include <pthread.h>
  typedef pthread_mutex_t cmi_mutex_t;
  typedef pthread_t cmi_thread_t;
  #define CMI_MUTEX_INITIALIZER PTHREAD_MUTEX_INITIALIZER
#endif

typedef void *(cmi_thread_func)(void *);

extern void cmi_mutex_lock(cmi_mutex_t *mutex);
extern void cmi_mutex_unlock(cmi_mutex_t *mutex);
extern int cmi_thread_create(cmi_thread_t *thread,
                             cmi_thread_func *func,
                             void *arg);
extern int cmi_thread_join(cmi_thread_t thread);
[[noreturn]] extern void cmi_thread_exit(void);

#endif /* CIMBA_CMI_THREAD_H */
