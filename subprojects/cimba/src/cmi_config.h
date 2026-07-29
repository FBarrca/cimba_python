/*
 * cmi_config.h - preprocessor macros to identify architecture, compiler, and
 *                operating system, defining macros for portability.
 *
 * Copyright (c) Asbjørn M. Bonvik 1994, 1995, 2025-26.
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

#ifndef CIMBA_CMI_CONFIG_H
#define CIMBA_CMI_CONFIG_H

/*
 * Identify the processor architecture.
 */
#define CMI_AMD64 1
#define CMI_ARM64 2
#if (defined (__amd64__) || defined (__amd64) || defined (__x86_64__) || \
     defined (__x86_64) || defined (_M_X64) || defined (_M_AMD64))
  #define CMI_ARCH CMI_AMD64
#elif defined (__aarch64__) || defined (__arm64__) || defined (_M_ARM64)
  #define CMI_ARCH CMI_ARM64
#else
  #error "Platform architecture not yet supported."
#endif

/*
 * Identify the operating system.
 */
#define CMI_LINUX 1
#define CMI_WINDOWS 2
#define CMI_MACOS 3
#if defined (__linux__) || defined (__linux) || defined (linux)
  #define CMI_OS CMI_LINUX
#elif defined (_WIN64) || defined (_WIN32) || defined (__WIN32__)
  #define CMI_OS CMI_WINDOWS
#elif defined (__APPLE__) && defined (__MACH__)
  #define CMI_OS CMI_MACOS
#else
  #error "Platform operating system not yet supported."
#endif

/*
 * Identify the compiler. So far, only GCC and Clang are supported.
 * Test for Clang first, in case it defines __GNUC__ for compatibility reasons.
 */
#define CMI_GCC 1
#define CMI_CLANG 2
#define CMI_MSVC 3
#if defined (__clang__)
  #define CMI_COMPILER CMI_CLANG
#elif defined (__GNUC__)
  #define CMI_COMPILER CMI_GCC
#elif defined (_MSC_VER)
  #define CMI_COMPILER CMI_MSVC
#else
  #error "Compiler not yet supported."
#endif

#if CMI_OS == CMI_LINUX && (CMI_COMPILER == CMI_GCC || CMI_COMPILER == CMI_CLANG)
    #define CMB_THREAD_LOCAL _Thread_local __attribute__((tls_model("initial-exec")))
#elif CMI_COMPILER == CMI_MSVC
    #define CMB_THREAD_LOCAL __declspec(thread)
#else
    #define CMB_THREAD_LOCAL _Thread_local
#endif


#endif /* CIMBA_CMI_CONFIG_H */
