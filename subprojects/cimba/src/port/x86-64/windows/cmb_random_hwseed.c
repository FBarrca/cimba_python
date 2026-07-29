/*
 * cmb_random_hwseed.c - Windows specific hardware seed
 *
 * Copyright (c) Asbjørn M. Bonvik 2025.
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

#define _CRT_RAND_S
#include <stdlib.h>

#include "cmb_assert.h"
#include "cmb_random.h"

/* Windows-specific code to get a suitable 64-bit seed from the OS CSPRNG. */
uint64_t cmb_random_hwseed(void)
{
    unsigned int high = 0u;
    unsigned int low = 0u;
    const errno_t high_result = rand_s(&high);
    const errno_t low_result = rand_s(&low);
    cmb_assert_always(high_result == 0);
    cmb_assert_always(low_result == 0);

    const uint64_t seed = ((uint64_t)high << 32u) | (uint64_t)low;
    cmb_assert_debug(seed != 0u);
    return seed;
}
