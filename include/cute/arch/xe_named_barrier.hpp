/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <sycl/sycl.hpp>
#include <cassert>

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)

//
// Use OpenCL named barrier initialization to trigger compiler
// allocate named barrier.
//
extern SYCL_EXTERNAL void named_barrier_init(int id);

#endif

namespace cute {

// Initialize named barrier resources.
// N is the number of named barriers to allocate (barrier IDs 1 to N).
template <int N>
CUTE_HOST_DEVICE void named_barrier_init() {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
  if constexpr (N - 1 > 0)
    named_barrier_init<N - 1>();

  ::named_barrier_init(N);
#endif
}

// TODO(Jiexin): Intel Xe named barrier hardware supports separate producer/consumer roles
// via nbarrier.signal with type field (type=1 for producer, type=2 for consumer) and
// independent producer_count / consumer_count. This enables fine-grained async
// producer-consumer synchronization that has no direct equivalent in NVIDIA's bar.sync.
// We should design a new SYCL-TLA specific API (not reusing NVIDIA CUTLASS interfaces)
// to expose this capability.

// Intel Xe subgroup (hardware thread) size.
inline constexpr uint32_t XeSubgroupSize = 16;

// Named barrier arrive (baseline form).
// All participating threads are Producer_Consumer (type=0).
// num_producers = num_consumers = num_threads.
// num_threads is work-item count and must be a multiple of XeSubgroupSize.
CUTE_HOST_DEVICE void named_barrier_arrive(uint8_t id, uint32_t num_threads) {
#ifdef __SYCL_DEVICE_ONLY__
  assert(num_threads % XeSubgroupSize == 0 &&
         "Intel Xe named barrier requires num_threads to be a multiple of XeSubgroupSize (16).");
  uint8_t num_subgroups = static_cast<uint8_t>(num_threads / XeSubgroupSize);
  asm volatile(
      "nbarrier.signal %0(0,0)<0;1,0> %1(0,0)<0;1,0>\n"
      :: "rw"(id), "rw"(num_subgroups));
#endif
}

// Named barrier wait.
// Blocks until all producers and consumers have signaled barrier <id>.
CUTE_HOST_DEVICE void named_barrier_wait(uint8_t id) {
#ifdef __SYCL_DEVICE_ONLY__
  asm volatile(
      "nbarrier.wait %0(0,0)<0;1,0>\n"
      :: "rw"(id));
#endif
}

// Named barrier arrive_and_wait (signal + wait combined).
// Baseline form: type=0, num_producers = num_consumers = num_threads.
// num_threads is work-item count and must be a multiple of XeSubgroupSize.
CUTE_HOST_DEVICE void named_barrier_arrive_and_wait(uint8_t id, uint32_t num_threads) {
#ifdef __SYCL_DEVICE_ONLY__
  named_barrier_arrive(id, num_threads);
  named_barrier_wait(id);
#endif
}

} // end namespace cute
