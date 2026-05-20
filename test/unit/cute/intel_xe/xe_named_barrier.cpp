/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
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

#include "utils.hpp"
#include <cute/arch/xe_named_barrier.hpp>
#include <cutlass/arch/barrier.h>

using namespace cute;
using namespace cutlass;
using namespace compat::experimental;

constexpr int NUM_THREADS = 32; // 2 subgroups of 16

////////////////////////////////////////////////////////////////////////////////////////////////////
// Test 1: cute-level API - named_barrier_arrive_and_wait
// Tests cute::named_barrier_arrive_and_wait which exercises arrive_and_wait_internal path.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class...> class CuteArriveAndWaitTestName;

void cute_arrive_and_wait_kernel(int* output) {
  auto* shared_data = compat::local_mem<int[NUM_THREADS]>();

  int tid = static_cast<int>(ThreadIdxX());

  cute::named_barrier_init<1>();

  // First half writes to SLM
  if (tid < NUM_THREADS / 2) {
    shared_data[tid] = tid + 100;
  }

  // All threads synchronize via cute-level arrive_and_wait
  cute::named_barrier_arrive_and_wait(0, NUM_THREADS);

  // All threads read (second half reads first half's data, first half reads own)
  if (tid >= NUM_THREADS / 2) {
    output[tid] = shared_data[tid - NUM_THREADS / 2];
  } else {
    output[tid] = shared_data[tid];
  }
}

TEST(Xe_Named_Barrier, cute_arrive_and_wait) {
  cutlass::host_vector<int> host_output(NUM_THREADS, 0);
  cutlass::device_vector<int> device_output(NUM_THREADS, 0);

  launch<cute_arrive_and_wait_kernel, CuteArriveAndWaitTestName<>>(
      launch_policy{
          compat::dim3(1), compat::dim3(NUM_THREADS),
          kernel_properties{sycl_exp::sub_group_size<SUBGROUP_SIZE>}},
      device_output.data());

  compat::wait_and_throw();
  host_output = device_output;

  for (int i = 0; i < NUM_THREADS; ++i) {
    int src = (i >= NUM_THREADS / 2) ? (i - NUM_THREADS / 2) : i;
    EXPECT_EQ(host_output[i], src + 100);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Test 2: cute-level API - named_barrier_arrive + named_barrier_wait (separate arrive/wait)
// Tests cute::named_barrier_arrive which exercises arrive_internal path.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class...> class CuteSignalWaitTestName;

void cute_signal_wait_kernel(int* output) {
  auto* shared_data = compat::local_mem<int[NUM_THREADS]>();

  int tid = static_cast<int>(ThreadIdxX());

  cute::named_barrier_init<1>();

  // First half writes
  if (tid < NUM_THREADS / 2) {
    shared_data[tid] = tid + 300;
  }

  // All threads signal (arrive) then wait separately
  cute::named_barrier_arrive(0, NUM_THREADS);
  cute::named_barrier_wait(0);

  // Second half reads first half's data
  if (tid >= NUM_THREADS / 2) {
    output[tid] = shared_data[tid - NUM_THREADS / 2];
  } else {
    output[tid] = shared_data[tid];
  }
}

TEST(Xe_Named_Barrier, cute_signal_and_wait) {
  cutlass::host_vector<int> host_output(NUM_THREADS, 0);
  cutlass::device_vector<int> device_output(NUM_THREADS, 0);

  launch<cute_signal_wait_kernel, CuteSignalWaitTestName<>>(
      launch_policy{
          compat::dim3(1), compat::dim3(NUM_THREADS),
          kernel_properties{sycl_exp::sub_group_size<SUBGROUP_SIZE>}},
      device_output.data());

  compat::wait_and_throw();
  host_output = device_output;

  for (int i = 0; i < NUM_THREADS; ++i) {
    int src = (i >= NUM_THREADS / 2) ? (i - NUM_THREADS / 2) : i;
    EXPECT_EQ(host_output[i], src + 300);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Test 3: NamedBarrier class - static sync() and arrive_and_wait() API
// Tests cutlass::arch::NamedBarrier::sync() which internally calls arrive_and_wait_internal
// with ReservedNamedBarrierCount offset.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class...> class ClassStaticSyncTestName;

void class_static_sync_kernel(int* output) {
  auto* shared_data = compat::local_mem<int[NUM_THREADS]>();

  int tid = static_cast<int>(ThreadIdxX());

  // NamedBarrier user API uses barrier IDs offset by ReservedNamedBarrierCount (=8)
  // Allocate 1 user barrier (total = ReservedNamedBarrierCount + 1 = 9).
  cutlass::arch::NamedBarrier::init<1>();

  // Instance bound to user barrier 0 (hw id = 8) with NUM_THREADS participants.
  cutlass::arch::NamedBarrier barrier(NUM_THREADS, 0u);

  // All threads write
  shared_data[tid] = tid * 2;

  // Synchronize via the instance member function.
  barrier.sync();

  // Circular shift read
  int read_idx = (tid + NUM_THREADS / 2) % NUM_THREADS;
  output[tid] = shared_data[read_idx];
}

TEST(Xe_Named_Barrier, class_static_sync) {
  cutlass::host_vector<int> host_output(NUM_THREADS, 0);
  cutlass::device_vector<int> device_output(NUM_THREADS, 0);

  launch<class_static_sync_kernel, ClassStaticSyncTestName<>>(
      launch_policy{
          compat::dim3(1), compat::dim3(NUM_THREADS),
          kernel_properties{sycl_exp::sub_group_size<SUBGROUP_SIZE>}},
      device_output.data());

  compat::wait_and_throw();
  host_output = device_output;

  for (int i = 0; i < NUM_THREADS; ++i) {
    int read_idx = (i + NUM_THREADS / 2) % NUM_THREADS;
    EXPECT_EQ(host_output[i], read_idx * 2);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Test 4: NamedBarrier class - instance arrive() + wait (tests arrive_internal path)
// Uses NamedBarrier instance to call arrive() (signal only) followed by a wait.
// Two barriers coordinate a two-phase pipeline.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class...> class ClassInstanceArriveTestName;

void class_instance_arrive_kernel(int* output) {
  auto* shared_A = compat::local_mem<int[NUM_THREADS]>();
  auto* shared_B = compat::local_mem<int[NUM_THREADS]>();

  int tid = static_cast<int>(ThreadIdxX());

  // Need 2 user barriers (total = ReservedNamedBarrierCount + 2 = 10).
  cutlass::arch::NamedBarrier::init<2>();

  // Two barrier instances for two-phase synchronization
  cutlass::arch::NamedBarrier barrier0(NUM_THREADS, 0u);  // hw id = 8
  cutlass::arch::NamedBarrier barrier1(NUM_THREADS, 1u);  // hw id = 9

  // Phase 1: First half writes to shared_A
  if (tid < NUM_THREADS / 2) {
    shared_A[tid] = tid + 10;
  }

  // All threads arrive_and_wait on barrier0
  barrier0.arrive_and_wait();

  // Phase 2: Second half reads shared_A and writes shared_B
  if (tid >= NUM_THREADS / 2) {
    shared_B[tid] = shared_A[tid - NUM_THREADS / 2] + 1000;
  }

  // Second-phase synchronization on barrier1 via the NamedBarrier class API.
  barrier1.arrive_and_wait();

  // Phase 3: First half reads shared_B
  if (tid < NUM_THREADS / 2) {
    output[tid] = shared_B[tid + NUM_THREADS / 2];
  } else {
    output[tid] = shared_B[tid];
  }
}

TEST(Xe_Named_Barrier, class_instance_arrive) {
  cutlass::host_vector<int> host_output(NUM_THREADS, 0);
  cutlass::device_vector<int> device_output(NUM_THREADS, 0);

  launch<class_instance_arrive_kernel, ClassInstanceArriveTestName<>>(
      launch_policy{
          compat::dim3(1), compat::dim3(NUM_THREADS),
          kernel_properties{sycl_exp::sub_group_size<SUBGROUP_SIZE>}},
      device_output.data());

  compat::wait_and_throw();
  host_output = device_output;

  for (int i = 0; i < NUM_THREADS; ++i) {
    int src = (i >= NUM_THREADS / 2) ? (i - NUM_THREADS / 2) : i;
    EXPECT_EQ(host_output[i], src + 10 + 1000);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Test 5: GEMM-like producer-consumer pattern (barrier threads < kernel threads)
//
// Simulates a simplified GEMM mainloop:
// - Total kernel threads = 64 (4 subgroups)
// - Producer threads (subgroups 0-1, tid 0-31): load A and B tiles into SLM
// - Consumer threads (subgroups 2-3, tid 32-63): compute C = A * B (element-wise)
//
// Named barrier 0 synchronizes only the 32 producer threads to ensure both A and B
// tiles are fully loaded before signaling consumers. Named barrier 1 synchronizes all
// 64 threads to ensure tiles are visible to consumers before compute.
//
// This mirrors CUTLASS GEMM where load (producer) and math (consumer) thread groups
// use different barriers with different participant counts.
////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr int GEMM_TOTAL_THREADS = 64;     // 4 subgroups
constexpr int GEMM_PRODUCER_THREADS = 32;  // 2 subgroups (load A and B)
constexpr int TILE_SIZE = 16;              // tile dimension

template <class...> class GemmProducerConsumerTestName;

void gemm_producer_consumer_kernel(int* output) {
  auto* tile_A = compat::local_mem<int[TILE_SIZE]>();
  auto* tile_B = compat::local_mem<int[TILE_SIZE]>();
  auto* tile_C = compat::local_mem<int[TILE_SIZE]>();

  int tid = static_cast<int>(ThreadIdxX());

  // Allocate barriers:
  // user barrier 0 (hw id 8): producer-only sync (32 threads / 16 = 2 subgroups)
  // user barrier 1 (hw id 9): all-thread sync (64 threads / 16 = 4 subgroups)
  cutlass::arch::NamedBarrier::init<2>();

  // Two barrier instances with different participant counts.
  cutlass::arch::NamedBarrier producer_barrier(GEMM_PRODUCER_THREADS, 0u);  // hw id = 8
  cutlass::arch::NamedBarrier all_barrier(GEMM_TOTAL_THREADS, 1u);           // hw id = 9

  bool is_producer = (tid < GEMM_PRODUCER_THREADS);

  if (is_producer) {
    // Producers: subgroup 0 loads tile_A, subgroup 1 loads tile_B
    int local_id = tid % TILE_SIZE;
    if (tid < TILE_SIZE) {
      tile_A[local_id] = local_id + 1;  // A[i] = i + 1
    } else {
      tile_B[local_id] = local_id + 2;  // B[i] = i + 2
    }

    // Producer-only barrier: ensure both A and B are fully loaded
    // Only 32 threads (2 subgroups) participate - this is the key test:
    // barrier threads (32) < total kernel threads (64)
    producer_barrier.sync();
  }

  // All 64 threads barrier to make SLM writes visible to consumers
  all_barrier.arrive_and_wait();

  if (!is_producer) {
    // Consumers: compute C[i] = A[i] * B[i]
    int local_id = (tid - GEMM_PRODUCER_THREADS) % TILE_SIZE;
    if (tid < GEMM_PRODUCER_THREADS + TILE_SIZE) {
      tile_C[local_id] = tile_A[local_id] * tile_B[local_id];
    }
  }

  // Final barrier: ensure consumers finish writing tile_C
  all_barrier.arrive_and_wait();

  // Producers write results to global memory
  if (tid < TILE_SIZE) {
    output[tid] = tile_C[tid];
  }
}

TEST(Xe_Named_Barrier, gemm_producer_consumer) {
  cutlass::host_vector<int> host_output(TILE_SIZE, 0);
  cutlass::device_vector<int> device_output(TILE_SIZE, 0);

  launch<gemm_producer_consumer_kernel, GemmProducerConsumerTestName<>>(
      launch_policy{
          compat::dim3(1), compat::dim3(GEMM_TOTAL_THREADS),
          kernel_properties{sycl_exp::sub_group_size<SUBGROUP_SIZE>}},
      device_output.data());

  compat::wait_and_throw();
  host_output = device_output;

  // C[i] = A[i] * B[i] = (i+1) * (i+2)
  for (int i = 0; i < TILE_SIZE; ++i) {
    EXPECT_EQ(host_output[i], (i + 1) * (i + 2));
  }
}
