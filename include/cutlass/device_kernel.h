/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
/*! \file
    \brief Template for generic CUTLASS kernel.
*/

#pragma once

#include <cutlass/detail/helper_macros.hpp> // CUTLASS_HOST_DEVICE
#include <cutlass/platform/platform.h> // uint64_t

// __grid_constant__ was introduced in CUDA 11.7.
#if ((__CUDACC_VER_MAJOR__ >= 12) || ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 7)))
#  define CUTLASS_GRID_CONSTANT_SUPPORTED
#endif

// __grid_constant__ can be enabled only on SM70+
#if defined(CUTLASS_GRID_CONSTANT_SUPPORTED) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
#  define CUTLASS_GRID_CONSTANT_ENABLED
#endif

#if ! defined(CUTLASS_GRID_CONSTANT)
#  if defined(CUTLASS_GRID_CONSTANT_ENABLED)
#    define CUTLASS_GRID_CONSTANT __grid_constant__
#  else
#    define CUTLASS_GRID_CONSTANT
#  endif
#endif

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

template <typename T>   struct Type2Type  {  using type=T;                    };
// using the simple type to replace the complex type to reduce this symbol size
template <typename  T>                                                                        struct GetUnderlyingKernel                              : public Type2Type<T>               {};
template <uint64_t shader_guid, unsigned index, template <uint64_t, unsigned> class Wrapper > struct GetUnderlyingKernel<Wrapper<shader_guid,index>>  : public Wrapper<shader_guid,index> {};
template <typename  T>                                                                        using  GetUnderlyingKernel_t                            = typename GetUnderlyingKernel<T>::type;


////////////////////////////////////////////////////////////////////////////////

/// Generic CUTLASS kernel template.
#if defined(CUTLASS_ENABLE_SYCL) && !defined(SYCL_EXT_ONEAPI_WORK_GROUP_SCRATCH_MEMORY)
template <typename Operator>
void Kernel(typename Operator::Params params, char* smem) {
  // Dynamic shared memory base pointer
  int* SharedStorageBase = reinterpret_cast<int*>(smem);
  // Declare pointer to dynamic shared memory.
  typename Operator::SharedStorage *shared_storage =
      reinterpret_cast<typename Operator::SharedStorage *>(SharedStorageBase);

  Operator op;

  op(params, *shared_storage);
  cutlass::arch::synclog_print();
}
#else
template <typename Operator>
CUTLASS_GLOBAL
void Kernel(typename Operator::Params params) {
  // Dynamic shared memory base pointer
#if defined(CUTLASS_ENABLE_SYCL)
  int* SharedStorageBase = static_cast<int*>(
      sycl::ext::oneapi::experimental::get_work_group_scratch_memory());
#else
  extern __shared__ int SharedStorageBase[];
#endif
  // Declare pointer to dynamic shared memory.
  typename Operator::SharedStorage *shared_storage =
      reinterpret_cast<typename Operator::SharedStorage *>(SharedStorageBase);

  Operator op;

  op(params, *shared_storage);
  cutlass::arch::synclog_print();
}
#endif

/// Generic CUTLASS kernel template.
#if defined(CUTLASS_ENABLE_SYCL) && !defined(SYCL_EXT_ONEAPI_WORK_GROUP_SCRATCH_MEMORY)
template <typename Operator>
void Kernel2(typename Operator::Params params, char* smem) {
  // Dynamic shared memory base pointer
  int* SharedStorageBase = reinterpret_cast<int*>(smem);
  // Declare pointer to dynamic shared memory.
  typename Operator::SharedStorage *shared_storage =
      reinterpret_cast<typename Operator::SharedStorage *>(SharedStorageBase);

  Operator::invoke(params, *shared_storage);
  cutlass::arch::synclog_print();

}
#else
template <typename Operator>
CUTLASS_GLOBAL
void Kernel2(typename Operator::Params params) {
  // Dynamic shared memory base pointer
#if defined(CUTLASS_ENABLE_SYCL)
  int* SharedStorageBase = static_cast<int*>(
      sycl::ext::oneapi::experimental::get_work_group_scratch_memory());

#else
  extern __shared__ int SharedStorageBase[];
#endif
  // Declare pointer to dynamic shared memory.
  typename Operator::SharedStorage *shared_storage =
      reinterpret_cast<typename Operator::SharedStorage *>(SharedStorageBase);

  Operator::invoke(params, *shared_storage);
  cutlass::arch::synclog_print();

}
#endif


////////////////////////////////////////////////////////////////////////////////
//
// 3.0 specific launch
//
////////////////////////////////////////////////////////////////////////////////

/// Generic CUTLASS kernel template.
#if defined(CUTLASS_ENABLE_SYCL) && !defined(SYCL_EXT_ONEAPI_WORK_GROUP_SCRATCH_MEMORY)
template <typename Operator>
void device_kernel(typename Operator::Params const& params, sycl::local_ptr<char> smem)
{
  Operator op;
  op(params, smem);
  cutlass::arch::synclog_print();
}
#else
template <typename Operator>
CUTLASS_GLOBAL
#ifdef __CUDACC__
// Enclosing this in __CUDACC__ suppresses MSVC warnings.
__launch_bounds__(Operator::MaxThreadsPerBlock, Operator::MinBlocksPerMultiprocessor)
#endif // __CUDACC__
#if defined(CUTLASS_ENABLE_SYCL)
void device_kernel(typename Operator::Params const& params)
#else
void device_kernel(CUTLASS_GRID_CONSTANT typename Operator::Params const params)
#endif
{
  // Dynamic shared memory base pointer
#if defined(CUTLASS_ENABLE_SYCL)
  char* smem = static_cast<char*>(sycl::ext::oneapi::experimental::get_work_group_scratch_memory());
#else
  extern __shared__ char smem[];
#endif
  Operator op;
  op(params, smem);
  cutlass::arch::synclog_print();
}
#endif // defined(CUTLASS_ENABLE_SYCL) && !defined(SYCL_EXT_ONEAPI_WORK_GROUP_SCRATCH_MEMORY)

////////////////////////////////////////////////////////////////////////////////
} /// namespace cutlass
