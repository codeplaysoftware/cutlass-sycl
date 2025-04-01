/***************************************************************************************************
 * Copyright (c) 2025 - 2025 Codeplay Software Ltd. All rights reserved.
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
    \brief Tests for Xe tf32_tf32_fp32
*/


#include "cutlass/cutlass.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "default_gemm_configuration.hpp"

#include "gemm_testbed_3x.hpp"

/* TODO(Codeplay): TF32 copy builtins don't work well with GEMM. Needs more investigation
TEST(XE_Device_Gemm_tf32t_tf32t_f32t_tensor_op_f32, 256x256x32) {
  using Config = cutlass::gemm::device::DefaultGemmConfigurationToCutlass3Types<
    cutlass::arch::OpClassTensorOp, cutlass::arch::IntelPVC,
    tfloat32_t, cutlass::layout::RowMajor,
    tfloat32_t, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int,int>,
      Config::CollectiveMainloop,
      Config::CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestXe<Gemm>());
}

 TODO(Codeplay): missing copy transpose builtin and prefetch builtin
TEST(XE_Device_Gemm_tf32n_tf32t_f32t_tensor_op_f32, 256x256x32) {
  using Config = cutlass::gemm::device::DefaultGemmConfigurationToCutlass3Types<
    cutlass::arch::OpClassTensorOp, cutlass::arch::IntelPVC,
    tfloat32_t, cutlass::layout::ColumnMajor,
    tfloat32_t, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int,int>,
      Config::CollectiveMainloop,
      Config::CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestXe<Gemm>());
}

TEST(XE_Device_Gemm_tf32t_tf32n_f32t_tensor_op_f32, 256x256x32) {
  using Config = cutlass::gemm::device::DefaultGemmConfigurationToCutlass3Types<
    cutlass::arch::OpClassTensorOp, cutlass::arch::IntelPVC,
    tfloat32_t, cutlass::layout::RowMajor,
    tfloat32_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int,int>,
      Config::CollectiveMainloop,
      Config::CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestXe<Gemm>());
}

TEST(XE_Device_Gemm_tf32n_tf32n_f32t_tensor_op_f32, 256x256x32) {
  using Config = cutlass::gemm::device::DefaultGemmConfigurationToCutlass3Types<
    cutlass::arch::OpClassTensorOp, cutlass::arch::IntelPVC,
    tfloat32_t, cutlass::layout::ColumnMajor,
    tfloat32_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int,int>,
      Config::CollectiveMainloop,
      Config::CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestXe<Gemm>());
}
*/
