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
    \brief Tests for Xe flash attention bf16
*/

#include "flash_attention_testbed_3x.hpp"

namespace cutlass {

template<typename TileShape, typename TiledMma, bool HasCausalMask>
struct XE_Flash_Attention {
  using LayoutQ = cutlass::layout::RowMajor;
  using LayoutK = cutlass::layout::ColumnMajor;
  using LayoutV = cutlass::layout::RowMajor;
  using LayoutO = cutlass::layout::RowMajor;

  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementInputQ = bfloat16_t;
  using ElementInputKV = bfloat16_t;
  using ElementOutput = float;

  static constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelPVC<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelPVCEpilogue;

  using GmemTiledCopyQ = XE_2D_U16x16x32_LD_N;
  using GmemTiledCopyK = XE_2D_U16x16x16_LD_T;
  using GmemTiledCopyV = XE_2D_U16x32x32_LD_V;
  using GmemTiledCopyStore = XE_2D_U32x8x16_ST_N;
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogueAttention<
        EpilogueDispatchPolicy, TileShape, ElementAccumulator, cutlass::gemm::TagToStrideC_t<LayoutO>, ElementOutput,
        GmemTiledCopyStore>;
  using CollectiveSoftmaxEpilogue = cutlass::epilogue::collective::CollectiveSoftmaxEpilogue<
        HasCausalMask, EpilogueDispatchPolicy, ElementAccumulator>;

  // Mainloop
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMmaAttention<
        GEMMDispatchPolicy, TileShape, ElementInputQ, cutlass::gemm::TagToStrideA_t<LayoutQ>, ElementInputKV,
        cutlass::gemm::TagToStrideB_t<LayoutK>, ElementInputKV, cutlass::gemm::TagToStrideB_t<LayoutV>, TiledMma,
        GmemTiledCopyQ, // Q
        GmemTiledCopyK, // K
        GmemTiledCopyV, // V,
        HasCausalMask>;

    using Kernel = cutlass::gemm::kernel::GemmUniversalAttention<Shape<int, int, int, int, int, int>, CollectiveMainloop,
                                                                     CollectiveSoftmaxEpilogue, CollectiveEpilogue>;
};

TEST(XE_Flash_Attention_bf16, causal_with_head_size_64) {
  using TileShape = Shape<_128, _64, _64, _64>;
  using TiledMma =
        typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
                                      Layout<Shape<_128, _64, _64>>,
                                      Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>>::TiledMMA;

  using Kernel = XE_Flash_Attention<TileShape, TiledMma, true>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestAll<Kernel>(64));
}

TEST(XE_Flash_Attention_bf16, noncausal_with_head_size_64) {
  using TileShape = Shape<_128, _64, _64, _64>;
  using TiledMma =
        typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
                                      Layout<Shape<_128, _64, _64>>,
                                      Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>>::TiledMMA;

  using Kernel = XE_Flash_Attention<TileShape, TiledMma, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestAll<Kernel>(64));
}

TEST(XE_Flash_Attention_bf16, causal_with_head_size_96) {
  using TileShape = Shape<_128, _64, _64, _64>;
  using TiledMma =
        typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
                                      Layout<Shape<_128, _64, _64>>,
                                      Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>>::TiledMMA;

  using Kernel = XE_Flash_Attention<TileShape, TiledMma, true>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestAll<Kernel>(96));
}

TEST(XE_Flash_Attention_bf16, noncausal_with_head_size_96) {
  using TileShape = Shape<_128, _64, _64, _64>;
  using TiledMma =
        typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
                                      Layout<Shape<_128, _64, _64>>,
                                      Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>>::TiledMMA;

  using Kernel = XE_Flash_Attention<TileShape, TiledMma, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestAll<Kernel>(96));
}

TEST(XE_Flash_Attention_bf16, causal_with_head_size_128) {
  using TileShape = Shape<_128, _128, _64, _64>;
    using TiledMma =
        typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
                                      Layout<Shape<_128, _128, _64>>,
                                      Layout<Shape<_8, _2, _1>, Stride<_2, _1, _1>>>::TiledMMA;

  using Kernel = XE_Flash_Attention<TileShape, TiledMma, true>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestAll<Kernel>(128));
}

TEST(XE_Flash_Attention_bf16, noncausal_with_head_size_128) {
  using TileShape = Shape<_128, _128, _64, _64>;
    using TiledMma =
        typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
                                      Layout<Shape<_128, _128, _64>>,
                                      Layout<Shape<_8, _2, _1>, Stride<_2, _1, _1>>>::TiledMMA;

  using Kernel = XE_Flash_Attention<TileShape, TiledMma, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestAll<Kernel>(128));
}

TEST(XE_Flash_Attention_bf16, causal_with_head_size_192) {
  using TileShape = Shape<_128, _128, _64, _64>;
    using TiledMma =
        typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
                                      Layout<Shape<_128, _128, _64>>,
                                      Layout<Shape<_8, _2, _1>, Stride<_2, _1, _1>>>::TiledMMA;

  using Kernel = XE_Flash_Attention<TileShape, TiledMma, true>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestAll<Kernel>(192));
}

TEST(XE_Flash_Attention_bf16, noncausal_with_head_size_192) {
  using TileShape = Shape<_128, _128, _64, _64>;
    using TiledMma =
        typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
                                      Layout<Shape<_128, _128, _64>>,
                                      Layout<Shape<_8, _2, _1>, Stride<_2, _1, _1>>>::TiledMMA;

  using Kernel = XE_Flash_Attention<TileShape, TiledMma, false>::Kernel;
  EXPECT_TRUE(test::flash_attention::TestAll<Kernel>(192));
}

} // namespace cutlass
