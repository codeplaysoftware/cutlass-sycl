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
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor_predicate.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int Stages,
  class TileShape_,
  class ElementAOptionalTuple,
  class StrideA_,
  class ElementBOptionalTuple,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
    MainloopIntelPVCMixedPrecision<Stages>,
    TileShape_,
    ElementAOptionalTuple,
    StrideA_,
    ElementBOptionalTuple,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_>
{
private:
  enum class ConversionMode {
    DirectConvert,
    ConvertAndScale,
    ConvertAndScaleWithZero
  };

  using ScaleA = detail::deduce_mixed_width_dtype_t<1, ElementAOptionalTuple>;
  using ScaleB = detail::deduce_mixed_width_dtype_t<1, ElementBOptionalTuple>;
  using ZeroA = detail::deduce_mixed_width_dtype_t<2, ElementAOptionalTuple>;
  using ZeroB = detail::deduce_mixed_width_dtype_t<2, ElementBOptionalTuple>;

public:
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopIntelPVCMixedPrecision<Stages>;
  using WorkgroupTileShape = TileShape_;

  
  static_assert(cute::is_tuple<ElementAOptionalTuple>::value ^ cute::is_tuple<ElementBOptionalTuple>::value, 
    "Either A OR B must be a tuple. It must take the from {ElementOperand, [ElementScale],"
    "[ElementZero]}. Inputs in [] are optional.");

  using ElementA = detail::deduce_mixed_width_dtype_t<0, ElementAOptionalTuple>;
  using ElementB = detail::deduce_mixed_width_dtype_t<0, ElementBOptionalTuple>;
  static constexpr bool IsATransformed = cute::is_tuple<ElementAOptionalTuple>::value;
  using ElementScale = cute::conditional_t<IsATransformed, ScaleA, ScaleB>;
  using ElementZero = cute::conditional_t<IsATransformed, ZeroA, ZeroB>;
  using ElementMMA = cute::conditional_t<IsATransformed, ElementB, ElementA>;
  using ElementQuant = cute::conditional_t<IsATransformed, ElementA, ElementB>;

  static_assert(cute::is_same_v<ElementMMA, ElementScale> || cute::is_same_v<ElementScale, void>, "Quantization scale type must match MMA type.");
  static_assert(cute::is_same_v<ElementMMA, ElementZero> || cute::is_same_v<ElementZero, void>, "Quantization zero point must match MMA type.");

  // For cases where we can't have a void type, we can use this to allow the code to compile when the scale / zero is void.
  using NonVoidElementScale = cute::conditional_t<cute::is_void_v<ElementScale>, ElementMMA, ElementScale>;
  using NonVoidElementZero = cute::conditional_t<cute::is_void_v<ElementZero>, ElementMMA, ElementZero>;

  using StrideA = StrideA_;
  using StrideB = StrideB_;

  // These are always MN major
  using StrideScale = cute::Stride<_1, int64_t, int64_t>;
  // For cases where we can't have a void scale, we can use this to allow the code to compile when the scale is void.
  using NonVoidStrideScale = cute::conditional_t<
      cute::is_void_v<StrideScale>, cute::Stride<_1, int64_t, int64_t>, StrideScale>;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;

  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using GmemTiledCopyScale = XE_2D_U16x1x32_LD_N;  // TODO(joe): generalize

  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  static_assert(cute::sizeof_bits_v<ElementA> >= 8 &&
                    cute::sizeof_bits_v<ElementB> >= 8,
                "Subbyte types not supported.");
  static_assert(!cute::is_same_v<ElementA, ElementB>, "Mixed precision GEMM requires different types for A and B!");

private:
   
  static constexpr ConversionMode 
  get_conversion_mode() {
    if constexpr (cute::is_void_v<ElementScale>) {
      return ConversionMode::DirectConvert;
    } 
    else if constexpr (cute::is_void_v<ElementZero>) {
      return ConversionMode::ConvertAndScale;
    }
    else {
      return ConversionMode::ConvertAndScaleWithZero;
    }
  }

  static constexpr ConversionMode KernelConversionMode = get_conversion_mode();
  static constexpr bool ModeHasScales = KernelConversionMode == ConversionMode::ConvertAndScale ||
                                        KernelConversionMode == ConversionMode::ConvertAndScaleWithZero;

public:
  static constexpr int SubgroupSize = DispatchPolicy::SubgroupSize;

  using MmaAtomShape = typename TiledMma::AtomShape_MNK;

  static constexpr auto BLK_M = get<0>(WorkgroupTileShape{});
  static constexpr auto BLK_N = get<1>(WorkgroupTileShape{});
  static constexpr auto BLK_K = get<2>(WorkgroupTileShape{});
  
  static constexpr auto ATOM_M = get<1>(typename TiledMma::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_N = get<2>(typename TiledMma::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_K = get<3>(typename TiledMma::ThrLayoutVMNK{}.shape());

  static constexpr auto SG_M = ceil_div(BLK_M, ATOM_M);
  static constexpr auto SG_N = ceil_div(BLK_N, ATOM_N);
  static constexpr auto SG_K = ceil_div(BLK_K, ATOM_K);
  using SubgroupTileShape = Shape<decltype(SG_M), decltype(SG_N), decltype(SG_K)>;

  static constexpr size_t cacheline_bytes = 64;
  static constexpr auto block_size_w_a = cute::min(SG_K, cacheline_bytes / sizeof(ElementA));
  static constexpr auto block_size_w_b = cute::min(SG_N, cacheline_bytes / sizeof(ElementB));
  static constexpr auto nums_block_w_a = ceil_div(SG_K, block_size_w_a);
  static constexpr auto nums_block_w_b = ceil_div(SG_N, block_size_w_b);
  using PrefetchAThrShape = Shape<Int<ATOM_N /cute::gcd(ATOM_N, nums_block_w_a)>, Int<cute::gcd(ATOM_N, nums_block_w_a)>>;
  using PrefetchBThrShape = Shape<Int<ATOM_M /cute::gcd(ATOM_M, nums_block_w_b)>, Int<cute::gcd(ATOM_M, nums_block_w_b)>>;
  using PrefetchATileSize = decltype(ceil_div(Shape<Int<SG_M>, Int<SG_K>>{},PrefetchAThrShape{}));
  using PrefetchBTileSize = decltype(ceil_div(Shape<Int<SG_K>, Int<SG_N>>{},PrefetchBThrShape{}));
  
  static constexpr uint32_t MaxThreadsPerBlock = size(TiledMma{});
  using traits_load_A = Copy_Traits<GmemTiledCopyA, StrideA>;
  using atom_load_A = Copy_Atom<traits_load_A, ElementA>;

  using traits_load_B = Copy_Traits<GmemTiledCopyB, StrideB>;
  using atom_load_B = Copy_Atom<traits_load_B, ElementB>;

  using traits_load_scale = Copy_Traits<GmemTiledCopyScale, StrideScale>;
  using atom_load_scale = Copy_Atom<traits_load_scale, NonVoidElementScale>;

  using XE_Prefetch_A = decltype(cute::detail::prefetch_selector<PrefetchATileSize, ElementA>());
  using XE_Prefetch_B = decltype(cute::detail::prefetch_selector<PrefetchBTileSize, ElementB>());

  using  TensorMKL = decltype(make_tensor(make_gmem_ptr(static_cast<ElementA const*>(nullptr)), make_shape(0,0,0), StrideA{}));   //(m, k)
  using  TensorNKL = decltype(make_tensor(make_gmem_ptr(static_cast<ElementB const*>(nullptr)), make_shape(0,0,0), StrideB{}));   //(n, k)
  using  TensorScale = decltype(make_tensor(make_gmem_ptr(static_cast<NonVoidElementScale const*>(nullptr)), make_shape(0,0,0), StrideScale{}));   //(m OR n, k)
  using  TensorZero = decltype(make_tensor(make_gmem_ptr(static_cast<NonVoidElementZero const*>(nullptr)), make_shape(0,0,0), StrideScale{}));   //(m OR n, k)
 
  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
    ElementScale const* ptr_S = nullptr;
    StrideScale dS;
    int group_size = 1; // Avoid /0 when no scales // TODO(joe): Is this needed?
    ElementZero const* ptr_Z = nullptr;
  };

  // TODO(joe): Can/should I specialize the Params struct based on `ConversionMode`?
  // Could this save passing some args & registers etc?
  struct Params {
    TensorMKL mA;
    TensorNKL mB;
    TensorScale mScale;
    TensorZero mZero;
    int64_t scale_k;
    int group_size;
  };

  //
  // Methods
  //

  CollectiveMma() = default;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const &problem_shape,
                          Arguments const &args, void *workspace) {
    (void)workspace;

    auto [M, N, K, L] = problem_shape;

    auto mA_mkl =
        make_tensor(make_gmem_ptr(static_cast<ElementA const *>(args.ptr_A)),
                    make_layout(make_shape(M, K, L), args.dA));

    auto mB_nkl =
        make_tensor(make_gmem_ptr(static_cast<ElementB const *>(args.ptr_B)),
                    make_layout(make_shape(N, K, L), args.dB));

    // These are unused in ConversionMode::DirectConvert
    auto scale_k = cute::ceil_div(K, args.group_size);
    auto mScale = make_tensor(
        make_gmem_ptr(static_cast<NonVoidElementScale const *>(args.ptr_S)),
        make_layout(make_shape(IsATransformed ? M : N, scale_k, L), args.dS));

    auto mZero =
        make_tensor(make_gmem_ptr(static_cast<NonVoidElementZero const *>(args.ptr_Z)),
                    make_layout(make_shape(IsATransformed ? M : N, scale_k, L), args.dS));
    
    return Params{mA_mkl, mB_nkl, mScale, mZero, scale_k, args.group_size};
  }

  // Helper functions to select packing for conversion
  template <class SrcType,
            class DstType,
            int Cosize>
  struct select_packing { // Naive packing policy
    static constexpr auto value() {
      return Int<cute::gcd(Cosize, 32 / cute::min(sizeof_bits_v<SrcType>, sizeof_bits_v<DstType>))>{};
    }
  };

  /// Utilities to transform A.
  template <class EngineIn,
            class EngineOut, 
            class EngineScales, 
            class LayoutIn,
            class LayoutOut,
            class LayoutScales,
            class... Ts>
  CUTLASS_DEVICE
  void transform_A(
    Tensor<EngineIn, LayoutIn> const& tCrA_load, 
    Tensor<EngineOut, LayoutOut>& tCrA_mma,
    Tensor<EngineScales, LayoutScales>& tCrS_input) {

    static_assert(is_rmem<EngineIn>::value, "Input tensor for A conversion must come from registers");
    static_assert(is_rmem<EngineOut>::value, "Output tensor for A conversion must come from registers");
    static_assert(cosize_v<LayoutIn> == cosize_v<LayoutOut>);
    static_assert(size_v<LayoutIn> == cosize_v<LayoutIn>);
    static_assert(size_v<LayoutOut> == cosize_v<LayoutOut>);
    using SrcType = typename EngineIn::value_type;
    using DstType = typename EngineOut::value_type;

    auto const& src = tCrA_load(_, _, _);
    auto const& dst = tCrA_mma(_, _, _);
    auto pSrc = raw_pointer_cast(src.data());
    auto pDst = const_cast<DstType*>(raw_pointer_cast(dst.data()));
    constexpr int num_elements = decltype(size(src))::value;

    constexpr int pack = decltype(select_packing<SrcType, DstType, num_elements>::value())::value;
    using Converter = cutlass::NumericArrayConverter<DstType, SrcType, pack, cutlass::FloatRoundStyle::round_to_nearest>;
    using SrcArray = cutlass::Array<SrcType, pack>;
    using DstArray = cutlass::Array<DstType, pack>;
    constexpr int iters = num_elements / pack;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < iters; ++i) {
      SrcArray const* pSrcArr = reinterpret_cast<SrcArray const*>(pSrc) + i;
      DstArray* pDstArr = reinterpret_cast<DstArray*>(pDst) + i;
      *pDstArr = Converter::convert(*pSrcArr);
    }
    if(ModeHasScales){
      for (int i = 0; i < 16; ++i){
        for (int j = 0; j < 2; ++j){
          // TODO(joe): Finish this off...
         tCrA_mma(_,_,j)[i] *= shfl_sync(0xFFFFFFFF, tCrS_input(j), i);
        }
      }
    }
  }

  /// Perform a subgroup-scoped matrix multiply-accumulate
  template <
    int PrefetchStrideA,
    int PrefetchStrideB,
    class FrgTensorD,
    class TensorA,
    class TensorB,
    class FrgTensorC,
    class KTileIterator,
    class ResidueMNK,
    class BlkCoord
  >
  CUTLASS_DEVICE void
  operator() (
      FrgTensorD &accum,
      TensorA gA,
      TensorB gB,
      FrgTensorC const &src_accum,
      KTileIterator k_tile_iter, int k_tile_count,
      ResidueMNK residue_mnk,
      BlkCoord const &blk_coord,
      int const &K_start,
      int thread_idx,
      char *smem_buf,
      Params const& mainloop) 
  {
    static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");

    (void)residue_mnk;
    (void)thread_idx;
    (void)smem_buf;

    auto [m_idx, n_idx, k_idx, l_idx] = blk_coord;
  #ifdef CUTLASS_SYCL_SWITCH_WG
    const int m_coord = n_idx * BLK_M + (get_sub_group_id() / ATOM_N) * SG_M;
    const int n_coord = m_idx * BLK_N + (get_sub_group_id() % ATOM_N) * SG_N;
  #else
    const int m_coord = m_idx * BLK_M + (get_sub_group_id() / ATOM_N) * SG_M;
    const int n_coord = n_idx * BLK_N + (get_sub_group_id() % ATOM_N) * SG_N;
  #endif
    const int l_coord = l_idx;

    auto tiled_copy_a = make_xe_2d_copy(atom_load_A{}.with(mainloop.mA),
                                             Layout<Shape<_1, Int<SubgroupSize>>>{});
    auto tiled_copy_b = make_xe_2d_copy(atom_load_B{}.with(mainloop.mB),
                                             Layout<Shape<_1, Int<SubgroupSize>>>{});
    auto tiled_copy_scale = make_xe_2d_copy(atom_load_scale{}.with(mainloop.mScale),
                                             Layout<Shape<_1, Int<SubgroupSize>>>{});

    // Partition the copying of A and B tiles across the threads
    auto thr_copy_A = tiled_copy_a.get_slice(thread_idx);
    auto thr_copy_B = tiled_copy_b.get_slice(thread_idx);
    auto thr_copy_scale = tiled_copy_scale.get_slice(thread_idx);

    // Instantiate the MMA object and get thread slice
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(thread_idx);

    // Partition fragment
    Tensor fragment_A = thr_mma.partition_fragment_A(gA(_, _, 0)); // (M_atom, M_iter, K_iter)
    Tensor fragment_B = thr_mma.partition_fragment_B(gB(_, _, 0)); // (K_atom, N_iter, K_iter)

    // If IsATransformed, we need modes M_atom, and M_iter from fragment_A layout
    // Else we need mode N_iter from fragment_B layout.
    // TODO(joe): handle B version (!IsATransformed), and generalize/cuteify!
    Tensor fragment_scale_input = make_tensor<NonVoidElementScale>(make_layout(Shape<_2, _1, _1>{}));

    // narrow input fragment
    Tensor tCrA_input = make_tensor<ElementA>(fragment_A.layout());

    static_assert(std::is_same_v<typename decltype(tCrA_input)::value_type, ElementA>);
    static_assert(std::is_same_v<typename decltype(fragment_A)::value_type, ElementB>);

    // Retile for copy
    auto copy_tCrA = thr_copy_A.retile_D(tCrA_input);
    Tensor copy_tCrB = thr_copy_B.retile_D(fragment_B);
    Tensor copy_tCrS = thr_copy_scale.retile_D(fragment_scale_input);

    // Retile for cute::gemm
    Tensor mma_tCrA = thr_copy_A.retile_MMA(thr_mma, fragment_A);
    Tensor mma_tCrB = thr_copy_B.retile_MMA(thr_mma, fragment_B);

  #if CUTLASS_ENABLE_DEBUG_PRINTS
    if (cutlass::thread(LOG_THREAD, LOG_GROUP)) {
        print("======================= A: \n");
        print("  gA : "); print(gA); print("\n");
        print("copy_tCrA : "); print(copy_tCrA); print("\n");
        print("  mma_tCrA : "); print(mma_tCrA); print("\n");

        print("=====================  B :\n");
        print("  gB : "); print(gB); print("\n");
        print("copy_tCrB : "); print(copy_tCrB); print("\n");
        print("  mma_tCrB : "); print(mma_tCrB); print("\n");

        print("=====================  Config: \n");
        print("  threads per workgroup : "); print(MaxThreadsPerBlock); print("\n");
        print("  SubgroupTileShape : "); print(SubgroupTileShape{}); print("\n");

        print(" PrefetchAThrShape :    ");print(PrefetchAThrShape{});print("\n");
        print(" PrefetchBThrShape :    ");print(PrefetchBThrShape{});print("\n");
        print(" PrefetchATileSize :    ");print(PrefetchATileSize{});print("\n");
        print(" PrefetchBTileSize :    ");print(PrefetchBTileSize{});print("\n");
      }
  #endif

    //
    // Mainloop
    //
    Tensor block2d_copy_iter_a = tiled_copy_a.get_pvc_tensor(make_coord(m_coord, 0, l_coord), copy_tCrA.shape());
    auto copy_iter_a = append_pvc_tensor<1>(block2d_copy_iter_a, k_tile_count, BLK_K);

    Tensor block2d_copy_iter_b = tiled_copy_b.get_pvc_tensor(make_coord(n_coord, 0, l_coord), copy_tCrB.shape());
    auto copy_iter_b = append_pvc_tensor<1>(block2d_copy_iter_b, k_tile_count, BLK_K);

    Tensor copy_iter_s = make_tensor(make_inttuple_iter(make_coord(m_coord, 0, l_coord)),
                                             make_layout(make_shape(_1{}, _1{}, _1{}, k_tile_count), 
                                                         make_stride(_0{}, E<0>{} * _32{}, _0{}, E<1>{} * _1{})));

    const int k_start_idx = crd2idx((*k_tile_iter), make_shape(K_start));
    int prefetch_k = 0;

    Tensor block2d_prefetch_iter_a = XE_Prefetch_A{}.get_pvc_tensor(
                               make_coord(m_coord + (get_sub_group_id() % ATOM_N) / get<1>(PrefetchAThrShape{}) * get<0>(PrefetchATileSize{}),
                                          (k_start_idx + (get_sub_group_id() % ATOM_N) % get<1>(PrefetchAThrShape{})) * PrefetchStrideA,
                                          l_coord),
                               make_shape(_1{}, _1{}, _1{}));
    auto prefetch_iter_a = append_pvc_tensor<1>(block2d_prefetch_iter_a, k_tile_count, BLK_K);

    Tensor block2d_prefetch_iter_b = XE_Prefetch_B{}.get_pvc_tensor(
                               make_coord((get_sub_group_id() / ATOM_N / get<1>(PrefetchBThrShape{}) + k_start_idx) * PrefetchStrideB,
                                           n_coord + (get_sub_group_id() / ATOM_N) % get<1>(PrefetchBThrShape{}) * get<1>(PrefetchBTileSize{}),
                                           l_coord),
                               make_shape(_1{}, _1{}, _1{}));
    auto prefetch_iter_b = append_pvc_tensor<0>(block2d_prefetch_iter_b, k_tile_count, BLK_K);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < DispatchPolicy::Stages; i++, prefetch_k++) {
      if constexpr(cute::detail::has_prefetch<GmemTiledCopyA>) {
        prefetch(tiled_copy_a, prefetch_iter_a(_,_,_,prefetch_k));
      }
      if constexpr(cute::detail::has_prefetch<GmemTiledCopyB>) {
        prefetch(tiled_copy_b, prefetch_iter_b(_,_,_,prefetch_k));
      }
    }

    const int k_reload_factor = mainloop.group_size / BLK_K; 

    CUTLASS_PRAGMA_UNROLL
    for (int k_tile = 0, k = k_start_idx; k_tile < k_tile_count; ++k_tile, ++k, ++prefetch_k) {
      // Copy gmem to rmem for the first k_tile
      copy(tiled_copy_a, copy_iter_a(_,_,_,k), copy_tCrA);
      copy(tiled_copy_b, copy_iter_b(_,_,_,k), copy_tCrB);

      if constexpr(ModeHasScales){
        copy(tiled_copy_scale, copy_iter_s(_, _, _, k_reload_factor), copy_tCrS);
      }
      transform_A(tCrA_input, mma_tCrA, fragment_scale_input);

      if(prefetch_k < k_tile_count) {
        if constexpr(cute::detail::has_prefetch<GmemTiledCopyA>) {
          prefetch(tiled_copy_a, prefetch_iter_a(_,_,_,prefetch_k));
        }
        if constexpr(cute::detail::has_prefetch<GmemTiledCopyB>) {
          prefetch(tiled_copy_b, prefetch_iter_b(_,_,_,prefetch_k));
        } 
      }

      cute::gemm(tiled_mma, mma_tCrA, mma_tCrB, accum);
    }
  }
};


} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
