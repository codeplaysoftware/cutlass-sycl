/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
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

namespace cutlass::flash_attention::collective {
using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename To_type, typename Engine, typename Layout>
CUTLASS_DEVICE auto convert_type(Tensor<Engine, Layout> const &tensor) {
  using From_type = typename Engine::value_type;
  constexpr int numel = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
  auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
  return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class DispatchPolicy, class ProblemShapeType_, class TileShape_, class ElementQ_, class StrideQ_, class ElementK_, class StrideK_,
          class ElementV_, class StrideV_, class TiledMma_, class GmemTiledCopyQ_, class GmemTiledCopyK_,
          class GmemTiledCopyV_, bool CausalMask_>
struct FlashDecodeMma {
  static_assert(cutlass::detail::dependent_false<ElementQ_>, "Could not find a mainloop specialization.");
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <int Stages, class ProblemShapeType_, class TileShape_, class ElementQ_, class StrideQ_, class ElementK_, class StrideK_,
          class ElementV_, class StrideV_, class TiledMma_, class GmemTiledCopyQ_, class GmemTiledCopyK_,
          class GmemTiledCopyV_, bool CausalMask_>
struct FlashDecodeMma<gemm::MainloopIntelXeXMX16<Stages>, ProblemShapeType_, TileShape_, ElementQ_, StrideQ_, ElementK_, StrideK_, ElementV_,
                              StrideV_, TiledMma_, GmemTiledCopyQ_, GmemTiledCopyK_, GmemTiledCopyV_, CausalMask_> {
  //
  // Type Aliases
  //
  using DispatchPolicy = gemm::MainloopIntelXeXMX16<Stages>;
  using TileShape = TileShape_;
  using WorkgroupTileShape = TileShape; // <BLK_N_K, BLK_N_V, BLK_K_PV, BLK_K_QK>
  using ProblemShapeType = ProblemShapeType_;
  using ElementQ = ElementQ_;
  using StrideQ = StrideQ_;
  using ElementK = ElementK_;
  using StrideK = StrideK_;
  using ElementV = ElementV_;
  using StrideV = StrideV_;
  using TiledMma = TiledMma_;
  using TiledMmaQVO = TiledMma;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyQ = GmemTiledCopyQ_;
  using GmemTiledCopyK = GmemTiledCopyK_;
  using GmemTiledCopyV = GmemTiledCopyV_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  static constexpr bool CausalMask = CausalMask_;
  static constexpr int SubgroupSize = DispatchPolicy::SubgroupSize;

  using MmaAtomShape = typename TiledMma::AtomShape_MNK;

  static constexpr auto PV_ATOM_M = get<1>(typename TiledMmaQVO::ThrLayoutVMNK{}.shape());
  static constexpr auto PV_ATOM_N = get<2>(typename TiledMmaQVO::ThrLayoutVMNK{}.shape());
  static constexpr auto PV_ATOM_K = get<3>(typename TiledMmaQVO::ThrLayoutVMNK{}.shape());
  // <8, 512, 64>
  using TileShapeQK = decltype(Shape<decltype(get<0>(MmaAtomShape())), decltype(get<0>(WorkgroupTileShape{})), decltype(get<3>(WorkgroupTileShape{}))>{}); // <BLK_M_Q, BLK_N_QK, BLK_K_QK>

  static constexpr auto QK_BLK_M = get<0>(TileShapeQK{});
  static constexpr auto QK_BLK_N = get<1>(TileShapeQK{});
  static constexpr auto QK_BLK_K = get<2>(TileShapeQK{});

  using Q_Atom_Shape = decltype(Shape<_1, Int<PV_ATOM_M>, _1>{});
  using Q_Atom_Stride = decltype(Stride<_1, Int<PV_ATOM_N>, _1>{});

  // This TiledMma is required to serve the specific tiling requirements for QK GEMM.
  // This is due to the consumption of the same part of matrix Q by all subgroups within a workgroup.
  using TiledMmaQ = typename TiledMMAHelper<typename TiledMmaQVO::Atom, 
                                            Layout<TileShapeQK>,
                                            Layout<Q_Atom_Shape, Q_Atom_Stride>>::TiledMMA;

  static constexpr auto QK_ATOM_M = get<1>(typename TiledMmaQ::ThrLayoutVMNK{}.shape()); // 1
  static constexpr auto QK_ATOM_N = get<2>(typename TiledMmaQ::ThrLayoutVMNK{}.shape()); // 8
  static constexpr auto QK_ATOM_K = get<3>(typename TiledMmaQ::ThrLayoutVMNK{}.shape()); // 1
//(512,64) /(8/1) -> (64,64)
  using TempShapeQK = decltype(cute::shape_div(take<1, 3>(TileShapeQK{}), take<2, 4>(typename TiledMmaQ::ThrLayoutVMNK{}.shape())));
  // (8, 64,64)
  using SubgroupTileShapeQK = decltype(make_shape(get<0>(TileShapeQK{}), get<0>(TempShapeQK{}), get<1>(TempShapeQK{})));
  using FragsShapeQK = decltype(cute::shape_div(take<0, 2>(SubgroupTileShapeQK{}), take<0, 2>(typename TiledMmaQ::AtomShape_MNK())));

  static constexpr auto QK_SG_M = get<0>(SubgroupTileShapeQK{});
  static constexpr auto QK_SG_N = get<1>(SubgroupTileShapeQK{});
  static constexpr auto QK_SG_K = get<2>(SubgroupTileShapeQK{});
  // <512, (64/128), 64>
  using TileShapePV = decltype(Shape<decltype(get<0>(TileShapeQK{})), decltype(get<1>(WorkgroupTileShape{})), decltype(QK_SG_N)>{}); // <BLK_M_PV, BLK_N_V, SG_N_QK>

  static constexpr auto PV_BLK_N = get<1>(TileShapePV{});
  static constexpr auto PV_BLK_K = get<2>(TileShapePV{});

  using TempShapePV = decltype(cute::shape_div(take<1, 3>(TileShapePV{}), take<2, 4>(typename TiledMmaQVO::ThrLayoutVMNK{}.shape())));
  using SubgroupTileShapePV = decltype(make_shape(get<0>(TileShapePV{}), get<0>(TempShapePV{}), get<1>(TempShapePV{})));
  using FragsShapePV = decltype(cute::shape_div(take<0, 2>(SubgroupTileShapePV{}), take<0, 2>(typename TiledMmaQVO::AtomShape_MNK())));

  static constexpr bool is_var_len = cutlass::fmha::collective::is_variable_length_v<tuple_element_t<3, ProblemShapeType>>;

  static constexpr uint32_t MaxThreadsPerBlock = size(TiledMmaQVO{});
  using CopyThreadShape = Shape<_1, Int<SubgroupSize>>;
  
  using traits_load_Q = Copy_Traits<GmemTiledCopyQ, StrideQ>;
  using atom_load_Q = Copy_Atom<traits_load_Q, ElementQ>;
  using val_layout_load_Q = decltype(make_layout(shape_div(typename traits_load_Q::BlockShape{}, CopyThreadShape{})));
  using XE_Copy_Q = decltype(make_tiled_copy(atom_load_Q{}, Layout<CopyThreadShape>{}, val_layout_load_Q{}));

  using traits_load_K = Copy_Traits<GmemTiledCopyK, StrideK>;
  using atom_load_K = Copy_Atom<traits_load_K, ElementK>;
  using val_layout_load_K = decltype(make_layout(shape_div(typename traits_load_K::BlockShape{}, CopyThreadShape{})));
  using XE_Copy_K = decltype(make_tiled_copy(atom_load_K{}, Layout<CopyThreadShape>{}, val_layout_load_K{}));

  using traits_load_V = Copy_Traits<GmemTiledCopyV, StrideV>;
  using atom_load_V = Copy_Atom<traits_load_V, ElementV>;
  using val_layout_load_V = decltype(make_layout(shape_div(typename traits_load_V::BlockShape{}, CopyThreadShape{})));
  using XE_Copy_V = decltype(make_tiled_copy(atom_load_V{}, Layout<CopyThreadShape>{}, val_layout_load_V{}));

  // Host side kernel arguments
  struct Arguments {
    ElementQ const *ptr_Q;
    StrideQ dQ;
    ElementK const *ptr_K;
    StrideK dK;
    ElementV const *ptr_V;
    StrideV dV;
    ElementK const* ptr_K_cache;
    StrideK dK_cache;
    ElementV const* ptr_V_cache;
    StrideV dV_cache;
  };

  struct Params {
    XE_Copy_Q gmem_tiled_copy_q;
    XE_Copy_K gmem_tiled_copy_k;
    XE_Copy_V gmem_tiled_copy_v;
    XE_Copy_K gmem_tiled_copy_k_cache;
    XE_Copy_V gmem_tiled_copy_v_cache;
  };

  //
  // Methods
  //

  FlashDecodeMma() = default;

  static constexpr Params to_underlying_arguments(ProblemShapeType const &problem_shape, Arguments const &args,
                                                  void *workspace) {
    (void)workspace;

    auto [batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, seq_len_kv_cache, head_size_qk, head_size_vo] = problem_shape;

    auto tensorQ = make_tensor(make_gmem_ptr(args.ptr_Q), make_layout(make_shape(seq_len_qo, head_size_qk, batch * num_heads_q), args.dQ));
    auto tensorK = make_tensor(make_gmem_ptr(args.ptr_K), make_layout(make_shape(seq_len_kv, head_size_qk, batch * num_heads_kv), args.dK));
    auto tensorV = make_tensor(make_gmem_ptr(args.ptr_V), make_layout(make_shape(head_size_vo, seq_len_kv, batch * num_heads_kv), args.dV));
    auto tensorK_cache = make_tensor(make_gmem_ptr(args.ptr_K_cache), make_layout(make_shape(seq_len_kv_cache, head_size_qk, batch * num_heads_kv), args.dK_cache));
    auto tensorV_cache = make_tensor(make_gmem_ptr(args.ptr_V_cache), make_layout(make_shape(head_size_vo, seq_len_kv_cache, batch * num_heads_kv), args.dV_cache));

    XE_Copy_Q copyQ{XE_Copy_Q{}.with(tensorQ)};
    XE_Copy_K copyK{XE_Copy_K{}.with(tensorK)};
    XE_Copy_V copyV{XE_Copy_V{}.with(tensorV)};
    XE_Copy_K copyK_cache{XE_Copy_K{}.with(tensorK_cache)};
    XE_Copy_V copyV_cache{XE_Copy_V{}.with(tensorV_cache)};
  
    return Params{copyQ, copyK, copyV, copyK_cache, copyV_cache};
  }

  template <class FragQccum, class TensorQ, class TensorK, class FragSrc>
  CUTLASS_DEVICE void mmaQK(FragQccum &accum, TensorQ gQ, TensorK gK, FragSrc const &frag_src,
                            int const &k_tile_count, Params const &params, bool is_KV_cache) {

    auto& gmem_tiled_copy_k = is_KV_cache ? params.gmem_tiled_copy_k_cache : params.gmem_tiled_copy_k;

    int thread_idx = static_cast<int>(ThreadIdxX());
    auto thr_copy_Q = params.gmem_tiled_copy_q.get_slice(thread_idx);
    auto thr_copy_K = gmem_tiled_copy_k.get_slice(thread_idx);
    // Instantiate the MMA object
    TiledMmaQ tiled_mma;
    // To make all threads in a warp have the same global tensors pass in the index of thread 0 in each warp
    auto sg = syclcompat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_id()[0] * DispatchPolicy::SubgroupSize;
    auto thread_mma_k = tiled_mma.get_slice(first_thread_in_sg_idx);
    auto thread_mma_q = tiled_mma.get_slice(first_thread_in_sg_idx);

    // Partition
    Tensor tCgQ = thread_mma_q.partition_A(gQ);
    Tensor tCgK = thread_mma_k.partition_B(gK);

    // Create fragments
    Tensor tCrQ = make_tensor<ElementQ>(make_fragment_layout(params.gmem_tiled_copy_q, take<0,3>(tCgQ.shape())));
    Tensor tCrK = make_tensor<ElementK>(make_fragment_layout(gmem_tiled_copy_k, take<0,3>(tCgK.shape())));
    
    // Retile registers for copies
    Tensor tQrQ = thr_copy_Q.retile_D(tCrQ);
    Tensor tKrK = thr_copy_K.retile_D(tCrK);

    // Retile global tile for copies
    Tensor tQgQ = thr_copy_Q.retile_S(tCgQ);
    Tensor tKgK = thr_copy_K.retile_S(tCgK);

#if CUTLASS_ENABLE_DEBUG_PRINTS
#define PRINT(x) print(#x ": "); print(x); print("\n");
  if (cute::thread(LOG_THREAD, LOG_GROUP)) {
    print("======================= Q: \n");
    PRINT(gQ);
    PRINT(tCrQ);
    PRINT(tCgQ);
    PRINT(tQrQ);
    PRINT(tQgQ);

    print("=====================  K :\n");
    PRINT(gK);
    PRINT(tCrK);
    PRINT(tCgK);
    PRINT(tKrK);
    PRINT(tKgK);

    print("=====================  Config: \n");
    PRINT(MaxThreadsPerBlock);
    PRINT(SubgroupTileShapeQK{});
    PRINT(accum);
  }
  #undef PRINT
#endif

    //
    // Mainloop
    //

    TiledMmaQVO tiled_mma2;
    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
      copy(params.gmem_tiled_copy_q, tQgQ(_,_,_,k_tile), tQrQ);
      copy(gmem_tiled_copy_k, tKgK(_,_,_,k_tile), tKrK);
      cute::gemm(tiled_mma2, accum, tCrQ, tCrK, frag_src);
    }
  }

  template <class FragQccum, class FragS, class TensorV, class FragSrc>
  CUTLASS_DEVICE void mmaPV(FragQccum &accum, FragS const &tSr, TensorV gV,
                            FragSrc const &frag_src, Params const &params, bool is_KV_cache) {

    auto& gmem_tiled_copy_v = is_KV_cache ? params.gmem_tiled_copy_v_cache : params.gmem_tiled_copy_v;

    int thread_idx = static_cast<int>(ThreadIdxX());
    // Instantiate the MMA object
    TiledMmaQVO tiled_mma;
    auto sg = syclcompat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_id()[0] * DispatchPolicy::SubgroupSize;
    auto thread_mma = tiled_mma.get_slice(first_thread_in_sg_idx);  
    Tensor tCgV = thread_mma.partition_B(gV);
    Tensor tCrV = make_tensor<ElementV>(make_fragment_layout(gmem_tiled_copy_v, tCgV.shape()));

    // Partition the copying of A and B tiles across the threads
    auto gmem_thr_copy_V = gmem_tiled_copy_v.get_slice(thread_idx);
    Tensor tVrV = gmem_thr_copy_V.retile_D(tCrV);
    Tensor tVgV = gmem_thr_copy_V.retile_S(tCgV);

#if CUTLASS_ENABLE_DEBUG_PRINTS
#define PRINT(x) print(#x ": "); print(x); print("\n");
  if (cute::thread(LOG_THREAD, LOG_GROUP)) {
    print("=====================  V :\n");
    PRINT(gV);
    PRINT(tCrV);
    PRINT(tCgV);
    PRINT(tVrV);
    PRINT(tVgV);

    print("=====================  Config: \n");
    PRINT(MaxThreadsPerBlock);
    PRINT(SubgroupTileShapePV{});
    PRINT(accum);
  }
  #undef PRINT
#endif

    // 7) Convert S to P (FP32 -> BF16)
    Tensor tPr = convert_type<typename TiledMmaQVO::ValTypeA>(tSr);

    //
    // Mainloop
    //
    copy(gmem_tiled_copy_v, tVgV, tVrV);
    cute::gemm(tiled_mma, accum, tPr, tCrV, frag_src);
  }

  // SequenceLengthShape = Shape<int, int, int>
  // For Fixed Sequence Length, ProblemShape = Shape<int, int, int, int, int, int, int, int>
  // For Variable Sequence Length, ProblemShape = Shape<int, int, int, VariableSeqlen, VariableSeqlen, VariableSeqlen, int, int>
  template <class ProblemShape, class SequenceLengthShape>
  CUTLASS_DEVICE static constexpr Params get_updated_copies(Params const& params, ProblemShape const& problem_shape,
                                                            SequenceLengthShape const& sequence_length_shape, int const& l_coord) {
    if constexpr (!is_var_len) {
      return params;
    } else {
      auto [num_heads_q, num_heads_kv, head_size_qk, head_size_vo] = select<1, 2, 6, 7>(problem_shape);
      auto [seq_len_qo, seq_len_kv, seq_len_kv_cache] = sequence_length_shape;

      auto qo_cumulative_length = get<3>(problem_shape).cumulative_length;
      auto kv_cumulative_length = get<4>(problem_shape).cumulative_length;
      auto kv_cache_cumulative_length = get<5>(problem_shape).cumulative_length;

      int offset_q = num_heads_q * head_size_qk * qo_cumulative_length[l_coord];
      int offset_k = num_heads_kv * head_size_qk * kv_cumulative_length[l_coord];
      int offset_v = num_heads_kv * head_size_vo * kv_cumulative_length[l_coord];
      int offset_k_cache = seq_len_kv_cache == 0 ? 0 : num_heads_kv * head_size_qk * kv_cache_cumulative_length[l_coord];
      int offset_v_cache = seq_len_kv_cache == 0 ? 0 : num_heads_kv * head_size_vo * kv_cache_cumulative_length[l_coord];

      auto q_traits = static_cast<traits_load_Q const&>(params.gmem_tiled_copy_q);
      const ElementQ* q_ptr = (const ElementQ*)q_traits.base_ptr;

      auto k_traits = static_cast<traits_load_K const&>(params.gmem_tiled_copy_k);
      const ElementK* k_ptr = (const ElementK*)k_traits.base_ptr;

      auto v_traits = static_cast<traits_load_V const&>(params.gmem_tiled_copy_v);
      const ElementV* v_ptr = (const ElementV*)v_traits.base_ptr;

      auto k_traits_cache = static_cast<traits_load_K const&>(params.gmem_tiled_copy_k_cache);
      const ElementK* k_cache_ptr = (const ElementK*)k_traits_cache.base_ptr;

      auto v_traits_cache = static_cast<traits_load_V const&>(params.gmem_tiled_copy_v_cache);
      const ElementV* v_cache_ptr = (const ElementV*)v_traits_cache.base_ptr;

      auto shape_q = make_shape(static_cast<int>(seq_len_qo), head_size_qk, num_heads_q);
      StrideQ stride_q = cutlass::make_cute_packed_stride(StrideQ{}, shape_q);

      auto shape_k = make_shape(static_cast<int>(seq_len_kv), head_size_qk, num_heads_kv);
      StrideK stride_k = cutlass::make_cute_packed_stride(StrideK{}, shape_k);

      auto shape_v = make_shape(head_size_vo, static_cast<int>(seq_len_kv), num_heads_kv);
      StrideV stride_v = cutlass::make_cute_packed_stride(StrideV{}, shape_v);

      auto shape_k_cache = make_shape(static_cast<int>(seq_len_kv_cache), head_size_qk, num_heads_kv);
      StrideK stride_k_cache = cutlass::make_cute_packed_stride(StrideK{}, shape_k_cache);

      auto shape_v_cache = make_shape(head_size_vo, static_cast<int>(seq_len_kv_cache), num_heads_kv);
      StrideV stride_v_cache = cutlass::make_cute_packed_stride(StrideV{}, shape_v_cache);

      auto tensorQ = make_tensor(make_gmem_ptr(q_ptr + offset_q), make_layout(shape_q, stride_q));
      auto tensorK = make_tensor(make_gmem_ptr(k_ptr + offset_k), make_layout(shape_k, stride_k));
      auto tensorV = make_tensor(make_gmem_ptr(v_ptr + offset_v), make_layout(shape_v, stride_v));
      auto tensorK_cache = make_tensor(make_gmem_ptr(k_cache_ptr + offset_k_cache), make_layout(shape_k_cache, stride_k_cache));
      auto tensorV_cache = make_tensor(make_gmem_ptr(v_cache_ptr + offset_v_cache), make_layout(shape_v_cache, stride_v_cache));

      XE_Copy_Q copyQ{XE_Copy_Q{}.with(tensorQ)};
      XE_Copy_K copyK{XE_Copy_K{}.with(tensorK)};
      XE_Copy_V copyV{XE_Copy_V{}.with(tensorV)};
      XE_Copy_K copyK_cache{XE_Copy_K{}.with(tensorK_cache)};
      XE_Copy_V copyV_cache{XE_Copy_V{}.with(tensorV_cache)};

      return Params{copyQ, copyK, copyV, copyK_cache, copyV_cache};
    }
  }
};

} // namespace cutlass::flash_attention::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
