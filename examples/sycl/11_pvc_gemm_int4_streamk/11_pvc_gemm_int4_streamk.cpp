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
/***************************************
* Mixed Precision PVC Gemm Example For int4_t (RowMajor A) x (ColumnMajor B)
*
* This example demonstrates how to dispatch a mixed precision GEMM on PVC, with optional dequantization.
* The GemmMode enum describes the 3 modes of operation:
*
* Note: due to a bug in the IGC compiler, it's currently necessary to build this example with the following
* environment variable set:
*   export IGC_allowDecompose2DBlockFuncs=0
*/

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include <cute/tensor.hpp>
#include <random>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "sycl_common.hpp"
#include "helper.h"
#include "cutlass/util/mixed_dtype_utils.hpp"

#define MByte (1024 * 1024)

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////

using MmaType = _Float16;
using QuantType = int4_t;//_BitInt(4);

// Command line options parsing
struct Options {

  bool help;
  bool error;

  bool splitk, dp;

  int m, n, k, l, iterations;
  int g, warmup;
  float alpha, beta;
  int flush_cache, cache_cnt, l3_cache, splits;

  Options():
    help(false),
    error(false),
    m(5120), n(4096), k(4096), l(1), iterations(20),
    g(128), alpha(1.f), beta(0.f), warmup(0), flush_cache(0),
    cache_cnt(3), splitk(true), dp(false), splits(2)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    if (cmd.check_cmd_line_flag("splitk")) {
      splitk = true;
    }

    if (cmd.check_cmd_line_flag("dp")) {
      dp = true;
    }

    cmd.get_cmd_line_argument("m", m, 32);
    cmd.get_cmd_line_argument("n", n, 4096);
    cmd.get_cmd_line_argument("k", k, 4096);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("g", g, 128);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
    cmd.get_cmd_line_argument("warmup", warmup, 0);
    cmd.get_cmd_line_argument("flush_cache", flush_cache, 0);
    cmd.get_cmd_line_argument("cache_cnt", cache_cnt, 3);
    cmd.get_cmd_line_argument("l3_cache", l3_cache, 192);
    cmd.get_cmd_line_argument("splits", splits, 2);
    cmd.get_cmd_line_argument("splitk", splitk, true);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "PVC int4_t StreamK GEMM Mixed Type Example\n\n"
        << "Options:\n\n"
        << "  --help                      If specified, displays this usage statement\n\n"
        << "  --dp                        If specified, uses Data Parallel decomposition\n"
        << "  --splitk                    If specified, uses SplitK decomposition\n"
        << "  --m=<int>                   Sets the M extent of the GEMM\n"
        << "  --n=<int>                   Sets the N extent of the GEMM\n"
        << "  --k=<int>                   Sets the K extent of the GEMM\n"
        << "  --l=<int>                   Sets the L extent (batch count) of the GEMM\n"
        << "  --splits=<int>              Sets the splitting factor for GEMM\n"
        << "  --alpha=<s32>               Epilogue scalar alpha\n"
        << "  --beta=<s32>                Epilogue scalar beta\n\n"
        << "  --iterations=<int>          Iterations\n\n";

    return out;
  }
};

// Factory structs to factor out boilerplate code
namespace helpers{
using namespace cutlass::gemm;
template <typename DispatchPolicy, typename TileShape, typename LayoutA,
          typename LayoutB, typename TiledMMA, typename GmemTiledCopyA,
          typename GmemTiledCopyB>
struct MixedCollectiveMmaBuilder {

  template <typename ElementA, typename ElementB>
  using CollectiveMma = collective::CollectiveMma<
      DispatchPolicy, TileShape, ElementA, LayoutA, ElementB, LayoutB, TiledMMA,
      GmemTiledCopyA, void, void, cute::identity, GmemTiledCopyB, void, void,
      cute::identity>;
};

template <typename ProblemShape, typename CollectiveEpilogue>
struct MixedGemmUniversalAdapterBuilder {
  template <typename CollectiveMainloop>
  using GemmUniversalAdapter =
      device::GemmUniversalAdapter<kernel::GemmUniversal<
          ProblemShape, CollectiveMainloop, CollectiveEpilogue, cutlass::gemm::StreamKScheduler>>;
};
}
///////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class Gemm
>
struct ExampleRunner {

  using CollectiveMainloop = typename Gemm::CollectiveMainloop;
  using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;
  using LayoutD = typename Gemm::LayoutD;

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementAcc = typename Gemm::ElementAccumulator;
  using ElementMMA = MmaType;
  using ElementQuant = QuantType;

  using ElementScale = typename CollectiveMainloop::NonVoidElementScale;
  using ElementZero = typename CollectiveMainloop::NonVoidElementZero;
  // Scale and Zero share a stride since the layout and shapes must be the same.
  using StrideScale = typename CollectiveMainloop::StrideScale;
  using StrideZero = StrideScale; 

  using ElementC = typename Gemm::ElementC;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;
  using ElementAccumulator = typename CollectiveEpilogue::ElementAccumulator;

  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  //
  // Data members
  //

  /// Initialization
  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;
  StrideScale stride_S;

  uint64_t seed = 0;

  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementMMA> block_A_dq; // Dequantized copy of A for validation
  cutlass::DeviceAllocation<ElementMMA> block_B_dq; // Dequantized copy of B for validation
  cutlass::DeviceAllocation<ElementScale> block_scale;
  cutlass::DeviceAllocation<ElementZero> block_zero;
  cutlass::DeviceAllocation<ElementC> block_C;
  cutlass::DeviceAllocation<ElementOutput> block_D;
  cutlass::DeviceAllocation<ElementOutput> block_ref_D;

  //
  // Methods
  //

  template <class T> static void fill_matrix( std::vector<T> &M) {
    std::random_device dev;
    std::mt19937 rng(dev());

    T start, end;

    if constexpr (std::is_same_v<T, tfloat32_t> || std::is_same_v<T, half_t>
                   || std::is_same_v<T, bfloat16_t> || std::is_same_v<T, float>) {
      start = (T)0.0;
      end = (T)1.0;
    } else if constexpr (std::is_same_v<T, int8_t>) {
      start = (T)(-5);
      end = (T)5;
    } else if constexpr (std::is_same_v<T, uint8_t>) {
      start = (T)0;
      end = (T)5;
    } else {
      CUTE_STATIC_ASSERT(false, "you must set coreect start/end value to initialize data");
    }

    std::uniform_real_distribution<float> dist((T)start, (T)end);
    for (int i = 0; i < M.size(); i++)
      M[i] = static_cast<T>(dist(rng));
  }

  void flush_cache(int l3_cache_size) {
    std::vector<uint8_t> host_cache;
    cutlass::DeviceAllocation<uint8_t> dev_cache_block;
    dev_cache_block.reset(l3_cache_size + 64);
    host_cache = std::vector<uint8_t>((size_t)dev_cache_block.size());
    // fill_matrix(host_cache);
    syclcompat::memcpy(dev_cache_block.get(), host_cache.data(),
                       dev_cache_block.size());
    syclcompat::wait();

    auto q = syclcompat::get_default_queue();

    using cache_dtype = uint32_t;
    cache_dtype* mem_to = (cache_dtype*)dev_cache_block.get();
    cache_dtype* mem_from = (cache_dtype*)(dev_cache_block.get() + sizeof(cache_dtype));

    q.parallel_for(sycl::nd_range<1>(l3_cache_size / sizeof(cache_dtype), 1024), [=](auto idx) {
      int i = idx.get_global_id();
      *mem_to += mem_from[i];
    });

    q.wait();
  }

  bool verify(const Options &options) {
      
    //
    // Compute reference output (default gemm kernel w/ ElementA == ElementB)
    //

    using GmemTiledCopyA = XE_2D_U16x32x32_LD_N;
    using GmemTiledCopyB = XE_2D_U16x16x16_LD_T;

    // Workgroup-level tile
    using TileShape = Shape<_256, _256, _32>;

    using TiledMma =
        typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32F16F16F32_TT>, Layout<TileShape>,
                                      Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

    constexpr int PipelineStages = 3;
    using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16<PipelineStages>;
    using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;

    using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementOutput, ElementCompute,
            ElementAccumulator, ElementAccumulator, cutlass::FloatRoundStyle::round_to_nearest>;

    using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape,
            decltype(tile_shape(TiledMma()))>;

    using CollectiveEpilogueRef = cutlass::epilogue::collective::CollectiveEpilogue<
            EpilogueDispatchPolicy,
            TileShape,
            ElementAccumulator,
            cutlass::gemm::TagToStrideC_t<LayoutC>,
            ElementOutput,
            cutlass::gemm::TagToStrideC_t<LayoutD>,
            FusionCallBacks,
            XE_2D_U32x8x16_LD_N,
            void, void,
            XE_2D_U32x8x16_ST_N,
            void, void>;

    // Mainloop
    using CollectiveMainloopRef = cutlass::gemm::collective::CollectiveMma<
            GEMMDispatchPolicy,
            TileShape,
            ElementMMA,
            cutlass::gemm::TagToStrideA_t<LayoutA>,
            ElementMMA,
            cutlass::gemm::TagToStrideB_t<LayoutB>,
            TiledMma,
            GmemTiledCopyA, void, void, cute::identity,  // A
            GmemTiledCopyB, void, void, cute::identity   // B
    >;

    using GemmKernelRef = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloopRef,
    CollectiveEpilogueRef
    >;

    using GemmRef = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelRef>;

    typename GemmRef::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {options.m, options.n, options.k, options.l},
      {block_A_dq.get(), stride_A, block_B_dq.get(), stride_B},
      {{options.alpha, options.beta}, block_C.get(), stride_C, block_ref_D.get(), stride_D}
    };

    // Run the gemm where the scaling is performed outside of the kernel.
    GemmRef gemm_ref;
    size_t workspace_size = GemmRef::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    CUTLASS_CHECK(gemm_ref.can_implement(arguments));
    CUTLASS_CHECK(gemm_ref.initialize(arguments, workspace.get()));
    CUTLASS_CHECK(gemm_ref.run());

    // compare_reference
    ElementOutput const epsilon(1e-2f);
    ElementOutput const non_zero_floor(1e-4f);
    bool passed = cutlass::reference::device::BlockCompareRelativelyEqual(block_ref_D.get(),
                      block_D.get(), block_D.size(), epsilon, non_zero_floor);
    return passed;
  }

  template <class Element>
  bool initialize_scale(
    cutlass::DeviceAllocation<Element>& block, 
    Options const& options) {

#ifdef INT4_DEBUG
    std::vector<Element> stage(block.size(), Element(1.0f));
    for (int i =0; i < 1; i++) {
      for (int j =0; j < 4096; j++) {
        stage[i * 4096 +j] = (Element)((j + 2) % 7);
      }
    }
    block.copy_from_host(stage.data());
#else
    float elt_max_f = float(cutlass::platform::numeric_limits<ElementQuant>::max());
    const float max_dequant_val = 4.f;
    const float min_dequant_val = 0.5f;

    float scope_max(max_dequant_val / elt_max_f);
    float scope_min(min_dequant_val / elt_max_f);

    cutlass::reference::device::BlockFillRandomUniform(
      block.get(), block.size(), seed, Element(scope_max), Element(scope_min));
#endif

return true;
  }

  template <class Element>
  bool initialize_zero(
    cutlass::DeviceAllocation<Element>& block,
    Options const& options) {
    cutlass::reference::device::BlockFillRandomUniform(
      block.get(), block.size(), seed, Element(2.0f), Element(-2.0f));
    return true;
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize(Options const& options) {
    auto [M, N, K, L] = ProblemShapeType{options.m, options.n, options.k, options.l};

    const int scale_k = cute::ceil_div(options.k, options.g);
    const int dq_mn_size = options.n;
    auto shape_A = cute::make_shape(M, K, L);
    auto shape_B = cute::make_shape(N, K, L);
    auto shape_CD = cute::make_shape(M, N, L);
    auto shape_scale_zero = cute::make_shape(dq_mn_size, scale_k, L);

    stride_A = cutlass::make_cute_packed_stride(StrideA{}, shape_A);
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, shape_CD);
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, shape_CD);
    stride_S = cutlass::make_cute_packed_stride(StrideScale{}, shape_scale_zero);

    block_A.reset(M * K * L);
    block_A_dq.reset(M * K * L);

    if (options.flush_cache) {
      auto l3_cache_size = options.l3_cache * MByte;
      auto elements = max(K * N * L, l3_cache_size * 8 / sizeof_bits_v<ElementQuant>);

      block_B.reset(elements * options.cache_cnt);
      block_B_dq.reset(elements * options.cache_cnt);
    } else {
      block_B.reset(K * N * L);
      block_B_dq.reset(K * N * L);
    }

    block_C.reset(M * N * L);
    block_D.reset(M * N * L);
    block_ref_D.reset(M * N * L);
    block_scale.reset(scale_k * L * dq_mn_size);
    block_zero.reset(scale_k * L * dq_mn_size);

    initialize_mixed_dtype_block(block_A, block_A_dq, seed + 2023);
    initialize_mixed_dtype_block(block_B, block_B_dq, seed + 2022);
    initialize_block(block_C, seed + 2021);

    initialize_scale(block_scale, options);
    initialize_zero(block_zero, options);

    auto layout_A = make_layout(shape_A, stride_A);
    auto layout_B = make_layout(shape_B, stride_B);
    auto layout_scale_zero = make_layout(shape_scale_zero, stride_S);

    // Note that we are overwriting the relevant `block_X_dq` here, both were
    // filled by initialize_mixed_dtype_block above
    cutlass::dequantize(block_B_dq.get(), block_B.get(), layout_B,
                      block_scale.get(), block_zero.get(), layout_scale_zero,
                      options.g);
  }

  cutlass::Status run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
    auto l3_cache_size = options.l3_cache * MByte;

    ProblemShapeType problem_size = ProblemShapeType{options.m, options.n, options.k, options.l};

    initialize(options);

    typename Gemm::GemmKernel::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        {block_A.get(), stride_A, block_B.get(), stride_B, block_scale.get(),
         stride_S, options.g, block_zero.get()},
        {{options.alpha, options.beta},
         block_C.get(),
         stride_C,
         block_D.get(),
         stride_D},
        hw_info,
        {options.splits, // Setting splits > 1 will force SplitK decomposition
          // Set the decomposition mode based on user provided options
          options.dp ? cutlass::gemm::kernel::detail::PersistentTileSchedulerXeStreamKParams::DecompositionMode::DataParallel :
          options.splitk ? cutlass::gemm::kernel::detail::PersistentTileSchedulerXeStreamKParams::DecompositionMode::SplitK :
                              cutlass::gemm::kernel::detail::PersistentTileSchedulerXeStreamKParams::DecompositionMode::StreamK}
      };

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    if (gemm_op.can_implement(arguments) != cutlass::Status::kSuccess){
      std::cout << "Invalid Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
      std::exit(1);
    }

    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));

    // Run the GEMM
    CUTLASS_CHECK(gemm_op.run());

    syclcompat::wait();

    // Verify that the result is correct
    bool passed = verify(options);

    std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;

    if(!passed) return cutlass::Status::kErrorInternal;

    float total_time = 0.f;
    if (options.warmup >= options.iterations) {
      return cutlass::Status::kErrorInternal;
    }

    double tflops = (2.0 * options.m * options.n * options.k * options.l) * 1e-12;
    double hbm = (sizeof_bits_v<ElementA> * options.m * options.k / 8 +
                  sizeof_bits_v<ElementB> * options.k * options.n / 8 +
                  sizeof_bits_v<ElementOutput> * options.m * options.n / 8) * 1e-9;

    std::cout << "\nProblem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
    printf("--l=%d --iterations=%d --flush_cache=%d\n", options.l, options.iterations, options.flush_cache);
    printf("--warmup=%d, --cache_cnt=%d, --l3_cache_size=%d\n\n", options.warmup, options.cache_cnt, l3_cache_size);

    if (options.iterations > 0) {
      for (int i = 0; i < options.iterations; ++i) {
        // flush_cache(l3_cache_size);
        if (options.flush_cache != 0) {
          if (i < options.warmup) {
            CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
          } else {
            auto l3_cache_size = options.l3_cache * MByte;
            auto elements = max(options.k * options.n * options.l, l3_cache_size * 8 / sizeof_bits_v<ElementQuant>);

            typename Gemm::GemmKernel::Arguments arguments1{
              cutlass::gemm::GemmUniversalMode::kGemm,
              problem_size,
              {block_A.get(), stride_A, block_B.get() + ((i - options.warmup + 1) % options.cache_cnt) * elements / 2,
                stride_B, block_scale.get(), stride_S, options.g, block_zero.get()},
              {{options.alpha, options.beta},
              block_C.get(),
              stride_C,
              block_D.get(),
              stride_D},
              hw_info};
              CUTLASS_CHECK(gemm_op.initialize(arguments1, workspace.get()));
          }
        } else {
          CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
        }

        GPU_Clock timer;
        timer.start();
        gemm_op.run();
        // syclcompat::wait();
        auto ctime = timer.seconds();

        if (i >= options.warmup) {
          total_time += ctime;
        }
  
        printf("Cutlass GEMM Performance [%d]:     [%4.3f]TFlop/s  [%4.3f]GB/s  (%6.4f)ms\n", i, tflops / ctime, hbm / ctime, ctime*1000);
      }

      float cute_time = total_time / (options.iterations - options.warmup);

      printf("Cutlass GEMM Performance average:     [%4.3f]TFlop/s  [%4.3f]GB/s  (%6.4f)ms\n", tflops / cute_time, hbm / cute_time, cute_time*1000);
    }

    return cutlass::Status::kSuccess;
  }

};

int main(int argc, const char** argv)
{
  //
  // Parse options
  //

  Options options;

  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  //
  // Run examples
  //

  // The KernelHardwareInfo struct holds the number of EUs on the GPU with a given device ID. This
  // information is used by the underlying kernel.
  cutlass::KernelHardwareInfo hw_info;

  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  bool passed;

  // The code section below describes datatype for input, output matrices and computation between
  // elements in input matrices.
  using ElementAccumulator = float;      // <- data type of accumulator
  using ElementComputeEpilogue = float;  // <- data type of epilogue operations
  using ElementInputA = QuantType;       // <- data type of elements in input matrix A
  using ElementInputB = MmaType;         // <- data type of elements in input matrix B
  using ElementOutput = float;           // <- data type of elements in output matrix D

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using ElementZero = MmaType;
  using ElementScale = MmaType;

  // Note: XE_2D_U18x32x32_LD_N is incompatible with our bf16 MMA atoms
  using GmemTiledCopyA = XE_2D_U4x16x16_LD_T;
  using GmemTiledCopyB = XE_2D_U16x32x16_LD_N;
  static_assert(sizeof(ElementInputA) == 1, "ElementA width must match GmemTiledCopyA U8");

  // Workgroup-level tile
  using TileShape = Shape<_256, _128, _16>;

  using TiledMma =
      typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32F16F16F32_TT>, Layout<TileShape>,
                                    Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  constexpr int PipelineStages = 4;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16MixedPrecision<PipelineStages, cutlass::gemm::KernelXeCooperative>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementOutput, ElementComputeEpilogue,
          ElementAccumulator, ElementAccumulator, cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape,
          decltype(tile_shape(TiledMma()))>;
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
          EpilogueDispatchPolicy,
          TileShape,
          ElementAccumulator,
          cutlass::gemm::TagToStrideC_t<LayoutC>,
          ElementOutput,
          cutlass::gemm::TagToStrideC_t<LayoutD>,
          FusionCallBacks,
          XE_2D_U32x8x16_LD_N,
          void, void,
          XE_2D_U32x8x16_ST_N,
          void, void>;

  // Use the helpers to avoid template arg repetition
  using GemmAdapterBuilder = helpers::MixedGemmUniversalAdapterBuilder<Shape<int, int, int, int>, CollectiveEpilogue>;

  using MixedBuilderQuantB =
      helpers::MixedCollectiveMmaBuilder<GEMMDispatchPolicy, TileShape,
                                cutlass::gemm::TagToStrideA_t<LayoutA>,
                                cutlass::gemm::TagToStrideB_t<LayoutB>,
                                TiledMma, GmemTiledCopyB, GmemTiledCopyA>;

  using MainloopBConvertAndScaleWithZeroPoint =
      MixedBuilderQuantB::CollectiveMma<
          ElementInputB, cute::tuple<ElementInputA, ElementScale, ElementZero>>;
  using GemmBConvertAndScaleWithZeroPoint =
      GemmAdapterBuilder::GemmUniversalAdapter<
          MainloopBConvertAndScaleWithZeroPoint>;

  std::cout << "Running in ConvertAndScaleWithZeroPoint mode." << std::endl;
  CUTLASS_CHECK(ExampleRunner<GemmBConvertAndScaleWithZeroPoint>{}.run(options, hw_info));

  return 0;
}
