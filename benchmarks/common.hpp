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

#include <benchmark/benchmark.h>
#include <iostream>
#include <sstream>
#include <fstream>

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
  static inline std::size_t get_llc_size() {
    #if defined(CUTLASS_ENABLE_SYCL)
      return syclcompat::get_default_queue().get_device().get_info<sycl::info::device::global_mem_cache_size>();   
    #else
      cudaDeviceProp prop_struct;
      auto result = cudaGetDeviceProperties(&prop_struct, 0);
      if (result != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(result));
      }
      return static_cast<std::size_t>(prop_struct.l2CacheSize);
    #endif
  }


namespace benchmark {

  template<typename Options>
  class BenchmarkRegistry {
    using BM_Lambda = std::function<void(::benchmark::State& state, Options const&, cutlass::KernelHardwareInfo const &)>;
    std::map<const std::string, BM_Lambda> benchmarks;

    static BenchmarkRegistry& get_instance() {
      static BenchmarkRegistry runner;
      return runner;
    }

    BenchmarkRegistry() = default;
  public:
    BenchmarkRegistry(BenchmarkRegistry const&) = delete;
    void operator=(BenchmarkRegistry const&) = delete;

    static auto const& get_benchmark(std::string const& name) {
      auto& benchs = get_instance().benchmarks;
      auto it = benchs.find(name);
      if (it == benchs.end()) {
        throw std::runtime_error("Benchmark not found");
      }
      return it->second;
    }

    static void Register(std::string const& key, BM_Lambda const func) {
      auto& benchs = get_instance().benchmarks;
      if (benchs.find(key) == benchs.end()) {
        benchs.insert(std::make_pair(key, func));
      } else {
        std::cerr << "Benchmark " << key << " duplicated." << std::endl;
      }
    }
  };
} // namespace benchmark
} // namespace cutlass



///////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct BenckmarkOptions {

  bool help;
  bool error;
  std::string config_file;

  BenckmarkOptions():
          help(false),
          error(false)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("config_file", config_file);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "Benchmark\n\n"
        << "Options:\n\n"
        << "  --config_file=/path/to/config_file.in\n\n";

    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename BenchOptions>
auto benchmark_main(int argc, const char **argv) -> int {
  BenchOptions options;

  options.parse(argc, argv);

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
  const auto benchmark_config = argv[0];
  auto runner = cutlass::benchmark::BenchmarkRegistry<BenchOptions>::get_benchmark(benchmark_config);

  std::stringstream benchmark_name;
  benchmark_name << benchmark_config << "/" << options.benchmark_name();
  ::benchmark::RegisterBenchmark(benchmark_name.str(), runner, options, hw_info)->UseManualTime();
  return 0;
}


template <typename BenchOptions>
void register_benchmarks(std::string line) {
  // Split the line into arguments
  std::istringstream iss(line);
  std::vector<std::string> args;
  std::string arg;

  while (iss >> arg) {
    args.push_back(arg);
  }

  // Prepare argc and argv for secondary_main
  int line_argc = static_cast<int>(args.size());
  std::vector<const char*> line_argv(line_argc);

  for (int i = 0; i < line_argc; ++i) {
    line_argv[i] = args[i].c_str();
  }

  // Call the secondary main function with the parsed arguments
  benchmark_main<BenchOptions>(line_argc, line_argv.data());
}
