#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "forgeir/core/status.hpp"
#include "forgeir/runtime/runtime_session.hpp"

namespace forgeir {

struct RuntimeBenchmarkOptions {
    std::uint64_t warmup_count{5};
    std::uint64_t measured_iteration_count{20};
};

struct OperationTimingSamples {
    std::string operation_id;
    std::string operation_type;
    std::string kernel;
    std::vector<std::int64_t> output_shape;
    std::vector<double> elapsed_microseconds;
};

struct RuntimeBenchmarkSamples {
    std::string clock_source{"std::chrono::steady_clock"};
    bool clock_is_steady{true};
    std::uint64_t warmup_count{0};
    std::uint64_t measured_iteration_count{0};
    std::uint64_t planned_arena_bytes{0};
    std::uint64_t peak_live_bytes{0};
    std::uint64_t naive_allocation_bytes{0};
    std::vector<double> end_to_end_microseconds;
    std::vector<OperationTimingSamples> operations;
};

[[nodiscard]] Result<RuntimeBenchmarkSamples>
benchmark_runtime(RuntimeSession& session,
                  const std::unordered_map<std::string, ExternalTensor>& inputs,
                  const std::unordered_map<std::string, ExternalTensor>& parameters,
                  const RuntimeBenchmarkOptions& options = RuntimeBenchmarkOptions{});

} // namespace forgeir
