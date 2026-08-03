#include "forgeir/runtime/benchmark.hpp"

#include <chrono>
#include <cstddef>
#include <string>
#include <utility>

namespace forgeir {
namespace {

constexpr std::uint64_t kMaximumWarmupCount = 10000;
constexpr std::uint64_t kMaximumMeasuredIterationCount = 100000;

Status validate_options(const RuntimeBenchmarkOptions& options) {
    if (options.warmup_count > kMaximumWarmupCount) {
        return Status::error(StatusCode::invalid_argument,
                             "benchmark warm-up count exceeds the safety bound");
    }
    if (options.measured_iteration_count == 0 ||
        options.measured_iteration_count > kMaximumMeasuredIterationCount) {
        return Status::error(StatusCode::invalid_argument,
                             "benchmark measured iteration count must be in [1, 100000]");
    }
    return Status::ok_status();
}

Status execution_failure(const Status& status, const std::string& phase,
                         const std::uint64_t iteration) {
    return Status::error(status.code(), "benchmark " + phase + " iteration " +
                                            std::to_string(iteration) +
                                            " failed: " + status.message());
}

Status append_operation_samples(RuntimeBenchmarkSamples& samples,
                                const std::vector<ExecutionTraceRecord>& trace) {
    if (samples.operations.empty()) {
        samples.operations.reserve(trace.size());
        for (const ExecutionTraceRecord& record : trace) {
            OperationTimingSamples operation;
            operation.operation_id = record.operation_id;
            operation.operation_type = record.operation_type;
            operation.kernel = record.kernel;
            operation.output_shape = record.output_shape;
            operation.elapsed_microseconds.reserve(
                static_cast<std::size_t>(samples.measured_iteration_count));
            samples.operations.push_back(std::move(operation));
        }
    }
    if (trace.size() != samples.operations.size()) {
        return Status::error(StatusCode::internal,
                             "runtime operation trace length changed between iterations");
    }
    for (std::size_t index = 0; index < trace.size(); ++index) {
        const ExecutionTraceRecord& record = trace[index];
        OperationTimingSamples& operation = samples.operations[index];
        if (record.operation_id != operation.operation_id ||
            record.operation_type != operation.operation_type ||
            record.kernel != operation.kernel || record.output_shape != operation.output_shape) {
            return Status::error(StatusCode::internal,
                                 "runtime operation trace contract changed between iterations");
        }
        operation.elapsed_microseconds.push_back(record.elapsed_microseconds);
    }
    return Status::ok_status();
}

} // namespace

Result<RuntimeBenchmarkSamples>
benchmark_runtime(RuntimeSession& session,
                  const std::unordered_map<std::string, ExternalTensor>& inputs,
                  const std::unordered_map<std::string, ExternalTensor>& parameters,
                  const RuntimeBenchmarkOptions& options) {
    const Status option_status = validate_options(options);
    if (!option_status.ok()) {
        return option_status;
    }

    for (std::uint64_t iteration = 0; iteration < options.warmup_count; ++iteration) {
        const Status status = session.execute(inputs, parameters);
        if (!status.ok()) {
            return execution_failure(status, "warm-up", iteration);
        }
    }

    RuntimeBenchmarkSamples samples;
    samples.clock_is_steady = std::chrono::steady_clock::is_steady;
    samples.warmup_count = options.warmup_count;
    samples.measured_iteration_count = options.measured_iteration_count;
    samples.planned_arena_bytes = session.memory_plan().arena_size_bytes;
    samples.peak_live_bytes = session.memory_plan().peak_live_bytes;
    samples.naive_allocation_bytes = session.memory_plan().naive_allocation_bytes;
    samples.end_to_end_microseconds.reserve(
        static_cast<std::size_t>(options.measured_iteration_count));

    for (std::uint64_t iteration = 0; iteration < options.measured_iteration_count; ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        const Status status = session.execute(inputs, parameters);
        const auto end = std::chrono::steady_clock::now();
        if (!status.ok()) {
            return execution_failure(status, "measured", iteration);
        }
        samples.end_to_end_microseconds.push_back(
            std::chrono::duration<double, std::micro>(end - start).count());
        const Status trace_status = append_operation_samples(samples, session.trace());
        if (!trace_status.ok()) {
            return trace_status;
        }
    }
    return samples;
}

} // namespace forgeir
