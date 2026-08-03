#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "forgeir/core/status.hpp"
#include "forgeir/ir/graph.hpp"
#include "forgeir/runtime/cpu_backend.hpp"
#include "forgeir/runtime/memory_planner.hpp"
#include "forgeir/runtime/tensor_storage.hpp"

namespace forgeir {

struct ExternalTensor {
    const float* data{nullptr};
    std::vector<std::int64_t> shape;
    std::uint64_t byte_size{0};
    bool contiguous{true};
};

struct HostTensor {
    std::vector<std::int64_t> shape;
    std::vector<float> values;
};

struct ExecutionTraceRecord {
    std::string operation_id;
    std::string operation_type;
    std::string kernel;
    std::vector<std::int64_t> output_shape;
    double elapsed_microseconds{0.0};
    bool has_arena_offset{false};
    std::uint64_t arena_offset{0};
};

class RuntimeSession {
  public:
    [[nodiscard]] static Result<std::shared_ptr<RuntimeSession>>
    load(const std::string& graph_path, std::string_view backend_name = "cpu");

    [[nodiscard]] Status execute(const std::unordered_map<std::string, ExternalTensor>& inputs,
                                 const std::unordered_map<std::string, ExternalTensor>& parameters,
                                 const std::vector<std::string>& capture_value_ids = {});
    [[nodiscard]] Result<std::unordered_map<std::string, HostTensor>> get_outputs() const;
    [[nodiscard]] Result<HostTensor> get_value(const std::string& value_id) const;
    [[nodiscard]] const std::vector<ExecutionTraceRecord>& trace() const noexcept;
    [[nodiscard]] const Graph& graph() const noexcept;
    [[nodiscard]] const MemoryPlan& memory_plan() const noexcept;

  private:
    RuntimeSession(Graph graph, MemoryPlan memory_plan, TensorStorage arena,
                   std::unique_ptr<Backend> backend);

    Graph graph_;
    MemoryPlan memory_plan_;
    TensorStorage arena_;
    std::unique_ptr<Backend> backend_;
    std::vector<ExecutionTraceRecord> trace_;
    std::unordered_map<std::string, HostTensor> captured_values_;
    bool executed_{false};
};

} // namespace forgeir
