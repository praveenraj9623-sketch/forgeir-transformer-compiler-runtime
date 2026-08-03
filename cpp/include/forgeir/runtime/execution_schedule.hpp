#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "forgeir/core/status.hpp"
#include "forgeir/ir/graph.hpp"

namespace forgeir {

struct ScheduledOperation {
    std::uint64_t index{0};
    std::string operation_id;
    std::string semantic_name;
    OperationType type{OperationType::input};
    std::vector<std::string> input_ids;
    std::vector<std::string> output_ids;
};

struct ExecutionSchedule {
    std::string graph_hash;
    std::vector<ScheduledOperation> operations;
};

[[nodiscard]] Result<ExecutionSchedule> build_execution_schedule(const Graph& graph);
[[nodiscard]] nlohmann::json execution_schedule_json(const ExecutionSchedule& schedule);

} // namespace forgeir
