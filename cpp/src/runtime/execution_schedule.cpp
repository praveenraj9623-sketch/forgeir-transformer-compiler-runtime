#include "forgeir/runtime/execution_schedule.hpp"

#include <limits>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace forgeir {
namespace {

bool has_side_effect(const Operation& operation) {
    const auto marker = operation.attributes().find("side_effect");
    return marker != operation.attributes().end() && marker->is_boolean() && marker->get<bool>();
}

Status add_edge(const std::size_t producer, const std::size_t consumer,
                std::set<std::pair<std::size_t, std::size_t>>& edges,
                std::vector<std::uint64_t>& indegrees,
                std::vector<std::vector<std::size_t>>& dependents) {
    if (!edges.emplace(producer, consumer).second) {
        return Status::ok_status();
    }
    if (indegrees[consumer] == std::numeric_limits<std::uint64_t>::max()) {
        return Status::error(StatusCode::overflow, "schedule dependency count overflow");
    }
    ++indegrees[consumer];
    dependents[producer].push_back(consumer);
    return Status::ok_status();
}

} // namespace

Result<ExecutionSchedule> build_execution_schedule(const Graph& graph) {
    if (graph.operations().size() > std::numeric_limits<std::uint64_t>::max()) {
        return Status::error(StatusCode::overflow,
                             "operation count exceeds execution schedule index capacity");
    }

    std::unordered_map<std::string, std::size_t> producer_by_value;
    for (std::size_t index = 0; index < graph.operations().size(); ++index) {
        for (const std::string& output_id : graph.operations()[index].output_ids()) {
            if (!producer_by_value.emplace(output_id, index).second) {
                return Status::error(StatusCode::failed_precondition,
                                     "value " + output_id +
                                         " has multiple producers while scheduling");
            }
        }
    }

    std::vector<std::uint64_t> indegrees(graph.operations().size(), 0);
    std::vector<std::vector<std::size_t>> dependents(graph.operations().size());
    std::set<std::pair<std::size_t, std::size_t>> edges;
    for (std::size_t consumer = 0; consumer < graph.operations().size(); ++consumer) {
        for (const std::string& input_id : graph.operations()[consumer].input_ids()) {
            const auto producer = producer_by_value.find(input_id);
            if (producer == producer_by_value.end()) {
                return Status::error(StatusCode::failed_precondition,
                                     "operation " + graph.operations()[consumer].id() +
                                         " has no producer for input " + input_id);
            }
            const Status edge_status =
                add_edge(producer->second, consumer, edges, indegrees, dependents);
            if (!edge_status.ok()) {
                return edge_status;
            }
        }
    }

    std::size_t previous_side_effect = graph.operations().size();
    for (std::size_t index = 0; index < graph.operations().size(); ++index) {
        if (!has_side_effect(graph.operations()[index])) {
            continue;
        }
        if (previous_side_effect != graph.operations().size()) {
            const Status edge_status =
                add_edge(previous_side_effect, index, edges, indegrees, dependents);
            if (!edge_status.ok()) {
                return edge_status;
            }
        }
        previous_side_effect = index;
    }

    std::set<std::pair<std::string, std::size_t>> ready;
    for (std::size_t index = 0; index < graph.operations().size(); ++index) {
        if (indegrees[index] == 0) {
            ready.emplace(graph.operations()[index].id(), index);
        }
    }

    ExecutionSchedule schedule;
    schedule.graph_hash = graph.graph_hash();
    schedule.operations.reserve(graph.operations().size());
    while (!ready.empty()) {
        const std::size_t operation_index = ready.begin()->second;
        ready.erase(ready.begin());
        const Operation& operation = graph.operations()[operation_index];
        const auto schedule_index = static_cast<std::uint64_t>(schedule.operations.size());
        schedule.operations.push_back(
            ScheduledOperation{schedule_index, operation.id(), operation.semantic_name(),
                               operation.type(), operation.input_ids(), operation.output_ids()});
        for (const std::size_t dependent : dependents[operation_index]) {
            if (indegrees[dependent] == 0) {
                return Status::error(StatusCode::internal,
                                     "schedule dependency underflow while ordering operations");
            }
            --indegrees[dependent];
            if (indegrees[dependent] == 0) {
                ready.emplace(graph.operations()[dependent].id(), dependent);
            }
        }
    }
    if (schedule.operations.size() != graph.operations().size()) {
        return Status::error(StatusCode::failed_precondition,
                             "graph contains a cycle or conflicting side-effect order");
    }
    return schedule;
}

nlohmann::json execution_schedule_json(const ExecutionSchedule& schedule) {
    nlohmann::json operations = nlohmann::json::array();
    for (const ScheduledOperation& operation : schedule.operations) {
        operations.push_back(nlohmann::json{{"index", operation.index},
                                            {"operation_id", operation.operation_id},
                                            {"semantic_name", operation.semantic_name},
                                            {"type", to_string(operation.type)},
                                            {"inputs", operation.input_ids},
                                            {"outputs", operation.output_ids}});
    }
    return nlohmann::json{{"schedule_schema_version", "1.0"},
                          {"graph_hash", schedule.graph_hash},
                          {"operation_count", schedule.operations.size()},
                          {"operations", std::move(operations)}};
}

} // namespace forgeir
