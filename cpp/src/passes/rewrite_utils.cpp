#include "rewrite_utils.hpp"

#include <algorithm>
#include <set>
#include <utility>
#include <vector>

namespace forgeir {

std::unordered_map<std::string, std::size_t> operation_index_by_output(const Graph& graph) {
    std::unordered_map<std::string, std::size_t> indices;
    for (std::size_t index = 0; index < graph.operations().size(); ++index) {
        for (const std::string& output : graph.operations()[index].output_ids()) {
            indices.emplace(output, index);
        }
    }
    return indices;
}

std::unordered_map<std::string, std::size_t> value_index_by_id(const Graph& graph) {
    std::unordered_map<std::string, std::size_t> indices;
    for (std::size_t index = 0; index < graph.values().size(); ++index) {
        indices.emplace(graph.values()[index].id(), index);
    }
    return indices;
}

std::unordered_map<std::string, std::size_t> value_use_counts(const Graph& graph) {
    std::unordered_map<std::string, std::size_t> counts;
    for (const Operation& operation : graph.operations()) {
        for (const std::string& input : operation.input_ids()) {
            ++counts[input];
        }
    }
    return counts;
}

bool is_declared_output(const Graph& graph, const std::string_view value_id) {
    return std::find(graph.output_ids().begin(), graph.output_ids().end(), value_id) !=
           graph.output_ids().end();
}

bool has_side_effect(const Operation& operation) {
    const auto marker = operation.attributes().find("side_effect");
    return marker != operation.attributes().end() && marker->is_boolean() && marker->get<bool>();
}

void replace_all_uses(Graph& graph, const std::string_view from, const std::string_view to) {
    for (Operation& operation : graph.mutable_operations()) {
        for (std::string& input : operation.mutable_input_ids()) {
            if (input == from) {
                input = std::string(to);
            }
        }
    }
}

void erase_operations(Graph& graph, const std::unordered_set<std::string>& operation_ids) {
    auto& operations = graph.mutable_operations();
    operations.erase(std::remove_if(operations.begin(), operations.end(),
                                    [&operation_ids](const Operation& operation) {
                                        return operation_ids.find(operation.id()) !=
                                               operation_ids.end();
                                    }),
                     operations.end());
}

void erase_values(Graph& graph, const std::unordered_set<std::string>& value_ids) {
    auto& values = graph.mutable_values();
    values.erase(std::remove_if(values.begin(), values.end(),
                                [&value_ids](const Value& value) {
                                    return value_ids.find(value.id()) != value_ids.end();
                                }),
                 values.end());
}

Result<bool> stable_topological_order(Graph& graph) {
    const auto producers = operation_index_by_output(graph);
    const std::vector<Operation>& operations = graph.operations();
    std::vector<std::size_t> indegrees(operations.size(), 0);
    std::vector<std::vector<std::size_t>> dependents(operations.size());
    for (std::size_t consumer = 0; consumer < operations.size(); ++consumer) {
        std::unordered_set<std::size_t> dependencies;
        for (const std::string& input : operations[consumer].input_ids()) {
            const auto producer = producers.find(input);
            if (producer == producers.end()) {
                return Status::error(StatusCode::failed_precondition,
                                     "operation " + operations[consumer].id() +
                                         " has an input without a producer: " + input);
            }
            dependencies.insert(producer->second);
        }
        for (const std::size_t producer : dependencies) {
            ++indegrees[consumer];
            dependents[producer].push_back(consumer);
        }
    }
    std::size_t previous_side_effect = operations.size();
    for (std::size_t index = 0; index < operations.size(); ++index) {
        if (!has_side_effect(operations[index])) {
            continue;
        }
        if (previous_side_effect != operations.size() &&
            std::find(dependents[previous_side_effect].begin(),
                      dependents[previous_side_effect].end(),
                      index) == dependents[previous_side_effect].end()) {
            dependents[previous_side_effect].push_back(index);
            ++indegrees[index];
        }
        previous_side_effect = index;
    }

    std::set<std::pair<std::string, std::size_t>> ready;
    for (std::size_t index = 0; index < operations.size(); ++index) {
        if (indegrees[index] == 0) {
            ready.emplace(operations[index].id(), index);
        }
    }
    std::vector<Operation> ordered;
    ordered.reserve(operations.size());
    while (!ready.empty()) {
        const std::size_t index = ready.begin()->second;
        ready.erase(ready.begin());
        ordered.push_back(operations[index]);
        for (const std::size_t dependent : dependents[index]) {
            --indegrees[dependent];
            if (indegrees[dependent] == 0) {
                ready.emplace(operations[dependent].id(), dependent);
            }
        }
    }
    if (ordered.size() != operations.size()) {
        return Status::error(StatusCode::failed_precondition,
                             "graph contains a cycle after rewriting");
    }

    bool changed = false;
    for (std::size_t index = 0; index < ordered.size(); ++index) {
        changed = changed || ordered[index].id() != operations[index].id();
    }
    if (changed) {
        graph.mutable_operations() = std::move(ordered);
    }
    return changed;
}

} // namespace forgeir
