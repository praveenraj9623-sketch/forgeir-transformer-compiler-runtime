#include "forgeir/passes/dead_code_elimination_pass.hpp"

#include <algorithm>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "rewrite_utils.hpp"

namespace forgeir {

std::string_view DeadCodeEliminationPass::name() const noexcept {
    return "DeadCodeEliminationPass";
}

PassResult DeadCodeEliminationPass::run(Graph& graph) {
    const auto producers = operation_index_by_output(graph);
    std::vector<std::string> worklist(graph.output_ids().begin(), graph.output_ids().end());
    worklist.insert(worklist.end(), graph.input_ids().begin(), graph.input_ids().end());
    for (const Operation& operation : graph.operations()) {
        if (has_side_effect(operation)) {
            worklist.insert(worklist.end(), operation.output_ids().begin(),
                            operation.output_ids().end());
        }
    }

    std::unordered_set<std::string> live_values;
    std::unordered_set<std::string> live_operations;
    while (!worklist.empty()) {
        std::string value_id = std::move(worklist.back());
        worklist.pop_back();
        if (!live_values.insert(value_id).second) {
            continue;
        }
        const auto producer = producers.find(value_id);
        if (producer == producers.end()) {
            continue;
        }
        const Operation& operation = graph.operations()[producer->second];
        if (!live_operations.insert(operation.id()).second) {
            continue;
        }
        worklist.insert(worklist.end(), operation.input_ids().begin(), operation.input_ids().end());
        live_values.insert(operation.output_ids().begin(), operation.output_ids().end());
    }

    std::vector<std::string> removed_operations;
    std::vector<std::string> removed_values;
    std::unordered_set<std::string> operation_ids;
    std::unordered_set<std::string> value_ids;
    for (const Operation& operation : graph.operations()) {
        if (live_operations.find(operation.id()) == live_operations.end()) {
            removed_operations.push_back(operation.id());
            operation_ids.insert(operation.id());
        }
    }
    for (const Value& value : graph.values()) {
        if (live_values.find(value.id()) == live_values.end()) {
            removed_values.push_back(value.id());
            value_ids.insert(value.id());
        }
    }
    if (operation_ids.empty() && value_ids.empty()) {
        return PassResult::success(false);
    }

    erase_operations(graph, operation_ids);
    erase_values(graph, value_ids);
    RewriteRecord rewrite{
        "",
        "",
        "removed operations and values not required by declared outputs, declared "
        "inputs, or side effects",
        {},
        std::move(removed_operations),
        {},
        std::move(removed_values)};
    return PassResult::success(true, {}, {std::move(rewrite)});
}

} // namespace forgeir
