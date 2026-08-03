#include "forgeir/passes/redundant_reshape_elimination_pass.hpp"

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "rewrite_utils.hpp"

namespace forgeir {

std::string_view RedundantReshapeEliminationPass::name() const noexcept {
    return "RedundantReshapeEliminationPass";
}

PassResult RedundantReshapeEliminationPass::run(Graph& graph) {
    const auto values = value_index_by_id(graph);
    std::unordered_set<std::string> removed_operations;
    std::unordered_set<std::string> removed_values;
    std::vector<RewriteRecord> rewrites;
    for (const Operation& operation : graph.operations()) {
        if (operation.type() != OperationType::reshape || has_side_effect(operation)) {
            continue;
        }
        const std::string& input_id = operation.input_ids().front();
        const std::string& output_id = operation.output_ids().front();
        if (is_declared_output(graph, output_id)) {
            continue;
        }
        const auto input = values.find(input_id);
        const auto output = values.find(output_id);
        if (input == values.end() || output == values.end() ||
            graph.values()[input->second].descriptor().shape().dimensions() !=
                graph.values()[output->second].descriptor().shape().dimensions()) {
            continue;
        }
        replace_all_uses(graph, output_id, input_id);
        removed_operations.insert(operation.id());
        removed_values.insert(output_id);
        rewrites.push_back(RewriteRecord{"",
                                         "",
                                         "removed a reshape whose input and output shapes "
                                         "are identical",
                                         {},
                                         {operation.id()},
                                         {},
                                         {output_id}});
    }
    erase_operations(graph, removed_operations);
    erase_values(graph, removed_values);
    const bool changed = !rewrites.empty();
    return PassResult::success(changed, {}, std::move(rewrites));
}

} // namespace forgeir
