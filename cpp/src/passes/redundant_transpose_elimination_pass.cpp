#include "forgeir/passes/redundant_transpose_elimination_pass.hpp"

#include <cstdint>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "rewrite_utils.hpp"

namespace forgeir {

std::string_view RedundantTransposeEliminationPass::name() const noexcept {
    return "RedundantTransposeEliminationPass";
}

PassResult RedundantTransposeEliminationPass::run(Graph& graph) {
    std::unordered_set<std::string> removed_operations;
    std::unordered_set<std::string> removed_values;
    std::vector<RewriteRecord> rewrites;
    for (const Operation& operation : graph.operations()) {
        if (operation.type() != OperationType::transpose || has_side_effect(operation)) {
            continue;
        }
        const std::string& output_id = operation.output_ids().front();
        if (is_declared_output(graph, output_id)) {
            continue;
        }
        const auto permutation = operation.attributes().find("permutation");
        if (permutation == operation.attributes().end() || !permutation->is_array()) {
            continue;
        }
        bool identity = true;
        for (std::size_t index = 0; index < permutation->size(); ++index) {
            const nlohmann::json& dimension = (*permutation)[index];
            identity = identity && dimension.is_number_integer() &&
                       dimension.get<std::int64_t>() == static_cast<std::int64_t>(index);
        }
        if (!identity) {
            continue;
        }
        const std::string& input_id = operation.input_ids().front();
        replace_all_uses(graph, output_id, input_id);
        removed_operations.insert(operation.id());
        removed_values.insert(output_id);
        rewrites.push_back(RewriteRecord{"",
                                         "",
                                         "removed an identity transpose permutation",
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
