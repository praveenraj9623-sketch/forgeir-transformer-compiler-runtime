#include "forgeir/passes/canonicalisation_pass.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "rewrite_utils.hpp"

namespace forgeir {

std::string_view CanonicalisationPass::name() const noexcept { return "CanonicalisationPass"; }

PassResult CanonicalisationPass::run(Graph& graph) {
    std::vector<RewriteRecord> rewrites;
    const auto values = value_index_by_id(graph);
    for (Operation& operation : graph.mutable_operations()) {
        nlohmann::json& attributes = operation.mutable_attributes();
        bool modified = false;
        std::string reason;

        if (operation.type() == OperationType::softmax ||
            operation.type() == OperationType::rms_norm) {
            auto axis = attributes.find("axis");
            if (axis != attributes.end() && axis->is_number_integer()) {
                const std::int64_t original = axis->get<std::int64_t>();
                if (original < 0) {
                    const std::string& input_id = operation.input_ids().front();
                    const auto value = values.find(input_id);
                    if (value != values.end()) {
                        const auto rank = static_cast<std::int64_t>(
                            graph.values()[value->second].descriptor().shape().rank());
                        *axis = original + rank;
                        modified = true;
                        reason = "normalised negative axis to its nonnegative canonical index";
                    }
                }
            }
        }
        if (operation.type() == OperationType::gelu && !attributes.contains("approximate")) {
            attributes["approximate"] = "none";
            modified = true;
            reason = "materialised the canonical exact GELU attribute";
        }
        if (operation.type() == OperationType::softmax && !attributes.contains("stable")) {
            attributes["stable"] = true;
            modified = true;
            reason = "materialised the canonical stable Softmax attribute";
        }
        if (operation.type() == OperationType::causal_mask && !attributes.contains("diagonal")) {
            attributes["diagonal"] = 0;
            modified = true;
            reason = "materialised the canonical causal-mask diagonal";
        }
        const auto side_effect = attributes.find("side_effect");
        if (side_effect != attributes.end() && side_effect->is_boolean() &&
            !side_effect->get<bool>()) {
            attributes.erase(side_effect);
            modified = true;
            reason = "removed a redundant false side-effect marker";
        }
        if (modified) {
            rewrites.push_back(
                RewriteRecord{"", "", std::move(reason), {}, {}, {operation.id()}, {}});
        }
    }

    auto ordering = stable_topological_order(graph);
    if (!ordering.ok()) {
        const Operation& operation = graph.operations().front();
        std::vector<Diagnostic> diagnostics{Diagnostic{
            DiagnosticSeverity::error, "canonicalisation.order", ordering.status().message(),
            operation.id(), operation.id(), operation.output_ids().front()}};
        return PassResult::failure(ordering.status(), std::move(diagnostics), std::move(rewrites));
    }
    if (ordering.value()) {
        std::vector<std::string> modified;
        modified.reserve(graph.operations().size());
        for (const Operation& operation : graph.operations()) {
            modified.push_back(operation.id());
        }
        rewrites.push_back(RewriteRecord{"",
                                         "",
                                         "restored stable topological operation order",
                                         {},
                                         {},
                                         std::move(modified),
                                         {}});
    }
    const bool changed = !rewrites.empty();
    return PassResult::success(changed, {}, std::move(rewrites));
}

} // namespace forgeir
