#include "forgeir/passes/fuse_bias_gelu_pass.hpp"

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "rewrite_utils.hpp"

namespace forgeir {
namespace {

bool exact_gelu(const Operation& operation) {
    const auto approximate = operation.attributes().find("approximate");
    return approximate == operation.attributes().end() ||
           (approximate->is_string() && approximate->get<std::string>() == "none");
}

} // namespace

std::string_view FuseBiasGELUPass::name() const noexcept { return "FuseBiasGELUPass"; }

PassResult FuseBiasGELUPass::run(Graph& graph) {
    const auto producers = operation_index_by_output(graph);
    const auto values = value_index_by_id(graph);
    const auto uses = value_use_counts(graph);
    std::unordered_set<std::string> removed_operations;
    std::unordered_set<std::string> removed_values;
    std::vector<RewriteRecord> rewrites;

    for (std::size_t gelu_index = 0; gelu_index < graph.operations().size(); ++gelu_index) {
        const Operation& gelu = graph.operations()[gelu_index];
        if (gelu.type() != OperationType::gelu || gelu.input_ids().size() != 1 ||
            has_side_effect(gelu) || !exact_gelu(gelu)) {
            continue;
        }
        const std::string add_output = gelu.input_ids().front();
        const auto add_producer = producers.find(add_output);
        if (add_producer == producers.end()) {
            continue;
        }
        const Operation& add = graph.operations()[add_producer->second];
        if (add.type() != OperationType::add || add.input_ids().size() != 2 ||
            has_side_effect(add) || uses.find(add_output) == uses.end() ||
            uses.at(add_output) != 1 || is_declared_output(graph, add_output)) {
            continue;
        }

        std::size_t base_index = 0;
        std::string bias_id;
        std::string base_output;
        bool matched_base = false;
        for (std::size_t input_index = 0; input_index < 2; ++input_index) {
            const std::string& candidate_output = add.input_ids()[input_index];
            const auto base_producer = producers.find(candidate_output);
            if (base_producer == producers.end()) {
                continue;
            }
            const Operation& candidate = graph.operations()[base_producer->second];
            if ((candidate.type() == OperationType::linear ||
                 candidate.type() == OperationType::mat_mul) &&
                candidate.input_ids().size() == 2 && !has_side_effect(candidate)) {
                base_index = base_producer->second;
                base_output = candidate_output;
                bias_id = add.input_ids()[1 - input_index];
                matched_base = true;
                break;
            }
        }
        if (!matched_base || uses.find(base_output) == uses.end() || uses.at(base_output) != 1 ||
            is_declared_output(graph, base_output)) {
            continue;
        }
        const auto bias_value = values.find(bias_id);
        const auto add_value = values.find(add_output);
        if (bias_value == values.end() || add_value == values.end()) {
            continue;
        }
        const Value& bias = graph.values()[bias_value->second];
        const auto& output_shape =
            graph.values()[add_value->second].descriptor().shape().dimensions();
        if ((bias.kind() != ValueKind::parameter && bias.kind() != ValueKind::constant) ||
            output_shape.empty() ||
            bias.descriptor().shape().dimensions() !=
                std::vector<std::int64_t>{output_shape.back()}) {
            continue;
        }

        const Operation& base = graph.operations()[base_index];
        if (base.type() == OperationType::linear) {
            const auto bias_attribute = base.attributes().find("bias");
            if (bias_attribute == base.attributes().end() || !bias_attribute->is_boolean() ||
                bias_attribute->get<bool>()) {
                continue;
            }
        }

        std::vector<std::string> fused_inputs = base.input_ids();
        fused_inputs.push_back(bias_id);
        nlohmann::json fused_attributes = base.attributes();
        fused_attributes["bias"] = true;
        fused_attributes["fused_activation"] = "GELU";
        fused_attributes["fused_activation_approximate"] = "none";
        fused_attributes["fused_from"] = {base.id(), add.id(), gelu.id()};

        Operation& fused = graph.mutable_operations()[gelu_index];
        fused.set_type(base.type());
        fused.set_input_ids(std::move(fused_inputs));
        fused.set_attributes(std::move(fused_attributes));

        removed_operations.insert(base.id());
        removed_operations.insert(add.id());
        removed_values.insert(base_output);
        removed_values.insert(add_output);
        rewrites.push_back(RewriteRecord{
            "",
            "",
            "fused a single-use projection, rank-one bias Add, and exact GELU while preserving the "
            "terminal GELU output value",
            {},
            {base.id(), add.id()},
            {gelu.id()},
            {base_output, add_output}});
    }

    erase_operations(graph, removed_operations);
    erase_values(graph, removed_values);
    const bool changed = !rewrites.empty();
    return PassResult::success(changed, {}, std::move(rewrites));
}

} // namespace forgeir
