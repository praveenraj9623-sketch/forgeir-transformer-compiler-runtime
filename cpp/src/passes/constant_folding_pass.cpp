#include "forgeir/passes/constant_folding_pass.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rewrite_utils.hpp"

namespace forgeir {
namespace {

std::optional<nlohmann::json> fold_scalar(const Operation& operation,
                                          const std::vector<nlohmann::json>& operands) {
    const auto number = [](const nlohmann::json& value) {
        return value.is_number_integer() || value.is_number_unsigned() || value.is_number_float();
    };
    if (operation.type() == OperationType::gelu && operands.size() == 1 &&
        operands.front().is_number()) {
        const double value = operands.front().get<double>();
        return nlohmann::json(0.5 * value * (1.0 + std::erf(value / std::sqrt(2.0))));
    }
    if (operands.size() != 2 || !number(operands[0]) || !number(operands[1])) {
        return std::nullopt;
    }
    const bool integral = operands[0].is_number_integer() && operands[1].is_number_integer();
    if (operation.type() == OperationType::add) {
        if (integral) {
            const auto left = operands[0].get<std::int64_t>();
            const auto right = operands[1].get<std::int64_t>();
            const long double result = static_cast<long double>(left) + right;
            if (result < static_cast<long double>(std::numeric_limits<std::int64_t>::min()) ||
                result > static_cast<long double>(std::numeric_limits<std::int64_t>::max())) {
                return std::nullopt;
            }
            return nlohmann::json(static_cast<std::int64_t>(result));
        }
        return nlohmann::json(operands[0].get<double>() + operands[1].get<double>());
    }
    if (operation.type() == OperationType::multiply) {
        if (integral) {
            const auto left = operands[0].get<std::int64_t>();
            const auto right = operands[1].get<std::int64_t>();
            const long double result = static_cast<long double>(left) * right;
            if (result < static_cast<long double>(std::numeric_limits<std::int64_t>::min()) ||
                result > static_cast<long double>(std::numeric_limits<std::int64_t>::max())) {
                return std::nullopt;
            }
            return nlohmann::json(static_cast<std::int64_t>(result));
        }
        return nlohmann::json(operands[0].get<double>() * operands[1].get<double>());
    }
    if (operation.type() == OperationType::divide) {
        const double divisor = operands[1].get<double>();
        if (divisor == 0.0) {
            return std::nullopt;
        }
        return nlohmann::json(operands[0].get<double>() / divisor);
    }
    return std::nullopt;
}

} // namespace

std::string_view ConstantFoldingPass::name() const noexcept { return "ConstantFoldingPass"; }

PassResult ConstantFoldingPass::run(Graph& graph) {
    std::vector<RewriteRecord> rewrites;
    auto producers = operation_index_by_output(graph);
    auto values = value_index_by_id(graph);

    for (Operation& operation : graph.mutable_operations()) {
        if (operation.type() != OperationType::add && operation.type() != OperationType::multiply &&
            operation.type() != OperationType::divide && operation.type() != OperationType::gelu) {
            continue;
        }
        if (has_side_effect(operation)) {
            continue;
        }
        const auto output_value = values.find(operation.output_ids().front());
        if (output_value == values.end()) {
            continue;
        }
        const auto element_count =
            graph.values()[output_value->second].descriptor().element_count();
        if (!element_count.ok() || element_count.value() > kMaximumFoldedTensorElements ||
            element_count.value() != 1 ||
            graph.values()[output_value->second].descriptor().shape().rank() != 0) {
            continue;
        }

        std::vector<nlohmann::json> operands;
        bool all_constant = true;
        for (const std::string& input : operation.input_ids()) {
            const auto producer = producers.find(input);
            if (producer == producers.end()) {
                all_constant = false;
                break;
            }
            const Operation& constant = graph.operations()[producer->second];
            const auto value = constant.attributes().find("value");
            if (constant.type() != OperationType::constant ||
                value == constant.attributes().end() || value->is_array() || value->is_object()) {
                all_constant = false;
                break;
            }
            operands.push_back(*value);
        }
        if (!all_constant) {
            continue;
        }
        const auto folded = fold_scalar(operation, operands);
        if (!folded.has_value()) {
            continue;
        }

        const std::string operation_id = operation.id();
        const std::string original_type(to_string(operation.type()));
        operation.set_type(OperationType::constant);
        operation.set_input_ids({});
        operation.set_attributes(
            nlohmann::json{{"value", folded.value()},
                           {"folded_from", original_type},
                           {"maximum_folded_elements", kMaximumFoldedTensorElements}});
        Value& output = graph.mutable_values()[output_value->second];
        if (!is_declared_output(graph, output.id())) {
            output.set_kind(ValueKind::constant);
        }
        rewrites.push_back(RewriteRecord{
            "", "", "folded a bounded scalar constant expression", {}, {}, {operation_id}, {}});
        producers = operation_index_by_output(graph);
    }
    const bool changed = !rewrites.empty();
    return PassResult::success(changed, {}, std::move(rewrites));
}

} // namespace forgeir
