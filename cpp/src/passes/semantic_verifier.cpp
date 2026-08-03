#include "semantic_verifier.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace forgeir {
namespace {

using Dimensions = std::vector<std::int64_t>;

class GraphContext {
  public:
    explicit GraphContext(const Graph& graph) {
        values_.reserve(graph.values().size());
        for (const Value& value : graph.values()) {
            values_.emplace(value.id(), std::cref(value));
        }
    }

    [[nodiscard]] const Value& value(const std::string& id) const { return values_.at(id).get(); }

  private:
    std::unordered_map<std::string, std::reference_wrapper<const Value>> values_;
};

Diagnostic error_diagnostic(std::string code, std::string message, const Operation& operation,
                            const std::string& value_id) {
    return Diagnostic{DiagnosticSeverity::error,
                      std::move(code),
                      std::move(message),
                      operation.id(),
                      operation.id(),
                      value_id};
}

PassResult result_from_diagnostics(std::vector<Diagnostic> diagnostics) {
    if (diagnostics.empty()) {
        return PassResult::success(false);
    }
    const std::string failure_message = diagnostics.front().message;
    return PassResult::failure(Status::error(StatusCode::failed_precondition, failure_message),
                               std::move(diagnostics));
}

bool dimensions_equal(const Shape& shape, const Dimensions& expected) {
    return shape.dimensions() == expected;
}

std::string dimensions_text(const Dimensions& dimensions) {
    std::string text{"["};
    for (std::size_t index = 0; index < dimensions.size(); ++index) {
        if (index != 0) {
            text += ',';
        }
        text += std::to_string(dimensions[index]);
    }
    text += ']';
    return text;
}

Result<std::int64_t> integer_attribute(const Operation& operation, const std::string_view name) {
    const auto iterator = operation.attributes().find(name);
    if (iterator == operation.attributes().end() ||
        (!iterator->is_number_integer() && !iterator->is_number_unsigned())) {
        return Status::error(StatusCode::invalid_argument,
                             "attribute '" + std::string(name) + "' must be an integer");
    }
    if (iterator->is_number_unsigned()) {
        const auto value = iterator->get<std::uint64_t>();
        if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
            return Status::error(StatusCode::overflow,
                                 "attribute '" + std::string(name) + "' exceeds int64_t");
        }
        return static_cast<std::int64_t>(value);
    }
    return iterator->get<std::int64_t>();
}

Result<Dimensions> integer_array_attribute(const Operation& operation,
                                           const std::string_view name) {
    const auto iterator = operation.attributes().find(name);
    if (iterator == operation.attributes().end() || !iterator->is_array()) {
        return Status::error(StatusCode::invalid_argument,
                             "attribute '" + std::string(name) + "' must be an integer array");
    }
    Dimensions values;
    values.reserve(iterator->size());
    for (const nlohmann::json& item : *iterator) {
        if (!item.is_number_integer() && !item.is_number_unsigned()) {
            return Status::error(StatusCode::invalid_argument, "attribute '" + std::string(name) +
                                                                   "' must contain only integers");
        }
        if (item.is_number_unsigned()) {
            const auto value = item.get<std::uint64_t>();
            if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
                return Status::error(StatusCode::overflow,
                                     "attribute '" + std::string(name) +
                                         "' contains a value exceeding int64_t");
            }
            values.push_back(static_cast<std::int64_t>(value));
        } else {
            values.push_back(item.get<std::int64_t>());
        }
    }
    return values;
}

Result<Dimensions> broadcast_dimensions(const Dimensions& left, const Dimensions& right) {
    const std::size_t rank = std::max(left.size(), right.size());
    Dimensions result(rank, 1);
    for (std::size_t offset = 0; offset < rank; ++offset) {
        const std::int64_t left_dimension =
            offset < left.size() ? left[left.size() - 1 - offset] : 1;
        const std::int64_t right_dimension =
            offset < right.size() ? right[right.size() - 1 - offset] : 1;
        if (left_dimension != right_dimension && left_dimension != 1 && right_dimension != 1) {
            return Status::error(StatusCode::failed_precondition,
                                 "dimensions " + std::to_string(left_dimension) + " and " +
                                     std::to_string(right_dimension) + " cannot broadcast");
        }
        result[rank - 1 - offset] = std::max(left_dimension, right_dimension);
    }
    return result;
}

Result<Dimensions> matmul_dimensions(const Dimensions& left, const Dimensions& right) {
    if (left.size() < 2 || right.size() < 2) {
        return Status::error(StatusCode::failed_precondition,
                             "MatMul inputs must each have rank at least two");
    }
    if (left.back() != right[right.size() - 2]) {
        return Status::error(StatusCode::failed_precondition,
                             "MatMul contracting dimensions do not match");
    }
    const Dimensions left_batch(left.begin(), left.end() - 2);
    const Dimensions right_batch(right.begin(), right.end() - 2);
    auto batch = broadcast_dimensions(left_batch, right_batch);
    if (!batch.ok()) {
        return batch.status();
    }
    Dimensions result = batch.take_value();
    result.push_back(left[left.size() - 2]);
    result.push_back(right.back());
    return result;
}

void validate_output_shape(const Operation& operation, const Value& output,
                           const Dimensions& expected, std::vector<Diagnostic>& diagnostics,
                           const std::string_view code = "shape.output.mismatch") {
    if (!dimensions_equal(output.descriptor().shape(), expected)) {
        diagnostics.push_back(error_diagnostic(
            std::string(code),
            "operation " + operation.id() + " output " + output.id() + " declares shape " +
                dimensions_text(output.descriptor().shape().dimensions()) +
                " but inference gives " + dimensions_text(expected),
            operation, output.id()));
    }
}

void verify_linear_shape(const Operation& operation, const GraphContext& context,
                         std::vector<Diagnostic>& diagnostics) {
    const Value& input = context.value(operation.input_ids()[0]);
    const Value& weight = context.value(operation.input_ids()[1]);
    const Value& output = context.value(operation.output_ids()[0]);
    const Dimensions& input_shape = input.descriptor().shape().dimensions();
    const Dimensions& weight_shape = weight.descriptor().shape().dimensions();
    if (input_shape.empty()) {
        diagnostics.push_back(error_diagnostic("shape.linear.input_rank",
                                               "operation " + operation.id() + " input " +
                                                   input.id() + " must have rank at least one",
                                               operation, input.id()));
        return;
    }
    if (weight_shape.size() != 2) {
        diagnostics.push_back(error_diagnostic("shape.linear.weight_rank",
                                               "operation " + operation.id() + " weight " +
                                                   weight.id() + " must have rank two",
                                               operation, weight.id()));
        return;
    }

    const auto bias_attribute = operation.attributes().find("bias");
    if (bias_attribute == operation.attributes().end() || !bias_attribute->is_boolean()) {
        diagnostics.push_back(error_diagnostic("shape.linear.bias_attribute",
                                               "operation " + operation.id() +
                                                   " must declare boolean attribute 'bias'",
                                               operation, output.id()));
        return;
    }
    const bool has_bias = bias_attribute->get<bool>();
    const std::size_t expected_operands = has_bias ? 3U : 2U;
    if (operation.input_ids().size() != expected_operands) {
        diagnostics.push_back(error_diagnostic("shape.linear.bias_operand",
                                               "operation " + operation.id() +
                                                   " bias attribute requires " +
                                                   std::to_string(expected_operands) + " operands",
                                               operation, output.id()));
        return;
    }

    const std::int64_t out_features = weight_shape[0];
    const std::int64_t in_features = weight_shape[1];
    if (input_shape.back() != in_features) {
        diagnostics.push_back(error_diagnostic(
            "shape.linear.contract",
            "operation " + operation.id() + " input " + input.id() + " final dimension " +
                std::to_string(input_shape.back()) + " does not match weight " + weight.id() +
                " input dimension " + std::to_string(in_features),
            operation, output.id()));
        return;
    }

    auto declared_in_features = integer_attribute(operation, "in_features");
    auto declared_out_features = integer_attribute(operation, "out_features");
    if (!declared_in_features.ok() || !declared_out_features.ok() ||
        declared_in_features.value() != in_features ||
        declared_out_features.value() != out_features) {
        diagnostics.push_back(
            error_diagnostic("shape.linear.feature_attributes",
                             "operation " + operation.id() +
                                 " in_features/out_features attributes must match the weight shape",
                             operation, weight.id()));
        return;
    }

    if (has_bias) {
        const Value& bias = context.value(operation.input_ids()[2]);
        if (bias.descriptor().shape().dimensions() != Dimensions{out_features}) {
            diagnostics.push_back(error_diagnostic("shape.linear.bias",
                                                   "operation " + operation.id() + " bias " +
                                                       bias.id() + " must have shape [" +
                                                       std::to_string(out_features) + "]",
                                                   operation, bias.id()));
            return;
        }
    }

    Dimensions expected = input_shape;
    expected.back() = out_features;
    validate_output_shape(operation, output, expected, diagnostics);
}

void verify_rms_norm_shape(const Operation& operation, const GraphContext& context,
                           std::vector<Diagnostic>& diagnostics) {
    const Value& input = context.value(operation.input_ids()[0]);
    const Value& weight = context.value(operation.input_ids()[1]);
    const Value& output = context.value(operation.output_ids()[0]);
    const Dimensions& input_shape = input.descriptor().shape().dimensions();
    if (input_shape.empty()) {
        diagnostics.push_back(error_diagnostic("shape.rms_norm.input_rank",
                                               "operation " + operation.id() + " input " +
                                                   input.id() + " must have a final dimension",
                                               operation, input.id()));
        return;
    }
    if (weight.descriptor().shape().dimensions() != Dimensions{input_shape.back()}) {
        diagnostics.push_back(error_diagnostic("shape.rms_norm.weight",
                                               "operation " + operation.id() + " weight " +
                                                   weight.id() +
                                                   " must match the input final dimension",
                                               operation, weight.id()));
        return;
    }
    auto axis = integer_attribute(operation, "axis");
    const auto rank = static_cast<std::int64_t>(input_shape.size());
    if (!axis.ok() || (axis.value() != -1 && axis.value() != rank - 1)) {
        diagnostics.push_back(error_diagnostic("shape.rms_norm.axis",
                                               "operation " + operation.id() +
                                                   " RMSNorm axis must select the final dimension",
                                               operation, output.id()));
        return;
    }
    const auto epsilon = operation.attributes().find("epsilon");
    if (epsilon == operation.attributes().end() || !epsilon->is_number() ||
        epsilon->get<double>() <= 0.0) {
        diagnostics.push_back(
            error_diagnostic("shape.rms_norm.epsilon",
                             "operation " + operation.id() + " RMSNorm epsilon must be positive",
                             operation, output.id()));
        return;
    }
    validate_output_shape(operation, output, input_shape, diagnostics);
}

void verify_reshape_shape(const Operation& operation, const GraphContext& context,
                          std::vector<Diagnostic>& diagnostics) {
    const Value& input = context.value(operation.input_ids()[0]);
    const Value& output = context.value(operation.output_ids()[0]);
    auto target = integer_array_attribute(operation, "shape");
    if (!target.ok()) {
        diagnostics.push_back(error_diagnostic(
            "shape.reshape.attribute",
            "operation " + operation.id() +
                " has invalid explicit reshape dimensions: " + target.status().message(),
            operation, output.id()));
        return;
    }
    if (std::any_of(target.value().begin(), target.value().end(),
                    [](const std::int64_t dimension) { return dimension <= 0; })) {
        diagnostics.push_back(error_diagnostic(
            "shape.reshape.ambiguous",
            "operation " + operation.id() +
                " reshape dimensions must be explicit and positive; inferred dimensions are not "
                "accepted",
            operation, output.id()));
        return;
    }
    const auto target_shape = Shape::create(target.value());
    const auto input_elements = input.descriptor().element_count();
    const auto target_elements = target_shape.ok() ? target_shape.value().element_count()
                                                   : Result<std::uint64_t>(target_shape.status());
    if (!input_elements.ok() || !target_elements.ok() ||
        input_elements.value() != target_elements.value()) {
        diagnostics.push_back(error_diagnostic("shape.reshape.element_count",
                                               "operation " + operation.id() + " reshape from " +
                                                   input.id() + " to " + output.id() +
                                                   " must preserve element count",
                                               operation, output.id()));
        return;
    }
    validate_output_shape(operation, output, target.value(), diagnostics);
}

void verify_transpose_shape(const Operation& operation, const GraphContext& context,
                            std::vector<Diagnostic>& diagnostics) {
    const Value& input = context.value(operation.input_ids()[0]);
    const Value& output = context.value(operation.output_ids()[0]);
    const Dimensions& input_shape = input.descriptor().shape().dimensions();
    auto permutation = integer_array_attribute(operation, "permutation");
    if (!permutation.ok() || permutation.value().size() != input_shape.size()) {
        diagnostics.push_back(error_diagnostic(
            "shape.transpose.rank",
            "operation " + operation.id() + " permutation rank must match input " + input.id(),
            operation, output.id()));
        return;
    }
    std::unordered_set<std::int64_t> seen;
    Dimensions expected;
    expected.reserve(input_shape.size());
    for (const std::int64_t index : permutation.value()) {
        if (index < 0 || index >= static_cast<std::int64_t>(input_shape.size()) ||
            !seen.insert(index).second) {
            diagnostics.push_back(
                error_diagnostic("shape.transpose.permutation",
                                 "operation " + operation.id() +
                                     " permutation must contain every input dimension exactly once",
                                 operation, output.id()));
            return;
        }
        expected.push_back(input_shape[static_cast<std::size_t>(index)]);
    }
    validate_output_shape(operation, output, expected, diagnostics);
}

void verify_operation_shape(const Operation& operation, const GraphContext& context,
                            std::vector<Diagnostic>& diagnostics) {
    const Value& output = context.value(operation.output_ids()[0]);
    switch (operation.type()) {
    case OperationType::input:
    case OperationType::parameter:
    case OperationType::constant:
        return;
    case OperationType::mat_mul: {
        const Value& left = context.value(operation.input_ids()[0]);
        const Value& right = context.value(operation.input_ids()[1]);
        auto expected = matmul_dimensions(left.descriptor().shape().dimensions(),
                                          right.descriptor().shape().dimensions());
        if (!expected.ok()) {
            diagnostics.push_back(error_diagnostic(
                "shape.matmul.contract",
                "operation " + operation.id() + " with operands " + left.id() + " and " +
                    right.id() + " is invalid: " + expected.status().message(),
                operation, output.id()));
            return;
        }
        if (operation.input_ids().size() == 3) {
            const Value& bias = context.value(operation.input_ids()[2]);
            const auto bias_attribute = operation.attributes().find("bias");
            if (bias_attribute == operation.attributes().end() || !bias_attribute->is_boolean() ||
                !bias_attribute->get<bool>() ||
                bias.descriptor().shape().dimensions() != Dimensions{expected.value().back()}) {
                diagnostics.push_back(
                    error_diagnostic("shape.matmul.bias",
                                     "operation " + operation.id() + " fused bias " + bias.id() +
                                         " must be rank one and match the MatMul final dimension",
                                     operation, bias.id()));
                return;
            }
        }
        validate_output_shape(operation, output, expected.value(), diagnostics);
        return;
    }
    case OperationType::linear:
        verify_linear_shape(operation, context, diagnostics);
        return;
    case OperationType::add:
    case OperationType::multiply:
    case OperationType::divide: {
        const Value& left = context.value(operation.input_ids()[0]);
        const Value& right = context.value(operation.input_ids()[1]);
        auto expected = broadcast_dimensions(left.descriptor().shape().dimensions(),
                                             right.descriptor().shape().dimensions());
        if (!expected.ok()) {
            diagnostics.push_back(error_diagnostic(
                "shape.broadcast.incompatible",
                "operation " + operation.id() + " operands " + left.id() + " and " + right.id() +
                    " cannot broadcast: " + expected.status().message(),
                operation, output.id()));
            return;
        }
        validate_output_shape(operation, output, expected.value(), diagnostics);
        return;
    }
    case OperationType::rms_norm:
        verify_rms_norm_shape(operation, context, diagnostics);
        return;
    case OperationType::gelu:
        validate_output_shape(
            operation, output,
            context.value(operation.input_ids()[0]).descriptor().shape().dimensions(), diagnostics);
        return;
    case OperationType::softmax: {
        const Value& input = context.value(operation.input_ids()[0]);
        const auto rank = static_cast<std::int64_t>(input.descriptor().shape().rank());
        auto axis = integer_attribute(operation, "axis");
        if (rank == 0 || !axis.ok() || axis.value() < -rank || axis.value() >= rank) {
            diagnostics.push_back(error_diagnostic("shape.softmax.axis",
                                                   "operation " + operation.id() +
                                                       " Softmax axis is outside input " +
                                                       input.id() + " rank",
                                                   operation, output.id()));
            return;
        }
        validate_output_shape(operation, output, input.descriptor().shape().dimensions(),
                              diagnostics);
        return;
    }
    case OperationType::reshape:
        verify_reshape_shape(operation, context, diagnostics);
        return;
    case OperationType::transpose:
        verify_transpose_shape(operation, context, diagnostics);
        return;
    case OperationType::causal_mask: {
        const Value& input = context.value(operation.input_ids()[0]);
        const Dimensions& dimensions = input.descriptor().shape().dimensions();
        if (dimensions.size() < 2 || dimensions[dimensions.size() - 2] != dimensions.back()) {
            diagnostics.push_back(error_diagnostic("shape.causal_mask.attention",
                                                   "operation " + operation.id() +
                                                       " attention scores " + input.id() +
                                                       " must have equal query and key dimensions",
                                                   operation, output.id()));
            return;
        }
        validate_output_shape(operation, output, dimensions, diagnostics);
        return;
    }
    }
}

bool is_numeric(const DataType type) {
    return type == DataType::float32 || type == DataType::int64;
}

void require_same_dtype(const Operation& operation, const GraphContext& context,
                        const bool require_float, std::vector<Diagnostic>& diagnostics) {
    const Value& output = context.value(operation.output_ids()[0]);
    const DataType expected = context.value(operation.input_ids()[0]).descriptor().data_type();
    for (const std::string& input_id : operation.input_ids()) {
        if (context.value(input_id).descriptor().data_type() != expected) {
            diagnostics.push_back(error_diagnostic("dtype.inputs.mismatch",
                                                   "operation " + operation.id() + " input " +
                                                       input_id +
                                                       " dtype does not match its other operands",
                                                   operation, output.id()));
            return;
        }
    }
    if ((require_float && expected != DataType::float32) ||
        (!require_float && !is_numeric(expected))) {
        diagnostics.push_back(error_diagnostic("dtype.operation.unsupported",
                                               "operation " + operation.id() +
                                                   " does not support dtype " +
                                                   std::string(to_string(expected)),
                                               operation, output.id()));
        return;
    }
    if (output.descriptor().data_type() != expected) {
        diagnostics.push_back(error_diagnostic(
            "dtype.output.mismatch",
            "operation " + operation.id() + " output " + output.id() + " declares dtype " +
                std::string(to_string(output.descriptor().data_type())) +
                " but propagation gives " + std::string(to_string(expected)),
            operation, output.id()));
    }
}

void verify_constant_dtype(const Operation& operation, const GraphContext& context,
                           std::vector<Diagnostic>& diagnostics) {
    const Value& output = context.value(operation.output_ids()[0]);
    const auto value = operation.attributes().find("value");
    if (value == operation.attributes().end()) {
        diagnostics.push_back(error_diagnostic(
            "dtype.constant.value", "operation " + operation.id() + " has no constant value",
            operation, output.id()));
        return;
    }
    DataType expected = DataType::boolean;
    if (value->is_boolean()) {
        expected = DataType::boolean;
    } else if (value->is_number_integer() || value->is_number_unsigned()) {
        expected = DataType::int64;
    } else if (value->is_number_float()) {
        expected = DataType::float32;
    } else {
        diagnostics.push_back(
            error_diagnostic("dtype.constant.ambiguous",
                             "operation " + operation.id() +
                                 " constant dtype cannot be inferred unambiguously from its value",
                             operation, output.id()));
        return;
    }
    if (output.descriptor().shape().rank() != 0) {
        diagnostics.push_back(error_diagnostic("dtype.constant.shape",
                                               "operation " + operation.id() +
                                                   " scalar constant output " + output.id() +
                                                   " must have scalar shape",
                                               operation, output.id()));
        return;
    }
    if (output.descriptor().data_type() != expected) {
        diagnostics.push_back(error_diagnostic("dtype.constant.mismatch",
                                               "operation " + operation.id() + " output " +
                                                   output.id() +
                                                   " dtype does not match the constant value",
                                               operation, output.id()));
    }
}

void verify_operation_dtype(const Operation& operation, const GraphContext& context,
                            std::vector<Diagnostic>& diagnostics) {
    switch (operation.type()) {
    case OperationType::input:
    case OperationType::parameter:
        return;
    case OperationType::constant:
        verify_constant_dtype(operation, context, diagnostics);
        return;
    case OperationType::mat_mul:
    case OperationType::linear:
    case OperationType::add:
    case OperationType::multiply:
        require_same_dtype(operation, context, false, diagnostics);
        return;
    case OperationType::divide:
    case OperationType::rms_norm:
    case OperationType::gelu:
    case OperationType::softmax:
    case OperationType::causal_mask:
        require_same_dtype(operation, context, true, diagnostics);
        return;
    case OperationType::reshape:
    case OperationType::transpose: {
        const Value& input = context.value(operation.input_ids()[0]);
        const Value& output = context.value(operation.output_ids()[0]);
        if (output.descriptor().data_type() != input.descriptor().data_type()) {
            diagnostics.push_back(error_diagnostic("dtype.output.mismatch",
                                                   "operation " + operation.id() + " output " +
                                                       output.id() + " must preserve input " +
                                                       input.id() + " dtype",
                                                   operation, output.id()));
        }
        return;
    }
    }
}

} // namespace

PassResult verify_graph_shapes(const Graph& graph) {
    const GraphContext context(graph);
    std::vector<Diagnostic> diagnostics;
    for (const Operation& operation : graph.operations()) {
        verify_operation_shape(operation, context, diagnostics);
    }
    return result_from_diagnostics(std::move(diagnostics));
}

PassResult verify_graph_dtypes(const Graph& graph) {
    const GraphContext context(graph);
    std::vector<Diagnostic> diagnostics;
    for (const Operation& operation : graph.operations()) {
        verify_operation_dtype(operation, context, diagnostics);
    }
    return result_from_diagnostics(std::move(diagnostics));
}

} // namespace forgeir
