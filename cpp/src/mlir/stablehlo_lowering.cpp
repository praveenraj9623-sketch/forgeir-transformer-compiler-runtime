#include "forgeir/mlir/stablehlo_lowering.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "forgeir/passes/verification.hpp"

namespace forgeir {
namespace {

using Dimensions = std::vector<std::int64_t>;

std::string join(const std::vector<std::string>& values, const std::string_view separator) {
    std::ostringstream stream;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            stream << separator;
        }
        stream << values[index];
    }
    return stream.str();
}

std::string integer_list(const Dimensions& values) {
    std::vector<std::string> text;
    text.reserve(values.size());
    for (const std::int64_t value : values) {
        text.push_back(std::to_string(value));
    }
    return join(text, ", ");
}

std::string array_attribute(const Dimensions& values) {
    if (values.empty()) {
        return "array<i64>";
    }
    return "array<i64: " + integer_list(values) + ">";
}

std::string mlir_element_type(const DataType data_type) {
    switch (data_type) {
    case DataType::boolean:
        return "i1";
    case DataType::float32:
        return "f32";
    case DataType::int64:
        return "i64";
    }
    return "invalid";
}

std::string ranked_tensor_type(const DataType data_type, const Dimensions& dimensions) {
    std::ostringstream stream;
    stream << "tensor<";
    for (const std::int64_t dimension : dimensions) {
        stream << dimension << 'x';
    }
    stream << mlir_element_type(data_type) << '>';
    return stream.str();
}

std::string ranked_tensor_type(const Value& value) {
    return ranked_tensor_type(value.descriptor().data_type(),
                              value.descriptor().shape().dimensions());
}

std::string float_literal(const double value) {
    std::ostringstream stream;
    stream << std::scientific << std::setprecision(9) << value;
    return stream.str();
}

class StablehloLowerer {
  public:
    explicit StablehloLowerer(const Graph& graph) : graph_(graph) {
        values_.reserve(graph.values().size());
        for (const Value& value : graph.values()) {
            values_.emplace(value.id(), &value);
        }
    }

    MlirLoweringResult run() {
        emit_header();
        for (const Operation& operation : graph_.operations()) {
            emit_comment(operation);
            if (!emit_operation(operation)) {
                return {failure_status_, {}, std::move(diagnostics_)};
            }
        }
        emit_return();
        stream_ << "  }\n}\n";
        return {Status::ok_status(), stream_.str(), {}};
    }

  private:
    const Value& value(const std::string& id) const { return *values_.at(id); }

    std::string ssa(const std::string& value_id) const { return '%' + value_id; }

    std::string temporary() { return "%forgeir" + std::to_string(next_temporary_++); }

    void emit_header() {
        std::vector<std::string> arguments;
        for (const Operation& operation : graph_.operations()) {
            if (operation.type() != OperationType::input &&
                operation.type() != OperationType::parameter) {
                continue;
            }
            for (const std::string& output_id : operation.output_ids()) {
                arguments.push_back(ssa(output_id) + ": " + ranked_tensor_type(value(output_id)));
            }
        }

        std::vector<std::string> output_types;
        output_types.reserve(graph_.output_ids().size());
        for (const std::string& output_id : graph_.output_ids()) {
            output_types.push_back(ranked_tensor_type(value(output_id)));
        }

        stream_ << "// ForgeIR StableHLO textual bridge\n"
                << "// graph_schema_version = " << graph_.schema_version() << "\n"
                << "// graph_hash = " << graph_.graph_hash() << "\n"
                << "module {\n"
                << "  func.func @main(" << join(arguments, ", ") << ')';
        if (output_types.size() == 1) {
            stream_ << " -> " << output_types.front();
        } else if (!output_types.empty()) {
            stream_ << " -> (" << join(output_types, ", ") << ')';
        }
        stream_ << " {\n";
    }

    void emit_comment(const Operation& operation) {
        stream_ << "    // forgeir.op_id = " << operation.id()
                << "; forgeir.operation = " << to_string(operation.type()) << "\n";
    }

    void emit_return() {
        std::vector<std::string> operands;
        std::vector<std::string> types;
        operands.reserve(graph_.output_ids().size());
        types.reserve(graph_.output_ids().size());
        for (const std::string& output_id : graph_.output_ids()) {
            operands.push_back(ssa(output_id));
            types.push_back(ranked_tensor_type(value(output_id)));
        }
        stream_ << "    \"func.return\"(" << join(operands, ", ") << ") : (" << join(types, ", ")
                << ") -> ()\n";
    }

    bool fail(const Operation& operation, const std::string& code, const std::string& message,
              const std::string& value_id) {
        failure_status_ = Status::error(StatusCode::unsupported, message);
        diagnostics_.push_back(Diagnostic{DiagnosticSeverity::error, code, message, operation.id(),
                                          operation.id(), value_id});
        return false;
    }

    void emit_generic(const std::string& result, const std::string_view operation_name,
                      const std::vector<std::string>& operands,
                      const std::vector<std::string>& operand_types, const std::string& result_type,
                      const std::string& attributes = {}) {
        stream_ << "    " << result << " = \"" << operation_name << "\"(" << join(operands, ", ")
                << ')';
        if (!attributes.empty()) {
            stream_ << " {" << attributes << '}';
        }
        stream_ << " : (" << join(operand_types, ", ") << ") -> " << result_type << "\n";
    }

    std::string emit_dense_constant(const std::string& literal, const DataType data_type,
                                    const Dimensions& dimensions) {
        const std::string result = temporary();
        const std::string type = ranked_tensor_type(data_type, dimensions);
        emit_generic(result, "stablehlo.constant", {}, {}, type,
                     "value = dense<" + literal + "> : " + type);
        return result;
    }

    std::string emit_float_constant(const double literal, const Dimensions& dimensions) {
        return emit_dense_constant(float_literal(literal), DataType::float32, dimensions);
    }

    std::string
    emit_broadcast(const std::string& operand, const DataType data_type,
                   const Dimensions& input_dimensions, const Dimensions& output_dimensions,
                   const std::optional<Dimensions>& explicit_dimensions = std::nullopt) {
        if (input_dimensions == output_dimensions) {
            return operand;
        }
        Dimensions dimensions;
        if (explicit_dimensions.has_value()) {
            dimensions = explicit_dimensions.value();
        } else {
            const std::size_t offset = output_dimensions.size() - input_dimensions.size();
            dimensions.reserve(input_dimensions.size());
            for (std::size_t index = 0; index < input_dimensions.size(); ++index) {
                dimensions.push_back(static_cast<std::int64_t>(offset + index));
            }
        }
        const std::string result = temporary();
        emit_generic(result, "stablehlo.broadcast_in_dim", {operand},
                     {ranked_tensor_type(data_type, input_dimensions)},
                     ranked_tensor_type(data_type, output_dimensions),
                     "broadcast_dimensions = " + array_attribute(dimensions));
        return result;
    }

    void emit_binary(const std::string& result, const std::string_view operation_name,
                     const std::string& left, const std::string& right, const DataType data_type,
                     const Dimensions& dimensions) {
        const std::string type = ranked_tensor_type(data_type, dimensions);
        emit_generic(result, operation_name, {left, right}, {type, type}, type);
    }

    std::string emit_reduce(const std::string& input, const std::string& initial_value,
                            const DataType data_type, const Dimensions& input_dimensions,
                            const Dimensions& result_dimensions, const Dimensions& axes,
                            const std::string_view reducer) {
        const std::string result = temporary();
        const std::string input_type = ranked_tensor_type(data_type, input_dimensions);
        const std::string scalar_type = ranked_tensor_type(data_type, {});
        const std::string result_type = ranked_tensor_type(data_type, result_dimensions);
        const std::string reduced = temporary();
        stream_ << "    " << result << " = \"stablehlo.reduce\"(" << input << ", " << initial_value
                << ") ({\n"
                << "    ^bb0(%lhs: " << scalar_type << ", %rhs: " << scalar_type << "):\n"
                << "      " << reduced << " = \"stablehlo." << reducer << "\"(%lhs, %rhs) : ("
                << scalar_type << ", " << scalar_type << ") -> " << scalar_type << "\n"
                << "      \"stablehlo.return\"(" << reduced << ") : (" << scalar_type << ") -> ()\n"
                << "    }) {dimensions = " << array_attribute(axes) << "} : (" << input_type << ", "
                << scalar_type << ") -> " << result_type << "\n";
        return result;
    }

    bool emit_elementwise(const Operation& operation, const std::string_view operation_name) {
        const Value& left_value = value(operation.input_ids()[0]);
        const Value& right_value = value(operation.input_ids()[1]);
        const Value& output_value = value(operation.output_ids()[0]);
        const Dimensions& output_dimensions = output_value.descriptor().shape().dimensions();
        const DataType data_type = output_value.descriptor().data_type();
        const std::string left =
            emit_broadcast(ssa(left_value.id()), data_type,
                           left_value.descriptor().shape().dimensions(), output_dimensions);
        const std::string right =
            emit_broadcast(ssa(right_value.id()), data_type,
                           right_value.descriptor().shape().dimensions(), output_dimensions);
        emit_binary(ssa(output_value.id()), operation_name, left, right, data_type,
                    output_dimensions);
        return true;
    }

    bool emit_matmul(const Operation& operation, const bool linear) {
        const Value& left_value = value(operation.input_ids()[0]);
        const Value& right_value = value(operation.input_ids()[1]);
        const Value& output_value = value(operation.output_ids()[0]);
        const DataType data_type = output_value.descriptor().data_type();
        const Dimensions& left_dimensions = left_value.descriptor().shape().dimensions();
        const Dimensions& right_dimensions = right_value.descriptor().shape().dimensions();
        const Dimensions& output_dimensions = output_value.descriptor().shape().dimensions();

        std::string left = ssa(left_value.id());
        std::string right = ssa(right_value.id());
        Dimensions normalized_left = left_dimensions;
        Dimensions normalized_right = right_dimensions;
        Dimensions lhs_batching;
        Dimensions rhs_batching;
        Dimensions lhs_contracting;
        Dimensions rhs_contracting;

        if (linear) {
            normalized_right = {right_dimensions[1], right_dimensions[0]};
            right = temporary();
            emit_generic(right, "stablehlo.transpose", {ssa(right_value.id())},
                         {ranked_tensor_type(data_type, right_dimensions)},
                         ranked_tensor_type(data_type, normalized_right),
                         "permutation = " + array_attribute({1, 0}));
            lhs_contracting = {static_cast<std::int64_t>(left_dimensions.size() - 1)};
            rhs_contracting = {0};
        } else {
            const std::size_t batch_rank = output_dimensions.size() - 2;
            Dimensions target_left(output_dimensions.begin(),
                                   output_dimensions.begin() +
                                       static_cast<std::ptrdiff_t>(batch_rank));
            target_left.push_back(left_dimensions[left_dimensions.size() - 2]);
            target_left.push_back(left_dimensions.back());
            Dimensions target_right(output_dimensions.begin(),
                                    output_dimensions.begin() +
                                        static_cast<std::ptrdiff_t>(batch_rank));
            target_right.push_back(right_dimensions[right_dimensions.size() - 2]);
            target_right.push_back(right_dimensions.back());

            Dimensions left_mapping;
            const std::size_t left_batch_rank = left_dimensions.size() - 2;
            for (std::size_t index = 0; index < left_batch_rank; ++index) {
                left_mapping.push_back(
                    static_cast<std::int64_t>(batch_rank - left_batch_rank + index));
            }
            left_mapping.push_back(static_cast<std::int64_t>(batch_rank));
            left_mapping.push_back(static_cast<std::int64_t>(batch_rank + 1));
            Dimensions right_mapping;
            const std::size_t right_batch_rank = right_dimensions.size() - 2;
            for (std::size_t index = 0; index < right_batch_rank; ++index) {
                right_mapping.push_back(
                    static_cast<std::int64_t>(batch_rank - right_batch_rank + index));
            }
            right_mapping.push_back(static_cast<std::int64_t>(batch_rank));
            right_mapping.push_back(static_cast<std::int64_t>(batch_rank + 1));

            left = emit_broadcast(left, data_type, left_dimensions, target_left, left_mapping);
            right = emit_broadcast(right, data_type, right_dimensions, target_right, right_mapping);
            normalized_left = std::move(target_left);
            normalized_right = std::move(target_right);
            for (std::size_t index = 0; index < batch_rank; ++index) {
                lhs_batching.push_back(static_cast<std::int64_t>(index));
                rhs_batching.push_back(static_cast<std::int64_t>(index));
            }
            lhs_contracting = {static_cast<std::int64_t>(batch_rank + 1)};
            rhs_contracting = {static_cast<std::int64_t>(batch_rank)};
        }

        const bool has_bias = operation.input_ids().size() == 3;
        const auto fused = operation.attributes().find("fused_activation");
        const bool has_fused_gelu = fused != operation.attributes().end();
        if (has_fused_gelu && (!fused->is_string() || fused->get<std::string>() != "GELU")) {
            return fail(operation, "mlir.unsupported_fused_activation",
                        "operation " + operation.id() + " requests an unsupported fused activation",
                        output_value.id());
        }
        const std::string dot_result =
            has_bias || has_fused_gelu ? temporary() : ssa(output_value.id());
        const std::string dimension_numbers =
            "dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [" +
            integer_list(lhs_batching) + "], rhs_batching_dimensions = [" +
            integer_list(rhs_batching) + "], lhs_contracting_dimensions = [" +
            integer_list(lhs_contracting) + "], rhs_contracting_dimensions = [" +
            integer_list(rhs_contracting) + "]>";
        emit_generic(dot_result, "stablehlo.dot_general", {left, right},
                     {ranked_tensor_type(data_type, normalized_left),
                      ranked_tensor_type(data_type, normalized_right)},
                     ranked_tensor_type(data_type, output_dimensions), dimension_numbers);

        std::string current = dot_result;
        if (has_bias) {
            const Value& bias = value(operation.input_ids()[2]);
            const Dimensions& bias_dimensions = bias.descriptor().shape().dimensions();
            const Dimensions bias_mapping{static_cast<std::int64_t>(output_dimensions.size() - 1)};
            const std::string broadcast_bias = emit_broadcast(
                ssa(bias.id()), data_type, bias_dimensions, output_dimensions, bias_mapping);
            const std::string bias_result = has_fused_gelu ? temporary() : ssa(output_value.id());
            emit_binary(bias_result, "stablehlo.add", current, broadcast_bias, data_type,
                        output_dimensions);
            current = bias_result;
        }
        if (has_fused_gelu) {
            return emit_gelu(operation, current, ssa(output_value.id()), output_value);
        }
        return true;
    }

    bool emit_gelu(const Operation& operation, const std::string& input, const std::string& output,
                   const Value& output_value) {
        if (output_value.descriptor().data_type() != DataType::float32) {
            return fail(operation, "mlir.unsupported_dtype",
                        "operation " + operation.id() + " GELU lowering requires float32",
                        output_value.id());
        }
        const auto approximate = operation.attributes().find("approximate");
        if (approximate != operation.attributes().end() &&
            (!approximate->is_string() || approximate->get<std::string>() != "none")) {
            return fail(operation, "mlir.unsupported_gelu_convention",
                        "operation " + operation.id() +
                            " GELU lowering supports only approximate=none",
                        output_value.id());
        }
        const Dimensions& dimensions = output_value.descriptor().shape().dimensions();
        const std::string type = ranked_tensor_type(output_value);
        const std::string sqrt_two = emit_float_constant(std::sqrt(2.0), dimensions);
        const std::string scaled = temporary();
        emit_binary(scaled, "stablehlo.divide", input, sqrt_two, DataType::float32, dimensions);
        const std::string error_function = temporary();
        emit_generic(error_function, "chlo.erf", {scaled}, {type}, type);
        const std::string one = emit_float_constant(1.0, dimensions);
        const std::string shifted = temporary();
        emit_binary(shifted, "stablehlo.add", error_function, one, DataType::float32, dimensions);
        const std::string multiplied = temporary();
        emit_binary(multiplied, "stablehlo.multiply", input, shifted, DataType::float32,
                    dimensions);
        const std::string half = emit_float_constant(0.5, dimensions);
        emit_binary(output, "stablehlo.multiply", multiplied, half, DataType::float32, dimensions);
        return true;
    }

    bool emit_rms_norm(const Operation& operation) {
        const Value& input_value = value(operation.input_ids()[0]);
        const Value& weight_value = value(operation.input_ids()[1]);
        const Value& output_value = value(operation.output_ids()[0]);
        if (output_value.descriptor().data_type() != DataType::float32) {
            return fail(operation, "mlir.unsupported_dtype",
                        "operation " + operation.id() + " RMSNorm lowering requires float32",
                        output_value.id());
        }
        const Dimensions& dimensions = input_value.descriptor().shape().dimensions();
        Dimensions reduced_dimensions(dimensions.begin(), dimensions.end() - 1);
        const Dimensions reduction_axis{static_cast<std::int64_t>(dimensions.size() - 1)};
        const std::string input = ssa(input_value.id());
        const std::string square = temporary();
        emit_binary(square, "stablehlo.multiply", input, input, DataType::float32, dimensions);
        const std::string zero = emit_float_constant(0.0, {});
        const std::string sum = emit_reduce(square, zero, DataType::float32, dimensions,
                                            reduced_dimensions, reduction_axis, "add");
        const std::string count =
            emit_float_constant(static_cast<double>(dimensions.back()), reduced_dimensions);
        const std::string mean = temporary();
        emit_binary(mean, "stablehlo.divide", sum, count, DataType::float32, reduced_dimensions);
        const double epsilon = operation.attributes().at("epsilon").get<double>();
        const std::string epsilon_value = emit_float_constant(epsilon, reduced_dimensions);
        const std::string variance_epsilon = temporary();
        emit_binary(variance_epsilon, "stablehlo.add", mean, epsilon_value, DataType::float32,
                    reduced_dimensions);
        const std::string inverse_root = temporary();
        const std::string reduced_type = ranked_tensor_type(DataType::float32, reduced_dimensions);
        emit_generic(inverse_root, "stablehlo.rsqrt", {variance_epsilon}, {reduced_type},
                     reduced_type);
        Dimensions prefix_mapping;
        for (std::size_t index = 0; index + 1 < dimensions.size(); ++index) {
            prefix_mapping.push_back(static_cast<std::int64_t>(index));
        }
        const std::string inverse_broadcast = emit_broadcast(
            inverse_root, DataType::float32, reduced_dimensions, dimensions, prefix_mapping);
        const std::string normalized = temporary();
        emit_binary(normalized, "stablehlo.multiply", input, inverse_broadcast, DataType::float32,
                    dimensions);
        const Dimensions weight_mapping{static_cast<std::int64_t>(dimensions.size() - 1)};
        const std::string weight = emit_broadcast(ssa(weight_value.id()), DataType::float32,
                                                  weight_value.descriptor().shape().dimensions(),
                                                  dimensions, weight_mapping);
        emit_binary(ssa(output_value.id()), "stablehlo.multiply", normalized, weight,
                    DataType::float32, dimensions);
        return true;
    }

    bool emit_softmax(const Operation& operation) {
        const Value& input_value = value(operation.input_ids()[0]);
        const Value& output_value = value(operation.output_ids()[0]);
        if (output_value.descriptor().data_type() != DataType::float32) {
            return fail(operation, "mlir.unsupported_dtype",
                        "operation " + operation.id() + " Softmax lowering requires float32",
                        output_value.id());
        }
        const Dimensions& dimensions = input_value.descriptor().shape().dimensions();
        std::int64_t axis = operation.attributes().at("axis").get<std::int64_t>();
        if (axis < 0) {
            axis += static_cast<std::int64_t>(dimensions.size());
        }
        Dimensions reduced_dimensions = dimensions;
        reduced_dimensions.erase(reduced_dimensions.begin() + axis);
        Dimensions broadcast_mapping;
        for (std::int64_t index = 0; index < static_cast<std::int64_t>(dimensions.size());
             ++index) {
            if (index != axis) {
                broadcast_mapping.push_back(index);
            }
        }
        const std::string negative_infinity =
            emit_dense_constant("0xFF800000", DataType::float32, {});
        const std::string maximum =
            emit_reduce(ssa(input_value.id()), negative_infinity, DataType::float32, dimensions,
                        reduced_dimensions, {axis}, "maximum");
        const std::string maximum_broadcast = emit_broadcast(
            maximum, DataType::float32, reduced_dimensions, dimensions, broadcast_mapping);
        const std::string shifted = temporary();
        emit_binary(shifted, "stablehlo.subtract", ssa(input_value.id()), maximum_broadcast,
                    DataType::float32, dimensions);
        const std::string exponential = temporary();
        const std::string full_type = ranked_tensor_type(output_value);
        emit_generic(exponential, "stablehlo.exponential", {shifted}, {full_type}, full_type);
        const std::string zero = emit_float_constant(0.0, {});
        const std::string sum = emit_reduce(exponential, zero, DataType::float32, dimensions,
                                            reduced_dimensions, {axis}, "add");
        const std::string sum_broadcast = emit_broadcast(sum, DataType::float32, reduced_dimensions,
                                                         dimensions, broadcast_mapping);
        emit_binary(ssa(output_value.id()), "stablehlo.divide", exponential, sum_broadcast,
                    DataType::float32, dimensions);
        return true;
    }

    bool emit_constant(const Operation& operation) {
        const Value& output_value = value(operation.output_ids()[0]);
        const nlohmann::json& constant = operation.attributes().at("value");
        std::string literal;
        if (constant.is_boolean()) {
            literal = constant.get<bool>() ? "true" : "false";
        } else if (constant.is_number_integer() || constant.is_number_unsigned()) {
            literal = std::to_string(constant.get<std::int64_t>());
        } else if (constant.is_number_float()) {
            const double number = constant.get<double>();
            if (!std::isfinite(number)) {
                return fail(operation, "mlir.unsupported_constant",
                            "operation " + operation.id() + " has a non-finite JSON constant",
                            output_value.id());
            }
            literal = float_literal(number);
        } else {
            return fail(operation, "mlir.unsupported_constant",
                        "operation " + operation.id() + " has an unsupported constant value",
                        output_value.id());
        }
        const std::string type = ranked_tensor_type(output_value);
        emit_generic(ssa(output_value.id()), "stablehlo.constant", {}, {}, type,
                     "value = dense<" + literal + "> : " + type);
        return true;
    }

    bool emit_operation(const Operation& operation) {
        const Value& output_value = value(operation.output_ids()[0]);
        switch (operation.type()) {
        case OperationType::input:
        case OperationType::parameter:
            return true;
        case OperationType::constant:
            return emit_constant(operation);
        case OperationType::mat_mul:
            return emit_matmul(operation, false);
        case OperationType::linear:
            return emit_matmul(operation, true);
        case OperationType::add:
            return emit_elementwise(operation, "stablehlo.add");
        case OperationType::multiply:
            return emit_elementwise(operation, "stablehlo.multiply");
        case OperationType::divide:
            return emit_elementwise(operation, "stablehlo.divide");
        case OperationType::rms_norm:
            return emit_rms_norm(operation);
        case OperationType::gelu:
            return emit_gelu(operation, ssa(operation.input_ids()[0]), ssa(output_value.id()),
                             output_value);
        case OperationType::softmax:
            return emit_softmax(operation);
        case OperationType::reshape:
            emit_generic(ssa(output_value.id()), "stablehlo.reshape",
                         {ssa(operation.input_ids()[0])},
                         {ranked_tensor_type(value(operation.input_ids()[0]))},
                         ranked_tensor_type(output_value));
            return true;
        case OperationType::transpose: {
            const Dimensions permutation =
                operation.attributes().at("permutation").get<Dimensions>();
            emit_generic(
                ssa(output_value.id()), "stablehlo.transpose", {ssa(operation.input_ids()[0])},
                {ranked_tensor_type(value(operation.input_ids()[0]))},
                ranked_tensor_type(output_value), "permutation = " + array_attribute(permutation));
            return true;
        }
        case OperationType::causal_mask:
            return fail(operation, "mlir.unsupported_operation",
                        "operation " + operation.id() +
                            " (CausalMask) has no StableHLO lowering in Milestone 11",
                        output_value.id());
        }
        return fail(operation, "mlir.unsupported_operation",
                    "operation " + operation.id() + " has no StableHLO lowering",
                    output_value.id());
    }

    const Graph& graph_;
    std::unordered_map<std::string, const Value*> values_;
    std::ostringstream stream_;
    std::size_t next_temporary_{0};
    Status failure_status_{Status::ok_status()};
    std::vector<Diagnostic> diagnostics_;
};

std::optional<std::string> environment_path() {
#ifdef _WIN32
    char* value = nullptr;
    std::size_t length = 0;
    if (_dupenv_s(&value, &length, "PATH") != 0 || value == nullptr) {
        return std::nullopt;
    }
    const std::unique_ptr<char, decltype(&std::free)> owner(value, &std::free);
    return std::string(owner.get(), length == 0 ? 0 : length - 1);
#else
    const char* value = std::getenv("PATH");
    if (value == nullptr) {
        return std::nullopt;
    }
    return std::string(value);
#endif
}

std::optional<std::filesystem::path> find_validation_tool() {
    const std::optional<std::string> path_value = environment_path();
    if (!path_value.has_value()) {
        return std::nullopt;
    }
#ifdef _WIN32
    constexpr char separator = ';';
    const std::vector<std::string> names{"stablehlo-opt.exe", "stablehlo-opt", "mlir-opt.exe",
                                         "mlir-opt"};
#else
    constexpr char separator = ':';
    const std::vector<std::string> names{"stablehlo-opt", "mlir-opt"};
#endif
    std::stringstream paths(path_value.value());
    std::string directory;
    while (std::getline(paths, directory, separator)) {
        if (directory.size() >= 2 && directory.front() == '"' && directory.back() == '"') {
            directory = directory.substr(1, directory.size() - 2);
        }
        for (const std::string& name : names) {
            std::error_code error;
            const std::filesystem::path candidate = std::filesystem::path(directory) / name;
            if (std::filesystem::is_regular_file(candidate, error) && !error) {
                return std::filesystem::absolute(candidate);
            }
        }
    }
    return std::nullopt;
}

bool shell_safe_path(const std::filesystem::path& path) {
#ifdef _WIN32
    return path.string().find_first_of("\"%!\r\n") == std::string::npos;
#else
    return path.string().find_first_of("\r\n") == std::string::npos;
#endif
}

std::string shell_quote(const std::filesystem::path& path) {
#ifdef _WIN32
    return '"' + path.string() + '"';
#else
    std::string result{"'"};
    for (const char character : path.string()) {
        if (character == '\'') {
            result += "'\\''";
        } else {
            result += character;
        }
    }
    result += '\'';
    return result;
#endif
}

int run_tool(const std::filesystem::path& tool, const std::filesystem::path& input,
             const std::vector<std::string>& arguments, const std::filesystem::path& output,
             const std::filesystem::path& log) {
    if (!shell_safe_path(tool) || !shell_safe_path(input) || !shell_safe_path(output) ||
        !shell_safe_path(log)) {
        return -1;
    }
    std::ostringstream command;
    command << shell_quote(tool) << ' ' << shell_quote(input);
    for (const std::string& argument : arguments) {
        command << ' ' << argument;
    }
    command << " -o " << shell_quote(output) << " > " << shell_quote(log) << " 2>&1";
    return std::system(command.str().c_str());
}

} // namespace

MlirLoweringResult lower_to_stablehlo(const Graph& graph) {
    Graph verified_graph = graph;
    const VerificationReport verification = verify_graph(verified_graph);
    if (!verification.success()) {
        return {Status::error(StatusCode::failed_precondition,
                              "ForgeIR graph must pass semantic verification before MLIR lowering"),
                {},
                verification.pipeline.diagnostics};
    }
    return StablehloLowerer(verified_graph).run();
}

Status write_mlir_module(const std::filesystem::path& path, const std::string& module_text) {
    std::error_code error;
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path(), error);
        if (error) {
            return Status::error(StatusCode::internal,
                                 "unable to create MLIR output directory: " + error.message());
        }
    }
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        return Status::error(StatusCode::internal, "unable to open MLIR output: " + path.string());
    }
    stream << module_text;
    if (!stream) {
        return Status::error(StatusCode::internal, "unable to write MLIR output: " + path.string());
    }
    return Status::ok_status();
}

MlirToolValidationResult validate_mlir_module(const std::filesystem::path& module_path) {
    const std::optional<std::filesystem::path> tool = find_validation_tool();
    if (!tool.has_value()) {
        return {"tool unavailable",
                "",
                false,
                false,
                false,
                false,
                {},
                {},
                "stablehlo-opt and mlir-opt were not found on PATH"};
    }

    const std::filesystem::path base = module_path.parent_path() / module_path.stem();
    const std::filesystem::path verified_output = base.string() + ".verified.mlir";
    const std::filesystem::path canonical_output = base.string() + ".canonical.mlir";
    const std::filesystem::path log = base.string() + ".validation.log";
    MlirToolValidationResult result{
        "failed", tool->generic_string(), true, false, false, false, {}, log, {}};
    if (run_tool(tool.value(), module_path, {}, verified_output, log) != 0) {
        result.message = "external MLIR syntax verification failed; see diagnostic_log";
        return result;
    }
    result.syntax_verified = true;
    std::error_code remove_error;
    std::filesystem::remove(verified_output, remove_error);
    result.canonicalization_attempted = true;
    if (run_tool(tool.value(), module_path, {"--canonicalize", "--cse"}, canonical_output, log) ==
        0) {
        result.status = "validated";
        result.canonicalization_succeeded = true;
        result.canonical_output = canonical_output;
        result.message = "syntax verification and canonicalization/CSE succeeded";
    } else {
        result.status = "syntax verified";
        result.message =
            "syntax verification succeeded; canonicalization/CSE was unavailable or failed";
    }
    return result;
}

} // namespace forgeir
