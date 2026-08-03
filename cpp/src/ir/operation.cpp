#include "forgeir/ir/operation.hpp"

#include <string>
#include <utility>

namespace forgeir {

Result<OperationType> parse_operation_type(const std::string_view name) {
    if (name == "Input") {
        return OperationType::input;
    }
    if (name == "Parameter") {
        return OperationType::parameter;
    }
    if (name == "Constant") {
        return OperationType::constant;
    }
    if (name == "MatMul") {
        return OperationType::mat_mul;
    }
    if (name == "Linear") {
        return OperationType::linear;
    }
    if (name == "Add") {
        return OperationType::add;
    }
    if (name == "Mul") {
        return OperationType::multiply;
    }
    if (name == "Div") {
        return OperationType::divide;
    }
    if (name == "RMSNorm") {
        return OperationType::rms_norm;
    }
    if (name == "GELU") {
        return OperationType::gelu;
    }
    if (name == "Softmax") {
        return OperationType::softmax;
    }
    if (name == "Reshape") {
        return OperationType::reshape;
    }
    if (name == "Transpose") {
        return OperationType::transpose;
    }
    if (name == "CausalMask") {
        return OperationType::causal_mask;
    }
    return Status::error(StatusCode::unsupported,
                         "unsupported graph operation: " + std::string(name));
}

std::string_view to_string(const OperationType type) noexcept {
    switch (type) {
    case OperationType::input:
        return "Input";
    case OperationType::parameter:
        return "Parameter";
    case OperationType::constant:
        return "Constant";
    case OperationType::mat_mul:
        return "MatMul";
    case OperationType::linear:
        return "Linear";
    case OperationType::add:
        return "Add";
    case OperationType::multiply:
        return "Mul";
    case OperationType::divide:
        return "Div";
    case OperationType::rms_norm:
        return "RMSNorm";
    case OperationType::gelu:
        return "GELU";
    case OperationType::softmax:
        return "Softmax";
    case OperationType::reshape:
        return "Reshape";
    case OperationType::transpose:
        return "Transpose";
    case OperationType::causal_mask:
        return "CausalMask";
    }
    return "invalid";
}

std::optional<std::size_t> expected_input_count(const OperationType type) noexcept {
    switch (type) {
    case OperationType::input:
    case OperationType::parameter:
    case OperationType::constant:
        return 0;
    case OperationType::linear:
    case OperationType::add:
    case OperationType::multiply:
    case OperationType::divide:
    case OperationType::mat_mul:
    case OperationType::rms_norm:
        return 2;
    case OperationType::gelu:
    case OperationType::softmax:
    case OperationType::reshape:
    case OperationType::transpose:
    case OperationType::causal_mask:
        return 1;
    }
    return std::nullopt;
}

Operation::Operation(std::string id, const OperationType type, std::string semantic_name,
                     std::vector<std::string> input_ids, std::vector<std::string> output_ids,
                     nlohmann::json attributes)
    : id_(std::move(id)), type_(type), semantic_name_(std::move(semantic_name)),
      input_ids_(std::move(input_ids)), output_ids_(std::move(output_ids)),
      attributes_(std::move(attributes)) {}

const std::string& Operation::id() const noexcept { return id_; }

OperationType Operation::type() const noexcept { return type_; }

const std::string& Operation::semantic_name() const noexcept { return semantic_name_; }

const std::vector<std::string>& Operation::input_ids() const noexcept { return input_ids_; }

const std::vector<std::string>& Operation::output_ids() const noexcept { return output_ids_; }

const nlohmann::json& Operation::attributes() const noexcept { return attributes_; }

} // namespace forgeir
