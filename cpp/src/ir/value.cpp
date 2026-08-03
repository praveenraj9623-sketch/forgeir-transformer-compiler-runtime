#include "forgeir/ir/value.hpp"

#include <string>
#include <utility>

namespace forgeir {

Result<ValueKind> parse_value_kind(const std::string_view name) {
    if (name == "input") {
        return ValueKind::input;
    }
    if (name == "parameter") {
        return ValueKind::parameter;
    }
    if (name == "constant") {
        return ValueKind::constant;
    }
    if (name == "intermediate") {
        return ValueKind::intermediate;
    }
    if (name == "output") {
        return ValueKind::output;
    }
    return Status::error(StatusCode::invalid_argument, "invalid value kind: " + std::string(name));
}

std::string_view to_string(const ValueKind kind) noexcept {
    switch (kind) {
    case ValueKind::input:
        return "input";
    case ValueKind::parameter:
        return "parameter";
    case ValueKind::constant:
        return "constant";
    case ValueKind::intermediate:
        return "intermediate";
    case ValueKind::output:
        return "output";
    }
    return "invalid";
}

Value::Value(std::string id, std::string semantic_name, TensorDescriptor descriptor,
             const ValueKind kind)
    : id_(std::move(id)), semantic_name_(std::move(semantic_name)),
      descriptor_(std::move(descriptor)), kind_(kind) {}

const std::string& Value::id() const noexcept { return id_; }

const std::string& Value::semantic_name() const noexcept { return semantic_name_; }

const TensorDescriptor& Value::descriptor() const noexcept { return descriptor_; }

ValueKind Value::kind() const noexcept { return kind_; }

} // namespace forgeir
