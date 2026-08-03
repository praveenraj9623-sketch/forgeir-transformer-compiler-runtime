#pragma once

#include <string>
#include <string_view>

#include "forgeir/core/status.hpp"
#include "forgeir/ir/tensor_descriptor.hpp"

namespace forgeir {

enum class ValueKind { input, parameter, constant, intermediate, output };

[[nodiscard]] Result<ValueKind> parse_value_kind(std::string_view name);
[[nodiscard]] std::string_view to_string(ValueKind kind) noexcept;

class Value {
  public:
    Value(std::string id, std::string semantic_name, TensorDescriptor descriptor, ValueKind kind);

    [[nodiscard]] const std::string& id() const noexcept;
    [[nodiscard]] const std::string& semantic_name() const noexcept;
    [[nodiscard]] const TensorDescriptor& descriptor() const noexcept;
    [[nodiscard]] ValueKind kind() const noexcept;
    void set_kind(ValueKind kind) noexcept;

  private:
    std::string id_;
    std::string semantic_name_;
    TensorDescriptor descriptor_;
    ValueKind kind_;
};

} // namespace forgeir
