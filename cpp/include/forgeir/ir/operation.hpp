#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "forgeir/core/status.hpp"

namespace forgeir {

enum class OperationType {
    input,
    parameter,
    constant,
    mat_mul,
    linear,
    add,
    multiply,
    divide,
    rms_norm,
    gelu,
    softmax,
    reshape,
    transpose,
    causal_mask
};

[[nodiscard]] Result<OperationType> parse_operation_type(std::string_view name);
[[nodiscard]] std::string_view to_string(OperationType type) noexcept;
[[nodiscard]] std::optional<std::size_t> expected_input_count(OperationType type) noexcept;

class Operation {
  public:
    Operation(std::string id, OperationType type, std::string semantic_name,
              std::vector<std::string> input_ids, std::vector<std::string> output_ids,
              nlohmann::json attributes);

    [[nodiscard]] const std::string& id() const noexcept;
    [[nodiscard]] OperationType type() const noexcept;
    [[nodiscard]] const std::string& semantic_name() const noexcept;
    [[nodiscard]] const std::vector<std::string>& input_ids() const noexcept;
    [[nodiscard]] const std::vector<std::string>& output_ids() const noexcept;
    [[nodiscard]] const nlohmann::json& attributes() const noexcept;
    [[nodiscard]] std::vector<std::string>& mutable_input_ids() noexcept;
    [[nodiscard]] nlohmann::json& mutable_attributes() noexcept;
    void set_type(OperationType type) noexcept;
    void set_semantic_name(std::string semantic_name);
    void set_input_ids(std::vector<std::string> input_ids);
    void set_output_ids(std::vector<std::string> output_ids);
    void set_attributes(nlohmann::json attributes);

  private:
    std::string id_;
    OperationType type_;
    std::string semantic_name_;
    std::vector<std::string> input_ids_;
    std::vector<std::string> output_ids_;
    nlohmann::json attributes_;
};

} // namespace forgeir
