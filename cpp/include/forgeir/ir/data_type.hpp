#pragma once

#include <cstdint>
#include <string_view>

#include "forgeir/core/status.hpp"

namespace forgeir {

enum class DataType { boolean, float32, int64 };

[[nodiscard]] Result<DataType> parse_data_type(std::string_view name);
[[nodiscard]] std::string_view to_string(DataType data_type) noexcept;
[[nodiscard]] std::uint64_t data_type_byte_width(DataType data_type) noexcept;

} // namespace forgeir
