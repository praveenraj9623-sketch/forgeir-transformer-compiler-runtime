#pragma once

#include <string_view>

#include "forgeir/core/build_config.hpp"

namespace forgeir {

inline constexpr std::string_view kVersion{FORGEIR_VERSION};

[[nodiscard]] std::string_view version() noexcept;

} // namespace forgeir
