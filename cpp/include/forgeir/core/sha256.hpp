#pragma once

#include <string>
#include <string_view>

namespace forgeir {

[[nodiscard]] std::string sha256(std::string_view data);
[[nodiscard]] bool is_lowercase_sha256(std::string_view digest) noexcept;

} // namespace forgeir
