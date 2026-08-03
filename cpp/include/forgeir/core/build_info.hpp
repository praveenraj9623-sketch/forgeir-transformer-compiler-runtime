#pragma once

#include <optional>
#include <string>

namespace forgeir {

struct BuildInfo {
    std::string forgeir_version;
    std::string compiler;
    std::string build_type;
    std::string operating_system;
    std::string cpp_standard;
    std::optional<std::string> python_version;
    bool cuda_compiled;
    bool hip_compiled;
    bool mlir_compiled;
};

[[nodiscard]] BuildInfo
current_build_info(std::optional<std::string> python_version = std::nullopt);
[[nodiscard]] std::string build_info_json(const BuildInfo& build_info);

} // namespace forgeir
