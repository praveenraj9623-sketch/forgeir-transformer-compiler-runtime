#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "forgeir/core/status.hpp"
#include "forgeir/ir/diagnostic.hpp"
#include "forgeir/ir/graph.hpp"

namespace forgeir {

struct MlirLoweringResult {
    Status status;
    std::string module_text;
    std::vector<Diagnostic> diagnostics;

    [[nodiscard]] bool success() const noexcept { return status.ok(); }
};

struct MlirToolValidationResult {
    std::string status;
    std::string tool;
    bool external_validation_available{false};
    bool syntax_verified{false};
    bool canonicalization_attempted{false};
    bool canonicalization_succeeded{false};
    std::filesystem::path canonical_output;
    std::filesystem::path diagnostic_log;
    std::string message;
};

[[nodiscard]] MlirLoweringResult lower_to_stablehlo(const Graph& graph);
[[nodiscard]] Status write_mlir_module(const std::filesystem::path& path,
                                       const std::string& module_text);
[[nodiscard]] MlirToolValidationResult
validate_mlir_module(const std::filesystem::path& module_path);

} // namespace forgeir
