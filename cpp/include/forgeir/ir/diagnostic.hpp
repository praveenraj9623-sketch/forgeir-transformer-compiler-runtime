#pragma once

#include <string>
#include <string_view>

namespace forgeir {

enum class DiagnosticSeverity { error, warning };

struct Diagnostic {
    DiagnosticSeverity severity{DiagnosticSeverity::error};
    std::string code;
    std::string message;
    std::string entity_id;
    std::string operation_id;
    std::string value_id;
};

[[nodiscard]] constexpr std::string_view to_string(const DiagnosticSeverity severity) noexcept {
    return severity == DiagnosticSeverity::error ? "error" : "warning";
}

} // namespace forgeir
