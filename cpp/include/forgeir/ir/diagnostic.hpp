#pragma once

#include <string>

namespace forgeir {

enum class DiagnosticSeverity { error, warning };

struct Diagnostic {
    DiagnosticSeverity severity{DiagnosticSeverity::error};
    std::string code;
    std::string message;
    std::string entity_id;
};

} // namespace forgeir
