#include "forgeir/passes/pass.hpp"

#include <stdexcept>
#include <utility>

namespace forgeir {

PassResult PassResult::success(const bool changed, std::vector<Diagnostic> diagnostics,
                               std::vector<RewriteRecord> rewrites) {
    return {Status::ok_status(), changed, std::move(diagnostics), std::move(rewrites)};
}

PassResult PassResult::failure(Status status, std::vector<Diagnostic> diagnostics,
                               std::vector<RewriteRecord> rewrites) {
    if (status.ok()) {
        throw std::invalid_argument("a failed pass must return a non-success status");
    }
    return {std::move(status), false, std::move(diagnostics), std::move(rewrites)};
}

} // namespace forgeir
