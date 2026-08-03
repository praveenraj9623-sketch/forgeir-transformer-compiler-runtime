#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "forgeir/core/status.hpp"
#include "forgeir/ir/diagnostic.hpp"
#include "forgeir/ir/graph.hpp"

namespace forgeir {

struct RewriteRecord {
    std::string pass_name;
    std::string stage;
    std::string reason;
    std::vector<std::string> nodes_added;
    std::vector<std::string> nodes_removed;
    std::vector<std::string> nodes_modified;
    std::vector<std::string> values_removed;
};

struct PassResult {
    Status status;
    bool changed{false};
    std::vector<Diagnostic> diagnostics;
    std::vector<RewriteRecord> rewrites;

    [[nodiscard]] static PassResult success(bool changed = false,
                                            std::vector<Diagnostic> diagnostics = {},
                                            std::vector<RewriteRecord> rewrites = {});
    [[nodiscard]] static PassResult failure(Status status, std::vector<Diagnostic> diagnostics,
                                            std::vector<RewriteRecord> rewrites = {});
};

class Pass {
  public:
    virtual ~Pass() = default;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual PassResult run(Graph& graph) = 0;
};

} // namespace forgeir
