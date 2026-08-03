#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "forgeir/passes/pass.hpp"

namespace forgeir {

struct PassExecutionRecord {
    std::string pass_name;
    std::string stage;
    bool success{false};
    bool changed{false};
    std::size_t diagnostic_count{0};
    std::size_t nodes_added{0};
    std::size_t nodes_removed{0};
};

struct PassManagerResult {
    Status status;
    bool changed{false};
    std::vector<Diagnostic> diagnostics;
    std::vector<RewriteRecord> rewrites;
    std::vector<PassExecutionRecord> executions;

    [[nodiscard]] std::size_t error_count() const noexcept;
    [[nodiscard]] std::size_t warning_count() const noexcept;
};

class PassManager {
  public:
    void add_pass(std::unique_ptr<Pass> pass);
    [[nodiscard]] PassManagerResult run(Graph& graph) const;

  private:
    std::vector<std::unique_ptr<Pass>> passes_;
};

} // namespace forgeir
