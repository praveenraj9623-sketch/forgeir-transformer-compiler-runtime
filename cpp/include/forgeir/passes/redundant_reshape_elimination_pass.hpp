#pragma once

#include "forgeir/passes/pass.hpp"

namespace forgeir {

class RedundantReshapeEliminationPass final : public Pass {
  public:
    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] PassResult run(Graph& graph) override;
};

} // namespace forgeir
