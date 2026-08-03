#pragma once

#include <cstdint>

#include "forgeir/passes/pass.hpp"

namespace forgeir {

inline constexpr std::uint64_t kMaximumFoldedTensorElements = 1024;

class ConstantFoldingPass final : public Pass {
  public:
    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] PassResult run(Graph& graph) override;
};

} // namespace forgeir
