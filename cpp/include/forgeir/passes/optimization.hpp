#pragma once

#include <cstddef>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "forgeir/core/status.hpp"
#include "forgeir/ir/graph.hpp"
#include "forgeir/passes/pass_manager.hpp"

namespace forgeir {

enum class OptimizationLevel { o0, o1, o2 };

[[nodiscard]] Result<OptimizationLevel> parse_optimization_level(std::string_view level);
[[nodiscard]] std::string_view to_string(OptimizationLevel level) noexcept;

struct OptimizationResult {
    OptimizationLevel level{OptimizationLevel::o0};
    std::string input_graph_hash;
    std::string output_graph_hash;
    std::size_t operations_before{0};
    std::size_t operations_after{0};
    std::size_t values_before{0};
    std::size_t values_after{0};
    PassManagerResult pipeline;

    [[nodiscard]] bool success() const noexcept;
};

[[nodiscard]] OptimizationResult optimize_graph(Graph& graph, OptimizationLevel level);
[[nodiscard]] nlohmann::json optimization_report_json(const OptimizationResult& result);

} // namespace forgeir
