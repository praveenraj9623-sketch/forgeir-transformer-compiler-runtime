#pragma once

#include "forgeir/ir/graph.hpp"
#include "forgeir/passes/pass.hpp"

namespace forgeir {

[[nodiscard]] PassResult verify_graph_shapes(const Graph& graph);
[[nodiscard]] PassResult verify_graph_dtypes(const Graph& graph);

} // namespace forgeir
