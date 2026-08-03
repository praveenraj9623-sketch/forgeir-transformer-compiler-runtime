#include "forgeir/passes/dtype_propagation_pass.hpp"

#include "semantic_verifier.hpp"

namespace forgeir {

std::string_view DTypePropagationPass::name() const noexcept { return "DTypePropagationPass"; }

PassResult DTypePropagationPass::run(Graph& graph) { return verify_graph_dtypes(graph); }

} // namespace forgeir
