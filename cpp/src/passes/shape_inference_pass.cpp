#include "forgeir/passes/shape_inference_pass.hpp"

#include "semantic_verifier.hpp"

namespace forgeir {

std::string_view ShapeInferencePass::name() const noexcept { return "ShapeInferencePass"; }

PassResult ShapeInferencePass::run(Graph& graph) { return verify_graph_shapes(graph); }

} // namespace forgeir
