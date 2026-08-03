#include "forgeir/passes/graph_verifier_pass.hpp"

#include <iterator>
#include <utility>

#include "semantic_verifier.hpp"

namespace forgeir {

std::string_view GraphVerifierPass::name() const noexcept { return "GraphVerifierPass"; }

PassResult GraphVerifierPass::run(Graph& graph) {
    PassResult shape_result = verify_graph_shapes(graph);
    PassResult dtype_result = verify_graph_dtypes(graph);

    std::vector<Diagnostic> diagnostics;
    diagnostics.reserve(shape_result.diagnostics.size() + dtype_result.diagnostics.size());
    diagnostics.insert(diagnostics.end(), std::make_move_iterator(shape_result.diagnostics.begin()),
                       std::make_move_iterator(shape_result.diagnostics.end()));
    diagnostics.insert(diagnostics.end(), std::make_move_iterator(dtype_result.diagnostics.begin()),
                       std::make_move_iterator(dtype_result.diagnostics.end()));
    if (!shape_result.status.ok()) {
        return PassResult::failure(std::move(shape_result.status), std::move(diagnostics));
    }
    if (!dtype_result.status.ok()) {
        return PassResult::failure(std::move(dtype_result.status), std::move(diagnostics));
    }
    return PassResult::success(false, std::move(diagnostics));
}

} // namespace forgeir
