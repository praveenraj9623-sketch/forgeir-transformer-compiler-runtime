#include "forgeir/passes/optimization.hpp"

#include <memory>
#include <string>

#include "forgeir/ir/graph_writer.hpp"
#include "forgeir/passes/canonicalisation_pass.hpp"
#include "forgeir/passes/constant_folding_pass.hpp"
#include "forgeir/passes/dead_code_elimination_pass.hpp"
#include "forgeir/passes/fuse_bias_gelu_pass.hpp"
#include "forgeir/passes/graph_verifier_pass.hpp"
#include "forgeir/passes/redundant_reshape_elimination_pass.hpp"
#include "forgeir/passes/redundant_transpose_elimination_pass.hpp"
#include "forgeir/passes/verification.hpp"

namespace forgeir {

Result<OptimizationLevel> parse_optimization_level(const std::string_view level) {
    if (level == "O0") {
        return OptimizationLevel::o0;
    }
    if (level == "O1") {
        return OptimizationLevel::o1;
    }
    if (level == "O2") {
        return OptimizationLevel::o2;
    }
    return Status::error(StatusCode::invalid_argument, "optimization level must be O0, O1, or O2");
}

std::string_view to_string(const OptimizationLevel level) noexcept {
    switch (level) {
    case OptimizationLevel::o0:
        return "O0";
    case OptimizationLevel::o1:
        return "O1";
    case OptimizationLevel::o2:
        return "O2";
    }
    return "invalid";
}

bool OptimizationResult::success() const noexcept { return pipeline.status.ok(); }

OptimizationResult optimize_graph(Graph& graph, const OptimizationLevel level) {
    OptimizationResult result;
    result.level = level;
    result.input_graph_hash = graph.graph_hash();
    result.operations_before = graph.operations().size();
    result.values_before = graph.values().size();

    PassManager manager;
    if (level == OptimizationLevel::o0) {
        manager.add_pass(std::make_unique<GraphVerifierPass>());
    } else {
        manager.add_pass(std::make_unique<CanonicalisationPass>());
        manager.add_pass(std::make_unique<ConstantFoldingPass>());
        if (level == OptimizationLevel::o2) {
            manager.add_pass(std::make_unique<RedundantReshapeEliminationPass>());
            manager.add_pass(std::make_unique<RedundantTransposeEliminationPass>());
            manager.add_pass(std::make_unique<FuseBiasGELUPass>());
        }
        manager.add_pass(std::make_unique<DeadCodeEliminationPass>());
    }
    result.pipeline = manager.run(graph);
    if (result.pipeline.status.ok()) {
        refresh_graph_hash(graph);
    }
    result.output_graph_hash = graph.graph_hash();
    result.operations_after = graph.operations().size();
    result.values_after = graph.values().size();
    return result;
}

nlohmann::json optimization_report_json(const OptimizationResult& result) {
    VerificationReport pipeline_report{"1.0", result.output_graph_hash, result.pipeline};
    nlohmann::json pipeline = verification_report_json(pipeline_report);
    return nlohmann::json{{"optimization_report_schema_version", "1.0"},
                          {"level", to_string(result.level)},
                          {"success", result.success()},
                          {"changed", result.pipeline.changed},
                          {"input_graph_hash", result.input_graph_hash},
                          {"output_graph_hash", result.output_graph_hash},
                          {"operation_counts",
                           {{"before", result.operations_before},
                            {"after", result.operations_after},
                            {"change", static_cast<std::int64_t>(result.operations_after) -
                                           static_cast<std::int64_t>(result.operations_before)}}},
                          {"value_counts",
                           {{"before", result.values_before},
                            {"after", result.values_after},
                            {"change", static_cast<std::int64_t>(result.values_after) -
                                           static_cast<std::int64_t>(result.values_before)}}},
                          {"maximum_folded_tensor_elements", kMaximumFoldedTensorElements},
                          {"pipeline", std::move(pipeline)}};
}

} // namespace forgeir
