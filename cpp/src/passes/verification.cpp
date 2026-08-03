#include "forgeir/passes/verification.hpp"

#include <memory>
#include <string>

#include "forgeir/passes/dtype_propagation_pass.hpp"
#include "forgeir/passes/graph_verifier_pass.hpp"
#include "forgeir/passes/shape_inference_pass.hpp"

namespace forgeir {
namespace {

nlohmann::json diagnostic_json(const Diagnostic& diagnostic) {
    return nlohmann::json{{"severity", to_string(diagnostic.severity)},
                          {"code", diagnostic.code},
                          {"message", diagnostic.message},
                          {"operation_id", diagnostic.operation_id},
                          {"value_id", diagnostic.value_id}};
}

} // namespace

bool VerificationReport::success() const noexcept { return pipeline.status.ok(); }

VerificationReport verify_graph(Graph& graph) {
    PassManager manager;
    manager.add_pass(std::make_unique<ShapeInferencePass>());
    manager.add_pass(std::make_unique<DTypePropagationPass>());
    manager.add_pass(std::make_unique<GraphVerifierPass>());
    return VerificationReport{graph.schema_version(), graph.graph_hash(), manager.run(graph)};
}

nlohmann::json verification_report_json(const VerificationReport& report) {
    nlohmann::json diagnostics = nlohmann::json::array();
    for (const Diagnostic& diagnostic : report.pipeline.diagnostics) {
        diagnostics.push_back(diagnostic_json(diagnostic));
    }

    nlohmann::json executions = nlohmann::json::array();
    for (const PassExecutionRecord& execution : report.pipeline.executions) {
        executions.push_back(nlohmann::json{{"pass", execution.pass_name},
                                            {"stage", execution.stage},
                                            {"success", execution.success},
                                            {"changed", execution.changed},
                                            {"diagnostic_count", execution.diagnostic_count},
                                            {"nodes_added", execution.nodes_added},
                                            {"nodes_removed", execution.nodes_removed}});
    }

    nlohmann::json rewrites = nlohmann::json::array();
    for (const RewriteRecord& rewrite : report.pipeline.rewrites) {
        rewrites.push_back(nlohmann::json{{"pass", rewrite.pass_name},
                                          {"stage", rewrite.stage},
                                          {"reason", rewrite.reason},
                                          {"nodes_added", rewrite.nodes_added},
                                          {"nodes_removed", rewrite.nodes_removed},
                                          {"nodes_modified", rewrite.nodes_modified},
                                          {"values_removed", rewrite.values_removed}});
    }

    return nlohmann::json{
        {"report_schema_version", "1.0"},
        {"graph_schema_version", report.graph_schema_version},
        {"graph_hash", report.graph_hash},
        {"success", report.success()},
        {"changed", report.pipeline.changed},
        {"status",
         {{"code", static_cast<int>(report.pipeline.status.code())},
          {"message", report.pipeline.status.message()}}},
        {"diagnostic_counts",
         {{"error", report.pipeline.error_count()}, {"warning", report.pipeline.warning_count()}}},
        {"diagnostics", std::move(diagnostics)},
        {"rewrites", std::move(rewrites)},
        {"pass_executions", std::move(executions)}};
}

} // namespace forgeir
