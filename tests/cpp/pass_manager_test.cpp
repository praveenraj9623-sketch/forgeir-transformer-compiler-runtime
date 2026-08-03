#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "forgeir/core/sha256.hpp"
#include "forgeir/ir/graph_loader.hpp"
#include "forgeir/passes/pass_manager.hpp"
#include "forgeir/passes/verification.hpp"

namespace {

std::filesystem::path repository_path(const std::string& relative_path) {
    return std::filesystem::path(FORGEIR_SOURCE_DIR) / relative_path;
}

nlohmann::json load_golden_graph() {
    std::ifstream stream(repository_path("tests/golden/milestone_04_valid_graph.json"));
    return nlohmann::json::parse(stream);
}

void refresh_graph_hash(nlohmann::json& graph) {
    graph.erase("graph_hash");
    graph["graph_hash"] = forgeir::sha256(graph.dump());
}

forgeir::Graph load_graph(nlohmann::json graph) {
    refresh_graph_hash(graph);
    auto loaded = forgeir::GraphLoader::load_from_json(graph.dump());
    if (!loaded.ok()) {
        throw std::runtime_error(loaded.status().message());
    }
    return loaded.take_value();
}

void expect_verification_error(nlohmann::json graph, const std::string& expected_code) {
    forgeir::Graph loaded = load_graph(std::move(graph));
    const forgeir::VerificationReport report = forgeir::verify_graph(loaded);
    ASSERT_FALSE(report.success());
    ASSERT_GT(report.pipeline.error_count(), 0U);
    ASSERT_FALSE(report.pipeline.diagnostics.empty());
    EXPECT_EQ(report.pipeline.diagnostics.front().code, expected_code);
    for (const forgeir::Diagnostic& diagnostic : report.pipeline.diagnostics) {
        EXPECT_EQ(diagnostic.operation_id.rfind("op", 0), 0U);
        EXPECT_EQ(diagnostic.value_id.rfind("v", 0), 0U);
    }
}

class RecordingPass final : public forgeir::Pass {
  public:
    RecordingPass(std::string name, std::vector<std::string>& order, const bool changed,
                  const bool fail)
        : name_(std::move(name)), order_(order), changed_(changed), fail_(fail) {}

    std::string_view name() const noexcept override { return name_; }

    forgeir::PassResult run(forgeir::Graph&) override {
        order_.push_back(name_);
        if (fail_) {
            std::vector<forgeir::Diagnostic> diagnostics{
                forgeir::Diagnostic{forgeir::DiagnosticSeverity::error, "test.failure",
                                    "intentional test failure", "op0002", "op0002", "v0002"}};
            return forgeir::PassResult::failure(
                forgeir::Status::error(forgeir::StatusCode::failed_precondition,
                                       "intentional test failure"),
                std::move(diagnostics));
        }
        return forgeir::PassResult::success(changed_);
    }

  private:
    std::string name_;
    std::vector<std::string>& order_;
    bool changed_;
    bool fail_;
};

} // namespace

TEST(VerificationPipeline, VerifiesFullTransformerGraphWithoutDiagnostics) {
    auto graph = forgeir::GraphLoader::load_from_file(
        repository_path("artifacts/graphs/milestone_03/default/tiny_transformer_block.graph.json"));
    ASSERT_TRUE(graph.ok()) << graph.status().message();

    const forgeir::VerificationReport report = forgeir::verify_graph(graph.value());
    EXPECT_TRUE(report.success());
    EXPECT_FALSE(report.pipeline.changed);
    EXPECT_EQ(report.pipeline.error_count(), 0U);
    EXPECT_EQ(report.pipeline.warning_count(), 0U);
    EXPECT_EQ(report.pipeline.executions.size(), 9U);
}

TEST(VerificationPipeline, IsIdempotentAndDeterministicWhenRerun) {
    forgeir::Graph graph = load_graph(load_golden_graph());
    const nlohmann::json first = forgeir::verification_report_json(forgeir::verify_graph(graph));
    const nlohmann::json second = forgeir::verification_report_json(forgeir::verify_graph(graph));

    EXPECT_EQ(first, second);
    EXPECT_FALSE(first.at("changed").get<bool>());
}

TEST(PassManager, PreservesInsertionOrderAndPropagatesFailure) {
    forgeir::Graph graph = load_graph(load_golden_graph());
    std::vector<std::string> order;
    forgeir::PassManager manager;
    manager.add_pass(std::make_unique<RecordingPass>("first", order, true, false));
    manager.add_pass(std::make_unique<RecordingPass>("second", order, false, true));
    manager.add_pass(std::make_unique<RecordingPass>("third", order, false, false));

    const forgeir::PassManagerResult result = manager.run(graph);
    EXPECT_FALSE(result.status.ok());
    EXPECT_TRUE(result.changed);
    EXPECT_EQ(order, (std::vector<std::string>{"first", "second"}));
    ASSERT_EQ(result.diagnostics.size(), 1U);
    EXPECT_EQ(result.diagnostics.front().operation_id, "op0002");
    EXPECT_EQ(result.diagnostics.front().value_id, "v0002");
    ASSERT_EQ(result.executions.size(), 5U);
    EXPECT_EQ(result.executions.back().pass_name, "second");
    EXPECT_EQ(result.executions.back().stage, "pass");
}

TEST(ShapeInference, RejectsInvalidMatMulContractingDimensions) {
    auto graph = load_golden_graph();
    graph["operations"][2]["type"] = "MatMul";
    graph["operations"][2]["attributes"] = nlohmann::json::object();
    graph["values"][1]["shape"] = nlohmann::json::array({3, 2});
    expect_verification_error(std::move(graph), "shape.matmul.contract");
}

TEST(ShapeInference, RejectsInvalidLinearWeightDimensions) {
    auto graph = load_golden_graph();
    graph["values"][1]["shape"] = nlohmann::json::array({2, 3});
    expect_verification_error(std::move(graph), "shape.linear.contract");
}

TEST(ShapeInference, RejectsInvalidLinearBiasDimensions) {
    auto graph = load_golden_graph();
    graph["values"].push_back(nlohmann::json{{"id", "v0003"},
                                             {"semantic_name", "bias"},
                                             {"shape", {3}},
                                             {"dtype", "float32"},
                                             {"kind", "parameter"}});
    graph["operations"].push_back(nlohmann::json{{"id", "op0003"},
                                                 {"type", "Parameter"},
                                                 {"semantic_name", "bias"},
                                                 {"inputs", nlohmann::json::array()},
                                                 {"outputs", {"v0003"}},
                                                 {"attributes",
                                                  {{"archive", "weight_tensors.npz"},
                                                   {"archive_key", "bias"},
                                                   {"content_sha256", std::string(64, '0')}}}});
    graph["operations"][2]["inputs"].push_back("v0003");
    graph["operations"][2]["attributes"]["bias"] = true;
    expect_verification_error(std::move(graph), "shape.linear.bias");
}

TEST(ShapeInference, RejectsIncompatibleBroadcast) {
    auto graph = load_golden_graph();
    graph["operations"][2]["type"] = "Add";
    graph["operations"][2]["attributes"] = nlohmann::json::object();
    graph["values"][1]["shape"] = nlohmann::json::array({3, 3});
    expect_verification_error(std::move(graph), "shape.broadcast.incompatible");
}

TEST(ShapeInference, AcceptsValidMulBroadcast) {
    auto graph_json = load_golden_graph();
    graph_json["operations"][2]["type"] = "Mul";
    graph_json["operations"][2]["attributes"] = nlohmann::json::object();
    graph_json["values"][1]["shape"] = nlohmann::json::array({2});
    forgeir::Graph graph = load_graph(std::move(graph_json));
    const forgeir::VerificationReport report = forgeir::verify_graph(graph);
    EXPECT_TRUE(report.success());
    EXPECT_EQ(report.pipeline.error_count(), 0U);
}

TEST(ShapeInference, AcceptsValidFloatDivisionBroadcast) {
    auto graph_json = load_golden_graph();
    graph_json["operations"][2]["type"] = "Div";
    graph_json["operations"][2]["attributes"] = nlohmann::json::object();
    graph_json["values"][1]["shape"] = nlohmann::json::array({2});
    forgeir::Graph graph = load_graph(std::move(graph_json));
    const forgeir::VerificationReport report = forgeir::verify_graph(graph);
    EXPECT_TRUE(report.success());
    EXPECT_EQ(report.pipeline.error_count(), 0U);
}

TEST(ShapeInference, RejectsRmsNormWeightMismatch) {
    auto graph = load_golden_graph();
    graph["operations"][2]["type"] = "RMSNorm";
    graph["operations"][2]["attributes"] = {{"axis", -1}, {"epsilon", 1.0e-5}};
    graph["values"][1]["shape"] = nlohmann::json::array({2, 2});
    expect_verification_error(std::move(graph), "shape.rms_norm.weight");
}

TEST(ShapeInference, RejectsSoftmaxAxisOutsideRank) {
    auto graph = load_golden_graph();
    graph["operations"][2]["type"] = "Softmax";
    graph["operations"][2]["inputs"] = nlohmann::json::array({"v0000"});
    graph["operations"][2]["attributes"] = {{"axis", 2}, {"stable", true}};
    expect_verification_error(std::move(graph), "shape.softmax.axis");
}

TEST(ShapeInference, RejectsReshapeElementCountMismatch) {
    auto graph = load_golden_graph();
    graph["operations"][2]["type"] = "Reshape";
    graph["operations"][2]["inputs"] = nlohmann::json::array({"v0000"});
    graph["operations"][2]["attributes"] = {{"shape", {3}}};
    graph["values"][2]["shape"] = nlohmann::json::array({3});
    expect_verification_error(std::move(graph), "shape.reshape.element_count");
}

TEST(ShapeInference, RejectsInvalidTransposePermutation) {
    auto graph = load_golden_graph();
    graph["operations"][2]["type"] = "Transpose";
    graph["operations"][2]["inputs"] = nlohmann::json::array({"v0000"});
    graph["operations"][2]["attributes"] = {{"permutation", {0, 0}}};
    expect_verification_error(std::move(graph), "shape.transpose.permutation");
}

TEST(ShapeInference, RejectsNonSquareAttentionMaskInput) {
    auto graph = load_golden_graph();
    graph["operations"][2]["type"] = "CausalMask";
    graph["operations"][2]["inputs"] = nlohmann::json::array({"v0000"});
    graph["operations"][2]["attributes"] = {{"diagonal", 0}, {"masked_value", "-inf"}};
    expect_verification_error(std::move(graph), "shape.causal_mask.attention");
}

TEST(DTypePropagation, RejectsOutputDTypeMismatch) {
    auto graph = load_golden_graph();
    graph["values"][2]["dtype"] = "int64";
    expect_verification_error(std::move(graph), "dtype.output.mismatch");
}
