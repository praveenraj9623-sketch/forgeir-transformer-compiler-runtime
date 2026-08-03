#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "forgeir/ir/graph.hpp"
#include "forgeir/ir/graph_writer.hpp"
#include "forgeir/passes/canonicalisation_pass.hpp"
#include "forgeir/passes/constant_folding_pass.hpp"
#include "forgeir/passes/dead_code_elimination_pass.hpp"
#include "forgeir/passes/fuse_bias_gelu_pass.hpp"
#include "forgeir/passes/optimization.hpp"
#include "forgeir/passes/pass_manager.hpp"
#include "forgeir/passes/redundant_reshape_elimination_pass.hpp"
#include "forgeir/passes/redundant_transpose_elimination_pass.hpp"
#include "forgeir/passes/verification.hpp"

namespace {

using forgeir::DataType;
using forgeir::Graph;
using forgeir::Operation;
using forgeir::OperationType;
using forgeir::Value;
using forgeir::ValueKind;

constexpr const char* kZeroHash =
    "0000000000000000000000000000000000000000000000000000000000000000";

forgeir::Shape make_shape(std::vector<std::int64_t> dimensions) {
    auto shape = forgeir::Shape::create(std::move(dimensions));
    if (!shape.ok()) {
        throw std::runtime_error(shape.status().message());
    }
    return shape.take_value();
}

Value make_value(std::string id, std::string name, std::vector<std::int64_t> dimensions,
                 const ValueKind kind, const DataType dtype = DataType::float32) {
    return Value(std::move(id), std::move(name),
                 forgeir::TensorDescriptor(dtype, make_shape(std::move(dimensions))), kind);
}

Operation make_operation(std::string id, const OperationType type, std::string name,
                         std::vector<std::string> inputs, std::string output,
                         nlohmann::json attributes = nlohmann::json::object()) {
    return Operation(std::move(id), type, std::move(name), std::move(inputs), {std::move(output)},
                     std::move(attributes));
}

nlohmann::json parameter_attributes(const std::string& key) {
    return {{"archive", "weight_tensors.npz"}, {"archive_key", key}, {"content_sha256", kZeroHash}};
}

Graph make_graph(std::vector<Value> values, std::vector<Operation> operations,
                 std::vector<std::string> outputs) {
    return Graph("1.0", "0.1.0", kZeroHash, kZeroHash,
                 forgeir::WeightManifestReference{"manifest.json", kZeroHash, "weight_tensors.npz",
                                                  kZeroHash},
                 {"v0000"}, std::move(outputs), std::move(values), std::move(operations));
}

Graph canonicalisation_graph() {
    std::vector<Value> values;
    values.push_back(make_value("v0000", "input", {1, 2}, ValueKind::input));
    values.push_back(make_value("v0001", "unused_weight", {2}, ValueKind::parameter));
    values.push_back(make_value("v0002", "softmax_output", {1, 2}, ValueKind::output));
    std::vector<Operation> operations;
    operations.push_back(make_operation("op0000", OperationType::input, "input", {}, "v0000"));
    operations.push_back(make_operation("op0002", OperationType::softmax, "softmax", {"v0000"},
                                        "v0002", {{"axis", -1}}));
    operations.push_back(make_operation("op0001", OperationType::parameter, "unused_weight", {},
                                        "v0001", parameter_attributes("unused_weight")));
    return make_graph(std::move(values), std::move(operations), {"v0002"});
}

Graph constant_folding_graph(const bool include_dead_constant = false) {
    std::vector<Value> values;
    values.push_back(make_value("v0000", "input", {}, ValueKind::input));
    values.push_back(make_value("v0001", "left", {}, ValueKind::constant));
    values.push_back(make_value("v0002", "right", {}, ValueKind::constant));
    values.push_back(make_value("v0003", "constant_sum", {}, ValueKind::intermediate));
    values.push_back(make_value("v0004", "output", {}, ValueKind::output));
    std::vector<Operation> operations;
    operations.push_back(make_operation("op0000", OperationType::input, "input", {}, "v0000"));
    operations.push_back(
        make_operation("op0001", OperationType::constant, "left", {}, "v0001", {{"value", 1.25}}));
    operations.push_back(
        make_operation("op0002", OperationType::constant, "right", {}, "v0002", {{"value", 2.75}}));
    operations.push_back(
        make_operation("op0003", OperationType::add, "constant_sum", {"v0001", "v0002"}, "v0003"));
    operations.push_back(
        make_operation("op0004", OperationType::add, "output", {"v0000", "v0003"}, "v0004"));
    if (include_dead_constant) {
        values.push_back(make_value("v0005", "dead", {}, ValueKind::constant));
        operations.push_back(make_operation("op0005", OperationType::constant, "dead", {}, "v0005",
                                            {{"value", 9.0}}));
    }
    return make_graph(std::move(values), std::move(operations), {"v0004"});
}

Graph dead_code_graph(const bool mark_side_effect = false) {
    Graph graph = constant_folding_graph(true);
    if (mark_side_effect) {
        graph.mutable_operations().back().mutable_attributes()["side_effect"] = true;
    }
    return graph;
}

Graph redundant_reshape_graph() {
    std::vector<Value> values;
    values.push_back(make_value("v0000", "input", {1, 2}, ValueKind::input));
    values.push_back(make_value("v0001", "reshaped", {1, 2}, ValueKind::intermediate));
    values.push_back(make_value("v0002", "gelu_output", {1, 2}, ValueKind::output));
    std::vector<Operation> operations;
    operations.push_back(make_operation("op0000", OperationType::input, "input", {}, "v0000"));
    operations.push_back(make_operation("op0001", OperationType::reshape, "reshape", {"v0000"},
                                        "v0001", {{"shape", {1, 2}}}));
    operations.push_back(make_operation("op0002", OperationType::gelu, "gelu", {"v0001"}, "v0002",
                                        {{"approximate", "none"}}));
    return make_graph(std::move(values), std::move(operations), {"v0002"});
}

Graph redundant_transpose_graph() {
    std::vector<Value> values;
    values.push_back(make_value("v0000", "input", {1, 2}, ValueKind::input));
    values.push_back(make_value("v0001", "transposed", {1, 2}, ValueKind::intermediate));
    values.push_back(make_value("v0002", "gelu_output", {1, 2}, ValueKind::output));
    std::vector<Operation> operations;
    operations.push_back(make_operation("op0000", OperationType::input, "input", {}, "v0000"));
    operations.push_back(make_operation("op0001", OperationType::transpose, "transpose", {"v0000"},
                                        "v0001", {{"permutation", {0, 1}}}));
    operations.push_back(make_operation("op0002", OperationType::gelu, "gelu", {"v0001"}, "v0002",
                                        {{"approximate", "none"}}));
    return make_graph(std::move(values), std::move(operations), {"v0002"});
}

Graph fusion_graph(const OperationType projection_type = OperationType::linear,
                   const bool add_redundant_reshape = false,
                   const bool include_dead_constant = false) {
    std::vector<Value> values;
    values.push_back(make_value("v0000", "input", {1, 2}, ValueKind::input));
    std::string projection_input = "v0000";
    std::vector<Operation> operations;
    operations.push_back(make_operation("op0000", OperationType::input, "input", {}, "v0000"));
    if (add_redundant_reshape) {
        values.push_back(make_value("v0001", "reshaped", {1, 2}, ValueKind::intermediate));
        operations.push_back(make_operation("op0001", OperationType::reshape, "reshape", {"v0000"},
                                            "v0001", {{"shape", {1, 2}}}));
        projection_input = "v0001";
    }
    const std::string weight_id = add_redundant_reshape ? "v0002" : "v0001";
    const std::string bias_id = add_redundant_reshape ? "v0003" : "v0002";
    const std::string projection_output = add_redundant_reshape ? "v0004" : "v0003";
    const std::string add_output = add_redundant_reshape ? "v0005" : "v0004";
    const std::string gelu_output = add_redundant_reshape ? "v0006" : "v0005";
    const std::string parameter_op = add_redundant_reshape ? "op0002" : "op0001";
    const std::string bias_op = add_redundant_reshape ? "op0003" : "op0002";
    const std::string projection_op = add_redundant_reshape ? "op0004" : "op0003";
    const std::string add_op = add_redundant_reshape ? "op0005" : "op0004";
    const std::string gelu_op = add_redundant_reshape ? "op0006" : "op0005";
    values.push_back(make_value(weight_id, "weight", {2, 2}, ValueKind::parameter));
    values.push_back(make_value(bias_id, "bias", {2}, ValueKind::parameter));
    values.push_back(make_value(projection_output, "projection", {1, 2}, ValueKind::intermediate));
    values.push_back(make_value(add_output, "biased", {1, 2}, ValueKind::intermediate));
    values.push_back(make_value(gelu_output, "gelu_output", {1, 2}, ValueKind::output));
    operations.push_back(make_operation(parameter_op, OperationType::parameter, "weight", {},
                                        weight_id, parameter_attributes("weight")));
    operations.push_back(make_operation(bias_op, OperationType::parameter, "bias", {}, bias_id,
                                        parameter_attributes("bias")));
    nlohmann::json projection_attributes = nlohmann::json::object();
    if (projection_type == OperationType::linear) {
        projection_attributes = {{"bias", false}, {"in_features", 2}, {"out_features", 2}};
    }
    operations.push_back(make_operation(projection_op, projection_type, "projection",
                                        {projection_input, weight_id}, projection_output,
                                        std::move(projection_attributes)));
    operations.push_back(make_operation(add_op, OperationType::add, "bias_add",
                                        {projection_output, bias_id}, add_output));
    operations.push_back(make_operation(gelu_op, OperationType::gelu, "gelu", {add_output},
                                        gelu_output, {{"approximate", "none"}}));
    if (include_dead_constant) {
        const std::string dead_value = add_redundant_reshape ? "v0007" : "v0006";
        const std::string dead_op = add_redundant_reshape ? "op0007" : "op0006";
        values.push_back(make_value(dead_value, "dead", {}, ValueKind::constant));
        operations.push_back(make_operation(dead_op, OperationType::constant, "dead", {},
                                            dead_value, {{"value", 9.0}}));
    }
    return make_graph(std::move(values), std::move(operations), {gelu_output});
}

nlohmann::json graph_snapshot(const Graph& graph) {
    nlohmann::json operations = nlohmann::json::array();
    for (const Operation& operation : graph.operations()) {
        operations.push_back({{"id", operation.id()},
                              {"type", forgeir::to_string(operation.type())},
                              {"inputs", operation.input_ids()},
                              {"outputs", operation.output_ids()}});
    }
    return {{"operation_count", graph.operations().size()},
            {"value_count", graph.values().size()},
            {"outputs", graph.output_ids()},
            {"operations", std::move(operations)}};
}

const nlohmann::json& golden(const std::string& name) {
    static const nlohmann::json expectations = [] {
        const std::filesystem::path path = std::filesystem::path(FORGEIR_SOURCE_DIR) /
                                           "tests/golden/milestone_06_pass_expectations.json";
        std::ifstream stream(path);
        if (!stream) {
            throw std::runtime_error("cannot open optimization golden expectations");
        }
        return nlohmann::json::parse(stream);
    }();
    return expectations.at(name);
}

template <typename PassType> forgeir::PassManagerResult run_pass(Graph& graph) {
    forgeir::PassManager manager;
    manager.add_pass(std::make_unique<PassType>());
    return manager.run(graph);
}

template <typename PassType> void expect_idempotent(Graph& graph) {
    const forgeir::PassManagerResult second = run_pass<PassType>(graph);
    ASSERT_TRUE(second.status.ok()) << second.status.message();
    EXPECT_FALSE(second.changed);
    EXPECT_TRUE(second.rewrites.empty());
}

void expect_valid(Graph& graph) {
    const forgeir::VerificationReport report = forgeir::verify_graph(graph);
    ASSERT_TRUE(report.success()) << report.pipeline.status.message();
    EXPECT_EQ(report.pipeline.error_count(), 0U);
}

} // namespace

TEST(OptimizationGolden, CanonicalisationNormalizesAxesAttributesAndOrder) {
    Graph graph = canonicalisation_graph();
    const forgeir::PassManagerResult result = run_pass<forgeir::CanonicalisationPass>(graph);
    ASSERT_TRUE(result.status.ok()) << result.status.message();
    EXPECT_TRUE(result.changed);
    EXPECT_FALSE(result.rewrites.empty());
    EXPECT_EQ(graph.operations()[2].attributes().at("axis"), 1);
    EXPECT_TRUE(graph.operations()[2].attributes().at("stable").get<bool>());
    EXPECT_EQ(graph_snapshot(graph), golden("canonicalisation"));
    expect_valid(graph);
    expect_idempotent<forgeir::CanonicalisationPass>(graph);
}

TEST(OptimizationSafety, CanonicalisationPreservesRelativeSideEffectOrder) {
    Graph graph = canonicalisation_graph();
    graph.mutable_operations()[1].mutable_attributes()["side_effect"] = true;
    graph.mutable_operations()[2].mutable_attributes()["side_effect"] = true;
    const forgeir::PassManagerResult result = run_pass<forgeir::CanonicalisationPass>(graph);
    ASSERT_TRUE(result.status.ok()) << result.status.message();
    ASSERT_EQ(graph.operations().size(), 3U);
    EXPECT_EQ(graph.operations()[1].id(), "op0002");
    EXPECT_EQ(graph.operations()[2].id(), "op0001");
}

TEST(OptimizationGolden, ConstantFoldingFoldsOnlyTheBoundedScalarSubgraph) {
    Graph graph = constant_folding_graph();
    const forgeir::PassManagerResult result = run_pass<forgeir::ConstantFoldingPass>(graph);
    ASSERT_TRUE(result.status.ok()) << result.status.message();
    EXPECT_TRUE(result.changed);
    EXPECT_EQ(graph.operations()[3].type(), OperationType::constant);
    EXPECT_DOUBLE_EQ(graph.operations()[3].attributes().at("value").get<double>(), 4.0);
    EXPECT_EQ(graph.operations()[4].type(), OperationType::add);
    EXPECT_EQ(forgeir::kMaximumFoldedTensorElements, 1024U);
    EXPECT_EQ(graph_snapshot(graph), golden("constant_folding"));
    expect_valid(graph);
    expect_idempotent<forgeir::ConstantFoldingPass>(graph);

    Graph side_effect_graph = constant_folding_graph();
    side_effect_graph.mutable_operations()[3].mutable_attributes()["side_effect"] = true;
    const forgeir::PassManagerResult side_effect_result =
        run_pass<forgeir::ConstantFoldingPass>(side_effect_graph);
    EXPECT_TRUE(side_effect_result.status.ok());
    EXPECT_FALSE(side_effect_result.changed);
    EXPECT_EQ(side_effect_graph.operations()[3].type(), OperationType::add);
}

TEST(OptimizationGolden, DeadCodeEliminationPreservesOutputsAndSideEffects) {
    Graph graph = dead_code_graph();
    const forgeir::PassManagerResult result = run_pass<forgeir::DeadCodeEliminationPass>(graph);
    ASSERT_TRUE(result.status.ok()) << result.status.message();
    EXPECT_TRUE(result.changed);
    EXPECT_EQ(graph_snapshot(graph), golden("dead_code_elimination"));
    EXPECT_EQ(graph.output_ids(), (std::vector<std::string>{"v0004"}));
    expect_valid(graph);
    expect_idempotent<forgeir::DeadCodeEliminationPass>(graph);

    Graph side_effect_graph = dead_code_graph(true);
    const forgeir::PassManagerResult side_effect_result =
        run_pass<forgeir::DeadCodeEliminationPass>(side_effect_graph);
    EXPECT_TRUE(side_effect_result.status.ok());
    EXPECT_FALSE(side_effect_result.changed);
    EXPECT_EQ(side_effect_graph.operations().size(), 6U);
}

TEST(OptimizationGolden, RedundantReshapePreservesTheTerminalOutputValue) {
    Graph graph = redundant_reshape_graph();
    const forgeir::PassManagerResult result =
        run_pass<forgeir::RedundantReshapeEliminationPass>(graph);
    ASSERT_TRUE(result.status.ok()) << result.status.message();
    EXPECT_TRUE(result.changed);
    EXPECT_EQ(graph_snapshot(graph), golden("redundant_reshape"));
    EXPECT_EQ(graph.operations().back().input_ids().front(), "v0000");
    expect_valid(graph);
    expect_idempotent<forgeir::RedundantReshapeEliminationPass>(graph);
}

TEST(OptimizationGolden, RedundantTransposePreservesTheTerminalOutputValue) {
    Graph graph = redundant_transpose_graph();
    const forgeir::PassManagerResult result =
        run_pass<forgeir::RedundantTransposeEliminationPass>(graph);
    ASSERT_TRUE(result.status.ok()) << result.status.message();
    EXPECT_TRUE(result.changed);
    EXPECT_EQ(graph_snapshot(graph), golden("redundant_transpose"));
    EXPECT_EQ(graph.operations().back().input_ids().front(), "v0000");
    expect_valid(graph);
    expect_idempotent<forgeir::RedundantTransposeEliminationPass>(graph);
}

TEST(OptimizationGolden, FuseBiasGELUPreservesStableOutputIdentityForLinearAndMatMul) {
    Graph linear_graph = fusion_graph();
    const forgeir::PassManagerResult linear = run_pass<forgeir::FuseBiasGELUPass>(linear_graph);
    ASSERT_TRUE(linear.status.ok()) << linear.status.message();
    EXPECT_TRUE(linear.changed);
    EXPECT_EQ(graph_snapshot(linear_graph), golden("fuse_bias_gelu"));
    EXPECT_EQ(linear_graph.operations().back().attributes().at("fused_activation"), "GELU");
    EXPECT_EQ(linear_graph.values().back().semantic_name(), "gelu_output");
    expect_valid(linear_graph);
    expect_idempotent<forgeir::FuseBiasGELUPass>(linear_graph);

    Graph matmul_graph = fusion_graph(OperationType::mat_mul);
    const forgeir::PassManagerResult matmul = run_pass<forgeir::FuseBiasGELUPass>(matmul_graph);
    ASSERT_TRUE(matmul.status.ok()) << matmul.status.message();
    EXPECT_TRUE(matmul.changed);
    EXPECT_EQ(matmul_graph.operations().back().type(), OperationType::mat_mul);
    EXPECT_EQ(matmul_graph.operations().back().input_ids().size(), 3U);
    expect_valid(matmul_graph);
    expect_idempotent<forgeir::FuseBiasGELUPass>(matmul_graph);
}

TEST(OptimizationGolden, O1CombinedPipelineMatchesGoldenAndIsIdempotent) {
    Graph graph = constant_folding_graph(true);
    const forgeir::OptimizationResult first =
        forgeir::optimize_graph(graph, forgeir::OptimizationLevel::o1);
    ASSERT_TRUE(first.success()) << first.pipeline.status.message();
    EXPECT_TRUE(first.pipeline.changed);
    EXPECT_EQ(graph_snapshot(graph), golden("O1"));
    EXPECT_EQ(first.operations_before, 6U);
    EXPECT_EQ(first.operations_after, 3U);
    expect_valid(graph);

    const forgeir::OptimizationResult second =
        forgeir::optimize_graph(graph, forgeir::OptimizationLevel::o1);
    EXPECT_TRUE(second.success());
    EXPECT_FALSE(second.pipeline.changed);
    EXPECT_EQ(second.operations_before, second.operations_after);
}

TEST(OptimizationGolden, O0PerformsVerificationOnly) {
    Graph graph = fusion_graph();
    const nlohmann::json before = forgeir::graph_json(graph);
    const forgeir::OptimizationResult result =
        forgeir::optimize_graph(graph, forgeir::OptimizationLevel::o0);
    ASSERT_TRUE(result.success()) << result.pipeline.status.message();
    EXPECT_FALSE(result.pipeline.changed);
    EXPECT_EQ(result.operations_before, result.operations_after);
    EXPECT_EQ(graph_snapshot(graph), golden("O0"));
    nlohmann::json expected = before;
    expected["graph_hash"] = graph.graph_hash();
    EXPECT_EQ(forgeir::graph_json(graph), expected);
}

TEST(OptimizationGolden, O2CombinedPipelineMatchesGoldenAndRecordsEveryRewriteReason) {
    Graph graph = fusion_graph(OperationType::linear, true, true);
    const forgeir::OptimizationResult first =
        forgeir::optimize_graph(graph, forgeir::OptimizationLevel::o2);
    ASSERT_TRUE(first.success()) << first.pipeline.status.message();
    EXPECT_TRUE(first.pipeline.changed);
    EXPECT_EQ(graph_snapshot(graph), golden("O2"));
    EXPECT_EQ(first.operations_before, 8U);
    EXPECT_EQ(first.operations_after, 4U);
    ASSERT_FALSE(first.pipeline.rewrites.empty());
    for (const forgeir::RewriteRecord& rewrite : first.pipeline.rewrites) {
        EXPECT_FALSE(rewrite.reason.empty());
    }
    expect_valid(graph);

    const forgeir::OptimizationResult second =
        forgeir::optimize_graph(graph, forgeir::OptimizationLevel::o2);
    EXPECT_TRUE(second.success());
    EXPECT_FALSE(second.pipeline.changed);
    EXPECT_EQ(second.operations_before, second.operations_after);

    Graph repeated_graph = fusion_graph(OperationType::linear, true, true);
    const forgeir::OptimizationResult repeated =
        forgeir::optimize_graph(repeated_graph, forgeir::OptimizationLevel::o2);
    ASSERT_TRUE(repeated.success()) << repeated.pipeline.status.message();
    EXPECT_EQ(repeated.output_graph_hash, first.output_graph_hash);
    EXPECT_EQ(forgeir::graph_json(repeated_graph), forgeir::graph_json(graph));
}
