#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "forgeir/ir/graph_loader.hpp"
#include "forgeir/mlir/stablehlo_lowering.hpp"

namespace forgeir {
namespace {

Shape shape(std::vector<std::int64_t> dimensions) {
    auto result = Shape::create(std::move(dimensions));
    EXPECT_TRUE(result.ok());
    return result.take_value();
}

Value tensor_value(std::string id, std::vector<std::int64_t> dimensions, const ValueKind kind,
                   const DataType data_type = DataType::float32) {
    const std::string semantic_name = id;
    return Value(std::move(id), semantic_name,
                 TensorDescriptor(data_type, shape(std::move(dimensions))), kind);
}

WeightManifestReference empty_manifest() { return {"", "", "", ""}; }

Graph unary_graph(const OperationType type, std::vector<std::int64_t> input_shape,
                  std::vector<std::int64_t> output_shape, nlohmann::json attributes) {
    std::vector<Value> values;
    values.push_back(tensor_value("v0000", std::move(input_shape), ValueKind::input));
    values.push_back(tensor_value("v0001", std::move(output_shape), ValueKind::output));
    std::vector<Operation> operations;
    operations.emplace_back("op0000", OperationType::input, "input", std::vector<std::string>{},
                            std::vector<std::string>{"v0000"}, nlohmann::json::object());
    operations.emplace_back("op0001", type, "operation", std::vector<std::string>{"v0000"},
                            std::vector<std::string>{"v0001"}, std::move(attributes));
    return Graph("1.0", "0.1.0", std::string(64, '0'), std::string(64, '1'), empty_manifest(),
                 {"v0000"}, {"v0001"}, std::move(values), std::move(operations));
}

Graph binary_graph(const OperationType type, std::vector<std::int64_t> left_shape,
                   std::vector<std::int64_t> right_shape, std::vector<std::int64_t> output_shape) {
    std::vector<Value> values;
    values.push_back(tensor_value("v0000", std::move(left_shape), ValueKind::input));
    values.push_back(tensor_value("v0001", std::move(right_shape), ValueKind::input));
    values.push_back(tensor_value("v0002", std::move(output_shape), ValueKind::output));
    std::vector<Operation> operations;
    operations.emplace_back("op0000", OperationType::input, "left", std::vector<std::string>{},
                            std::vector<std::string>{"v0000"}, nlohmann::json::object());
    operations.emplace_back("op0001", OperationType::input, "right", std::vector<std::string>{},
                            std::vector<std::string>{"v0001"}, nlohmann::json::object());
    operations.emplace_back("op0002", type, "binary", std::vector<std::string>{"v0000", "v0001"},
                            std::vector<std::string>{"v0002"}, nlohmann::json::object());
    return Graph("1.0", "0.1.0", std::string(64, '0'), std::string(64, '1'), empty_manifest(),
                 {"v0000", "v0001"}, {"v0002"}, std::move(values), std::move(operations));
}

Graph rms_norm_graph() {
    std::vector<Value> values;
    values.push_back(tensor_value("v0000", {2, 4}, ValueKind::input));
    values.push_back(tensor_value("v0001", {4}, ValueKind::parameter));
    values.push_back(tensor_value("v0002", {2, 4}, ValueKind::output));
    std::vector<Operation> operations;
    operations.emplace_back("op0000", OperationType::input, "input", std::vector<std::string>{},
                            std::vector<std::string>{"v0000"}, nlohmann::json::object());
    operations.emplace_back("op0001", OperationType::parameter, "weight",
                            std::vector<std::string>{}, std::vector<std::string>{"v0001"},
                            nlohmann::json::object());
    operations.emplace_back(
        "op0002", OperationType::rms_norm, "rms_norm", std::vector<std::string>{"v0000", "v0001"},
        std::vector<std::string>{"v0002"}, nlohmann::json{{"axis", -1}, {"epsilon", 1.0e-5}});
    return Graph("1.0", "0.1.0", std::string(64, '0'), std::string(64, '1'), empty_manifest(),
                 {"v0000"}, {"v0002"}, std::move(values), std::move(operations));
}

Graph scalar_constant_graph() {
    std::vector<Value> values;
    values.push_back(tensor_value("v0000", {}, ValueKind::output));
    std::vector<Operation> operations;
    operations.emplace_back("op0000", OperationType::constant, "constant",
                            std::vector<std::string>{}, std::vector<std::string>{"v0000"},
                            nlohmann::json{{"value", 0.5}});
    return Graph("1.0", "0.1.0", std::string(64, '0'), std::string(64, '1'), empty_manifest(), {},
                 {"v0000"}, std::move(values), std::move(operations));
}

TEST(MlirBridge, GoldenLinearLoweringIsDeterministic) {
    const std::filesystem::path source_dir = FORGEIR_SOURCE_DIR;
    auto graph = GraphLoader::load_from_file(source_dir / "tests" / "golden" /
                                             "milestone_04_valid_graph.json");
    ASSERT_TRUE(graph.ok()) << graph.status().message();
    const MlirLoweringResult first = lower_to_stablehlo(graph.value());
    const MlirLoweringResult second = lower_to_stablehlo(graph.value());
    ASSERT_TRUE(first.success()) << first.status.message();
    ASSERT_TRUE(second.success()) << second.status.message();
    EXPECT_EQ(first.module_text, second.module_text);

    std::ifstream golden_stream(source_dir / "tests" / "golden" / "milestone_11_linear.mlir",
                                std::ios::binary);
    ASSERT_TRUE(golden_stream);
    const std::string golden{std::istreambuf_iterator<char>(golden_stream),
                             std::istreambuf_iterator<char>()};
    EXPECT_EQ(first.module_text, golden);
}

TEST(MlirBridge, EmitsExplicitRankedTypesAndSourceOperationIds) {
    Graph graph = unary_graph(OperationType::transpose, {2, 3}, {3, 2},
                              nlohmann::json{{"permutation", {1, 0}}});
    const MlirLoweringResult result = lower_to_stablehlo(graph);
    ASSERT_TRUE(result.success()) << result.status.message();
    EXPECT_NE(result.module_text.find("tensor<2x3xf32>"), std::string::npos);
    EXPECT_NE(result.module_text.find("tensor<3x2xf32>"), std::string::npos);
    EXPECT_NE(result.module_text.find("forgeir.op_id = op0001"), std::string::npos);
    EXPECT_NE(result.module_text.find("permutation = array<i64: 1, 0>"), std::string::npos);
}

TEST(MlirBridge, LowersRequiredStablehloOperationsAndBroadcasting) {
    const std::vector<std::pair<Graph, std::string>> cases{
        {scalar_constant_graph(), "stablehlo.constant"},
        {binary_graph(OperationType::mat_mul, {2, 3}, {3, 4}, {2, 4}), "stablehlo.dot_general"},
        {binary_graph(OperationType::add, {2, 4}, {4}, {2, 4}), "stablehlo.broadcast_in_dim"},
        {binary_graph(OperationType::multiply, {2, 4}, {2, 4}, {2, 4}), "stablehlo.multiply"},
        {binary_graph(OperationType::divide, {2, 4}, {2, 4}, {2, 4}), "stablehlo.divide"},
        {unary_graph(OperationType::reshape, {2, 4}, {4, 2}, nlohmann::json{{"shape", {4, 2}}}),
         "stablehlo.reshape"},
    };
    for (const auto& test_case : cases) {
        const MlirLoweringResult result = lower_to_stablehlo(test_case.first);
        ASSERT_TRUE(result.success()) << result.status.message();
        EXPECT_NE(result.module_text.find(test_case.second), std::string::npos);
    }
}

TEST(MlirBridge, DecomposesGeluRmsNormAndStableSoftmax) {
    Graph gelu =
        unary_graph(OperationType::gelu, {2, 4}, {2, 4}, nlohmann::json{{"approximate", "none"}});
    const MlirLoweringResult gelu_result = lower_to_stablehlo(gelu);
    ASSERT_TRUE(gelu_result.success()) << gelu_result.status.message();
    EXPECT_NE(gelu_result.module_text.find("chlo.erf"), std::string::npos);
    EXPECT_NE(gelu_result.module_text.find("stablehlo.divide"), std::string::npos);

    Graph rms_norm = rms_norm_graph();
    const MlirLoweringResult rms_result = lower_to_stablehlo(rms_norm);
    ASSERT_TRUE(rms_result.success()) << rms_result.status.message();
    EXPECT_NE(rms_result.module_text.find("stablehlo.reduce"), std::string::npos);
    EXPECT_NE(rms_result.module_text.find("stablehlo.rsqrt"), std::string::npos);

    Graph softmax =
        unary_graph(OperationType::softmax, {2, 4}, {2, 4}, nlohmann::json{{"axis", -1}});
    const MlirLoweringResult softmax_result = lower_to_stablehlo(softmax);
    ASSERT_TRUE(softmax_result.success()) << softmax_result.status.message();
    EXPECT_NE(softmax_result.module_text.find("stablehlo.maximum"), std::string::npos);
    EXPECT_NE(softmax_result.module_text.find("stablehlo.exponential"), std::string::npos);
    EXPECT_NE(softmax_result.module_text.find("stablehlo.subtract"), std::string::npos);
}

TEST(MlirBridge, UnsupportedOperationReturnsStructuredDiagnostic) {
    Graph graph = unary_graph(OperationType::causal_mask, {1, 1, 2, 2}, {1, 1, 2, 2},
                              nlohmann::json::object());
    const MlirLoweringResult result = lower_to_stablehlo(graph);
    ASSERT_FALSE(result.success());
    EXPECT_EQ(result.status.code(), StatusCode::unsupported);
    ASSERT_EQ(result.diagnostics.size(), 1U);
    EXPECT_EQ(result.diagnostics[0].code, "mlir.unsupported_operation");
    EXPECT_EQ(result.diagnostics[0].operation_id, "op0001");
    EXPECT_EQ(result.diagnostics[0].value_id, "v0001");
}

TEST(MlirBridge, ReportsToolUnavailableWithoutClaimingValidation) {
    const std::filesystem::path output =
        std::filesystem::temp_directory_path() / "forgeir_mlir_bridge_tool_probe.mlir";
    const Status write = write_mlir_module(output, "module {}\n");
    ASSERT_TRUE(write.ok()) << write.message();
    const MlirToolValidationResult validation = validate_mlir_module(output);
    if (validation.status == "tool unavailable") {
        EXPECT_FALSE(validation.external_validation_available);
        EXPECT_FALSE(validation.syntax_verified);
        EXPECT_FALSE(validation.canonicalization_succeeded);
    } else {
        EXPECT_TRUE(validation.external_validation_available);
    }
    std::error_code error;
    std::filesystem::remove(output, error);
}

} // namespace
} // namespace forgeir
