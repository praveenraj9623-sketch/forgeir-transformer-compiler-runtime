#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "forgeir/core/sha256.hpp"
#include "forgeir/ir/graph_loader.hpp"

namespace {

const std::filesystem::path& golden_graph_path() {
    static const std::filesystem::path path = std::filesystem::path(FORGEIR_SOURCE_DIR) / "tests" /
                                              "golden" / "milestone_04_valid_graph.json";
    return path;
}

nlohmann::json load_golden_json() {
    std::ifstream stream(golden_graph_path());
    return nlohmann::json::parse(stream);
}

void refresh_graph_hash(nlohmann::json& graph) {
    graph.erase("graph_hash");
    graph["graph_hash"] = forgeir::sha256(graph.dump());
}

void expect_failure_contains(const nlohmann::json& graph, const std::string& message_fragment) {
    const auto result = forgeir::GraphLoader::load_from_json(graph.dump());
    ASSERT_FALSE(result.ok());
    EXPECT_NE(result.status().message().find(message_fragment), std::string::npos)
        << result.status().message();
}

} // namespace

TEST(GraphLoader, LoadsValidGoldenGraphAndBuildsSummary) {
    const auto result = forgeir::GraphLoader::load_from_file(golden_graph_path());
    ASSERT_TRUE(result.ok()) << result.status().message();

    const auto summary = result.value().summary();
    ASSERT_TRUE(summary.ok()) << summary.status().message();
    EXPECT_EQ(summary.value().schema_version, "1.0");
    EXPECT_EQ(summary.value().input_count, 1U);
    EXPECT_EQ(summary.value().output_count, 1U);
    EXPECT_EQ(summary.value().value_count, 3U);
    EXPECT_EQ(summary.value().operation_count, 3U);
    EXPECT_EQ(summary.value().operation_histogram.at("Input"), 1U);
    EXPECT_EQ(summary.value().operation_histogram.at("Parameter"), 1U);
    EXPECT_EQ(summary.value().operation_histogram.at("Linear"), 1U);
    EXPECT_EQ(summary.value().estimated_parameter_bytes, 16U);
}

TEST(GraphLoader, RejectsMalformedJson) {
    const auto result = forgeir::GraphLoader::load_from_json("{not-json");
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), forgeir::StatusCode::parse_error);
}

TEST(GraphLoader, RejectsMissingOperandId) {
    auto graph = load_golden_json();
    graph["operations"][2]["inputs"][0] = "v9999";
    expect_failure_contains(graph, "missing operand v9999");
}

TEST(GraphLoader, RejectsMissingOutputId) {
    auto graph = load_golden_json();
    graph["operations"][2]["outputs"][0] = "v9999";
    expect_failure_contains(graph, "missing output v9999");
}

TEST(GraphLoader, RejectsCycle) {
    auto graph = load_golden_json();
    graph["operations"][2]["inputs"][0] = "v0002";
    expect_failure_contains(graph, "cycle");
}

TEST(GraphLoader, RejectsOverflowingDimensions) {
    auto graph = load_golden_json();
    graph["values"][0]["shape"] = nlohmann::json::array(
        {std::numeric_limits<std::int64_t>::max(), std::numeric_limits<std::int64_t>::max()});
    expect_failure_contains(graph, "element count");
}

TEST(GraphLoader, RejectsOverflowingTensorByteSize) {
    auto graph = load_golden_json();
    graph["values"][0]["shape"] =
        nlohmann::json::array({std::numeric_limits<std::int64_t>::max(), 2});
    expect_failure_contains(graph, "byte size");
}

TEST(GraphLoader, RejectsNegativeDimensions) {
    auto graph = load_golden_json();
    graph["values"][0]["shape"] = nlohmann::json::array({1, -1});
    expect_failure_contains(graph, "must not be negative");
}

TEST(GraphLoader, RejectsUnsupportedOperation) {
    auto graph = load_golden_json();
    graph["operations"][2]["type"] = "Convolution";
    expect_failure_contains(graph, "unsupported graph operation");
}

TEST(GraphLoader, RejectsDuplicateValueId) {
    auto graph = load_golden_json();
    graph["values"].push_back(graph["values"][0]);
    expect_failure_contains(graph, "duplicate value ID");
}

TEST(GraphLoader, RejectsInvalidDtype) {
    auto graph = load_golden_json();
    graph["values"][0]["dtype"] = "float16";
    expect_failure_contains(graph, "unsupported dtype");
}

TEST(GraphLoader, RejectsZeroDimensions) {
    auto graph = load_golden_json();
    graph["values"][0]["shape"] = nlohmann::json::array({1, 0});
    expect_failure_contains(graph, "zero shape dimensions");
}

TEST(GraphLoader, RejectsUnsupportedSchemaVersion) {
    auto graph = load_golden_json();
    graph["graph_schema_version"] = "2.0";
    expect_failure_contains(graph, "unsupported graph schema version");
}

TEST(GraphLoader, RejectsInvalidGraphHashSyntax) {
    auto graph = load_golden_json();
    graph["graph_hash"] = "not-a-sha256";
    expect_failure_contains(graph, "lowercase SHA-256");
}

TEST(GraphLoader, RejectsInvalidParameterContentHash) {
    auto graph = load_golden_json();
    graph["operations"][1]["attributes"]["content_sha256"] = "invalid";
    expect_failure_contains(graph, "parameter content_sha256");
}

TEST(GraphLoader, RejectsGraphHashMismatch) {
    auto graph = load_golden_json();
    graph["producer_version"] = "tampered";
    expect_failure_contains(graph, "does not match");
}

TEST(GraphLoader, RejectsDisconnectedRequiredOutput) {
    auto graph = load_golden_json();
    graph["operations"][2]["inputs"] = nlohmann::json::array({"v0001", "v0001"});
    refresh_graph_hash(graph);
    expect_failure_contains(graph, "disconnected");
}

TEST(GraphLoader, ProducesDeterministicTopologicalOperationOrder) {
    auto graph = load_golden_json();
    std::swap(graph["operations"][0], graph["operations"][2]);
    refresh_graph_hash(graph);

    const auto result = forgeir::GraphLoader::load_from_json(graph.dump());
    ASSERT_TRUE(result.ok()) << result.status().message();
    ASSERT_EQ(result.value().operations().size(), 3U);
    EXPECT_EQ(result.value().operations()[0].id(), "op0000");
    EXPECT_EQ(result.value().operations()[1].id(), "op0001");
    EXPECT_EQ(result.value().operations()[2].id(), "op0002");
}
