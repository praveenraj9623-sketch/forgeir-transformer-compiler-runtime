#include <cstdint>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "forgeir/ir/graph_loader.hpp"
#include "forgeir/runtime/execution_schedule.hpp"
#include "forgeir/runtime/memory_planner.hpp"

namespace {

using forgeir::DataType;
using forgeir::Graph;
using forgeir::Operation;
using forgeir::OperationType;
using forgeir::TensorLifetime;
using forgeir::Value;
using forgeir::ValueKind;

constexpr const char* kZeroHash =
    "0000000000000000000000000000000000000000000000000000000000000000";

forgeir::Shape make_shape(std::vector<std::int64_t> dimensions,
                          const bool allow_zero_dimensions = false) {
    auto shape = forgeir::Shape::create(std::move(dimensions), allow_zero_dimensions);
    if (!shape.ok()) {
        throw std::runtime_error(shape.status().message());
    }
    return shape.take_value();
}

Value make_value(std::string id, std::string name, std::vector<std::int64_t> dimensions,
                 const ValueKind kind, const DataType dtype = DataType::float32,
                 const bool allow_zero_dimensions = false) {
    return Value(
        std::move(id), std::move(name),
        forgeir::TensorDescriptor(dtype, make_shape(std::move(dimensions), allow_zero_dimensions)),
        kind);
}

Operation make_operation(std::string id, const OperationType type, std::string name,
                         std::vector<std::string> inputs, std::string output) {
    nlohmann::json attributes = nlohmann::json::object();
    if (type == OperationType::gelu) {
        attributes["approximate"] = "none";
    }
    return Operation(std::move(id), type, std::move(name), std::move(inputs), {std::move(output)},
                     std::move(attributes));
}

Graph make_graph(std::vector<Value> values, std::vector<Operation> operations,
                 std::vector<std::string> outputs) {
    return Graph("1.0", "0.1.0", kZeroHash, kZeroHash,
                 forgeir::WeightManifestReference{"manifest.json", kZeroHash, "weight_tensors.npz",
                                                  kZeroHash},
                 {"v0000"}, std::move(outputs), std::move(values), std::move(operations));
}

Graph make_chain_graph() {
    std::vector<Value> values;
    values.push_back(make_value("v0000", "input", {4}, ValueKind::input));
    values.push_back(make_value("v0001", "first", {4}, ValueKind::intermediate));
    values.push_back(make_value("v0002", "second", {4}, ValueKind::intermediate));
    values.push_back(make_value("v0003", "third", {4}, ValueKind::intermediate));
    values.push_back(make_value("v0004", "output", {4}, ValueKind::output));
    std::vector<Operation> operations;
    operations.push_back(make_operation("op0000", OperationType::input, "input", {}, "v0000"));
    operations.push_back(
        make_operation("op0001", OperationType::gelu, "first", {"v0000"}, "v0001"));
    operations.push_back(
        make_operation("op0002", OperationType::gelu, "second", {"v0001"}, "v0002"));
    operations.push_back(
        make_operation("op0003", OperationType::gelu, "third", {"v0002"}, "v0003"));
    operations.push_back(
        make_operation("op0004", OperationType::gelu, "output", {"v0003"}, "v0004"));
    return make_graph(std::move(values), std::move(operations), {"v0004"});
}

Graph make_best_fit_graph() {
    std::vector<Value> values;
    values.push_back(make_value("v0000", "large_input", {32}, ValueKind::input));
    values.push_back(make_value("v0001", "small_input", {16}, ValueKind::input));
    values.push_back(make_value("v0002", "large_temporary", {32}, ValueKind::intermediate));
    values.push_back(make_value("v0003", "separator", {16}, ValueKind::intermediate));
    values.push_back(make_value("v0004", "small_temporary", {16}, ValueKind::intermediate));
    values.push_back(make_value("v0005", "large_output", {32}, ValueKind::output));
    values.push_back(make_value("v0006", "small_output", {16}, ValueKind::output));
    values.push_back(make_value("v0007", "best_fit_request", {16}, ValueKind::intermediate));
    values.push_back(make_value("v0008", "combined_output", {16}, ValueKind::output));
    std::vector<Operation> operations;
    operations.push_back(
        make_operation("op0000", OperationType::input, "large_input", {}, "v0000"));
    operations.push_back(
        make_operation("op0001", OperationType::input, "small_input", {}, "v0001"));
    operations.push_back(
        make_operation("op0002", OperationType::gelu, "large_temporary", {"v0000"}, "v0002"));
    operations.push_back(
        make_operation("op0003", OperationType::gelu, "separator", {"v0001"}, "v0003"));
    operations.push_back(
        make_operation("op0004", OperationType::gelu, "small_temporary", {"v0001"}, "v0004"));
    operations.push_back(
        make_operation("op0005", OperationType::gelu, "large_output", {"v0002"}, "v0005"));
    operations.push_back(
        make_operation("op0006", OperationType::gelu, "small_output", {"v0004"}, "v0006"));
    operations.push_back(
        make_operation("op0007", OperationType::gelu, "best_fit_request", {"v0001"}, "v0007"));
    operations.push_back(make_operation("op0008", OperationType::add, "combined_output",
                                        {"v0003", "v0007"}, "v0008"));
    return Graph("1.0", "0.1.0", kZeroHash, kZeroHash,
                 forgeir::WeightManifestReference{"manifest.json", kZeroHash, "weight_tensors.npz",
                                                  kZeroHash},
                 {"v0000", "v0001"}, {"v0005", "v0006", "v0008"}, std::move(values),
                 std::move(operations));
}

const TensorLifetime& lifetime(const forgeir::MemoryPlan& plan, const std::string& value_id) {
    for (const TensorLifetime& tensor : plan.tensors) {
        if (tensor.value_id == value_id) {
            return tensor;
        }
    }
    throw std::runtime_error("missing tensor lifetime " + value_id);
}

forgeir::MemoryPlan plan_chain(const forgeir::MemoryPlannerOptions& options = {}) {
    Graph graph = make_chain_graph();
    auto plan = forgeir::plan_memory(graph, options);
    if (!plan.ok()) {
        throw std::runtime_error(plan.status().message());
    }
    return plan.take_value();
}

} // namespace

TEST(ExecutionSchedule, IsDeterministicTopologicalAndUsesStableIds) {
    Graph graph = make_chain_graph();
    std::reverse(graph.mutable_operations().begin(), graph.mutable_operations().end());
    const auto first = forgeir::build_execution_schedule(graph);
    const auto second = forgeir::build_execution_schedule(graph);
    ASSERT_TRUE(first.ok()) << first.status().message();
    ASSERT_TRUE(second.ok()) << second.status().message();
    EXPECT_EQ(forgeir::execution_schedule_json(first.value()),
              forgeir::execution_schedule_json(second.value()));
    ASSERT_EQ(first.value().operations.size(), 5U);
    for (std::size_t index = 0; index < first.value().operations.size(); ++index) {
        EXPECT_EQ(first.value().operations[index].index, index);
        EXPECT_EQ(first.value().operations[index].operation_id, "op000" + std::to_string(index));
    }
}

TEST(MemoryPlanner, ComputesDefinitionFirstUseFinalUseAndInclusiveLifetime) {
    const forgeir::MemoryPlan plan = plan_chain();
    const TensorLifetime& first = lifetime(plan, "v0001");
    EXPECT_EQ(first.definition_index, 1U);
    ASSERT_TRUE(first.first_use_index.has_value());
    EXPECT_EQ(first.first_use_index.value(), 2U);
    EXPECT_EQ(first.final_use_index, 2U);
    EXPECT_EQ(first.lifetime_start, 1U);
    EXPECT_EQ(first.lifetime_end, 2U);

    const TensorLifetime& input = lifetime(plan, "v0000");
    EXPECT_EQ(input.storage_class, forgeir::TensorStorageClass::external_input);
    EXPECT_TRUE(input.external);
    EXPECT_TRUE(input.protected_buffer);
    EXPECT_FALSE(input.arena_offset.has_value());
    EXPECT_EQ(input.final_use_index, 1U);
}

TEST(MemoryPlanner, ReusesOnlyNonOverlappingIntermediateAllocations) {
    const forgeir::MemoryPlan plan = plan_chain();
    const TensorLifetime& first = lifetime(plan, "v0001");
    const TensorLifetime& second = lifetime(plan, "v0002");
    const TensorLifetime& third = lifetime(plan, "v0003");
    ASSERT_TRUE(first.arena_offset.has_value());
    ASSERT_TRUE(second.arena_offset.has_value());
    ASSERT_TRUE(third.arena_offset.has_value());
    EXPECT_EQ(first.arena_offset, third.arena_offset);
    EXPECT_NE(first.arena_offset, second.arena_offset);
    EXPECT_LT(first.final_use_index, third.definition_index);
    EXPECT_TRUE(forgeir::verify_memory_plan(plan).ok());

    forgeir::MemoryPlan invalid = plan;
    for (TensorLifetime& tensor : invalid.tensors) {
        if (tensor.value_id == "v0002") {
            tensor.arena_offset = first.arena_offset;
        }
    }
    const forgeir::Status status = forgeir::verify_memory_plan(invalid);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.message().find("overlap"), std::string::npos);
}

TEST(MemoryPlanner, ProtectsDeclaredOutputsWithDedicatedRetainedStorage) {
    const forgeir::MemoryPlan plan = plan_chain();
    const TensorLifetime& output = lifetime(plan, "v0004");
    ASSERT_TRUE(output.arena_offset.has_value());
    EXPECT_EQ(output.storage_class, forgeir::TensorStorageClass::arena_output);
    EXPECT_TRUE(output.protected_buffer);
    EXPECT_EQ(output.final_use_index, plan.schedule.operations.size());
    for (const TensorLifetime& tensor : plan.tensors) {
        if (tensor.value_id != output.value_id && tensor.arena_offset.has_value()) {
            EXPECT_NE(tensor.arena_offset, output.arena_offset);
        }
    }
    forgeir::MemoryPlan invalid = plan;
    for (TensorLifetime& tensor : invalid.tensors) {
        if (tensor.value_id == "v0004") {
            tensor.arena_offset = lifetime(plan, "v0001").arena_offset;
        }
    }
    EXPECT_FALSE(forgeir::verify_memory_plan(invalid).ok());
}

TEST(MemoryPlanner, RespectsBackendOverrideAlignment) {
    forgeir::MemoryPlannerOptions options;
    options.backend = "test-backend";
    options.alignment_bytes = 128;
    const forgeir::MemoryPlan plan = plan_chain(options);
    EXPECT_EQ(plan.backend, "test-backend");
    EXPECT_EQ(plan.alignment_bytes, 128U);
    for (const TensorLifetime& tensor : plan.tensors) {
        EXPECT_EQ(tensor.alignment_bytes, 128U);
        if (tensor.arena_offset.has_value()) {
            EXPECT_EQ(tensor.arena_offset.value() % 128U, 0U);
        }
    }
}

TEST(MemoryPlanner, RejectsNonPowerOfTwoAlignment) {
    Graph graph = make_chain_graph();
    forgeir::MemoryPlannerOptions options;
    options.alignment_bytes = 96;
    const auto plan = forgeir::plan_memory(graph, options);
    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), forgeir::StatusCode::invalid_argument);
}

TEST(MemoryPlanner, ChoosesTheSmallestDeterministicBestFitBlock) {
    Graph graph = make_best_fit_graph();
    const auto plan = forgeir::plan_memory(graph);
    ASSERT_TRUE(plan.ok()) << plan.status().message();
    const TensorLifetime& large = lifetime(plan.value(), "v0002");
    const TensorLifetime& separator = lifetime(plan.value(), "v0003");
    const TensorLifetime& small = lifetime(plan.value(), "v0004");
    const TensorLifetime& request = lifetime(plan.value(), "v0007");
    EXPECT_EQ(large.arena_offset, 0U);
    EXPECT_EQ(separator.arena_offset, 128U);
    EXPECT_EQ(small.arena_offset, 192U);
    EXPECT_EQ(request.arena_offset, 192U);
    EXPECT_LT(small.final_use_index, request.definition_index);
}

TEST(MemoryPlanner, IsDeterministicAndNeverExceedsNaiveAlignedAllocation) {
    const forgeir::MemoryPlan first = plan_chain();
    const forgeir::MemoryPlan second = plan_chain();
    EXPECT_EQ(forgeir::memory_plan_json(first), forgeir::memory_plan_json(second));
    EXPECT_EQ(forgeir::memory_timeline_csv(first), forgeir::memory_timeline_csv(second));
    EXPECT_EQ(forgeir::memory_timeline_svg(first), forgeir::memory_timeline_svg(second));
    EXPECT_LE(first.arena_size_bytes, first.naive_allocation_bytes);
    EXPECT_LE(first.peak_live_bytes, first.naive_allocation_bytes);
    EXPECT_EQ(first.arena_size_bytes, 192U);
    EXPECT_EQ(first.naive_allocation_bytes, 256U);
}

TEST(MemoryPlanner, KeepsParametersAndConstantsImmutableAndExternal) {
    const auto graph =
        forgeir::GraphLoader::load_from_file(std::filesystem::path(FORGEIR_SOURCE_DIR) / "tests" /
                                             "golden" / "milestone_04_valid_graph.json");
    ASSERT_TRUE(graph.ok()) << graph.status().message();
    const auto plan = forgeir::plan_memory(graph.value());
    ASSERT_TRUE(plan.ok()) << plan.status().message();
    const TensorLifetime& parameter = lifetime(plan.value(), "v0001");
    EXPECT_EQ(parameter.storage_class, forgeir::TensorStorageClass::external_immutable);
    EXPECT_TRUE(parameter.external);
    EXPECT_TRUE(parameter.immutable);
    EXPECT_TRUE(parameter.protected_buffer);
    EXPECT_FALSE(parameter.arena_offset.has_value());
}

TEST(MemoryPlanner, RejectsTensorByteSizeOverflow) {
    const std::vector<std::int64_t> dimensions{std::numeric_limits<std::int64_t>::max(), 2};
    std::vector<Value> values;
    values.push_back(make_value("v0000", "input", dimensions, ValueKind::input));
    values.push_back(make_value("v0001", "output", dimensions, ValueKind::output));
    std::vector<Operation> operations;
    operations.push_back(make_operation("op0000", OperationType::input, "input", {}, "v0000"));
    operations.push_back(
        make_operation("op0001", OperationType::gelu, "output", {"v0000"}, "v0001"));
    Graph graph = make_graph(std::move(values), std::move(operations), {"v0001"});
    const auto plan = forgeir::plan_memory(graph);
    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), forgeir::StatusCode::overflow);
}

TEST(MemoryPlanner, RejectsUnresolvedDynamicOrZeroDimensions) {
    std::vector<Value> values;
    values.push_back(make_value("v0000", "input", {0}, ValueKind::input, DataType::float32, true));
    values.push_back(
        make_value("v0001", "output", {0}, ValueKind::output, DataType::float32, true));
    std::vector<Operation> operations;
    operations.push_back(make_operation("op0000", OperationType::input, "input", {}, "v0000"));
    operations.push_back(
        make_operation("op0001", OperationType::gelu, "output", {"v0000"}, "v0001"));
    Graph graph = make_graph(std::move(values), std::move(operations), {"v0001"});
    const auto plan = forgeir::plan_memory(graph);
    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), forgeir::StatusCode::failed_precondition);
    EXPECT_NE(plan.status().message().find("unresolved dynamic"), std::string::npos);
}

TEST(MemoryPlanner, RejectsMissingIntermediateLifetimeInformation) {
    Graph graph = make_chain_graph();
    graph.mutable_values().push_back(
        make_value("v0005", "dead_intermediate", {4}, ValueKind::intermediate));
    graph.mutable_operations().push_back(
        make_operation("op0005", OperationType::gelu, "dead_intermediate", {"v0000"}, "v0005"));
    const auto plan = forgeir::plan_memory(graph);
    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), forgeir::StatusCode::failed_precondition);
    EXPECT_NE(plan.status().message().find("missing lifetime use information"), std::string::npos);
}
