#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "forgeir/ir/operation.hpp"
#include "forgeir/runtime/cpu_backend.hpp"
#include "forgeir/runtime/runtime_session.hpp"
#include "forgeir/runtime/tensor_storage.hpp"

namespace {

forgeir::Operation make_operation(const forgeir::OperationType type, nlohmann::json attributes,
                                  const std::size_t input_count = 2) {
    std::vector<std::string> inputs;
    for (std::size_t index = 0; index < input_count; ++index) {
        inputs.push_back("v000" + std::to_string(index));
    }
    return forgeir::Operation("op0000", type, "test.operation", std::move(inputs), {"v0099"},
                              std::move(attributes));
}

} // namespace

TEST(TensorStorage, AllocatesAlignedRaiiStorage) {
    auto storage = forgeir::TensorStorage::allocate(257, 64);
    ASSERT_TRUE(storage.ok()) << storage.status().message();
    EXPECT_EQ(storage.value().size_bytes(), 257U);
    EXPECT_EQ(storage.value().alignment_bytes(), 64U);
    ASSERT_NE(storage.value().data(), nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(storage.value().data()) % 64U, 0U);
}

TEST(CpuReferenceMatMul, ComputesBatchedMatrixProducts) {
    const std::vector<float> left{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F,
                                  2.0F, 0.0F, 1.0F, 1.0F, 3.0F, 2.0F};
    const std::vector<float> right{1.0F, 2.0F, 0.0F, 1.0F, 1.0F, 0.0F,
                                   2.0F, 1.0F, 1.0F, 0.0F, 0.0F, 2.0F};
    std::vector<float> output(8, 0.0F);
    const forgeir::Operation operation =
        make_operation(forgeir::OperationType::mat_mul, nlohmann::json::object());
    const forgeir::CpuBackend backend(forgeir::CpuMatMulImplementation::reference);
    const forgeir::Status status = backend.execute(
        operation, {{left.data(), {2, 2, 3}, true}, {right.data(), {2, 3, 2}, true}},
        {output.data(), {2, 2, 2}, true});
    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_EQ(output, (std::vector<float>{4.0F, 4.0F, 10.0F, 13.0F, 4.0F, 4.0F, 5.0F, 5.0F}));
}

TEST(CpuReferenceMatMul, RejectsContractingDimensionMismatch) {
    const std::vector<float> left(6, 1.0F);
    const std::vector<float> right(8, 1.0F);
    std::vector<float> output(4, 0.0F);
    const forgeir::Operation operation =
        make_operation(forgeir::OperationType::mat_mul, nlohmann::json::object());
    const forgeir::CpuBackend backend(forgeir::CpuMatMulImplementation::reference);
    const forgeir::Status status =
        backend.execute(operation, {{left.data(), {2, 3}, true}, {right.data(), {4, 2}, true}},
                        {output.data(), {2, 2}, true});
    EXPECT_FALSE(status.ok());
}

TEST(CpuTiledMatMul, MatchesReferenceExactly) {
    std::vector<float> left(2U * 7U * 35U);
    std::vector<float> right(2U * 35U * 33U);
    for (std::size_t index = 0; index < left.size(); ++index) {
        left[index] = static_cast<float>(static_cast<int>(index % 17U) - 8) / 9.0F;
    }
    for (std::size_t index = 0; index < right.size(); ++index) {
        right[index] = static_cast<float>(static_cast<int>(index % 13U) - 6) / 7.0F;
    }
    std::vector<float> reference(2U * 7U * 33U, 0.0F);
    std::vector<float> tiled(reference.size(), 0.0F);
    const forgeir::Operation operation =
        make_operation(forgeir::OperationType::mat_mul, nlohmann::json::object());
    const forgeir::CpuBackend reference_backend(forgeir::CpuMatMulImplementation::reference);
    const forgeir::CpuBackend tiled_backend(forgeir::CpuMatMulImplementation::tiled);
    ASSERT_TRUE(reference_backend
                    .execute(operation,
                             {{left.data(), {2, 7, 35}, true}, {right.data(), {2, 35, 33}, true}},
                             {reference.data(), {2, 7, 33}, true})
                    .ok());
    ASSERT_TRUE(tiled_backend
                    .execute(operation,
                             {{left.data(), {2, 7, 35}, true}, {right.data(), {2, 35, 33}, true}},
                             {tiled.data(), {2, 7, 33}, true})
                    .ok());
    EXPECT_EQ(tiled, reference);
}

TEST(CpuOperators, LinearAndFusedBiasGeluUseExactConvention) {
    const std::vector<float> input{0.5F, -1.0F, 2.0F};
    const std::vector<float> weight{1.0F, 0.5F, -0.25F, -0.5F, 1.5F, 0.75F};
    const std::vector<float> bias{0.125F, -0.25F};
    std::vector<float> linear_output(2, 0.0F);
    auto linear_attributes =
        nlohmann::json{{"bias", false}, {"in_features", 3}, {"out_features", 2}};
    const forgeir::CpuBackend backend;
    ASSERT_TRUE(backend
                    .execute(make_operation(forgeir::OperationType::linear, linear_attributes),
                             {{input.data(), {1, 3}, true}, {weight.data(), {2, 3}, true}},
                             {linear_output.data(), {1, 2}, true})
                    .ok());
    EXPECT_FLOAT_EQ(linear_output[0], -0.5F);
    EXPECT_FLOAT_EQ(linear_output[1], -0.25F);

    auto fused_attributes = nlohmann::json{{"bias", true},
                                           {"in_features", 3},
                                           {"out_features", 2},
                                           {"fused_activation", "GELU"},
                                           {"fused_activation_approximate", "none"}};
    std::vector<float> fused_output(2, 0.0F);
    ASSERT_TRUE(backend
                    .execute(make_operation(forgeir::OperationType::linear, fused_attributes, 3),
                             {{input.data(), {1, 3}, true},
                              {weight.data(), {2, 3}, true},
                              {bias.data(), {2}, true}},
                             {fused_output.data(), {1, 2}, true})
                    .ok());
    const auto exact_gelu = [](const float value) {
        return 0.5F * value * (1.0F + std::erf(value / std::sqrt(2.0F)));
    };
    EXPECT_NEAR(fused_output[0], exact_gelu(-0.375F), 1.0e-7F);
    EXPECT_NEAR(fused_output[1], exact_gelu(-0.5F), 1.0e-7F);
}

TEST(CpuOperators, ElementwiseBroadcastAddMulAndDiv) {
    const std::vector<float> left{1.0F, 2.0F};
    const std::vector<float> right{2.0F, 4.0F, 8.0F};
    const forgeir::CpuBackend backend;
    const std::vector<std::pair<forgeir::OperationType, std::vector<float>>> cases{
        {forgeir::OperationType::add, {3.0F, 5.0F, 9.0F, 4.0F, 6.0F, 10.0F}},
        {forgeir::OperationType::multiply, {2.0F, 4.0F, 8.0F, 4.0F, 8.0F, 16.0F}},
        {forgeir::OperationType::divide, {0.5F, 0.25F, 0.125F, 1.0F, 0.5F, 0.25F}},
    };
    for (const auto& [type, expected] : cases) {
        std::vector<float> output(6, 0.0F);
        const forgeir::Status status =
            backend.execute(make_operation(type, nlohmann::json::object()),
                            {{left.data(), {2, 1}, true}, {right.data(), {1, 3}, true}},
                            {output.data(), {2, 3}, true});
        ASSERT_TRUE(status.ok()) << status.message();
        EXPECT_EQ(output, expected);
    }
}

TEST(CpuOperators, RmsNormAndGeluMatchDocumentedEquations) {
    const std::vector<float> input{1.0F, -2.0F, 3.0F, -4.0F};
    const std::vector<float> weight{1.0F, 0.5F, 1.5F, 2.0F};
    std::vector<float> normalized(4, 0.0F);
    const forgeir::CpuBackend backend;
    ASSERT_TRUE(backend
                    .execute(make_operation(forgeir::OperationType::rms_norm,
                                            {{"axis", -1}, {"epsilon", 1.0e-5}}),
                             {{input.data(), {1, 4}, true}, {weight.data(), {4}, true}},
                             {normalized.data(), {1, 4}, true})
                    .ok());
    const float inverse_rms = 1.0F / std::sqrt(7.5F + 1.0e-5F);
    for (std::size_t index = 0; index < input.size(); ++index) {
        EXPECT_NEAR(normalized[index], input[index] * inverse_rms * weight[index], 1.0e-7F);
    }

    std::vector<float> activated(4, 0.0F);
    ASSERT_TRUE(
        backend
            .execute(make_operation(forgeir::OperationType::gelu, {{"approximate", "none"}}, 1),
                     {{input.data(), {4}, true}}, {activated.data(), {4}, true})
            .ok());
    for (std::size_t index = 0; index < input.size(); ++index) {
        const float expected =
            0.5F * input[index] * (1.0F + std::erf(input[index] / std::sqrt(2.0F)));
        EXPECT_NEAR(activated[index], expected, 1.0e-7F);
    }
}

TEST(CpuOperators, SoftmaxIsStableAndPropagatesNonFiniteInputs) {
    const std::vector<float> input{1000.0F, 1001.0F, 999.0F};
    std::vector<float> output(3, 0.0F);
    const forgeir::CpuBackend backend;
    ASSERT_TRUE(backend
                    .execute(make_operation(forgeir::OperationType::softmax,
                                            {{"axis", 0}, {"stable", true}}, 1),
                             {{input.data(), {3}, true}}, {output.data(), {3}, true})
                    .ok());
    const float denominator = std::exp(-1.0F) + 1.0F + std::exp(-2.0F);
    EXPECT_NEAR(output[0], std::exp(-1.0F) / denominator, 1.0e-7F);
    EXPECT_NEAR(output[1], 1.0F / denominator, 1.0e-7F);
    EXPECT_NEAR(output[2], std::exp(-2.0F) / denominator, 1.0e-7F);

    const std::vector<float> non_finite{std::numeric_limits<float>::infinity(), 0.0F};
    std::vector<float> non_finite_output(2, 0.0F);
    ASSERT_TRUE(backend
                    .execute(make_operation(forgeir::OperationType::softmax,
                                            {{"axis", 0}, {"stable", true}}, 1),
                             {{non_finite.data(), {2}, true}},
                             {non_finite_output.data(), {2}, true})
                    .ok());
    EXPECT_TRUE(std::isnan(non_finite_output[0]));
    EXPECT_TRUE(std::isnan(non_finite_output[1]));
}

TEST(CpuOperators, ReshapeTransposeAndCausalMaskMaterializeContiguousOutputs) {
    const std::vector<float> input{0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    const forgeir::CpuBackend backend;
    std::vector<float> reshaped(6, 0.0F);
    ASSERT_TRUE(
        backend
            .execute(make_operation(forgeir::OperationType::reshape, {{"shape", {3, 2}}}, 1),
                     {{input.data(), {2, 3}, true}}, {reshaped.data(), {3, 2}, true})
            .ok());
    EXPECT_EQ(reshaped, input);

    std::vector<float> transposed(6, 0.0F);
    ASSERT_TRUE(backend
                    .execute(make_operation(forgeir::OperationType::transpose,
                                            {{"permutation", {1, 0}}}, 1),
                             {{input.data(), {2, 3}, true}}, {transposed.data(), {3, 2}, true})
                    .ok());
    EXPECT_EQ(transposed, (std::vector<float>{0.0F, 3.0F, 1.0F, 4.0F, 2.0F, 5.0F}));

    const std::vector<float> scores{0.0F, 1.0F, 2.0F,  3.0F,  4.0F,  5.0F,  6.0F,  7.0F,
                                    8.0F, 9.0F, 10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F};
    std::vector<float> masked(16, 0.0F);
    ASSERT_TRUE(backend
                    .execute(make_operation(forgeir::OperationType::causal_mask,
                                            {{"diagonal", 0}, {"masked_value", "-inf"}}, 1),
                             {{scores.data(), {4, 4}, true}}, {masked.data(), {4, 4}, true})
                    .ok());
    EXPECT_FLOAT_EQ(masked[0], 0.0F);
    EXPECT_TRUE(std::isinf(masked[1]) && masked[1] < 0.0F);
    EXPECT_FLOAT_EQ(masked[4], 4.0F);
    EXPECT_FLOAT_EQ(masked[5], 5.0F);
    EXPECT_TRUE(std::isinf(masked[6]) && masked[6] < 0.0F);
}

TEST(CpuOperators, RejectsNonContiguousInputAndUnknownBackend) {
    const std::vector<float> input{1.0F, 2.0F};
    std::vector<float> output(2, 0.0F);
    const forgeir::CpuBackend backend;
    const forgeir::Status status =
        backend.execute(make_operation(forgeir::OperationType::gelu, {{"approximate", "none"}}, 1),
                        {{input.data(), {2}, false}}, {output.data(), {2}, true});
    EXPECT_EQ(status.code(), forgeir::StatusCode::unsupported);
    const auto unknown = forgeir::BackendRegistry::create("gpu");
    EXPECT_FALSE(unknown.ok());
    EXPECT_EQ(unknown.status().code(), forgeir::StatusCode::unsupported);
}

TEST(CpuRuntimeSession, LoadsRealGraphWithMilestoneSevenArenaPlan) {
    const std::filesystem::path graph_path = std::filesystem::path(FORGEIR_SOURCE_DIR) /
                                             "artifacts" / "graphs" / "milestone_06" / "O2" /
                                             "tiny_transformer_block.graph.json";
    const auto session = forgeir::RuntimeSession::load(graph_path.string(), "cpu");
    ASSERT_TRUE(session.ok()) << session.status().message();
    EXPECT_EQ(session.value()->memory_plan().backend, "cpu");
    EXPECT_EQ(session.value()->memory_plan().alignment_bytes, 64U);
    EXPECT_EQ(session.value()->memory_plan().arena_size_bytes, 262144U);
    EXPECT_EQ(session.value()->memory_plan().schedule.operations.size(), 35U);
}
