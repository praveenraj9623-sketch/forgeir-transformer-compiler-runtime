#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "forgeir/backends/cuda/cuda_backend.hpp"
#include "forgeir/runtime/cpu_backend.hpp"

namespace forgeir {
namespace {

std::vector<float> deterministic_input(const std::int64_t rows, const std::int64_t width) {
    std::vector<float> result(static_cast<std::size_t>(rows * width));
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] = static_cast<float>(static_cast<std::int64_t>(index % 29) - 14) / 7.0F;
    }
    return result;
}

float exact_gelu(const float value) {
    constexpr float kInverseSqrtTwo = 0.70710678118654752440F;
    return 0.5F * value * (1.0F + std::erf(value * kInverseSqrtTwo));
}

void expect_close(const std::vector<float>& actual, const std::vector<float>& expected,
                  const float absolute_tolerance = 3.0e-6F,
                  const float relative_tolerance = 3.0e-5F) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float tolerance = absolute_tolerance + relative_tolerance * std::abs(expected[index]);
        EXPECT_LE(std::abs(actual[index] - expected[index]), tolerance) << "index " << index;
    }
}

void require_real_cuda_device() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
    ASSERT_GT(device_count, 0) << "FORGEIR_ENABLE_CUDA tests require real NVIDIA hardware";
}

TEST(CudaBackend, RegistrySelectsTheRealCudaBackend) {
    require_real_cuda_device();
    auto backend = BackendRegistry::create("cuda");
    ASSERT_TRUE(backend.ok()) << backend.status().message();
    EXPECT_EQ(backend.value()->name(), "cuda");
    const auto metadata = query_cuda_device_metadata();
    ASSERT_TRUE(metadata.ok()) << metadata.status().message();
    EXPECT_FALSE(metadata.value().gpu_model.empty());
    EXPECT_FALSE(metadata.value().nvcc_version.empty());
    EXPECT_GT(metadata.value().compute_capability_major, 0);
}

TEST(CudaBackend, HandwrittenKernelsCoverOddSmallAndLargeWidths) {
    require_real_cuda_device();
    for (const std::int64_t width : {1, 3, 7, 31, 257, 1023, 4096}) {
        constexpr std::int64_t rows = 3;
        const std::vector<float> input = deterministic_input(rows, width);
        std::vector<float> auxiliary(static_cast<std::size_t>(width));
        for (std::size_t index = 0; index < auxiliary.size(); ++index) {
            auxiliary[index] = 0.75F + static_cast<float>(index % 11) / 20.0F;
        }

        CudaHostRequest gelu_request{CudaKernelKind::gelu, input, {}, rows, width, 1.0e-5F, 2, 2};
        auto gelu = run_cuda_host_profile(gelu_request);
        ASSERT_TRUE(gelu.ok()) << gelu.status().message();
        std::vector<float> gelu_expected(input.size());
        std::transform(input.begin(), input.end(), gelu_expected.begin(), exact_gelu);
        expect_close(gelu.value().output, gelu_expected);

        CudaHostRequest fused_request{
            CudaKernelKind::fused_bias_gelu, input, auxiliary, rows, width, 1.0e-5F, 2, 2};
        auto fused = run_cuda_host_profile(fused_request);
        ASSERT_TRUE(fused.ok()) << fused.status().message();
        std::vector<float> fused_expected(input.size());
        for (std::size_t index = 0; index < input.size(); ++index) {
            fused_expected[index] = exact_gelu(input[index] + auxiliary[index % auxiliary.size()]);
        }
        expect_close(fused.value().output, fused_expected);

        CudaHostRequest rms_request{
            CudaKernelKind::rms_norm, input, auxiliary, rows, width, 1.0e-5F, 2, 2};
        auto rms = run_cuda_host_profile(rms_request);
        ASSERT_TRUE(rms.ok()) << rms.status().message();
        std::vector<float> rms_expected(input.size());
        for (std::int64_t row = 0; row < rows; ++row) {
            float square_sum = 0.0F;
            for (std::int64_t column = 0; column < width; ++column) {
                const float value = input[static_cast<std::size_t>(row * width + column)];
                square_sum += value * value;
            }
            const float inverse_root =
                1.0F / std::sqrt(square_sum / static_cast<float>(width) + 1.0e-5F);
            for (std::int64_t column = 0; column < width; ++column) {
                const auto index = static_cast<std::size_t>(row * width + column);
                rms_expected[index] = input[index] * inverse_root * auxiliary[column];
            }
        }
        expect_close(rms.value().output, rms_expected, 5.0e-6F, 5.0e-5F);

        CudaHostRequest softmax_request{
            CudaKernelKind::softmax, input, {}, rows, width, 1.0e-5F, 2, 2};
        auto softmax = run_cuda_host_profile(softmax_request);
        ASSERT_TRUE(softmax.ok()) << softmax.status().message();
        std::vector<float> softmax_expected(input.size());
        for (std::int64_t row = 0; row < rows; ++row) {
            const auto begin = input.begin() + row * width;
            const float maximum = *std::max_element(begin, begin + width);
            float sum = 0.0F;
            for (std::int64_t column = 0; column < width; ++column) {
                const auto index = static_cast<std::size_t>(row * width + column);
                softmax_expected[index] = std::exp(input[index] - maximum);
                sum += softmax_expected[index];
            }
            for (std::int64_t column = 0; column < width; ++column) {
                softmax_expected[static_cast<std::size_t>(row * width + column)] /= sum;
            }
        }
        expect_close(softmax.value().output, softmax_expected, 5.0e-6F, 5.0e-5F);
        EXPECT_EQ(softmax.value().kernel_milliseconds.size(), 2U);
        EXPECT_EQ(softmax.value().launch.block_size, kCudaKernelBlockSize);
        EXPECT_EQ(softmax.value().launch.grid_size, rows);
        EXPECT_EQ(softmax.value().launch.shared_memory_bytes, kCudaKernelBlockSize * sizeof(float));
    }
}

TEST(CudaBackend, SoftmaxHandlesExtremeFiniteValues) {
    require_real_cuda_device();
    const float largest = std::numeric_limits<float>::max();
    const std::vector<float> input{largest, largest, -largest, 1000.0F, -1000.0F, 0.0F, 1.0F};
    CudaHostRequest request{CudaKernelKind::softmax, input, {}, 1, 7, 1.0e-5F, 2, 2};
    const auto profile = run_cuda_host_profile(request);
    ASSERT_TRUE(profile.ok()) << profile.status().message();
    for (const float value : profile.value().output) {
        EXPECT_TRUE(std::isfinite(value));
        EXPECT_GE(value, 0.0F);
        EXPECT_LE(value, 1.0F);
    }
    float sum = 0.0F;
    for (const float value : profile.value().output) {
        sum += value;
    }
    EXPECT_NEAR(sum, 1.0F, 1.0e-6F);
}

TEST(CudaBackend, RejectsUnsupportedShapesBeforeAllocation) {
    CudaHostRequest request;
    request.kernel = CudaKernelKind::softmax;
    request.rows = 1;
    request.width = kCudaMaximumRowWidth + 1;
    request.input.resize(static_cast<std::size_t>(request.width));
    const auto profile = run_cuda_host_profile(request);
    ASSERT_FALSE(profile.ok());
    EXPECT_EQ(profile.status().code(), StatusCode::unsupported);
}

} // namespace
} // namespace forgeir
