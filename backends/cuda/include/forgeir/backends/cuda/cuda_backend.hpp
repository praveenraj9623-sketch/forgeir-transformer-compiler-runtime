#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "forgeir/core/status.hpp"
#include "forgeir/runtime/cpu_backend.hpp"

namespace forgeir {

inline constexpr std::uint32_t kCudaKernelBlockSize = 256;
inline constexpr std::int64_t kCudaMaximumRowWidth = 16384;
inline constexpr std::uintptr_t kCudaRequiredPointerAlignment = 16;

enum class CudaKernelKind { gelu, rms_norm, softmax, fused_bias_gelu };

struct CudaLaunchMetadata {
    std::uint32_t block_size{0};
    std::uint32_t grid_size{0};
    std::uint64_t shared_memory_bytes{0};
};

struct CudaDeviceMetadata {
    std::string gpu_model;
    int device_ordinal{0};
    int compute_capability_major{0};
    int compute_capability_minor{0};
    int cuda_runtime_version{0};
    int cuda_driver_version{0};
    std::string nvcc_version;
};

struct CudaProfileResult {
    std::vector<float> output;
    std::vector<float> kernel_milliseconds;
    std::uint64_t warmup_iterations{0};
    std::uint64_t measured_iterations{0};
    CudaLaunchMetadata launch;
    CudaDeviceMetadata device;
};

struct CudaHostRequest {
    CudaKernelKind kernel{CudaKernelKind::gelu};
    std::vector<float> input;
    std::vector<float> auxiliary;
    std::int64_t rows{0};
    std::int64_t width{0};
    float epsilon{1.0e-5F};
    std::uint64_t warmup_iterations{10};
    std::uint64_t measured_iterations{50};
};

[[nodiscard]] Result<CudaDeviceMetadata> query_cuda_device_metadata();
[[nodiscard]] Result<CudaProfileResult> run_cuda_host_profile(const CudaHostRequest& request);

class CudaBackend final : public Backend {
  public:
    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] Status execute(const Operation& operation,
                                 const std::vector<ConstTensorView>& inputs,
                                 TensorView output) const override;
};

} // namespace forgeir
