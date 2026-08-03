#include "forgeir/backends/cuda/cuda_backend.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifndef FORGEIR_NVCC_VERSION
#define FORGEIR_NVCC_VERSION "unknown"
#endif

namespace forgeir {
namespace {

constexpr cudaStream_t kDefaultCudaStream = nullptr;

Status cuda_status(const cudaError_t error, const std::string& context) {
    if (error == cudaSuccess) {
        return Status::ok_status();
    }
    return Status::error(StatusCode::internal, context + ": " +
                                                   std::string(cudaGetErrorName(error)) + " (" +
                                                   cudaGetErrorString(error) + ")");
}

struct DeviceDeleter {
    void operator()(float* address) const noexcept {
        if (address != nullptr && cudaFree(address) != cudaSuccess) {
            std::terminate();
        }
    }
};

using DevicePointer = std::unique_ptr<float, DeviceDeleter>;

struct EventDeleter {
    void operator()(CUevent_st* event) const noexcept {
        if (event != nullptr && cudaEventDestroy(event) != cudaSuccess) {
            std::terminate();
        }
    }
};

using EventPointer = std::unique_ptr<CUevent_st, EventDeleter>;

Result<DevicePointer> allocate_device(const std::uint64_t element_count, const std::string& role) {
    if (element_count > std::numeric_limits<std::uint64_t>::max() / sizeof(float)) {
        return Status::error(StatusCode::overflow, role + " byte size overflowed");
    }
    const std::uint64_t byte_size = element_count * sizeof(float);
    if (byte_size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return Status::error(StatusCode::overflow, role + " exceeds host size_t capacity");
    }
    float* address = nullptr;
    const Status allocation = cuda_status(
        cudaMalloc(reinterpret_cast<void**>(&address), static_cast<std::size_t>(byte_size)),
        "cudaMalloc for " + role);
    if (!allocation.ok()) {
        return allocation;
    }
    return DevicePointer(address);
}

Result<EventPointer> create_event(const std::string& role) {
    cudaEvent_t event = nullptr;
    const Status creation = cuda_status(cudaEventCreate(&event), "cudaEventCreate for " + role);
    if (!creation.ok()) {
        return creation;
    }
    return EventPointer(event);
}

Result<std::uint64_t> checked_element_count(const std::int64_t rows, const std::int64_t width) {
    if (rows <= 0 || width <= 0) {
        return Status::error(StatusCode::invalid_argument,
                             "CUDA tensors require positive rows and width");
    }
    const auto unsigned_rows = static_cast<std::uint64_t>(rows);
    const auto unsigned_width = static_cast<std::uint64_t>(width);
    if (unsigned_rows > std::numeric_limits<std::uint64_t>::max() / unsigned_width) {
        return Status::error(StatusCode::overflow, "CUDA tensor element count overflowed");
    }
    return unsigned_rows * unsigned_width;
}

Status validate_device_pointer(const void* address, const int expected_device,
                               const std::string& role) {
    if (address == nullptr) {
        return Status::error(StatusCode::invalid_argument, role + " has a null device pointer");
    }
    if (reinterpret_cast<std::uintptr_t>(address) % kCudaRequiredPointerAlignment != 0) {
        return Status::error(StatusCode::invalid_argument,
                             role + " does not satisfy the 16-byte alignment contract");
    }
    cudaPointerAttributes attributes{};
    const Status attribute_status = cuda_status(cudaPointerGetAttributes(&attributes, address),
                                                "cudaPointerGetAttributes for " + role);
    if (!attribute_status.ok()) {
        return attribute_status;
    }
    if (attributes.type != cudaMemoryTypeDevice && attributes.type != cudaMemoryTypeManaged) {
        return Status::error(StatusCode::invalid_argument,
                             role + " is not CUDA device or managed memory");
    }
    if (attributes.device != expected_device) {
        return Status::error(StatusCode::invalid_argument,
                             role + " belongs to a different CUDA device");
    }
    return Status::ok_status();
}

__device__ float block_reduce_sum(float value, float* shared) {
    const unsigned int lane = threadIdx.x;
    shared[lane] = value;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            shared[lane] += shared[lane + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

__device__ float block_reduce_max(float value, float* shared) {
    const unsigned int lane = threadIdx.x;
    shared[lane] = value;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            shared[lane] = fmaxf(shared[lane], shared[lane + stride]);
        }
        __syncthreads();
    }
    return shared[0];
}

__global__
__launch_bounds__(kCudaKernelBlockSize) void gelu_kernel(const float* input, float* output,
                                                         const std::uint64_t element_count) {
    const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < element_count) {
        constexpr float kInverseSqrtTwo = 0.70710678118654752440F;
        const float value = input[index];
        output[index] = 0.5F * value * (1.0F + erff(value * kInverseSqrtTwo));
    }
}

__global__ __launch_bounds__(kCudaKernelBlockSize) void fused_bias_gelu_kernel(
    const float* input, const float* bias, float* output, const std::uint64_t element_count,
    const std::int64_t width) {
    const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < element_count) {
        constexpr float kInverseSqrtTwo = 0.70710678118654752440F;
        const float value = input[index] + bias[index % static_cast<std::uint64_t>(width)];
        output[index] = 0.5F * value * (1.0F + erff(value * kInverseSqrtTwo));
    }
}

__global__ __launch_bounds__(kCudaKernelBlockSize) void rms_norm_kernel(const float* input,
                                                                        const float* weight,
                                                                        float* output,
                                                                        const std::int64_t width,
                                                                        const float epsilon) {
    extern __shared__ float shared[];
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    const std::int64_t base = row * width;
    float square_sum = 0.0F;
    for (std::int64_t column = static_cast<std::int64_t>(threadIdx.x); column < width;
         column += static_cast<std::int64_t>(blockDim.x)) {
        const float value = input[base + column];
        square_sum += value * value;
    }
    const float total = block_reduce_sum(square_sum, shared);
    const float inverse_root = rsqrtf(total / static_cast<float>(width) + epsilon);
    for (std::int64_t column = static_cast<std::int64_t>(threadIdx.x); column < width;
         column += static_cast<std::int64_t>(blockDim.x)) {
        output[base + column] = input[base + column] * inverse_root * weight[column];
    }
}

__global__
__launch_bounds__(kCudaKernelBlockSize) void row_softmax_kernel(const float* input, float* output,
                                                                const std::int64_t width) {
    extern __shared__ float shared[];
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    const std::int64_t base = row * width;
    float local_maximum = -CUDART_INF_F;
    for (std::int64_t column = static_cast<std::int64_t>(threadIdx.x); column < width;
         column += static_cast<std::int64_t>(blockDim.x)) {
        local_maximum = fmaxf(local_maximum, input[base + column]);
    }
    const float maximum = block_reduce_max(local_maximum, shared);
    float local_sum = 0.0F;
    for (std::int64_t column = static_cast<std::int64_t>(threadIdx.x); column < width;
         column += static_cast<std::int64_t>(blockDim.x)) {
        const float exponential = expf(input[base + column] - maximum);
        output[base + column] = exponential;
        local_sum += exponential;
    }
    const float sum = block_reduce_sum(local_sum, shared);
    for (std::int64_t column = static_cast<std::int64_t>(threadIdx.x); column < width;
         column += static_cast<std::int64_t>(blockDim.x)) {
        output[base + column] /= sum;
    }
}

Result<CudaLaunchMetadata> launch_kernel(const CudaKernelKind kernel, const float* input,
                                         const float* auxiliary, float* output,
                                         const std::int64_t rows, const std::int64_t width,
                                         const float epsilon, const cudaStream_t stream) {
    const auto count = checked_element_count(rows, width);
    if (!count.ok()) {
        return count.status();
    }
    const std::uint64_t elementwise_grid =
        (count.value() + kCudaKernelBlockSize - 1) / kCudaKernelBlockSize;
    const std::uint64_t row_grid = static_cast<std::uint64_t>(rows);
    const std::uint64_t selected_grid =
        kernel == CudaKernelKind::gelu || kernel == CudaKernelKind::fused_bias_gelu
            ? elementwise_grid
            : row_grid;
    if (selected_grid == 0 || selected_grid > std::numeric_limits<std::uint32_t>::max()) {
        return Status::error(StatusCode::overflow, "CUDA launch grid exceeds uint32 capacity");
    }
    const auto grid = static_cast<std::uint32_t>(selected_grid);
    const std::uint64_t shared_memory =
        kernel == CudaKernelKind::rms_norm || kernel == CudaKernelKind::softmax
            ? static_cast<std::uint64_t>(kCudaKernelBlockSize) * sizeof(float)
            : 0;
    switch (kernel) {
    case CudaKernelKind::gelu:
        gelu_kernel<<<grid, kCudaKernelBlockSize, 0, stream>>>(input, output, count.value());
        break;
    case CudaKernelKind::rms_norm:
        rms_norm_kernel<<<grid, kCudaKernelBlockSize, shared_memory, stream>>>(
            input, auxiliary, output, width, epsilon);
        break;
    case CudaKernelKind::softmax:
        row_softmax_kernel<<<grid, kCudaKernelBlockSize, shared_memory, stream>>>(input, output,
                                                                                  width);
        break;
    case CudaKernelKind::fused_bias_gelu:
        fused_bias_gelu_kernel<<<grid, kCudaKernelBlockSize, 0, stream>>>(input, auxiliary, output,
                                                                          count.value(), width);
        break;
    }
    const Status launch_status = cuda_status(cudaPeekAtLastError(), "CUDA kernel launch");
    if (!launch_status.ok()) {
        return launch_status;
    }
    return CudaLaunchMetadata{kCudaKernelBlockSize, grid, shared_memory};
}

Status validate_request(const CudaHostRequest& request, const std::uint64_t element_count) {
    if (request.width > kCudaMaximumRowWidth) {
        return Status::error(StatusCode::unsupported,
                             "CUDA row width exceeds the supported maximum of " +
                                 std::to_string(kCudaMaximumRowWidth));
    }
    if (request.input.size() != element_count) {
        return Status::error(StatusCode::invalid_argument,
                             "CUDA input size does not match rows times width");
    }
    if (request.warmup_iterations == 0 || request.measured_iterations == 0) {
        return Status::error(StatusCode::invalid_argument,
                             "CUDA profiling requires positive warm-up and measured iterations");
    }
    if (request.kernel == CudaKernelKind::rms_norm ||
        request.kernel == CudaKernelKind::fused_bias_gelu) {
        if (request.auxiliary.size() != static_cast<std::size_t>(request.width)) {
            return Status::error(StatusCode::invalid_argument,
                                 "CUDA weight or bias size must equal the row width");
        }
    } else if (!request.auxiliary.empty()) {
        return Status::error(StatusCode::invalid_argument,
                             "CUDA GELU/Softmax does not accept an auxiliary tensor");
    }
    if (request.kernel == CudaKernelKind::rms_norm &&
        (!std::isfinite(request.epsilon) || request.epsilon <= 0.0F)) {
        return Status::error(StatusCode::invalid_argument,
                             "CUDA RMSNorm epsilon must be positive and finite");
    }
    return Status::ok_status();
}

Status copy_to_device(float* destination, const std::vector<float>& source,
                      const std::string& role) {
    return cuda_status(cudaMemcpy(destination, source.data(), source.size() * sizeof(float),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy host-to-device for " + role);
}

} // namespace

Result<CudaDeviceMetadata> query_cuda_device_metadata() {
    int device = 0;
    Status status = cuda_status(cudaGetDevice(&device), "cudaGetDevice");
    if (!status.ok()) {
        return status;
    }
    cudaDeviceProp properties{};
    status = cuda_status(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    if (!status.ok()) {
        return status;
    }
    int runtime_version = 0;
    status = cuda_status(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
    if (!status.ok()) {
        return status;
    }
    int driver_version = 0;
    status = cuda_status(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
    if (!status.ok()) {
        return status;
    }
    return CudaDeviceMetadata{properties.name,     device,          properties.major,
                              properties.minor,    runtime_version, driver_version,
                              FORGEIR_NVCC_VERSION};
}

Result<CudaProfileResult> run_cuda_host_profile(const CudaHostRequest& request) {
    const auto element_count = checked_element_count(request.rows, request.width);
    if (!element_count.ok()) {
        return element_count.status();
    }
    const Status request_status = validate_request(request, element_count.value());
    if (!request_status.ok()) {
        return request_status;
    }
    auto input = allocate_device(element_count.value(), "input");
    auto output = allocate_device(element_count.value(), "output");
    if (!input.ok()) {
        return input.status();
    }
    if (!output.ok()) {
        return output.status();
    }
    DevicePointer auxiliary;
    if (!request.auxiliary.empty()) {
        auto allocation =
            allocate_device(static_cast<std::uint64_t>(request.auxiliary.size()), "weight or bias");
        if (!allocation.ok()) {
            return allocation.status();
        }
        auxiliary = allocation.take_value();
    }
    Status status = copy_to_device(input.value().get(), request.input, "input");
    if (!status.ok()) {
        return status;
    }
    if (!request.auxiliary.empty()) {
        status = copy_to_device(auxiliary.get(), request.auxiliary, "weight or bias");
        if (!status.ok()) {
            return status;
        }
    }
    const auto device = query_cuda_device_metadata();
    if (!device.ok()) {
        return device.status();
    }
    status = validate_device_pointer(input.value().get(), device.value().device_ordinal, "input");
    if (!status.ok()) {
        return status;
    }
    status = validate_device_pointer(output.value().get(), device.value().device_ordinal, "output");
    if (!status.ok()) {
        return status;
    }
    if (auxiliary) {
        status = validate_device_pointer(auxiliary.get(), device.value().device_ordinal,
                                         "weight or bias");
        if (!status.ok()) {
            return status;
        }
    }

    CudaLaunchMetadata launch;
    for (std::uint64_t iteration = 0; iteration < request.warmup_iterations; ++iteration) {
        auto launched = launch_kernel(request.kernel, input.value().get(), auxiliary.get(),
                                      output.value().get(), request.rows, request.width,
                                      request.epsilon, kDefaultCudaStream);
        if (!launched.ok()) {
            return launched.status();
        }
        launch = launched.value();
    }
    status = cuda_status(cudaDeviceSynchronize(), "cudaDeviceSynchronize after warm-up");
    if (!status.ok()) {
        return status;
    }

    auto start = create_event("profile start");
    auto stop = create_event("profile stop");
    if (!start.ok()) {
        return start.status();
    }
    if (!stop.ok()) {
        return stop.status();
    }
    std::vector<float> timings;
    timings.reserve(static_cast<std::size_t>(request.measured_iterations));
    for (std::uint64_t iteration = 0; iteration < request.measured_iterations; ++iteration) {
        status = cuda_status(cudaEventRecord(start.value().get(), kDefaultCudaStream),
                             "cudaEventRecord profile start");
        if (!status.ok()) {
            return status;
        }
        auto launched = launch_kernel(request.kernel, input.value().get(), auxiliary.get(),
                                      output.value().get(), request.rows, request.width,
                                      request.epsilon, kDefaultCudaStream);
        if (!launched.ok()) {
            return launched.status();
        }
        launch = launched.value();
        status = cuda_status(cudaEventRecord(stop.value().get(), kDefaultCudaStream),
                             "cudaEventRecord profile stop");
        if (!status.ok()) {
            return status;
        }
        status = cuda_status(cudaEventSynchronize(stop.value().get()),
                             "cudaEventSynchronize profile stop");
        if (!status.ok()) {
            return status;
        }
        float milliseconds = 0.0F;
        status = cuda_status(
            cudaEventElapsedTime(&milliseconds, start.value().get(), stop.value().get()),
            "cudaEventElapsedTime");
        if (!status.ok()) {
            return status;
        }
        timings.push_back(milliseconds);
    }

    std::vector<float> host_output(static_cast<std::size_t>(element_count.value()));
    status = cuda_status(cudaMemcpy(host_output.data(), output.value().get(),
                                    host_output.size() * sizeof(float), cudaMemcpyDeviceToHost),
                         "cudaMemcpy device-to-host for output");
    if (!status.ok()) {
        return status;
    }
    return CudaProfileResult{std::move(host_output),
                             std::move(timings),
                             request.warmup_iterations,
                             request.measured_iterations,
                             launch,
                             device.value()};
}

std::string_view CudaBackend::name() const noexcept { return "cuda"; }

Status CudaBackend::execute(const Operation& operation, const std::vector<ConstTensorView>& inputs,
                            TensorView output) const {
    if (!output.contiguous) {
        return Status::error(StatusCode::unsupported,
                             "operation " + operation.id() + " requires a contiguous CUDA output");
    }
    const auto count = checked_tensor_element_count(output.shape);
    if (!count.ok()) {
        return count.status();
    }
    if (output.shape.empty()) {
        return Status::error(StatusCode::unsupported,
                             "operation " + operation.id() +
                                 " CUDA kernels require a non-scalar tensor");
    }
    int device = 0;
    Status status = cuda_status(cudaGetDevice(&device), "cudaGetDevice before backend execution");
    if (!status.ok()) {
        return status;
    }
    status = validate_device_pointer(output.data, device, "operation output");
    if (!status.ok()) {
        return status;
    }
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        if (!inputs[index].contiguous) {
            return Status::error(StatusCode::unsupported, "operation " + operation.id() +
                                                              " input " + std::to_string(index) +
                                                              " is not contiguous");
        }
        status = validate_device_pointer(inputs[index].data, device,
                                         "operation input " + std::to_string(index));
        if (!status.ok()) {
            return status;
        }
    }
    const std::int64_t width = output.shape.back();
    if (width <= 0 || width > kCudaMaximumRowWidth ||
        count.value() % static_cast<std::uint64_t>(width) != 0) {
        return Status::error(StatusCode::unsupported,
                             "operation " + operation.id() + " has an unsupported CUDA row shape");
    }
    const std::uint64_t unsigned_rows = count.value() / static_cast<std::uint64_t>(width);
    if (unsigned_rows > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        return Status::error(StatusCode::overflow,
                             "operation " + operation.id() + " CUDA row count overflowed");
    }
    const auto rows = static_cast<std::int64_t>(unsigned_rows);
    CudaKernelKind kernel = CudaKernelKind::gelu;
    const float* auxiliary = nullptr;
    float epsilon = 1.0e-5F;
    if (operation.type() == OperationType::gelu && inputs.size() == 1 &&
        inputs[0].shape == output.shape) {
        const auto approximate = operation.attributes().find("approximate");
        if (approximate != operation.attributes().end() &&
            (!approximate->is_string() || approximate->get<std::string>() != "none")) {
            return Status::error(StatusCode::unsupported,
                                 "operation " + operation.id() +
                                     " CUDA GELU supports only approximate='none'");
        }
        kernel = CudaKernelKind::gelu;
    } else if (operation.type() == OperationType::rms_norm && inputs.size() == 2 &&
               inputs[0].shape == output.shape &&
               inputs[1].shape == std::vector<std::int64_t>{width}) {
        const auto epsilon_attribute = operation.attributes().find("epsilon");
        if (epsilon_attribute == operation.attributes().end() || !epsilon_attribute->is_number()) {
            return Status::error(StatusCode::invalid_argument,
                                 "operation " + operation.id() + " CUDA RMSNorm requires epsilon");
        }
        epsilon = epsilon_attribute->get<float>();
        if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
            return Status::error(StatusCode::invalid_argument,
                                 "operation " + operation.id() +
                                     " CUDA RMSNorm epsilon must be positive and finite");
        }
        kernel = CudaKernelKind::rms_norm;
        auxiliary = inputs[1].data;
    } else if (operation.type() == OperationType::softmax && inputs.size() == 1 &&
               inputs[0].shape == output.shape) {
        const auto axis_attribute = operation.attributes().find("axis");
        if (axis_attribute == operation.attributes().end() ||
            !axis_attribute->is_number_integer()) {
            return Status::error(StatusCode::invalid_argument,
                                 "operation " + operation.id() +
                                     " CUDA Softmax requires an integer axis");
        }
        std::int64_t axis = axis_attribute->get<std::int64_t>();
        if (axis < 0) {
            axis += static_cast<std::int64_t>(output.shape.size());
        }
        if (axis != static_cast<std::int64_t>(output.shape.size() - 1)) {
            return Status::error(StatusCode::unsupported,
                                 "operation " + operation.id() +
                                     " CUDA Softmax supports only the final axis");
        }
        kernel = CudaKernelKind::softmax;
    } else {
        return Status::error(StatusCode::unsupported,
                             "operation " + operation.id() +
                                 " is not supported by the Milestone 12 CUDA registry backend");
    }
    auto launched = launch_kernel(kernel, inputs[0].data, auxiliary, output.data, rows, width,
                                  epsilon, kDefaultCudaStream);
    if (!launched.ok()) {
        return launched.status();
    }
    return cuda_status(cudaDeviceSynchronize(),
                       "cudaDeviceSynchronize after registry backend execution");
}

} // namespace forgeir
