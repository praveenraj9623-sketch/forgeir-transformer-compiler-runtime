#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "forgeir/backends/cuda/cuda_backend.hpp"

namespace py = pybind11;

namespace {

forgeir::CudaKernelKind parse_kernel(const std::string& name) {
    if (name == "GELU") {
        return forgeir::CudaKernelKind::gelu;
    }
    if (name == "RMSNorm") {
        return forgeir::CudaKernelKind::rms_norm;
    }
    if (name == "Softmax") {
        return forgeir::CudaKernelKind::softmax;
    }
    if (name == "FusedBiasGELU") {
        return forgeir::CudaKernelKind::fused_bias_gelu;
    }
    throw py::value_error("unsupported CUDA kernel name: " + name);
}

std::vector<float> checked_host_array(const py::array& array, const std::string& role) {
    if (!array.dtype().is(py::dtype::of<float>())) {
        throw py::type_error(role + " must have dtype float32");
    }
    if ((array.flags() & py::array::c_style) == 0) {
        throw py::value_error(role + " must be C-contiguous; no implicit copy is performed");
    }
    if (array.size() <= 0) {
        throw py::value_error(role + " must not be empty");
    }
    const auto* begin = static_cast<const float*>(array.data());
    return std::vector<float>(begin, begin + array.size());
}

py::dict device_metadata(const forgeir::CudaDeviceMetadata& metadata) {
    py::dict result;
    result["gpu_model"] = metadata.gpu_model;
    result["device_ordinal"] = metadata.device_ordinal;
    result["compute_capability"] = std::to_string(metadata.compute_capability_major) + "." +
                                   std::to_string(metadata.compute_capability_minor);
    result["compute_capability_major"] = metadata.compute_capability_major;
    result["compute_capability_minor"] = metadata.compute_capability_minor;
    result["cuda_runtime_version"] = metadata.cuda_runtime_version;
    result["cuda_driver_version"] = metadata.cuda_driver_version;
    result["nvcc_version"] = metadata.nvcc_version;
    return result;
}

} // namespace

PYBIND11_MODULE(forgeir_cuda_py, module) {
    module.doc() = "ForgeIR handwritten CUDA kernel validation binding";
    module.attr("MAXIMUM_ROW_WIDTH") = forgeir::kCudaMaximumRowWidth;
    module.attr("KERNEL_BLOCK_SIZE") = forgeir::kCudaKernelBlockSize;

    module.def("device_metadata", []() {
        const auto metadata = forgeir::query_cuda_device_metadata();
        if (!metadata.ok()) {
            throw std::runtime_error(metadata.status().message());
        }
        return device_metadata(metadata.value());
    });

    module.def(
        "run_kernel",
        [](const std::string& kernel_name, const py::array& input, const py::object& auxiliary,
           const std::int64_t rows, const std::int64_t width, const float epsilon,
           const std::uint64_t warmup_iterations, const std::uint64_t measured_iterations) {
            forgeir::CudaHostRequest request;
            request.kernel = parse_kernel(kernel_name);
            request.input = checked_host_array(input, "input");
            if (!auxiliary.is_none()) {
                if (!py::isinstance<py::array>(auxiliary)) {
                    throw py::type_error("auxiliary must be a NumPy array or None");
                }
                request.auxiliary =
                    checked_host_array(py::reinterpret_borrow<py::array>(auxiliary), "auxiliary");
            }
            request.rows = rows;
            request.width = width;
            request.epsilon = epsilon;
            request.warmup_iterations = warmup_iterations;
            request.measured_iterations = measured_iterations;
            auto profile = forgeir::run_cuda_host_profile(request);
            if (!profile.ok()) {
                throw py::value_error(profile.status().message());
            }

            py::array_t<float> output(std::vector<py::ssize_t>{static_cast<py::ssize_t>(rows),
                                                               static_cast<py::ssize_t>(width)});
            std::memcpy(output.mutable_data(), profile.value().output.data(),
                        profile.value().output.size() * sizeof(float));
            py::dict launch;
            launch["block_size"] = profile.value().launch.block_size;
            launch["grid_size"] = profile.value().launch.grid_size;
            launch["shared_memory_bytes"] = profile.value().launch.shared_memory_bytes;

            py::dict result;
            result["output"] = std::move(output);
            result["kernel_milliseconds"] = profile.value().kernel_milliseconds;
            result["warmup_iterations"] = profile.value().warmup_iterations;
            result["measured_iterations"] = profile.value().measured_iterations;
            result["launch"] = std::move(launch);
            result["device"] = device_metadata(profile.value().device);
            return result;
        },
        py::arg("kernel_name"), py::arg("input"), py::arg("auxiliary"), py::arg("rows"),
        py::arg("width"), py::arg("epsilon") = 1.0e-5F, py::arg("warmup_iterations") = 10,
        py::arg("measured_iterations") = 50);
}
