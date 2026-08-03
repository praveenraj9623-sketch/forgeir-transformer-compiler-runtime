#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "forgeir/core/build_info.hpp"
#include "forgeir/core/version.hpp"
#include "forgeir/ir/graph_loader.hpp"
#include "forgeir/passes/verification.hpp"
#include "forgeir/runtime/benchmark.hpp"
#include "forgeir/runtime/runtime_session.hpp"

namespace py = pybind11;

namespace {

std::vector<std::int64_t> array_shape(const py::array& array) {
    std::vector<std::int64_t> shape;
    shape.reserve(static_cast<std::size_t>(array.ndim()));
    for (py::ssize_t axis = 0; axis < array.ndim(); ++axis) {
        const py::ssize_t dimension = array.shape(axis);
        if (dimension <= 0) {
            throw py::value_error("runtime tensors require strictly positive dimensions");
        }
        shape.push_back(static_cast<std::int64_t>(dimension));
    }
    return shape;
}

std::unordered_map<std::string, forgeir::ExternalTensor> external_tensors(const py::dict& tensors,
                                                                          const std::string& role) {
    std::unordered_map<std::string, forgeir::ExternalTensor> result;
    result.reserve(tensors.size());
    for (const auto& item : tensors) {
        if (!py::isinstance<py::str>(item.first) || !py::isinstance<py::array>(item.second)) {
            throw py::type_error(role + " must map string keys to NumPy arrays");
        }
        const std::string key = py::cast<std::string>(item.first);
        const py::array array = py::reinterpret_borrow<py::array>(item.second);
        if (!array.dtype().is(py::dtype::of<float>())) {
            throw py::type_error(role + " '" + key + "' must have dtype float32");
        }
        const bool contiguous = (array.flags() & py::array::c_style) != 0;
        if (!contiguous) {
            throw py::value_error(role + " '" + key +
                                  "' must be C-contiguous; no implicit copy "
                                  "is performed");
        }
        if (array.nbytes() < 0) {
            throw py::value_error(role + " '" + key + "' has an invalid byte size");
        }
        result.emplace(key, forgeir::ExternalTensor{
                                static_cast<const float*>(array.data()), array_shape(array),
                                static_cast<std::uint64_t>(array.nbytes()), true});
    }
    return result;
}

py::array host_tensor_array(const forgeir::HostTensor& tensor) {
    std::vector<py::ssize_t> shape;
    shape.reserve(tensor.shape.size());
    for (const std::int64_t dimension : tensor.shape) {
        shape.push_back(static_cast<py::ssize_t>(dimension));
    }
    py::array_t<float> result(shape);
    std::memcpy(result.mutable_data(), tensor.values.data(), tensor.values.size() * sizeof(float));
    return result;
}

} // namespace

PYBIND11_MODULE(forgeir_py, module) {
    module.doc() = "ForgeIR diagnostics, graph inspection, and float32 CPU execution binding";
    module.attr("__version__") = std::string{forgeir::version()};

    module.def("version", []() { return std::string{forgeir::version()}; });
    module.def("doctor", []() {
        const forgeir::BuildInfo info = forgeir::current_build_info(PY_VERSION);
        py::dict features;
        features["cuda"] = info.cuda_compiled;
        features["hip"] = info.hip_compiled;
        features["mlir"] = info.mlir_compiled;

        py::dict diagnostic;
        diagnostic["forgeir_version"] = info.forgeir_version;
        diagnostic["compiler"] = info.compiler;
        diagnostic["build_type"] = info.build_type;
        diagnostic["operating_system"] = info.operating_system;
        diagnostic["cpp_standard"] = info.cpp_standard;
        diagnostic["python_version"] = *info.python_version;
        diagnostic["features"] = std::move(features);
        return diagnostic;
    });
    module.def("graph_summary", [](const std::string& path) {
        const auto graph = forgeir::GraphLoader::load_from_file(path);
        if (!graph.ok()) {
            throw py::value_error(graph.status().message());
        }
        const auto summary = graph.value().summary();
        if (!summary.ok()) {
            throw py::value_error(summary.status().message());
        }

        py::dict histogram;
        for (const auto& [operation, count] : summary.value().operation_histogram) {
            histogram[py::str(operation)] = count;
        }
        py::dict result;
        result["schema_version"] = summary.value().schema_version;
        result["input_count"] = summary.value().input_count;
        result["output_count"] = summary.value().output_count;
        result["value_count"] = summary.value().value_count;
        result["operation_count"] = summary.value().operation_count;
        result["operation_histogram"] = std::move(histogram);
        result["estimated_parameter_bytes"] = summary.value().estimated_parameter_bytes;
        return result;
    });
    module.def("verify_graph", [](const std::string& path) {
        auto graph = forgeir::GraphLoader::load_from_file(path);
        if (!graph.ok()) {
            throw py::value_error(graph.status().message());
        }
        const forgeir::VerificationReport report = forgeir::verify_graph(graph.value());
        const std::string json_text = forgeir::verification_report_json(report).dump();
        return py::module_::import("json").attr("loads")(json_text);
    });

    py::class_<forgeir::RuntimeSession, std::shared_ptr<forgeir::RuntimeSession>>(module,
                                                                                  "RuntimeSession");
    module.def(
        "load_graph",
        [](const std::string& path, const std::string& backend) {
            auto session = forgeir::RuntimeSession::load(path, backend);
            if (!session.ok()) {
                throw py::value_error(session.status().message());
            }
            return session.take_value();
        },
        py::arg("path"), py::arg("backend") = "cpu");
    module.def(
        "execute",
        [](const std::shared_ptr<forgeir::RuntimeSession>& session, const py::dict& inputs,
           const py::dict& parameters, const std::vector<std::string>& capture_values) {
            if (!session) {
                throw py::value_error("runtime session must not be null");
            }
            const auto external_inputs = external_tensors(inputs, "input");
            const auto external_parameters = external_tensors(parameters, "parameter");
            const forgeir::Status status =
                session->execute(external_inputs, external_parameters, capture_values);
            if (!status.ok()) {
                throw py::value_error(status.message());
            }
        },
        py::arg("session"), py::arg("inputs"), py::arg("parameters"),
        py::arg("capture_values") = std::vector<std::string>{});
    module.def(
        "get_outputs",
        [](const std::shared_ptr<forgeir::RuntimeSession>& session,
           const std::vector<std::string>& value_ids) {
            if (!session) {
                throw py::value_error("runtime session must not be null");
            }
            py::dict result;
            if (value_ids.empty()) {
                const auto outputs = session->get_outputs();
                if (!outputs.ok()) {
                    throw py::value_error(outputs.status().message());
                }
                for (const auto& [value_id, tensor] : outputs.value()) {
                    result[py::str(value_id)] = host_tensor_array(tensor);
                }
                return result;
            }
            for (const std::string& value_id : value_ids) {
                const auto tensor = session->get_value(value_id);
                if (!tensor.ok()) {
                    throw py::value_error(tensor.status().message());
                }
                result[py::str(value_id)] = host_tensor_array(tensor.value());
            }
            return result;
        },
        py::arg("session"), py::arg("value_ids") = std::vector<std::string>{});
    module.def("get_trace", [](const std::shared_ptr<forgeir::RuntimeSession>& session) {
        if (!session) {
            throw py::value_error("runtime session must not be null");
        }
        py::list result;
        for (const forgeir::ExecutionTraceRecord& record : session->trace()) {
            py::dict item;
            item["operation_id"] = record.operation_id;
            item["operation_type"] = record.operation_type;
            item["kernel"] = record.kernel;
            item["output_shape"] = record.output_shape;
            item["elapsed_microseconds"] = record.elapsed_microseconds;
            item["arena_offset"] = record.has_arena_offset
                                       ? py::cast(record.arena_offset)
                                       : py::reinterpret_borrow<py::object>(Py_None);
            result.append(std::move(item));
        }
        return result;
    });
    module.def(
        "benchmark_execute",
        [](const std::shared_ptr<forgeir::RuntimeSession>& session, const py::dict& inputs,
           const py::dict& parameters, const std::uint64_t warmup_count,
           const std::uint64_t measured_iteration_count) {
            if (!session) {
                throw py::value_error("runtime session must not be null");
            }
            const auto external_inputs = external_tensors(inputs, "input");
            const auto external_parameters = external_tensors(parameters, "parameter");
            forgeir::RuntimeBenchmarkOptions options;
            options.warmup_count = warmup_count;
            options.measured_iteration_count = measured_iteration_count;
            auto benchmark =
                forgeir::benchmark_runtime(*session, external_inputs, external_parameters, options);
            if (!benchmark.ok()) {
                throw py::value_error(benchmark.status().message());
            }

            py::list operations;
            for (const forgeir::OperationTimingSamples& operation : benchmark.value().operations) {
                py::dict item;
                item["operation_id"] = operation.operation_id;
                item["operation_type"] = operation.operation_type;
                item["kernel"] = operation.kernel;
                item["output_shape"] = operation.output_shape;
                item["samples_microseconds"] = operation.elapsed_microseconds;
                operations.append(std::move(item));
            }
            py::dict result;
            result["clock_source"] = benchmark.value().clock_source;
            result["clock_is_steady"] = benchmark.value().clock_is_steady;
            result["warmup_count"] = benchmark.value().warmup_count;
            result["measured_iteration_count"] = benchmark.value().measured_iteration_count;
            result["planned_arena_bytes"] = benchmark.value().planned_arena_bytes;
            result["peak_live_bytes"] = benchmark.value().peak_live_bytes;
            result["naive_allocation_bytes"] = benchmark.value().naive_allocation_bytes;
            result["end_to_end_samples_microseconds"] = benchmark.value().end_to_end_microseconds;
            result["operations"] = std::move(operations);
            return result;
        },
        py::arg("session"), py::arg("inputs"), py::arg("parameters"), py::arg("warmup_count"),
        py::arg("measured_iteration_count"));
}
