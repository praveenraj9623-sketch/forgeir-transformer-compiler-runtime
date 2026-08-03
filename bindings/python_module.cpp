#include <string>
#include <utility>

#include <pybind11/pybind11.h>

#include "forgeir/core/build_info.hpp"
#include "forgeir/core/version.hpp"
#include "forgeir/ir/graph_loader.hpp"

namespace py = pybind11;

PYBIND11_MODULE(forgeir_py, module) {
    module.doc() = "ForgeIR diagnostics and read-only graph inspection binding";
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
}
