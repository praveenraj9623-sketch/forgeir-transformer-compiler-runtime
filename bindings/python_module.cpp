#include <string>
#include <utility>

#include <pybind11/pybind11.h>

#include "forgeir/core/build_info.hpp"
#include "forgeir/core/version.hpp"

namespace py = pybind11;

PYBIND11_MODULE(forgeir_py, module) {
    module.doc() = "ForgeIR Milestone 1 diagnostics binding";
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
}
