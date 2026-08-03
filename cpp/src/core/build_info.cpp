#include "forgeir/core/build_info.hpp"

#include <utility>

#include <nlohmann/json.hpp>

#include "forgeir/core/build_config.hpp"

namespace forgeir {

BuildInfo current_build_info(std::optional<std::string> python_version) {
    return BuildInfo{
        FORGEIR_VERSION,
        FORGEIR_COMPILER_ID,
        FORGEIR_BUILD_TYPE,
        FORGEIR_OPERATING_SYSTEM,
        "C++17",
        std::move(python_version),
        FORGEIR_CUDA_COMPILED != 0,
        FORGEIR_HIP_COMPILED != 0,
        FORGEIR_MLIR_COMPILED != 0,
    };
}

std::string build_info_json(const BuildInfo& build_info) {
    nlohmann::json python_version = nullptr;
    if (build_info.python_version.has_value()) {
        python_version = *build_info.python_version;
    }

    const nlohmann::json diagnostic{
        {"forgeir_version", build_info.forgeir_version},
        {"compiler", build_info.compiler},
        {"build_type", build_info.build_type},
        {"operating_system", build_info.operating_system},
        {"cpp_standard", build_info.cpp_standard},
        {"python_version", std::move(python_version)},
        {"features",
         {
             {"cuda", build_info.cuda_compiled},
             {"hip", build_info.hip_compiled},
             {"mlir", build_info.mlir_compiled},
         }},
    };
    return diagnostic.dump(2);
}

} // namespace forgeir
