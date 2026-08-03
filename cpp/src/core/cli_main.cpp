#include <iostream>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "forgeir/core/build_info.hpp"
#include "forgeir/core/status.hpp"
#include "forgeir/core/version.hpp"
#include "forgeir/ir/graph_loader.hpp"

namespace {

void print_usage() {
    std::cerr << "Usage: forgeir_cli --version | doctor | inspect-graph <path>\n";
}

nlohmann::json summary_json(const forgeir::GraphSummary& summary) {
    return nlohmann::json{{"schema_version", summary.schema_version},
                          {"input_count", summary.input_count},
                          {"output_count", summary.output_count},
                          {"value_count", summary.value_count},
                          {"operation_count", summary.operation_count},
                          {"operation_histogram", summary.operation_histogram},
                          {"estimated_parameter_bytes", summary.estimated_parameter_bytes}};
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc == 3 && std::string_view{argv[1]} == "inspect-graph") {
        const auto graph = forgeir::GraphLoader::load_from_file(argv[2]);
        if (!graph.ok()) {
            std::cerr << nlohmann::json{{"error", graph.status().message()},
                                        {"status_code", static_cast<int>(graph.status().code())}}
                             .dump()
                      << '\n';
            return static_cast<int>(graph.status().code());
        }
        const auto summary = graph.value().summary();
        if (!summary.ok()) {
            std::cerr << nlohmann::json{{"error", summary.status().message()},
                                        {"status_code", static_cast<int>(summary.status().code())}}
                             .dump()
                      << '\n';
            return static_cast<int>(summary.status().code());
        }
        std::cout << summary_json(summary.value()).dump(2) << '\n';
        return static_cast<int>(forgeir::StatusCode::success);
    }

    if (argc != 2) {
        print_usage();
        return static_cast<int>(forgeir::StatusCode::invalid_argument);
    }

    const std::string_view command{argv[1]};
    if (command == "--version") {
        std::cout << "ForgeIR " << forgeir::version() << '\n';
        return static_cast<int>(forgeir::StatusCode::success);
    }
    if (command == "doctor") {
        std::cout << forgeir::build_info_json(forgeir::current_build_info()) << '\n';
        return static_cast<int>(forgeir::StatusCode::success);
    }

    print_usage();
    return static_cast<int>(forgeir::StatusCode::invalid_argument);
}
