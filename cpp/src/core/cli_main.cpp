#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "forgeir/core/build_info.hpp"
#include "forgeir/core/status.hpp"
#include "forgeir/core/version.hpp"
#include "forgeir/ir/graph_loader.hpp"
#include "forgeir/ir/graph_writer.hpp"
#include "forgeir/passes/optimization.hpp"
#include "forgeir/passes/verification.hpp"

namespace {

void print_usage() {
    std::cerr << "Usage: forgeir_cli --version | doctor | inspect-graph <path> | verify <graph> "
                 "[--report <path>] | optimize <input> --level O0|O1|O2 --output <path>\n";
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

forgeir::Status write_json_report(const std::filesystem::path& path, const nlohmann::json& report) {
    std::error_code error;
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent, error);
        if (error) {
            return forgeir::Status::error(forgeir::StatusCode::internal,
                                          "unable to create report directory: " + error.message());
        }
    }
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        return forgeir::Status::error(forgeir::StatusCode::internal,
                                      "unable to open verification report: " + path.string());
    }
    stream << report.dump(2) << '\n';
    if (!stream) {
        return forgeir::Status::error(forgeir::StatusCode::internal,
                                      "unable to write verification report: " + path.string());
    }
    return forgeir::Status::ok_status();
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc == 7 && std::string_view{argv[1]} == "optimize" &&
        std::string_view{argv[3]} == "--level" && std::string_view{argv[5]} == "--output") {
        const auto level = forgeir::parse_optimization_level(argv[4]);
        if (!level.ok()) {
            std::cerr << nlohmann::json{{"error", level.status().message()},
                                        {"status_code", static_cast<int>(level.status().code())}}
                             .dump()
                      << '\n';
            return static_cast<int>(level.status().code());
        }
        const std::filesystem::path input_path = argv[2];
        const std::filesystem::path output_path = argv[6];
        std::error_code path_error;
        if (std::filesystem::equivalent(input_path, output_path, path_error) && !path_error) {
            std::cerr << nlohmann::json{{"error",
                                         "optimization output must differ from the input graph"},
                                        {"status_code",
                                         static_cast<int>(forgeir::StatusCode::invalid_argument)}}
                             .dump()
                      << '\n';
            return static_cast<int>(forgeir::StatusCode::invalid_argument);
        }
        auto graph = forgeir::GraphLoader::load_from_file(input_path);
        if (!graph.ok()) {
            std::cerr << nlohmann::json{{"error", graph.status().message()},
                                        {"status_code", static_cast<int>(graph.status().code())}}
                             .dump()
                      << '\n';
            return static_cast<int>(graph.status().code());
        }
        const forgeir::Graph before = graph.value();
        const forgeir::OptimizationResult optimization =
            forgeir::optimize_graph(graph.value(), level.value());
        nlohmann::json report = forgeir::optimization_report_json(optimization);
        const std::filesystem::path artifact_base = output_path.parent_path() / output_path.stem();
        const std::filesystem::path report_path = artifact_base.string() + ".pass_report.json";
        const std::filesystem::path before_dot_path = artifact_base.string() + ".before.dot";
        const std::filesystem::path after_dot_path = artifact_base.string() + ".after.dot";
        report["artifacts"] = {{"optimized_graph", output_path.generic_string()},
                               {"pass_report", report_path.generic_string()},
                               {"before_dot", before_dot_path.generic_string()},
                               {"after_dot", after_dot_path.generic_string()}};
        if (!optimization.success()) {
            std::cout << report.dump(2) << '\n';
            return static_cast<int>(optimization.pipeline.status.code());
        }

        const forgeir::Status graph_write =
            forgeir::write_canonical_graph(graph.value(), output_path);
        const forgeir::Status report_write = write_json_report(report_path, report);
        const forgeir::Status before_dot_write =
            forgeir::write_text_file(before_dot_path, forgeir::graph_dot(before));
        const forgeir::Status after_dot_write =
            forgeir::write_text_file(after_dot_path, forgeir::graph_dot(graph.value()));
        for (const forgeir::Status& status :
             {graph_write, report_write, before_dot_write, after_dot_write}) {
            if (!status.ok()) {
                std::cerr << nlohmann::json{{"error", status.message()},
                                            {"status_code", static_cast<int>(status.code())}}
                                 .dump()
                          << '\n';
                return static_cast<int>(status.code());
            }
        }
        const auto reloaded = forgeir::GraphLoader::load_from_file(output_path);
        if (!reloaded.ok()) {
            std::cerr << nlohmann::json{{"error", reloaded.status().message()},
                                        {"status_code", static_cast<int>(reloaded.status().code())}}
                             .dump()
                      << '\n';
            return static_cast<int>(reloaded.status().code());
        }
        std::cout << report.dump(2) << '\n';
        return static_cast<int>(forgeir::StatusCode::success);
    }

    const bool verify_to_stdout = argc == 3 && std::string_view{argv[1]} == "verify";
    const bool verify_to_file = argc == 5 && std::string_view{argv[1]} == "verify" &&
                                std::string_view{argv[3]} == "--report";
    if (verify_to_stdout || verify_to_file) {
        auto graph = forgeir::GraphLoader::load_from_file(argv[2]);
        if (!graph.ok()) {
            std::cerr << nlohmann::json{{"error", graph.status().message()},
                                        {"status_code", static_cast<int>(graph.status().code())}}
                             .dump()
                      << '\n';
            return static_cast<int>(graph.status().code());
        }
        const forgeir::VerificationReport verification = forgeir::verify_graph(graph.value());
        const nlohmann::json report = forgeir::verification_report_json(verification);
        std::cout << report.dump(2) << '\n';
        if (verify_to_file) {
            const forgeir::Status write_status = write_json_report(argv[4], report);
            if (!write_status.ok()) {
                std::cerr << nlohmann::json{{"error", write_status.message()},
                                            {"status_code", static_cast<int>(write_status.code())}}
                                 .dump()
                          << '\n';
                return static_cast<int>(write_status.code());
            }
        }
        return verification.success() ? static_cast<int>(forgeir::StatusCode::success)
                                      : static_cast<int>(forgeir::StatusCode::failed_precondition);
    }

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
