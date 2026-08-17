#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "forgeir/core/build_config.hpp"
#include "forgeir/core/build_info.hpp"
#include "forgeir/core/status.hpp"
#include "forgeir/core/version.hpp"
#include "forgeir/ir/graph_loader.hpp"
#include "forgeir/ir/graph_writer.hpp"
#include "forgeir/passes/optimization.hpp"
#include "forgeir/passes/verification.hpp"
#include "forgeir/runtime/memory_planner.hpp"

#if FORGEIR_MLIR_COMPILED
#include "forgeir/mlir/stablehlo_lowering.hpp"
#endif

namespace {

void print_usage() {
    std::cerr << "Usage: forgeir_cli --version | doctor | inspect-graph <path> | verify <graph> "
                 "[--report <path>] | optimize <input> --level O0|O1|O2 --output <path> | "
                 "plan-memory <graph> [--alignment <bytes>] [--output-dir <path>] | "
                 "emit-mlir <graph> --output <file>\n";
}

#if FORGEIR_MLIR_COMPILED
nlohmann::json diagnostic_json(const forgeir::Diagnostic& diagnostic) {
    return nlohmann::json{{"severity", forgeir::to_string(diagnostic.severity)},
                          {"code", diagnostic.code},
                          {"message", diagnostic.message},
                          {"operation_id", diagnostic.operation_id},
                          {"value_id", diagnostic.value_id}};
}
#endif

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

forgeir::Result<std::uint64_t> parse_alignment(const std::string_view text) {
    std::uint64_t value = 0;
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto parsed = std::from_chars(begin, end, value);
    if (parsed.ec != std::errc{} || parsed.ptr != end || value == 0) {
        return forgeir::Status::error(forgeir::StatusCode::invalid_argument,
                                      "alignment must be a positive integer byte count");
    }
    return value;
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc == 5 && std::string_view{argv[1]} == "emit-mlir" &&
        std::string_view{argv[3]} == "--output") {
#if FORGEIR_MLIR_COMPILED
        auto graph = forgeir::GraphLoader::load_from_file(argv[2]);
        if (!graph.ok()) {
            std::cerr << nlohmann::json{{"error", graph.status().message()},
                                        {"status_code", static_cast<int>(graph.status().code())}}
                             .dump()
                      << '\n';
            return static_cast<int>(graph.status().code());
        }
        const forgeir::MlirLoweringResult lowering = forgeir::lower_to_stablehlo(graph.value());
        if (!lowering.success()) {
            nlohmann::json diagnostics = nlohmann::json::array();
            for (const forgeir::Diagnostic& diagnostic : lowering.diagnostics) {
                diagnostics.push_back(diagnostic_json(diagnostic));
            }
            std::cerr << nlohmann::json{{"success", false},
                                        {"error", lowering.status.message()},
                                        {"status_code", static_cast<int>(lowering.status.code())},
                                        {"diagnostics", std::move(diagnostics)}}
                             .dump(2)
                      << '\n';
            return static_cast<int>(lowering.status.code());
        }
        const std::filesystem::path output_path = argv[4];
        const forgeir::Status write_status =
            forgeir::write_mlir_module(output_path, lowering.module_text);
        if (!write_status.ok()) {
            std::cerr << nlohmann::json{{"error", write_status.message()},
                                        {"status_code", static_cast<int>(write_status.code())}}
                             .dump()
                      << '\n';
            return static_cast<int>(write_status.code());
        }
        const forgeir::MlirToolValidationResult validation =
            forgeir::validate_mlir_module(output_path);
        nlohmann::json external{
            {"status", validation.status},
            {"tool",
             validation.tool.empty() ? nlohmann::json(nullptr) : nlohmann::json(validation.tool)},
            {"available", validation.external_validation_available},
            {"syntax_verified", validation.syntax_verified},
            {"canonicalization_attempted", validation.canonicalization_attempted},
            {"canonicalization_succeeded", validation.canonicalization_succeeded},
            {"message", validation.message}};
        if (!validation.canonical_output.empty()) {
            external["canonical_output"] = validation.canonical_output.generic_string();
        }
        if (!validation.diagnostic_log.empty()) {
            external["diagnostic_log"] = validation.diagnostic_log.generic_string();
        }
        const nlohmann::json report{{"success", validation.status != "failed"},
                                    {"graph_hash", graph.value().graph_hash()},
                                    {"output", output_path.generic_string()},
                                    {"external_validation", std::move(external)}};
        std::cout << report.dump(2) << '\n';
        return validation.status == "failed"
                   ? static_cast<int>(forgeir::StatusCode::failed_precondition)
                   : static_cast<int>(forgeir::StatusCode::success);
#else
        std::cerr
            << nlohmann::json{{"success", false},
                              {"error",
                               {{"code", "mlir_bridge_disabled"},
                                {"message", "MLIR textual lowering is disabled; configure with "
                                            "FORGEIR_ENABLE_MLIR=ON"}}},
                              {"status_code", static_cast<int>(forgeir::StatusCode::unsupported)}}
                   .dump(2)
            << '\n';
        return static_cast<int>(forgeir::StatusCode::unsupported);
#endif
    }

    if (argc >= 3 && std::string_view{argv[1]} == "plan-memory") {
        const std::filesystem::path graph_path = argv[2];
        std::filesystem::path output_directory =
            graph_path.parent_path() / (graph_path.stem().string() + ".memory_plan");
        forgeir::MemoryPlannerOptions options;
        bool alignment_seen = false;
        bool output_directory_seen = false;
        for (int index = 3; index < argc; index += 2) {
            if (index + 1 >= argc) {
                print_usage();
                return static_cast<int>(forgeir::StatusCode::invalid_argument);
            }
            const std::string_view option{argv[index]};
            if (option == "--alignment" && !alignment_seen) {
                auto alignment = parse_alignment(argv[index + 1]);
                if (!alignment.ok()) {
                    std::cerr << nlohmann::json{{"error", alignment.status().message()},
                                                {"status_code",
                                                 static_cast<int>(alignment.status().code())}}
                                     .dump()
                              << '\n';
                    return static_cast<int>(alignment.status().code());
                }
                options.alignment_bytes = alignment.value();
                alignment_seen = true;
            } else if (option == "--output-dir" && !output_directory_seen) {
                output_directory = argv[index + 1];
                output_directory_seen = true;
            } else {
                print_usage();
                return static_cast<int>(forgeir::StatusCode::invalid_argument);
            }
        }

        auto graph = forgeir::GraphLoader::load_from_file(graph_path);
        if (!graph.ok()) {
            std::cerr << nlohmann::json{{"error", graph.status().message()},
                                        {"status_code", static_cast<int>(graph.status().code())}}
                             .dump()
                      << '\n';
            return static_cast<int>(graph.status().code());
        }
        const forgeir::VerificationReport verification = forgeir::verify_graph(graph.value());
        if (!verification.success()) {
            std::cerr << forgeir::verification_report_json(verification).dump(2) << '\n';
            return static_cast<int>(verification.pipeline.status.code());
        }
        auto plan = forgeir::plan_memory(graph.value(), options);
        if (!plan.ok()) {
            std::cerr << nlohmann::json{{"error", plan.status().message()},
                                        {"status_code", static_cast<int>(plan.status().code())}}
                             .dump()
                      << '\n';
            return static_cast<int>(plan.status().code());
        }
        auto artifacts = forgeir::write_memory_plan_artifacts(plan.value(), output_directory);
        if (!artifacts.ok()) {
            std::cerr << nlohmann::json{{"error", artifacts.status().message()},
                                        {"status_code",
                                         static_cast<int>(artifacts.status().code())}}
                             .dump()
                      << '\n';
            return static_cast<int>(artifacts.status().code());
        }
        const nlohmann::json summary{
            {"graph_hash", plan.value().graph_hash},
            {"backend", plan.value().backend},
            {"alignment_bytes", plan.value().alignment_bytes},
            {"operation_count", plan.value().schedule.operations.size()},
            {"planned_bytes", plan.value().arena_size_bytes},
            {"peak_live_bytes", plan.value().peak_live_bytes},
            {"naive_allocation_bytes", plan.value().naive_allocation_bytes},
            {"reuse_ratio", plan.value().reuse_ratio},
            {"reuse_fraction", plan.value().reuse_fraction},
            {"artifacts",
             {{"schedule_json", artifacts.value().schedule_json.generic_string()},
              {"memory_plan_json", artifacts.value().memory_plan_json.generic_string()},
              {"timeline_csv", artifacts.value().timeline_csv.generic_string()},
              {"timeline_svg", artifacts.value().timeline_svg.generic_string()}}}};
        std::cout << summary.dump(2) << '\n';
        return static_cast<int>(forgeir::StatusCode::success);
    }

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
