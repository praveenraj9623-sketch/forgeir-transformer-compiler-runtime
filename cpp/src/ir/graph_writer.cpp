#include "forgeir/ir/graph_writer.hpp"

#include <fstream>
#include <string>
#include <unordered_map>

#include "forgeir/core/sha256.hpp"

namespace forgeir {
namespace {

nlohmann::json value_json(const Value& value) {
    return nlohmann::json{{"id", value.id()},
                          {"semantic_name", value.semantic_name()},
                          {"shape", value.descriptor().shape().dimensions()},
                          {"dtype", to_string(value.descriptor().data_type())},
                          {"kind", to_string(value.kind())}};
}

nlohmann::json operation_json(const Operation& operation) {
    return nlohmann::json{{"id", operation.id()},
                          {"type", to_string(operation.type())},
                          {"semantic_name", operation.semantic_name()},
                          {"inputs", operation.input_ids()},
                          {"outputs", operation.output_ids()},
                          {"attributes", operation.attributes()}};
}

std::string escape_dot(std::string text) {
    std::string escaped;
    escaped.reserve(text.size());
    for (const char character : text) {
        if (character == '"' || character == '\\') {
            escaped.push_back('\\');
        }
        escaped.push_back(character);
    }
    return escaped;
}

} // namespace

nlohmann::json graph_json(const Graph& graph) {
    nlohmann::json values = nlohmann::json::array();
    for (const Value& value : graph.values()) {
        values.push_back(value_json(value));
    }
    nlohmann::json operations = nlohmann::json::array();
    for (const Operation& operation : graph.operations()) {
        operations.push_back(operation_json(operation));
    }
    const WeightManifestReference& weights = graph.weight_manifest();
    return nlohmann::json{{"graph_schema_version", graph.schema_version()},
                          {"producer_version", graph.producer_version()},
                          {"model_configuration_hash", graph.model_configuration_hash()},
                          {"graph_hash", graph.graph_hash()},
                          {"inputs", graph.input_ids()},
                          {"outputs", graph.output_ids()},
                          {"values", std::move(values)},
                          {"operations", std::move(operations)},
                          {"weight_manifest",
                           {{"reference", weights.reference},
                            {"sha256", weights.sha256},
                            {"weight_archive", weights.archive},
                            {"weight_archive_sha256", weights.archive_sha256}}}};
}

void refresh_graph_hash(Graph& graph) {
    nlohmann::json payload = graph_json(graph);
    payload.erase("graph_hash");
    graph.set_graph_hash(sha256(payload.dump()));
}

Status write_text_file(const std::filesystem::path& path, const std::string_view text) {
    std::error_code error;
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path(), error);
        if (error) {
            return Status::error(StatusCode::internal,
                                 "unable to create output directory: " + error.message());
        }
    }
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        return Status::error(StatusCode::internal, "unable to open output file: " + path.string());
    }
    stream.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!stream) {
        return Status::error(StatusCode::internal, "unable to write output file: " + path.string());
    }
    return Status::ok_status();
}

Status write_canonical_graph(const Graph& graph, const std::filesystem::path& path) {
    return write_text_file(path, graph_json(graph).dump() + "\n");
}

std::string graph_dot(const Graph& graph) {
    std::unordered_map<std::string, std::string> producer_by_value;
    for (const Operation& operation : graph.operations()) {
        for (const std::string& output : operation.output_ids()) {
            producer_by_value.emplace(output, operation.id());
        }
    }

    std::string dot = "digraph ForgeIR {\n  rankdir=LR;\n";
    for (const Operation& operation : graph.operations()) {
        dot += "  \"" + escape_dot(operation.id()) + "\" [label=\"" +
               escape_dot(operation.id() + " " + std::string(to_string(operation.type()))) +
               "\"];\n";
    }
    for (const Operation& operation : graph.operations()) {
        for (const std::string& input : operation.input_ids()) {
            const auto producer = producer_by_value.find(input);
            if (producer != producer_by_value.end()) {
                dot += "  \"" + escape_dot(producer->second) + "\" -> \"" +
                       escape_dot(operation.id()) + "\" [label=\"" + escape_dot(input) + "\"];\n";
            }
        }
    }
    dot += "}\n";
    return dot;
}

} // namespace forgeir
