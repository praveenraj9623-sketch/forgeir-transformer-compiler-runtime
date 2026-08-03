#include "forgeir/ir/graph_loader.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "forgeir/core/sha256.hpp"

namespace forgeir {
namespace {

using Json = nlohmann::json;

Status invalid(std::string message) {
    return Status::error(StatusCode::invalid_argument, std::move(message));
}

bool has_exact_prefixed_id(const std::string_view id, const std::string_view prefix,
                           const std::size_t digit_count) {
    if (id.size() != prefix.size() + digit_count || id.substr(0, prefix.size()) != prefix) {
        return false;
    }
    return std::all_of(id.begin() + static_cast<std::ptrdiff_t>(prefix.size()), id.end(),
                       [](const char character) {
                           return std::isdigit(static_cast<unsigned char>(character)) != 0;
                       });
}

Result<std::string> required_string(const Json& object, const std::string_view key,
                                    const std::string_view context,
                                    const bool allow_empty = false) {
    const auto iterator = object.find(key);
    if (iterator == object.end() || !iterator->is_string()) {
        return invalid(std::string(context) + " requires string field '" + std::string(key) + "'");
    }
    std::string result = iterator->get<std::string>();
    if (!allow_empty && result.empty()) {
        return invalid(std::string(context) + " field '" + std::string(key) +
                       "' must not be empty");
    }
    return result;
}

Result<std::vector<std::string>> required_id_array(const Json& object, const std::string_view key,
                                                   const std::string_view context,
                                                   const bool allow_empty,
                                                   const bool require_unique) {
    const auto iterator = object.find(key);
    if (iterator == object.end() || !iterator->is_array()) {
        return invalid(std::string(context) + " requires array field '" + std::string(key) + "'");
    }
    if (!allow_empty && iterator->empty()) {
        return invalid(std::string(context) + " field '" + std::string(key) +
                       "' must not be empty");
    }

    std::vector<std::string> ids;
    ids.reserve(iterator->size());
    std::unordered_set<std::string> seen;
    for (const Json& item : *iterator) {
        if (!item.is_string()) {
            return invalid(std::string(context) + " field '" + std::string(key) +
                           "' must contain only value IDs");
        }
        std::string id = item.get<std::string>();
        if (!has_exact_prefixed_id(id, "v", 4)) {
            return invalid(std::string(context) + " contains invalid value ID: " + id);
        }
        if (require_unique && !seen.insert(id).second) {
            return invalid(std::string(context) + " contains duplicate value ID: " + id);
        }
        ids.push_back(std::move(id));
    }
    return ids;
}

Result<std::vector<std::int64_t>> parse_dimensions(const Json& value, const std::string& value_id) {
    const auto iterator = value.find("shape");
    if (iterator == value.end() || !iterator->is_array()) {
        return invalid("value " + value_id + " requires array field 'shape'");
    }

    std::vector<std::int64_t> dimensions;
    dimensions.reserve(iterator->size());
    for (const Json& dimension : *iterator) {
        if (!dimension.is_number_integer() && !dimension.is_number_unsigned()) {
            return invalid("value " + value_id + " has a non-integer shape dimension");
        }
        if (dimension.is_number_unsigned()) {
            const auto unsigned_value = dimension.get<std::uint64_t>();
            if (unsigned_value >
                static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
                return Status::error(StatusCode::overflow,
                                     "value " + value_id +
                                         " shape dimension exceeds int64_t capacity");
            }
            dimensions.push_back(static_cast<std::int64_t>(unsigned_value));
        } else {
            dimensions.push_back(dimension.get<std::int64_t>());
        }
    }
    return dimensions;
}

Result<std::vector<Value>>
parse_values(const Json& document, std::unordered_map<std::string, std::size_t>& value_indices) {
    const auto iterator = document.find("values");
    if (iterator == document.end() || !iterator->is_array() || iterator->empty()) {
        return invalid("graph field 'values' must be a non-empty array");
    }

    std::vector<Value> values;
    values.reserve(iterator->size());
    for (const Json& item : *iterator) {
        if (!item.is_object()) {
            return invalid("every graph value must be an object");
        }
        auto id = required_string(item, "id", "value");
        if (!id.ok()) {
            return id.status();
        }
        if (!has_exact_prefixed_id(id.value(), "v", 4)) {
            return invalid("invalid value ID: " + id.value());
        }
        if (value_indices.find(id.value()) != value_indices.end()) {
            return invalid("duplicate value ID: " + id.value());
        }
        auto semantic_name = required_string(item, "semantic_name", "value " + id.value());
        if (!semantic_name.ok()) {
            return semantic_name.status();
        }
        auto dtype_name = required_string(item, "dtype", "value " + id.value());
        if (!dtype_name.ok()) {
            return dtype_name.status();
        }
        auto data_type = parse_data_type(dtype_name.value());
        if (!data_type.ok()) {
            return data_type.status();
        }
        auto kind_name = required_string(item, "kind", "value " + id.value());
        if (!kind_name.ok()) {
            return kind_name.status();
        }
        auto kind = parse_value_kind(kind_name.value());
        if (!kind.ok()) {
            return kind.status();
        }
        auto dimensions = parse_dimensions(item, id.value());
        if (!dimensions.ok()) {
            return dimensions.status();
        }
        auto shape = Shape::create(dimensions.take_value());
        if (!shape.ok()) {
            return Status::error(shape.status().code(),
                                 "value " + id.value() + ": " + shape.status().message());
        }
        TensorDescriptor descriptor(data_type.value(), shape.take_value());
        const auto byte_size = descriptor.byte_size();
        if (!byte_size.ok()) {
            return Status::error(byte_size.status().code(),
                                 "value " + id.value() + ": " + byte_size.status().message());
        }

        const std::size_t index = values.size();
        value_indices.emplace(id.value(), index);
        values.emplace_back(id.take_value(), semantic_name.take_value(), std::move(descriptor),
                            kind.value());
    }
    return values;
}

Result<std::vector<Operation>>
parse_operations(const Json& document,
                 const std::unordered_map<std::string, std::size_t>& value_indices,
                 std::unordered_map<std::string, std::size_t>& producer_indices) {
    const auto iterator = document.find("operations");
    if (iterator == document.end() || !iterator->is_array() || iterator->empty()) {
        return invalid("graph field 'operations' must be a non-empty array");
    }

    std::vector<Operation> operations;
    operations.reserve(iterator->size());
    std::unordered_set<std::string> operation_ids;
    for (const Json& item : *iterator) {
        if (!item.is_object()) {
            return invalid("every graph operation must be an object");
        }
        auto id = required_string(item, "id", "operation");
        if (!id.ok()) {
            return id.status();
        }
        if (!has_exact_prefixed_id(id.value(), "op", 4)) {
            return invalid("invalid operation ID: " + id.value());
        }
        if (!operation_ids.insert(id.value()).second) {
            return invalid("duplicate operation ID: " + id.value());
        }
        auto type_name = required_string(item, "type", "operation " + id.value());
        if (!type_name.ok()) {
            return type_name.status();
        }
        auto type = parse_operation_type(type_name.value());
        if (!type.ok()) {
            return Status::error(type.status().code(),
                                 "operation " + id.value() + ": " + type.status().message());
        }
        auto semantic_name = required_string(item, "semantic_name", "operation " + id.value());
        if (!semantic_name.ok()) {
            return semantic_name.status();
        }
        auto inputs = required_id_array(item, "inputs", "operation " + id.value(), true, false);
        if (!inputs.ok()) {
            return inputs.status();
        }
        auto outputs = required_id_array(item, "outputs", "operation " + id.value(), false, true);
        if (!outputs.ok()) {
            return outputs.status();
        }
        const auto expected = expected_input_count(type.value());
        if (!expected.has_value() || inputs.value().size() != expected.value()) {
            return invalid("operation " + id.value() + " has an invalid input count for " +
                           type_name.value());
        }
        if (outputs.value().size() != 1) {
            return invalid("operation " + id.value() + " must produce exactly one value");
        }
        for (const std::string& input_id : inputs.value()) {
            if (value_indices.find(input_id) == value_indices.end()) {
                return invalid("operation " + id.value() + " references missing operand " +
                               input_id);
            }
        }
        for (const std::string& output_id : outputs.value()) {
            if (value_indices.find(output_id) == value_indices.end()) {
                return invalid("operation " + id.value() + " references missing output " +
                               output_id);
            }
            if (producer_indices.find(output_id) != producer_indices.end()) {
                return invalid("value " + output_id + " has multiple producing operations");
            }
            producer_indices.emplace(output_id, operations.size());
        }
        const auto attributes = item.find("attributes");
        if (attributes == item.end() || !attributes->is_object()) {
            return invalid("operation " + id.value() + " requires object field 'attributes'");
        }
        if (type.value() == OperationType::parameter) {
            const auto content_hash = attributes->find("content_sha256");
            if (content_hash == attributes->end() || !content_hash->is_string() ||
                !is_lowercase_sha256(content_hash->get<std::string>())) {
                return invalid("operation " + id.value() +
                               " parameter content_sha256 must be a lowercase SHA-256 digest");
            }
        }
        operations.emplace_back(id.take_value(), type.value(), semantic_name.take_value(),
                                inputs.take_value(), outputs.take_value(), *attributes);
    }
    return operations;
}

Status verify_value_roles(const std::vector<Value>& values,
                          const std::vector<Operation>& operations,
                          const std::unordered_map<std::string, std::size_t>& value_indices) {
    for (const Operation& operation : operations) {
        const ValueKind output_kind =
            values[value_indices.at(operation.output_ids().front())].kind();
        ValueKind expected_kind = ValueKind::intermediate;
        if (operation.type() == OperationType::input) {
            expected_kind = ValueKind::input;
        } else if (operation.type() == OperationType::parameter) {
            expected_kind = ValueKind::parameter;
        } else if (operation.type() == OperationType::constant) {
            expected_kind = ValueKind::constant;
        }
        if (expected_kind == ValueKind::intermediate) {
            if (output_kind != ValueKind::intermediate && output_kind != ValueKind::output) {
                return invalid("operation " + operation.id() + " produces an invalid value kind");
            }
        } else if (output_kind != expected_kind) {
            return invalid("operation " + operation.id() + " does not match its output value kind");
        }
    }
    return Status::ok_status();
}

Result<std::vector<Operation>> deterministic_topological_order(
    const std::vector<Operation>& operations,
    const std::unordered_map<std::string, std::size_t>& producer_indices) {
    std::vector<std::size_t> indegrees(operations.size(), 0);
    std::vector<std::vector<std::size_t>> dependents(operations.size());
    for (std::size_t consumer = 0; consumer < operations.size(); ++consumer) {
        std::unordered_set<std::size_t> dependencies;
        for (const std::string& input_id : operations[consumer].input_ids()) {
            dependencies.insert(producer_indices.at(input_id));
        }
        for (const std::size_t producer : dependencies) {
            ++indegrees[consumer];
            dependents[producer].push_back(consumer);
        }
    }

    std::set<std::pair<std::string, std::size_t>> ready;
    for (std::size_t index = 0; index < indegrees.size(); ++index) {
        if (indegrees[index] == 0) {
            ready.emplace(operations[index].id(), index);
        }
    }

    std::vector<Operation> ordered;
    ordered.reserve(operations.size());
    while (!ready.empty()) {
        const std::size_t index = ready.begin()->second;
        ready.erase(ready.begin());
        ordered.push_back(operations[index]);
        for (const std::size_t dependent : dependents[index]) {
            --indegrees[dependent];
            if (indegrees[dependent] == 0) {
                ready.emplace(operations[dependent].id(), dependent);
            }
        }
    }
    if (ordered.size() != operations.size()) {
        return Status::error(StatusCode::failed_precondition, "graph contains a cycle");
    }
    return ordered;
}

Status verify_required_outputs_connected(const std::vector<std::string>& graph_inputs,
                                         const std::vector<std::string>& graph_outputs,
                                         const std::vector<Operation>& operations) {
    std::unordered_set<std::string> reachable(graph_inputs.begin(), graph_inputs.end());
    for (const Operation& operation : operations) {
        bool operation_reachable =
            operation.type() == OperationType::input &&
            reachable.find(operation.output_ids().front()) != reachable.end();
        for (const std::string& input_id : operation.input_ids()) {
            operation_reachable =
                operation_reachable || reachable.find(input_id) != reachable.end();
        }
        if (operation_reachable) {
            reachable.insert(operation.output_ids().begin(), operation.output_ids().end());
        }
    }
    for (const std::string& output_id : graph_outputs) {
        if (reachable.find(output_id) == reachable.end()) {
            return Status::error(StatusCode::failed_precondition,
                                 "required output " + output_id +
                                     " is disconnected from graph inputs");
        }
    }
    return Status::ok_status();
}

Status validate_hash(const std::string& hash, const std::string_view field) {
    if (!is_lowercase_sha256(hash)) {
        return invalid(std::string(field) + " must be a lowercase SHA-256 digest");
    }
    return Status::ok_status();
}

} // namespace

Result<Graph> GraphLoader::load_from_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return Status::error(StatusCode::not_found, "unable to open graph file: " + path.string());
    }
    const std::string text((std::istreambuf_iterator<char>(stream)),
                           std::istreambuf_iterator<char>());
    if (!stream.eof() && stream.fail()) {
        return Status::error(StatusCode::internal, "unable to read graph file: " + path.string());
    }
    return load_from_json(text);
}

Result<Graph> GraphLoader::load_from_json(const std::string_view json_text) {
    Json document;
    try {
        document = Json::parse(json_text);
    } catch (const Json::parse_error& error) {
        return Status::error(StatusCode::parse_error,
                             "malformed graph JSON: " + std::string(error.what()));
    }

    try {
        if (!document.is_object()) {
            return invalid("graph document must be a JSON object");
        }
        auto schema_version = required_string(document, "graph_schema_version", "graph");
        if (!schema_version.ok()) {
            return schema_version.status();
        }
        if (schema_version.value() != "1.0") {
            return Status::error(StatusCode::unsupported,
                                 "unsupported graph schema version: " + schema_version.value());
        }
        auto producer_version = required_string(document, "producer_version", "graph");
        if (!producer_version.ok()) {
            return producer_version.status();
        }
        auto configuration_hash = required_string(document, "model_configuration_hash", "graph");
        if (!configuration_hash.ok()) {
            return configuration_hash.status();
        }
        auto graph_hash = required_string(document, "graph_hash", "graph");
        if (!graph_hash.ok()) {
            return graph_hash.status();
        }
        for (const auto& [hash, field] : std::vector<std::pair<std::string, std::string_view>>{
                 {configuration_hash.value(), "model_configuration_hash"},
                 {graph_hash.value(), "graph_hash"}}) {
            const Status status = validate_hash(hash, field);
            if (!status.ok()) {
                return status;
            }
        }

        const auto weight_manifest_json = document.find("weight_manifest");
        if (weight_manifest_json == document.end() || !weight_manifest_json->is_object()) {
            return invalid("graph requires object field 'weight_manifest'");
        }
        auto manifest_reference =
            required_string(*weight_manifest_json, "reference", "weight_manifest");
        auto manifest_hash = required_string(*weight_manifest_json, "sha256", "weight_manifest");
        auto archive = required_string(*weight_manifest_json, "weight_archive", "weight_manifest");
        auto archive_hash =
            required_string(*weight_manifest_json, "weight_archive_sha256", "weight_manifest");
        if (!manifest_reference.ok()) {
            return manifest_reference.status();
        }
        if (!manifest_hash.ok()) {
            return manifest_hash.status();
        }
        if (!archive.ok()) {
            return archive.status();
        }
        if (!archive_hash.ok()) {
            return archive_hash.status();
        }
        if (archive.value() != "weight_tensors.npz") {
            return invalid("weight_manifest weight_archive must be 'weight_tensors.npz'");
        }
        for (const auto& [hash, field] : std::vector<std::pair<std::string, std::string_view>>{
                 {manifest_hash.value(), "weight_manifest.sha256"},
                 {archive_hash.value(), "weight_manifest.weight_archive_sha256"}}) {
            const Status status = validate_hash(hash, field);
            if (!status.ok()) {
                return status;
            }
        }

        auto inputs = required_id_array(document, "inputs", "graph", false, true);
        auto outputs = required_id_array(document, "outputs", "graph", false, true);
        if (!inputs.ok()) {
            return inputs.status();
        }
        if (!outputs.ok()) {
            return outputs.status();
        }

        std::unordered_map<std::string, std::size_t> value_indices;
        auto values = parse_values(document, value_indices);
        if (!values.ok()) {
            return values.status();
        }
        for (const std::string& input_id : inputs.value()) {
            const auto found = value_indices.find(input_id);
            if (found == value_indices.end()) {
                return invalid("graph input references missing value " + input_id);
            }
            if (values.value()[found->second].kind() != ValueKind::input) {
                return invalid("graph input " + input_id + " is not an input value");
            }
        }
        for (const std::string& output_id : outputs.value()) {
            const auto found = value_indices.find(output_id);
            if (found == value_indices.end()) {
                return invalid("graph output references missing value " + output_id);
            }
            if (values.value()[found->second].kind() != ValueKind::output) {
                return invalid("graph output " + output_id + " is not an output value");
            }
        }

        std::unordered_map<std::string, std::size_t> producer_indices;
        auto operations = parse_operations(document, value_indices, producer_indices);
        if (!operations.ok()) {
            return operations.status();
        }
        for (const Value& value : values.value()) {
            if (producer_indices.find(value.id()) == producer_indices.end()) {
                return invalid("value " + value.id() + " has no producing operation");
            }
        }
        const Status role_status =
            verify_value_roles(values.value(), operations.value(), value_indices);
        if (!role_status.ok()) {
            return role_status;
        }
        auto ordered_operations =
            deterministic_topological_order(operations.value(), producer_indices);
        if (!ordered_operations.ok()) {
            return ordered_operations.status();
        }
        const Status connectivity_status = verify_required_outputs_connected(
            inputs.value(), outputs.value(), ordered_operations.value());
        if (!connectivity_status.ok()) {
            return connectivity_status;
        }

        Json hash_payload = document;
        hash_payload.erase("graph_hash");
        if (sha256(hash_payload.dump()) != graph_hash.value()) {
            return invalid("graph_hash does not match the canonical graph content");
        }

        WeightManifestReference weight_manifest{manifest_reference.take_value(),
                                                manifest_hash.take_value(), archive.take_value(),
                                                archive_hash.take_value()};
        return Graph(schema_version.take_value(), producer_version.take_value(),
                     configuration_hash.take_value(), graph_hash.take_value(),
                     std::move(weight_manifest), inputs.take_value(), outputs.take_value(),
                     values.take_value(), ordered_operations.take_value());
    } catch (const Json::exception& error) {
        return invalid("invalid graph JSON field: " + std::string(error.what()));
    }
}

} // namespace forgeir
