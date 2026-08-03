#include "forgeir/ir/graph.hpp"

#include <limits>
#include <utility>

namespace forgeir {

Graph::Graph(std::string schema_version, std::string producer_version,
             std::string model_configuration_hash, std::string graph_hash,
             WeightManifestReference weight_manifest, std::vector<std::string> input_ids,
             std::vector<std::string> output_ids, std::vector<Value> values,
             std::vector<Operation> operations)
    : schema_version_(std::move(schema_version)), producer_version_(std::move(producer_version)),
      model_configuration_hash_(std::move(model_configuration_hash)),
      graph_hash_(std::move(graph_hash)), weight_manifest_(std::move(weight_manifest)),
      input_ids_(std::move(input_ids)), output_ids_(std::move(output_ids)),
      values_(std::move(values)), operations_(std::move(operations)) {}

const std::string& Graph::schema_version() const noexcept { return schema_version_; }

const std::string& Graph::producer_version() const noexcept { return producer_version_; }

const std::string& Graph::model_configuration_hash() const noexcept {
    return model_configuration_hash_;
}

const std::string& Graph::graph_hash() const noexcept { return graph_hash_; }

const WeightManifestReference& Graph::weight_manifest() const noexcept { return weight_manifest_; }

const std::vector<std::string>& Graph::input_ids() const noexcept { return input_ids_; }

const std::vector<std::string>& Graph::output_ids() const noexcept { return output_ids_; }

const std::vector<Value>& Graph::values() const noexcept { return values_; }

const std::vector<Operation>& Graph::operations() const noexcept { return operations_; }

Result<GraphSummary> Graph::summary() const {
    GraphSummary result;
    result.schema_version = schema_version_;
    result.input_count = input_ids_.size();
    result.output_count = output_ids_.size();
    result.value_count = values_.size();
    result.operation_count = operations_.size();

    for (const Operation& operation : operations_) {
        ++result.operation_histogram[std::string(to_string(operation.type()))];
    }
    for (const Value& value : values_) {
        if (value.kind() != ValueKind::parameter) {
            continue;
        }
        const auto value_bytes = value.descriptor().byte_size();
        if (!value_bytes.ok()) {
            return value_bytes.status();
        }
        if (result.estimated_parameter_bytes >
            std::numeric_limits<std::uint64_t>::max() - value_bytes.value()) {
            return Status::error(StatusCode::overflow,
                                 "estimated parameter byte total exceeds uint64_t capacity");
        }
        result.estimated_parameter_bytes += value_bytes.value();
    }
    return result;
}

} // namespace forgeir
