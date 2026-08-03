#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "forgeir/core/status.hpp"
#include "forgeir/ir/operation.hpp"
#include "forgeir/ir/value.hpp"

namespace forgeir {

struct WeightManifestReference {
    std::string reference;
    std::string sha256;
    std::string archive;
    std::string archive_sha256;
};

struct GraphSummary {
    std::string schema_version;
    std::size_t input_count{0};
    std::size_t output_count{0};
    std::size_t value_count{0};
    std::size_t operation_count{0};
    std::map<std::string, std::size_t> operation_histogram;
    std::uint64_t estimated_parameter_bytes{0};
};

class Graph {
  public:
    Graph(std::string schema_version, std::string producer_version,
          std::string model_configuration_hash, std::string graph_hash,
          WeightManifestReference weight_manifest, std::vector<std::string> input_ids,
          std::vector<std::string> output_ids, std::vector<Value> values,
          std::vector<Operation> operations);

    [[nodiscard]] const std::string& schema_version() const noexcept;
    [[nodiscard]] const std::string& producer_version() const noexcept;
    [[nodiscard]] const std::string& model_configuration_hash() const noexcept;
    [[nodiscard]] const std::string& graph_hash() const noexcept;
    [[nodiscard]] const WeightManifestReference& weight_manifest() const noexcept;
    [[nodiscard]] const std::vector<std::string>& input_ids() const noexcept;
    [[nodiscard]] const std::vector<std::string>& output_ids() const noexcept;
    [[nodiscard]] const std::vector<Value>& values() const noexcept;
    [[nodiscard]] const std::vector<Operation>& operations() const noexcept;
    [[nodiscard]] Result<GraphSummary> summary() const;

  private:
    std::string schema_version_;
    std::string producer_version_;
    std::string model_configuration_hash_;
    std::string graph_hash_;
    WeightManifestReference weight_manifest_;
    std::vector<std::string> input_ids_;
    std::vector<std::string> output_ids_;
    std::vector<Value> values_;
    std::vector<Operation> operations_;
};

} // namespace forgeir
