#pragma once

#include <string>

#include <nlohmann/json.hpp>

#include "forgeir/ir/graph.hpp"
#include "forgeir/passes/pass_manager.hpp"

namespace forgeir {

struct VerificationReport {
    std::string graph_schema_version;
    std::string graph_hash;
    PassManagerResult pipeline;

    [[nodiscard]] bool success() const noexcept;
};

[[nodiscard]] VerificationReport verify_graph(Graph& graph);
[[nodiscard]] nlohmann::json verification_report_json(const VerificationReport& report);

} // namespace forgeir
