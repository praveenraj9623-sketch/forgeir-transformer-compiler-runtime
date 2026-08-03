#pragma once

#include <filesystem>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "forgeir/core/status.hpp"
#include "forgeir/ir/graph.hpp"

namespace forgeir {

[[nodiscard]] nlohmann::json graph_json(const Graph& graph);
[[nodiscard]] std::string graph_dot(const Graph& graph);
void refresh_graph_hash(Graph& graph);
[[nodiscard]] Status write_canonical_graph(const Graph& graph, const std::filesystem::path& path);
[[nodiscard]] Status write_text_file(const std::filesystem::path& path, std::string_view text);

} // namespace forgeir
