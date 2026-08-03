#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include "forgeir/core/status.hpp"
#include "forgeir/ir/graph.hpp"

namespace forgeir {

[[nodiscard]] std::unordered_map<std::string, std::size_t>
operation_index_by_output(const Graph& graph);
[[nodiscard]] std::unordered_map<std::string, std::size_t> value_index_by_id(const Graph& graph);
[[nodiscard]] std::unordered_map<std::string, std::size_t> value_use_counts(const Graph& graph);
[[nodiscard]] bool is_declared_output(const Graph& graph, std::string_view value_id);
[[nodiscard]] bool has_side_effect(const Operation& operation);
void replace_all_uses(Graph& graph, std::string_view from, std::string_view to);
void erase_operations(Graph& graph, const std::unordered_set<std::string>& operation_ids);
void erase_values(Graph& graph, const std::unordered_set<std::string>& value_ids);
[[nodiscard]] Result<bool> stable_topological_order(Graph& graph);

} // namespace forgeir
