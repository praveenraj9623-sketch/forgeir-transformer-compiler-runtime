#include "forgeir/runtime/memory_planner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "forgeir/ir/graph_writer.hpp"

namespace forgeir {
namespace {

struct FreeBlock {
    std::uint64_t offset{0};
    std::uint64_t size{0};
};

struct ActiveAllocation {
    std::size_t tensor_index{0};
    std::uint64_t final_use_index{0};
    std::uint64_t offset{0};
    std::uint64_t size{0};
};

Result<std::uint64_t> checked_add(const std::uint64_t left, const std::uint64_t right,
                                  const std::string& context) {
    if (left > std::numeric_limits<std::uint64_t>::max() - right) {
        return Status::error(StatusCode::overflow, context + " exceeds uint64_t capacity");
    }
    return left + right;
}

Result<std::uint64_t> align_up(const std::uint64_t value, const std::uint64_t alignment) {
    if (alignment == 0) {
        return Status::error(StatusCode::invalid_argument, "tensor alignment must be positive");
    }
    const std::uint64_t remainder = value % alignment;
    if (remainder == 0) {
        return value;
    }
    return checked_add(value, alignment - remainder, "aligned byte size");
}

bool is_power_of_two(const std::uint64_t value) { return value != 0 && (value & (value - 1)) == 0; }

bool is_declared_output(const Graph& graph, const std::string& value_id) {
    return std::find(graph.output_ids().begin(), graph.output_ids().end(), value_id) !=
           graph.output_ids().end();
}

Status coalesce_free_blocks(std::vector<FreeBlock>& blocks) {
    std::sort(blocks.begin(), blocks.end(), [](const FreeBlock& left, const FreeBlock& right) {
        return left.offset < right.offset;
    });
    std::vector<FreeBlock> coalesced;
    for (const FreeBlock& block : blocks) {
        auto previous_end =
            coalesced.empty()
                ? Result<std::uint64_t>(std::uint64_t{0})
                : checked_add(coalesced.back().offset, coalesced.back().size, "free block range");
        if (!previous_end.ok()) {
            return previous_end.status();
        }
        if (!coalesced.empty() && previous_end.value() == block.offset) {
            auto merged_size =
                checked_add(coalesced.back().size, block.size, "coalesced free block size");
            if (!merged_size.ok()) {
                return merged_size.status();
            }
            coalesced.back().size = merged_size.value();
        } else {
            coalesced.push_back(block);
        }
    }
    blocks = std::move(coalesced);
    return Status::ok_status();
}

Status release_expired(const std::uint64_t definition_index, std::vector<ActiveAllocation>& active,
                       std::vector<FreeBlock>& free_blocks) {
    std::vector<ActiveAllocation> retained;
    retained.reserve(active.size());
    for (const ActiveAllocation& allocation : active) {
        if (allocation.final_use_index < definition_index) {
            free_blocks.push_back(FreeBlock{allocation.offset, allocation.size});
        } else {
            retained.push_back(allocation);
        }
    }
    active = std::move(retained);
    return coalesce_free_blocks(free_blocks);
}

Result<std::uint64_t> allocate_best_fit(const std::uint64_t size,
                                        std::vector<FreeBlock>& free_blocks,
                                        std::uint64_t& arena_end) {
    std::size_t best = free_blocks.size();
    for (std::size_t index = 0; index < free_blocks.size(); ++index) {
        if (free_blocks[index].size < size) {
            continue;
        }
        if (best == free_blocks.size() || free_blocks[index].size < free_blocks[best].size ||
            (free_blocks[index].size == free_blocks[best].size &&
             free_blocks[index].offset < free_blocks[best].offset)) {
            best = index;
        }
    }
    if (best != free_blocks.size()) {
        const std::uint64_t offset = free_blocks[best].offset;
        if (free_blocks[best].size == size) {
            free_blocks.erase(free_blocks.begin() + static_cast<std::ptrdiff_t>(best));
        } else {
            const auto remaining_offset =
                checked_add(free_blocks[best].offset, size, "remaining free block offset");
            if (!remaining_offset.ok()) {
                return remaining_offset.status();
            }
            free_blocks[best].offset = remaining_offset.value();
            free_blocks[best].size -= size;
        }
        return offset;
    }

    const auto end = checked_add(arena_end, size, "planned arena size");
    if (!end.ok()) {
        return end.status();
    }
    const std::uint64_t offset = arena_end;
    arena_end = end.value();
    return offset;
}

Result<std::uint64_t> append_protected(const std::uint64_t size, std::uint64_t& arena_end) {
    const auto end = checked_add(arena_end, size, "planned arena size");
    if (!end.ok()) {
        return end.status();
    }
    const std::uint64_t offset = arena_end;
    arena_end = end.value();
    return offset;
}

bool lifetimes_overlap(const TensorLifetime& left, const TensorLifetime& right) {
    return !(left.lifetime_end < right.lifetime_start || right.lifetime_end < left.lifetime_start);
}

Result<bool> memory_ranges_overlap(const TensorLifetime& left, const TensorLifetime& right) {
    const auto left_end = checked_add(left.arena_offset.value(), left.aligned_byte_size,
                                      "allocation range for " + left.value_id);
    const auto right_end = checked_add(right.arena_offset.value(), right.aligned_byte_size,
                                       "allocation range for " + right.value_id);
    if (!left_end.ok()) {
        return left_end.status();
    }
    if (!right_end.ok()) {
        return right_end.status();
    }
    return left.arena_offset.value() < right_end.value() &&
           right.arena_offset.value() < left_end.value();
}

std::string csv_escape(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) {
        return value;
    }
    std::string escaped{"\""};
    for (const char character : value) {
        if (character == '"') {
            escaped += "\"\"";
        } else {
            escaped.push_back(character);
        }
    }
    escaped.push_back('"');
    return escaped;
}

std::string xml_escape(const std::string& value) {
    std::string escaped;
    for (const char character : value) {
        switch (character) {
        case '&':
            escaped += "&amp;";
            break;
        case '<':
            escaped += "&lt;";
            break;
        case '>':
            escaped += "&gt;";
            break;
        case '"':
            escaped += "&quot;";
            break;
        default:
            escaped.push_back(character);
            break;
        }
    }
    return escaped;
}

nlohmann::json optional_index_json(const std::optional<std::uint64_t>& value) {
    return value.has_value() ? nlohmann::json(value.value()) : nlohmann::json(nullptr);
}

} // namespace

std::string_view to_string(const TensorStorageClass storage_class) noexcept {
    switch (storage_class) {
    case TensorStorageClass::external_input:
        return "external_input";
    case TensorStorageClass::external_immutable:
        return "external_immutable";
    case TensorStorageClass::arena_reusable:
        return "arena_reusable";
    case TensorStorageClass::arena_output:
        return "arena_output";
    }
    return "invalid";
}

Result<MemoryPlan> plan_memory(const Graph& graph, const MemoryPlannerOptions& options) {
    if (options.backend.empty()) {
        return Status::error(StatusCode::invalid_argument,
                             "memory planner backend must not be empty");
    }
    if (!is_power_of_two(options.alignment_bytes)) {
        return Status::error(StatusCode::invalid_argument,
                             "tensor alignment must be a positive power of two");
    }
    auto schedule_result = build_execution_schedule(graph);
    if (!schedule_result.ok()) {
        return schedule_result.status();
    }
    ExecutionSchedule schedule = schedule_result.take_value();
    if (schedule.operations.size() >= std::numeric_limits<std::uint64_t>::max()) {
        return Status::error(StatusCode::overflow, "schedule length exceeds uint64_t capacity");
    }
    const auto retention_boundary = static_cast<std::uint64_t>(schedule.operations.size());

    std::unordered_map<std::string, std::uint64_t> definition_by_value;
    std::unordered_map<std::string, std::vector<std::uint64_t>> uses_by_value;
    for (const ScheduledOperation& operation : schedule.operations) {
        for (const std::string& output_id : operation.output_ids) {
            if (!definition_by_value.emplace(output_id, operation.index).second) {
                return Status::error(StatusCode::failed_precondition,
                                     "value " + output_id + " has duplicate definition indices");
            }
        }
        for (const std::string& input_id : operation.input_ids) {
            uses_by_value[input_id].push_back(operation.index);
        }
    }

    MemoryPlan plan;
    plan.graph_hash = graph.graph_hash();
    plan.backend = options.backend;
    plan.alignment_bytes = options.alignment_bytes;
    plan.schedule = std::move(schedule);
    plan.tensors.reserve(graph.values().size());
    for (const Value& value : graph.values()) {
        const auto definition = definition_by_value.find(value.id());
        if (definition == definition_by_value.end()) {
            return Status::error(StatusCode::failed_precondition,
                                 "missing lifetime definition for value " + value.id());
        }
        for (const std::int64_t dimension : value.descriptor().shape().dimensions()) {
            if (dimension <= 0) {
                return Status::error(StatusCode::failed_precondition,
                                     "value " + value.id() +
                                         " contains an unresolved dynamic or zero dimension");
            }
        }
        const auto bytes = value.descriptor().byte_size();
        if (!bytes.ok()) {
            return Status::error(bytes.status().code(),
                                 "value " + value.id() + ": " + bytes.status().message());
        }
        const auto aligned_bytes = align_up(bytes.value(), options.alignment_bytes);
        if (!aligned_bytes.ok()) {
            return Status::error(aligned_bytes.status().code(),
                                 "value " + value.id() + ": " + aligned_bytes.status().message());
        }

        const auto uses = uses_by_value.find(value.id());
        std::optional<std::uint64_t> first_use;
        std::optional<std::uint64_t> last_use;
        if (uses != uses_by_value.end() && !uses->second.empty()) {
            const auto bounds = std::minmax_element(uses->second.begin(), uses->second.end());
            first_use = *bounds.first;
            last_use = *bounds.second;
            if (first_use.value() < definition->second) {
                return Status::error(StatusCode::failed_precondition,
                                     "value " + value.id() +
                                         " is used before its scheduled definition");
            }
        }

        const bool declared_output =
            is_declared_output(graph, value.id()) || value.kind() == ValueKind::output;
        if (value.kind() == ValueKind::intermediate && !last_use.has_value() && !declared_output) {
            return Status::error(StatusCode::failed_precondition,
                                 "missing lifetime use information for intermediate value " +
                                     value.id());
        }

        TensorLifetime lifetime;
        lifetime.value_id = value.id();
        lifetime.semantic_name = value.semantic_name();
        lifetime.kind = value.kind();
        lifetime.definition_index = definition->second;
        lifetime.first_use_index = first_use;
        lifetime.final_use_index =
            declared_output ? retention_boundary : last_use.value_or(definition->second);
        lifetime.lifetime_start = definition->second;
        lifetime.lifetime_end = lifetime.final_use_index;
        lifetime.alignment_bytes = options.alignment_bytes;
        lifetime.byte_size = bytes.value();
        lifetime.aligned_byte_size = aligned_bytes.value();

        if (declared_output || value.kind() == ValueKind::output) {
            lifetime.storage_class = TensorStorageClass::arena_output;
            lifetime.protected_buffer = true;
        } else if (value.kind() == ValueKind::input) {
            lifetime.storage_class = TensorStorageClass::external_input;
            lifetime.external = true;
            lifetime.protected_buffer = true;
        } else if (value.kind() == ValueKind::parameter || value.kind() == ValueKind::constant) {
            lifetime.storage_class = TensorStorageClass::external_immutable;
            lifetime.external = true;
            lifetime.immutable = true;
            lifetime.protected_buffer = true;
        } else {
            lifetime.storage_class = TensorStorageClass::arena_reusable;
        }
        plan.tensors.push_back(std::move(lifetime));
    }

    std::vector<std::size_t> allocation_order;
    for (std::size_t index = 0; index < plan.tensors.size(); ++index) {
        if (!plan.tensors[index].external) {
            allocation_order.push_back(index);
            const auto naive =
                checked_add(plan.naive_allocation_bytes, plan.tensors[index].aligned_byte_size,
                            "naive allocation byte total");
            if (!naive.ok()) {
                return naive.status();
            }
            plan.naive_allocation_bytes = naive.value();
        }
    }
    std::sort(allocation_order.begin(), allocation_order.end(),
              [&plan](const std::size_t left, const std::size_t right) {
                  const TensorLifetime& left_value = plan.tensors[left];
                  const TensorLifetime& right_value = plan.tensors[right];
                  if (left_value.definition_index != right_value.definition_index) {
                      return left_value.definition_index < right_value.definition_index;
                  }
                  return left_value.value_id < right_value.value_id;
              });

    std::uint64_t arena_end = 0;
    std::vector<ActiveAllocation> active;
    std::vector<FreeBlock> free_blocks;
    for (const std::size_t tensor_index : allocation_order) {
        TensorLifetime& lifetime = plan.tensors[tensor_index];
        const Status release_status =
            release_expired(lifetime.definition_index, active, free_blocks);
        if (!release_status.ok()) {
            return release_status;
        }
        Result<std::uint64_t> offset =
            lifetime.storage_class == TensorStorageClass::arena_output
                ? append_protected(lifetime.aligned_byte_size, arena_end)
                : allocate_best_fit(lifetime.aligned_byte_size, free_blocks, arena_end);
        if (!offset.ok()) {
            return Status::error(offset.status().code(),
                                 "value " + lifetime.value_id + ": " + offset.status().message());
        }
        lifetime.arena_offset = offset.value();
        active.push_back(ActiveAllocation{tensor_index, lifetime.final_use_index, offset.value(),
                                          lifetime.aligned_byte_size});
    }
    plan.arena_size_bytes = arena_end;
    if (plan.arena_size_bytes > plan.naive_allocation_bytes) {
        return Status::error(StatusCode::failed_precondition,
                             "planned arena exceeds the naive aligned allocation total");
    }

    for (std::uint64_t index = 0; index <= retention_boundary; ++index) {
        std::uint64_t live_bytes = 0;
        for (const TensorLifetime& lifetime : plan.tensors) {
            if (lifetime.external || lifetime.lifetime_start > index ||
                lifetime.lifetime_end < index) {
                continue;
            }
            const auto live =
                checked_add(live_bytes, lifetime.aligned_byte_size, "peak live byte total");
            if (!live.ok()) {
                return live.status();
            }
            live_bytes = live.value();
        }
        plan.peak_live_bytes = std::max(plan.peak_live_bytes, live_bytes);
    }

    if (plan.arena_size_bytes != 0) {
        plan.reuse_ratio = static_cast<double>(plan.naive_allocation_bytes) /
                           static_cast<double>(plan.arena_size_bytes);
    }
    if (plan.naive_allocation_bytes != 0) {
        plan.reuse_fraction =
            static_cast<double>(plan.naive_allocation_bytes - plan.arena_size_bytes) /
            static_cast<double>(plan.naive_allocation_bytes);
    }

    const Status verification = verify_memory_plan(plan);
    if (!verification.ok()) {
        return verification;
    }
    return plan;
}

Status verify_memory_plan(const MemoryPlan& plan) {
    if (!is_power_of_two(plan.alignment_bytes)) {
        return Status::error(StatusCode::failed_precondition,
                             "memory plan alignment is not a positive power of two");
    }
    for (const TensorLifetime& tensor : plan.tensors) {
        if (tensor.alignment_bytes != plan.alignment_bytes || tensor.alignment_bytes == 0) {
            return Status::error(StatusCode::failed_precondition,
                                 "value " + tensor.value_id +
                                     " has inconsistent alignment metadata");
        }
        if (tensor.kind == ValueKind::input &&
            (!tensor.external || !tensor.protected_buffer || tensor.arena_offset.has_value())) {
            return Status::error(StatusCode::failed_precondition,
                                 "input value " + tensor.value_id +
                                     " must remain protected external storage");
        }
        if ((tensor.kind == ValueKind::parameter || tensor.kind == ValueKind::constant) &&
            (!tensor.external || !tensor.immutable || !tensor.protected_buffer ||
             tensor.arena_offset.has_value())) {
            return Status::error(StatusCode::failed_precondition,
                                 "immutable value " + tensor.value_id +
                                     " must remain protected external storage");
        }
        if (tensor.kind == ValueKind::output &&
            (tensor.storage_class != TensorStorageClass::arena_output || !tensor.protected_buffer ||
             tensor.external || !tensor.arena_offset.has_value() ||
             tensor.final_use_index != plan.schedule.operations.size())) {
            return Status::error(StatusCode::failed_precondition,
                                 "declared output value " + tensor.value_id +
                                     " must own retained arena storage");
        }
        if (tensor.external) {
            if (tensor.arena_offset.has_value()) {
                return Status::error(StatusCode::failed_precondition,
                                     "external value " + tensor.value_id +
                                         " unexpectedly owns an arena offset");
            }
            continue;
        }
        if (!tensor.arena_offset.has_value()) {
            return Status::error(StatusCode::failed_precondition,
                                 "arena value " + tensor.value_id + " has no allocation offset");
        }
        if (tensor.arena_offset.value() % tensor.alignment_bytes != 0) {
            return Status::error(StatusCode::failed_precondition,
                                 "arena value " + tensor.value_id + " violates alignment");
        }
        const auto end = checked_add(tensor.arena_offset.value(), tensor.aligned_byte_size,
                                     "allocation range for " + tensor.value_id);
        if (!end.ok()) {
            return end.status();
        }
        if (end.value() > plan.arena_size_bytes) {
            return Status::error(StatusCode::failed_precondition,
                                 "arena value " + tensor.value_id + " exceeds the planned arena");
        }
    }
    for (std::size_t left = 0; left < plan.tensors.size(); ++left) {
        if (!plan.tensors[left].arena_offset.has_value()) {
            continue;
        }
        for (std::size_t right = left + 1; right < plan.tensors.size(); ++right) {
            if (!plan.tensors[right].arena_offset.has_value()) {
                continue;
            }
            const auto ranges_overlap =
                memory_ranges_overlap(plan.tensors[left], plan.tensors[right]);
            if (!ranges_overlap.ok()) {
                return ranges_overlap.status();
            }
            const bool output_protected =
                plan.tensors[left].storage_class == TensorStorageClass::arena_output ||
                plan.tensors[right].storage_class == TensorStorageClass::arena_output;
            if (ranges_overlap.value() &&
                (lifetimes_overlap(plan.tensors[left], plan.tensors[right]) || output_protected)) {
                return Status::error(StatusCode::failed_precondition,
                                     "live values " + plan.tensors[left].value_id + " and " +
                                         plan.tensors[right].value_id + " overlap in arena memory");
            }
        }
    }
    if (plan.peak_live_bytes > plan.arena_size_bytes ||
        plan.arena_size_bytes > plan.naive_allocation_bytes ||
        plan.peak_live_bytes > plan.naive_allocation_bytes) {
        return Status::error(StatusCode::failed_precondition,
                             "planned memory exceeds the naive aligned allocation total");
    }
    return Status::ok_status();
}

nlohmann::json memory_plan_json(const MemoryPlan& plan) {
    nlohmann::json tensors = nlohmann::json::array();
    for (const TensorLifetime& tensor : plan.tensors) {
        tensors.push_back(nlohmann::json{
            {"value_id", tensor.value_id},
            {"semantic_name", tensor.semantic_name},
            {"kind", to_string(tensor.kind)},
            {"definition_index", tensor.definition_index},
            {"first_use_index", optional_index_json(tensor.first_use_index)},
            {"final_use_index", tensor.final_use_index},
            {"lifetime_interval", {{"start", tensor.lifetime_start}, {"end", tensor.lifetime_end}}},
            {"alignment_bytes", tensor.alignment_bytes},
            {"byte_size", tensor.byte_size},
            {"aligned_byte_size", tensor.aligned_byte_size},
            {"storage_class", to_string(tensor.storage_class)},
            {"external", tensor.external},
            {"immutable", tensor.immutable},
            {"protected", tensor.protected_buffer},
            {"arena_offset", tensor.arena_offset.has_value()
                                 ? nlohmann::json(tensor.arena_offset.value())
                                 : nlohmann::json(nullptr)}});
    }
    return nlohmann::json{{"memory_plan_schema_version", "1.0"},
                          {"graph_hash", plan.graph_hash},
                          {"backend", plan.backend},
                          {"alignment_bytes", plan.alignment_bytes},
                          {"schedule_operation_count", plan.schedule.operations.size()},
                          {"summary",
                           {{"planned_bytes", plan.arena_size_bytes},
                            {"arena_size_bytes", plan.arena_size_bytes},
                            {"peak_live_bytes", plan.peak_live_bytes},
                            {"naive_allocation_bytes", plan.naive_allocation_bytes},
                            {"reuse_ratio", plan.reuse_ratio},
                            {"reuse_fraction", plan.reuse_fraction}}},
                          {"tensors", std::move(tensors)}};
}

std::string memory_timeline_csv(const MemoryPlan& plan) {
    std::ostringstream stream;
    stream << "value_id,semantic_name,kind,definition_index,first_use_index,final_use_index,"
              "lifetime_start,lifetime_end,alignment_bytes,byte_size,aligned_byte_size,"
              "storage_class,arena_offset,external,immutable,protected\n";
    for (const TensorLifetime& tensor : plan.tensors) {
        stream << csv_escape(tensor.value_id) << ',' << csv_escape(tensor.semantic_name) << ','
               << to_string(tensor.kind) << ',' << tensor.definition_index << ',';
        if (tensor.first_use_index.has_value()) {
            stream << tensor.first_use_index.value();
        }
        stream << ',' << tensor.final_use_index << ',' << tensor.lifetime_start << ','
               << tensor.lifetime_end << ',' << tensor.alignment_bytes << ',' << tensor.byte_size
               << ',' << tensor.aligned_byte_size << ',' << to_string(tensor.storage_class) << ',';
        if (tensor.arena_offset.has_value()) {
            stream << tensor.arena_offset.value();
        }
        stream << ',' << (tensor.external ? "true" : "false") << ','
               << (tensor.immutable ? "true" : "false") << ','
               << (tensor.protected_buffer ? "true" : "false") << '\n';
    }
    return stream.str();
}

std::string memory_timeline_svg(const MemoryPlan& plan) {
    constexpr double width = 1200.0;
    constexpr double height = 720.0;
    constexpr double left = 90.0;
    constexpr double right = 30.0;
    constexpr double top = 70.0;
    constexpr double bottom = 70.0;
    const double plot_width = width - left - right;
    const double plot_height = height - top - bottom;
    const double schedule_extent =
        std::max(1.0, static_cast<double>(plan.schedule.operations.size()) + 1.0);
    const double arena_extent = std::max(1.0, static_cast<double>(plan.arena_size_bytes));

    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2);
    stream << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1200\" height=\"720\" "
              "viewBox=\"0 0 1200 720\">\n";
    stream << "<rect width=\"1200\" height=\"720\" fill=\"#ffffff\"/>\n";
    stream << "<text x=\"90\" y=\"30\" font-family=\"monospace\" font-size=\"20\" "
              "font-weight=\"bold\">ForgeIR static tensor-memory timeline</text>\n";
    stream << "<text x=\"90\" y=\"52\" font-family=\"monospace\" font-size=\"12\">graph "
           << xml_escape(plan.graph_hash) << " | alignment " << plan.alignment_bytes
           << " bytes | arena " << plan.arena_size_bytes << " bytes</text>\n";
    stream << "<rect x=\"" << left << "\" y=\"" << top << "\" width=\"" << plot_width
           << "\" height=\"" << plot_height << "\" fill=\"#f8fafc\" stroke=\"#334155\"/>\n";

    for (std::size_t index = 0; index <= plan.schedule.operations.size(); ++index) {
        const double x = left + static_cast<double>(index) / schedule_extent * plot_width;
        stream << "<line x1=\"" << x << "\" y1=\"" << top << "\" x2=\"" << x << "\" y2=\""
               << (top + plot_height) << "\" stroke=\"#e2e8f0\" stroke-width=\"1\"/>\n";
        if (index % 2 == 0 || index == plan.schedule.operations.size()) {
            stream << "<text x=\"" << x << "\" y=\"" << (top + plot_height + 20.0)
                   << "\" text-anchor=\"middle\" font-family=\"monospace\" font-size=\"10\">"
                   << index << "</text>\n";
        }
    }

    for (const TensorLifetime& tensor : plan.tensors) {
        if (!tensor.arena_offset.has_value()) {
            continue;
        }
        const double x =
            left + static_cast<double>(tensor.lifetime_start) / schedule_extent * plot_width;
        const double rectangle_width = (static_cast<double>(tensor.lifetime_end) -
                                        static_cast<double>(tensor.lifetime_start) + 1.0) /
                                       schedule_extent * plot_width;
        const double rectangle_height = std::max(
            2.0, static_cast<double>(tensor.aligned_byte_size) / arena_extent * plot_height);
        const double allocation_end = static_cast<double>(tensor.arena_offset.value()) +
                                      static_cast<double>(tensor.aligned_byte_size);
        const double y = top + plot_height - allocation_end / arena_extent * plot_height;
        const char* fill =
            tensor.storage_class == TensorStorageClass::arena_output ? "#f97316" : "#2563eb";
        stream << "<rect x=\"" << x << "\" y=\"" << y << "\" width=\""
               << std::max(2.0, rectangle_width) << "\" height=\"" << rectangle_height
               << "\" fill=\"" << fill
               << "\" fill-opacity=\"0.72\" stroke=\"#0f172a\" stroke-width=\"0.5\">";
        stream << "<title>" << xml_escape(tensor.value_id + " " + tensor.semantic_name) << " | ["
               << tensor.lifetime_start << ',' << tensor.lifetime_end << "] | offset "
               << tensor.arena_offset.value() << " | " << tensor.aligned_byte_size
               << " aligned bytes</title></rect>\n";
    }
    stream << "<text x=\"" << (left + plot_width / 2.0) << "\" y=\"" << (height - 18.0)
           << "\" text-anchor=\"middle\" font-family=\"monospace\" font-size=\"12\">"
              "schedule index (inclusive lifetime intervals)</text>\n";
    stream << "<text x=\"20\" y=\"" << (top + plot_height / 2.0) << "\" transform=\"rotate(-90 20 "
           << (top + plot_height / 2.0)
           << ")\" text-anchor=\"middle\" font-family=\"monospace\" font-size=\"12\">"
              "arena byte offset</text>\n";
    stream << "<rect x=\"900\" y=\"32\" width=\"12\" height=\"12\" fill=\"#2563eb\"/>"
              "<text x=\"918\" y=\"42\" font-family=\"monospace\" font-size=\"11\">"
              "reusable intermediate</text>\n";
    stream << "<rect x=\"1050\" y=\"32\" width=\"12\" height=\"12\" fill=\"#f97316\"/>"
              "<text x=\"1068\" y=\"42\" font-family=\"monospace\" font-size=\"11\">"
              "protected output</text>\n";
    stream << "</svg>\n";
    return stream.str();
}

Result<MemoryPlanArtifactPaths>
write_memory_plan_artifacts(const MemoryPlan& plan, const std::filesystem::path& output_directory) {
    if (output_directory.empty()) {
        return Status::error(StatusCode::invalid_argument,
                             "memory-plan output directory must not be empty");
    }
    MemoryPlanArtifactPaths paths{
        output_directory, output_directory / "schedule.json", output_directory / "memory_plan.json",
        output_directory / "timeline.csv", output_directory / "timeline.svg"};
    const std::vector<std::pair<std::filesystem::path, std::string>> artifacts{
        {paths.schedule_json, execution_schedule_json(plan.schedule).dump(2) + "\n"},
        {paths.memory_plan_json, memory_plan_json(plan).dump(2) + "\n"},
        {paths.timeline_csv, memory_timeline_csv(plan)},
        {paths.timeline_svg, memory_timeline_svg(plan)}};
    for (const auto& [path, contents] : artifacts) {
        const Status status = write_text_file(path, contents);
        if (!status.ok()) {
            return status;
        }
    }
    return paths;
}

} // namespace forgeir
