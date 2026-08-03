#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "forgeir/core/status.hpp"
#include "forgeir/ir/graph.hpp"
#include "forgeir/runtime/execution_schedule.hpp"

namespace forgeir {

inline constexpr std::uint64_t kDefaultCpuTensorAlignment = 64;

enum class TensorStorageClass { external_input, external_immutable, arena_reusable, arena_output };

[[nodiscard]] std::string_view to_string(TensorStorageClass storage_class) noexcept;

struct MemoryPlannerOptions {
    std::string backend{"cpu"};
    std::uint64_t alignment_bytes{kDefaultCpuTensorAlignment};
};

struct TensorLifetime {
    std::string value_id;
    std::string semantic_name;
    ValueKind kind{ValueKind::intermediate};
    std::uint64_t definition_index{0};
    std::optional<std::uint64_t> first_use_index;
    std::uint64_t final_use_index{0};
    std::uint64_t lifetime_start{0};
    std::uint64_t lifetime_end{0};
    std::uint64_t alignment_bytes{0};
    std::uint64_t byte_size{0};
    std::uint64_t aligned_byte_size{0};
    TensorStorageClass storage_class{TensorStorageClass::arena_reusable};
    bool external{false};
    bool immutable{false};
    bool protected_buffer{false};
    std::optional<std::uint64_t> arena_offset;
};

struct MemoryPlan {
    std::string graph_hash;
    std::string backend;
    std::uint64_t alignment_bytes{0};
    ExecutionSchedule schedule;
    std::vector<TensorLifetime> tensors;
    std::uint64_t arena_size_bytes{0};
    std::uint64_t peak_live_bytes{0};
    std::uint64_t naive_allocation_bytes{0};
    double reuse_ratio{1.0};
    double reuse_fraction{0.0};
};

struct MemoryPlanArtifactPaths {
    std::filesystem::path output_directory;
    std::filesystem::path schedule_json;
    std::filesystem::path memory_plan_json;
    std::filesystem::path timeline_csv;
    std::filesystem::path timeline_svg;
};

[[nodiscard]] Result<MemoryPlan>
plan_memory(const Graph& graph, const MemoryPlannerOptions& options = MemoryPlannerOptions{});
[[nodiscard]] Status verify_memory_plan(const MemoryPlan& plan);
[[nodiscard]] nlohmann::json memory_plan_json(const MemoryPlan& plan);
[[nodiscard]] std::string memory_timeline_csv(const MemoryPlan& plan);
[[nodiscard]] std::string memory_timeline_svg(const MemoryPlan& plan);
[[nodiscard]] Result<MemoryPlanArtifactPaths>
write_memory_plan_artifacts(const MemoryPlan& plan, const std::filesystem::path& output_directory);

} // namespace forgeir
