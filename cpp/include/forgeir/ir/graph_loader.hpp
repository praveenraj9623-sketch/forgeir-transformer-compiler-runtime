#pragma once

#include <filesystem>
#include <string_view>

#include "forgeir/core/status.hpp"
#include "forgeir/ir/graph.hpp"

namespace forgeir {

class GraphLoader {
  public:
    [[nodiscard]] static Result<Graph> load_from_file(const std::filesystem::path& path);
    [[nodiscard]] static Result<Graph> load_from_json(std::string_view json_text);
};

} // namespace forgeir
