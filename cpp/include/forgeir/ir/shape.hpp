#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "forgeir/core/status.hpp"

namespace forgeir {

class Shape {
  public:
    [[nodiscard]] static Result<Shape> create(std::vector<std::int64_t> dimensions,
                                              bool allow_zero_dimensions = false);

    [[nodiscard]] const std::vector<std::int64_t>& dimensions() const noexcept;
    [[nodiscard]] std::size_t rank() const noexcept;
    [[nodiscard]] Result<std::uint64_t> element_count() const;

  private:
    explicit Shape(std::vector<std::int64_t> dimensions);

    std::vector<std::int64_t> dimensions_;
};

} // namespace forgeir
