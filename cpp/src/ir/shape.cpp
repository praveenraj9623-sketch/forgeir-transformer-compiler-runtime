#include "forgeir/ir/shape.hpp"

#include <limits>
#include <string>
#include <utility>

namespace forgeir {

Shape::Shape(std::vector<std::int64_t> dimensions) : dimensions_(std::move(dimensions)) {}

Result<Shape> Shape::create(std::vector<std::int64_t> dimensions,
                            const bool allow_zero_dimensions) {
    for (const std::int64_t dimension : dimensions) {
        if (dimension < 0) {
            return Status::error(StatusCode::invalid_argument,
                                 "shape dimensions must not be negative");
        }
        if (dimension == 0 && !allow_zero_dimensions) {
            return Status::error(StatusCode::invalid_argument,
                                 "zero shape dimensions are not allowed by graph schema 1.0");
        }
    }
    return Shape(std::move(dimensions));
}

const std::vector<std::int64_t>& Shape::dimensions() const noexcept { return dimensions_; }

std::size_t Shape::rank() const noexcept { return dimensions_.size(); }

Result<std::uint64_t> Shape::element_count() const {
    std::uint64_t count = 1;
    for (const std::int64_t dimension : dimensions_) {
        const auto unsigned_dimension = static_cast<std::uint64_t>(dimension);
        if (unsigned_dimension != 0 &&
            count > std::numeric_limits<std::uint64_t>::max() / unsigned_dimension) {
            return Status::error(StatusCode::overflow,
                                 "shape element count exceeds uint64_t capacity");
        }
        count *= unsigned_dimension;
    }
    return count;
}

} // namespace forgeir
