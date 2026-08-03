#pragma once

#include <cstdint>

#include "forgeir/core/status.hpp"
#include "forgeir/ir/data_type.hpp"
#include "forgeir/ir/shape.hpp"

namespace forgeir {

class TensorDescriptor {
  public:
    TensorDescriptor(DataType data_type, Shape shape);

    [[nodiscard]] DataType data_type() const noexcept;
    [[nodiscard]] const Shape& shape() const noexcept;
    [[nodiscard]] Result<std::uint64_t> element_count() const;
    [[nodiscard]] Result<std::uint64_t> byte_size() const;

  private:
    DataType data_type_;
    Shape shape_;
};

} // namespace forgeir
