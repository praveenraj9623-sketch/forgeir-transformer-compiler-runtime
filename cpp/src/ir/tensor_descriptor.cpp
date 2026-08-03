#include "forgeir/ir/tensor_descriptor.hpp"

#include <limits>
#include <utility>

namespace forgeir {

TensorDescriptor::TensorDescriptor(const DataType data_type, Shape shape)
    : data_type_(data_type), shape_(std::move(shape)) {}

DataType TensorDescriptor::data_type() const noexcept { return data_type_; }

const Shape& TensorDescriptor::shape() const noexcept { return shape_; }

Result<std::uint64_t> TensorDescriptor::element_count() const { return shape_.element_count(); }

Result<std::uint64_t> TensorDescriptor::byte_size() const {
    const auto count = element_count();
    if (!count.ok()) {
        return count.status();
    }
    const std::uint64_t width = data_type_byte_width(data_type_);
    if (width == 0 || count.value() > std::numeric_limits<std::uint64_t>::max() / width) {
        return Status::error(StatusCode::overflow, "tensor byte size exceeds uint64_t capacity");
    }
    return count.value() * width;
}

} // namespace forgeir
