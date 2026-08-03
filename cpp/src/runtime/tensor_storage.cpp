#include "forgeir/runtime/tensor_storage.hpp"

#include <algorithm>
#include <limits>
#include <new>

namespace forgeir {
namespace {

bool is_power_of_two(const std::uint64_t value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

} // namespace

void TensorStorage::AlignedDeleter::operator()(std::byte* address) const noexcept {
    if (address != nullptr) {
        ::operator delete(address, std::align_val_t(alignment));
    }
}

TensorStorage::TensorStorage(AlignedPointer data, const std::uint64_t size_bytes,
                             const std::uint64_t alignment_bytes)
    : data_(std::move(data)), size_bytes_(size_bytes), alignment_bytes_(alignment_bytes) {}

Result<TensorStorage> TensorStorage::allocate(const std::uint64_t size_bytes,
                                              const std::uint64_t alignment_bytes) {
    if (!is_power_of_two(alignment_bytes) || alignment_bytes < alignof(float)) {
        return Status::error(StatusCode::invalid_argument,
                             "tensor-storage alignment must be a power of two and at least " +
                                 std::to_string(alignof(float)) + " bytes");
    }
    if (alignment_bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        size_bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return Status::error(StatusCode::overflow,
                             "tensor-storage request exceeds host size_t capacity");
    }
    if (size_bytes == 0) {
        return TensorStorage(
            AlignedPointer(nullptr, AlignedDeleter{static_cast<std::size_t>(alignment_bytes)}), 0,
            alignment_bytes);
    }

    const auto alignment = static_cast<std::size_t>(alignment_bytes);
    const auto size = static_cast<std::size_t>(size_bytes);
    std::byte* address = nullptr;
    try {
        address = static_cast<std::byte*>(::operator new(size, std::align_val_t(alignment)));
    } catch (const std::bad_alloc&) {
        return Status::error(StatusCode::internal, "aligned tensor-storage allocation failed for " +
                                                       std::to_string(size_bytes) + " bytes");
    }
    std::fill_n(address, size, std::byte{0});
    return TensorStorage(AlignedPointer(address, AlignedDeleter{alignment}), size_bytes,
                         alignment_bytes);
}

std::byte* TensorStorage::data() noexcept { return data_.get(); }

const std::byte* TensorStorage::data() const noexcept { return data_.get(); }

std::uint64_t TensorStorage::size_bytes() const noexcept { return size_bytes_; }

std::uint64_t TensorStorage::alignment_bytes() const noexcept { return alignment_bytes_; }

ConstTensorView TensorView::as_const() const { return {data, shape, contiguous}; }

Result<std::uint64_t> checked_tensor_element_count(const std::vector<std::int64_t>& shape) {
    std::uint64_t count = 1;
    for (const std::int64_t dimension : shape) {
        if (dimension <= 0) {
            return Status::error(StatusCode::invalid_argument,
                                 "tensor dimensions must be strictly positive");
        }
        const auto unsigned_dimension = static_cast<std::uint64_t>(dimension);
        if (count > std::numeric_limits<std::uint64_t>::max() / unsigned_dimension) {
            return Status::error(StatusCode::overflow, "tensor element count overflows uint64_t");
        }
        count *= unsigned_dimension;
    }
    return count;
}

Result<std::uint64_t> checked_float32_byte_size(const std::vector<std::int64_t>& shape) {
    const auto count = checked_tensor_element_count(shape);
    if (!count.ok()) {
        return count.status();
    }
    if (count.value() > std::numeric_limits<std::uint64_t>::max() / sizeof(float)) {
        return Status::error(StatusCode::overflow, "float32 tensor byte size overflows uint64_t");
    }
    return count.value() * sizeof(float);
}

} // namespace forgeir
