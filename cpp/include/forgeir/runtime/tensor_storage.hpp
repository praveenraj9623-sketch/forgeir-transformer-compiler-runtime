#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "forgeir/core/status.hpp"

namespace forgeir {

class TensorStorage {
  public:
    TensorStorage() = default;
    TensorStorage(const TensorStorage&) = delete;
    TensorStorage& operator=(const TensorStorage&) = delete;
    TensorStorage(TensorStorage&&) noexcept = default;
    TensorStorage& operator=(TensorStorage&&) noexcept = default;

    [[nodiscard]] static Result<TensorStorage> allocate(std::uint64_t size_bytes,
                                                        std::uint64_t alignment_bytes);

    [[nodiscard]] std::byte* data() noexcept;
    [[nodiscard]] const std::byte* data() const noexcept;
    [[nodiscard]] std::uint64_t size_bytes() const noexcept;
    [[nodiscard]] std::uint64_t alignment_bytes() const noexcept;

  private:
    struct AlignedDeleter {
        std::size_t alignment{alignof(std::max_align_t)};
        void operator()(std::byte* address) const noexcept;
    };

    using AlignedPointer = std::unique_ptr<std::byte, AlignedDeleter>;

    TensorStorage(AlignedPointer data, std::uint64_t size_bytes, std::uint64_t alignment_bytes);

    AlignedPointer data_{nullptr, AlignedDeleter{}};
    std::uint64_t size_bytes_{0};
    std::uint64_t alignment_bytes_{0};
};

struct ConstTensorView {
    const float* data{nullptr};
    std::vector<std::int64_t> shape;
    bool contiguous{true};
};

struct TensorView {
    float* data{nullptr};
    std::vector<std::int64_t> shape;
    bool contiguous{true};

    [[nodiscard]] ConstTensorView as_const() const;
};

[[nodiscard]] Result<std::uint64_t>
checked_tensor_element_count(const std::vector<std::int64_t>& shape);
[[nodiscard]] Result<std::uint64_t>
checked_float32_byte_size(const std::vector<std::int64_t>& shape);

} // namespace forgeir
