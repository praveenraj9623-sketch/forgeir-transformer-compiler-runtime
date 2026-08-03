#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "forgeir/core/status.hpp"
#include "forgeir/ir/operation.hpp"
#include "forgeir/runtime/tensor_storage.hpp"

namespace forgeir {

enum class CpuMatMulImplementation { reference, tiled };

class Backend {
  public:
    virtual ~Backend() = default;
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual Status execute(const Operation& operation,
                                         const std::vector<ConstTensorView>& inputs,
                                         TensorView output) const = 0;
};

class CpuBackend final : public Backend {
  public:
    explicit CpuBackend(
        CpuMatMulImplementation matmul_implementation = CpuMatMulImplementation::tiled);

    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] Status execute(const Operation& operation,
                                 const std::vector<ConstTensorView>& inputs,
                                 TensorView output) const override;

  private:
    CpuMatMulImplementation matmul_implementation_;
};

class BackendRegistry {
  public:
    [[nodiscard]] static Result<std::unique_ptr<Backend>>
    create(std::string_view backend_name,
           CpuMatMulImplementation matmul_implementation = CpuMatMulImplementation::tiled);
};

} // namespace forgeir
