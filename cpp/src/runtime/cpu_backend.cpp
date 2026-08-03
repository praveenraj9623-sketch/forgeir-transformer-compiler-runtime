#include "forgeir/runtime/cpu_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "forgeir/core/build_config.hpp"

#if FORGEIR_CUDA_COMPILED
#include "forgeir/backends/cuda/cuda_backend.hpp"
#endif

namespace forgeir {
namespace {

using Dimensions = std::vector<std::int64_t>;

Status invalid_operation(const Operation& operation, const std::string& message) {
    return Status::error(StatusCode::invalid_argument,
                         "operation " + operation.id() + " (" +
                             std::string(to_string(operation.type())) + "): " + message);
}

Status validate_view(const Operation& operation, const ConstTensorView& view,
                     const std::string& role) {
    if (!view.contiguous) {
        return Status::error(StatusCode::unsupported,
                             "operation " + operation.id() + " rejects non-contiguous " + role);
    }
    const auto count = checked_tensor_element_count(view.shape);
    if (!count.ok()) {
        return invalid_operation(operation, role + ": " + count.status().message());
    }
    if (view.data == nullptr && count.value() != 0) {
        return invalid_operation(operation, role + " has a null data address");
    }
    return Status::ok_status();
}

Status validate_view(const Operation& operation, const TensorView& view, const std::string& role) {
    return validate_view(operation, view.as_const(), role);
}

Result<std::vector<std::uint64_t>> row_major_strides(const Dimensions& shape) {
    std::vector<std::uint64_t> strides(shape.size(), 1);
    std::uint64_t running = 1;
    for (std::size_t offset = 0; offset < shape.size(); ++offset) {
        const std::size_t index = shape.size() - 1 - offset;
        strides[index] = running;
        const auto dimension = static_cast<std::uint64_t>(shape[index]);
        if (running > std::numeric_limits<std::uint64_t>::max() / dimension) {
            return Status::error(StatusCode::overflow, "tensor stride calculation overflowed");
        }
        running *= dimension;
    }
    return strides;
}

float exact_gelu(const float value) {
    constexpr float kInverseSqrtTwo = 0.70710678118654752440F;
    return 0.5F * value * (1.0F + std::erf(value * kInverseSqrtTwo));
}

Status apply_fused_bias_gelu(const Operation& operation, const std::vector<ConstTensorView>& inputs,
                             TensorView output) {
    if (inputs.size() != 3) {
        return invalid_operation(operation, "FusedBiasGELU requires exactly three input tensors");
    }
    const auto& bias = inputs[2];
    if (bias.shape.size() != 1 || output.shape.empty() || bias.shape[0] != output.shape.back()) {
        return invalid_operation(operation,
                                 "FusedBiasGELU bias must match the final output dimension");
    }
    const auto activation = operation.attributes().find("fused_activation");
    const auto approximation = operation.attributes().find("fused_activation_approximate");
    if (activation == operation.attributes().end() || !activation->is_string() ||
        activation->get<std::string>() != "GELU" || approximation == operation.attributes().end() ||
        !approximation->is_string() || approximation->get<std::string>() != "none") {
        return Status::error(StatusCode::unsupported,
                             "operation " + operation.id() +
                                 " has an unsupported fused activation contract");
    }
    const auto count = checked_tensor_element_count(output.shape);
    if (!count.ok()) {
        return count.status();
    }
    const auto width = static_cast<std::uint64_t>(output.shape.back());
    for (std::uint64_t index = 0; index < count.value(); ++index) {
        output.data[index] = exact_gelu(output.data[index] + bias.data[index % width]);
    }
    return Status::ok_status();
}

Result<Dimensions> broadcast_shape(const Dimensions& left, const Dimensions& right) {
    const std::size_t rank = std::max(left.size(), right.size());
    Dimensions output(rank, 1);
    for (std::size_t offset = 0; offset < rank; ++offset) {
        const std::int64_t left_dimension =
            offset < left.size() ? left[left.size() - 1 - offset] : 1;
        const std::int64_t right_dimension =
            offset < right.size() ? right[right.size() - 1 - offset] : 1;
        if (left_dimension != right_dimension && left_dimension != 1 && right_dimension != 1) {
            return Status::error(StatusCode::failed_precondition,
                                 "elementwise operands cannot broadcast");
        }
        output[rank - 1 - offset] = std::max(left_dimension, right_dimension);
    }
    return output;
}

std::uint64_t broadcast_input_index(const std::uint64_t output_index,
                                    const Dimensions& output_shape,
                                    const std::vector<std::uint64_t>& output_strides,
                                    const Dimensions& input_shape,
                                    const std::vector<std::uint64_t>& input_strides) {
    const std::size_t rank_delta = output_shape.size() - input_shape.size();
    std::uint64_t input_index = 0;
    for (std::size_t output_axis = 0; output_axis < output_shape.size(); ++output_axis) {
        if (output_axis < rank_delta) {
            continue;
        }
        const std::size_t input_axis = output_axis - rank_delta;
        if (input_shape[input_axis] == 1) {
            continue;
        }
        const std::uint64_t coordinate = (output_index / output_strides[output_axis]) %
                                         static_cast<std::uint64_t>(output_shape[output_axis]);
        input_index += coordinate * input_strides[input_axis];
    }
    return input_index;
}

template <typename Function>
Status elementwise_binary(const Operation& operation, const std::vector<ConstTensorView>& inputs,
                          TensorView output, Function function) {
    if (inputs.size() != 2) {
        return invalid_operation(operation, "binary operation requires two inputs");
    }
    const auto expected = broadcast_shape(inputs[0].shape, inputs[1].shape);
    if (!expected.ok() || expected.value() != output.shape) {
        return invalid_operation(operation, "output shape does not match broadcast result");
    }
    const auto output_strides = row_major_strides(output.shape);
    const auto left_strides = row_major_strides(inputs[0].shape);
    const auto right_strides = row_major_strides(inputs[1].shape);
    const auto count = checked_tensor_element_count(output.shape);
    if (!output_strides.ok() || !left_strides.ok() || !right_strides.ok() || !count.ok()) {
        return Status::error(StatusCode::overflow,
                             "elementwise index calculation exceeded uint64_t capacity");
    }
    for (std::uint64_t index = 0; index < count.value(); ++index) {
        const std::uint64_t left_index = broadcast_input_index(
            index, output.shape, output_strides.value(), inputs[0].shape, left_strides.value());
        const std::uint64_t right_index = broadcast_input_index(
            index, output.shape, output_strides.value(), inputs[1].shape, right_strides.value());
        output.data[index] = function(inputs[0].data[left_index], inputs[1].data[right_index]);
    }
    return Status::ok_status();
}

Result<std::uint64_t> dimension_product(const Dimensions& dimensions) {
    return checked_tensor_element_count(dimensions);
}

std::uint64_t batch_base_offset(const std::uint64_t batch_index,
                                const Dimensions& output_batch_shape,
                                const std::vector<std::uint64_t>& output_batch_strides,
                                const Dimensions& input_batch_shape,
                                const std::vector<std::uint64_t>& input_strides) {
    const std::size_t rank_delta = output_batch_shape.size() - input_batch_shape.size();
    std::uint64_t result = 0;
    for (std::size_t output_axis = 0; output_axis < output_batch_shape.size(); ++output_axis) {
        if (output_axis < rank_delta) {
            continue;
        }
        const std::size_t input_axis = output_axis - rank_delta;
        if (input_batch_shape[input_axis] == 1) {
            continue;
        }
        const std::uint64_t coordinate =
            (batch_index / output_batch_strides[output_axis]) %
            static_cast<std::uint64_t>(output_batch_shape[output_axis]);
        result += coordinate * input_strides[input_axis];
    }
    return result;
}

Status reference_matmul(const Operation& operation, const ConstTensorView& left,
                        const ConstTensorView& right, TensorView output) {
    if (left.shape.size() < 2 || right.shape.size() < 2 || output.shape.size() < 2) {
        return invalid_operation(operation, "MatMul requires rank-two or batched rank-two tensors");
    }
    const std::int64_t rows = left.shape[left.shape.size() - 2];
    const std::int64_t contracting = left.shape.back();
    const std::int64_t right_contracting = right.shape[right.shape.size() - 2];
    const std::int64_t columns = right.shape.back();
    if (contracting != right_contracting || output.shape[output.shape.size() - 2] != rows ||
        output.shape.back() != columns) {
        return invalid_operation(operation, "MatMul shape contract is inconsistent");
    }
    const Dimensions left_batch(left.shape.begin(), left.shape.end() - 2);
    const Dimensions right_batch(right.shape.begin(), right.shape.end() - 2);
    const auto expected_batch = broadcast_shape(left_batch, right_batch);
    const Dimensions output_batch(output.shape.begin(), output.shape.end() - 2);
    if (!expected_batch.ok() || expected_batch.value() != output_batch) {
        return invalid_operation(operation, "MatMul batch dimensions cannot broadcast");
    }
    const auto output_batch_strides = row_major_strides(output_batch);
    const auto left_strides = row_major_strides(left.shape);
    const auto right_strides = row_major_strides(right.shape);
    const auto batch_count = dimension_product(output_batch);
    if (!output_batch_strides.ok() || !left_strides.ok() || !right_strides.ok() ||
        !batch_count.ok()) {
        return Status::error(StatusCode::overflow, "MatMul index calculation overflowed");
    }
    const auto rows_u = static_cast<std::uint64_t>(rows);
    const auto columns_u = static_cast<std::uint64_t>(columns);
    const auto contracting_u = static_cast<std::uint64_t>(contracting);
    for (std::uint64_t batch = 0; batch < batch_count.value(); ++batch) {
        const std::uint64_t left_base = batch_base_offset(
            batch, output_batch, output_batch_strides.value(), left_batch, left_strides.value());
        const std::uint64_t right_base = batch_base_offset(
            batch, output_batch, output_batch_strides.value(), right_batch, right_strides.value());
        const std::uint64_t output_base = batch * rows_u * columns_u;
        for (std::uint64_t row = 0; row < rows_u; ++row) {
            for (std::uint64_t column = 0; column < columns_u; ++column) {
                float sum = 0.0F;
                for (std::uint64_t inner = 0; inner < contracting_u; ++inner) {
                    sum += left.data[left_base + row * contracting_u + inner] *
                           right.data[right_base + inner * columns_u + column];
                }
                output.data[output_base + row * columns_u + column] = sum;
            }
        }
    }
    return Status::ok_status();
}

Status tiled_matmul(const Operation& operation, const ConstTensorView& left,
                    const ConstTensorView& right, TensorView output) {
    if (left.shape.size() < 2 || right.shape.size() < 2 || output.shape.size() < 2) {
        return invalid_operation(operation, "MatMul requires rank-two or batched rank-two tensors");
    }
    const std::int64_t rows = left.shape[left.shape.size() - 2];
    const std::int64_t contracting = left.shape.back();
    const std::int64_t right_contracting = right.shape[right.shape.size() - 2];
    const std::int64_t columns = right.shape.back();
    if (contracting != right_contracting || output.shape[output.shape.size() - 2] != rows ||
        output.shape.back() != columns) {
        return invalid_operation(operation, "MatMul shape contract is inconsistent");
    }
    const Dimensions left_batch(left.shape.begin(), left.shape.end() - 2);
    const Dimensions right_batch(right.shape.begin(), right.shape.end() - 2);
    const auto expected_batch = broadcast_shape(left_batch, right_batch);
    const Dimensions output_batch(output.shape.begin(), output.shape.end() - 2);
    if (!expected_batch.ok() || expected_batch.value() != output_batch) {
        return invalid_operation(operation, "MatMul batch dimensions cannot broadcast");
    }
    const auto output_batch_strides = row_major_strides(output_batch);
    const auto left_strides = row_major_strides(left.shape);
    const auto right_strides = row_major_strides(right.shape);
    const auto batch_count = dimension_product(output_batch);
    const auto output_count = checked_tensor_element_count(output.shape);
    if (!output_batch_strides.ok() || !left_strides.ok() || !right_strides.ok() ||
        !batch_count.ok() || !output_count.ok()) {
        return Status::error(StatusCode::overflow, "MatMul index calculation overflowed");
    }
    std::fill_n(output.data, static_cast<std::size_t>(output_count.value()), 0.0F);
    constexpr std::uint64_t kTile = 32;
    const auto rows_u = static_cast<std::uint64_t>(rows);
    const auto columns_u = static_cast<std::uint64_t>(columns);
    const auto contracting_u = static_cast<std::uint64_t>(contracting);
    for (std::uint64_t batch = 0; batch < batch_count.value(); ++batch) {
        const std::uint64_t left_base = batch_base_offset(
            batch, output_batch, output_batch_strides.value(), left_batch, left_strides.value());
        const std::uint64_t right_base = batch_base_offset(
            batch, output_batch, output_batch_strides.value(), right_batch, right_strides.value());
        const std::uint64_t output_base = batch * rows_u * columns_u;
        for (std::uint64_t row_tile = 0; row_tile < rows_u; row_tile += kTile) {
            const std::uint64_t row_end = std::min(row_tile + kTile, rows_u);
            for (std::uint64_t column_tile = 0; column_tile < columns_u; column_tile += kTile) {
                const std::uint64_t column_end = std::min(column_tile + kTile, columns_u);
                for (std::uint64_t inner_tile = 0; inner_tile < contracting_u;
                     inner_tile += kTile) {
                    const std::uint64_t inner_end = std::min(inner_tile + kTile, contracting_u);
                    for (std::uint64_t row = row_tile; row < row_end; ++row) {
                        for (std::uint64_t column = column_tile; column < column_end; ++column) {
                            float sum = output.data[output_base + row * columns_u + column];
                            for (std::uint64_t inner = inner_tile; inner < inner_end; ++inner) {
                                sum += left.data[left_base + row * contracting_u + inner] *
                                       right.data[right_base + inner * columns_u + column];
                            }
                            output.data[output_base + row * columns_u + column] = sum;
                        }
                    }
                }
            }
        }
    }
    return Status::ok_status();
}

Status linear(const Operation& operation, const std::vector<ConstTensorView>& inputs,
              TensorView output, const CpuMatMulImplementation implementation) {
    if (inputs.size() != 2 && inputs.size() != 3) {
        return invalid_operation(operation, "Linear requires two inputs or a fused bias input");
    }
    const auto& input = inputs[0];
    const auto& weight = inputs[1];
    if (input.shape.empty() || weight.shape.size() != 2 || output.shape.empty()) {
        return invalid_operation(operation, "Linear requires an input and a rank-two weight");
    }
    const auto in_features = static_cast<std::uint64_t>(input.shape.back());
    const auto out_features = static_cast<std::uint64_t>(weight.shape[0]);
    if (weight.shape[1] != input.shape.back() || output.shape.back() != weight.shape[0]) {
        return invalid_operation(operation, "Linear feature dimensions do not match");
    }
    const auto output_count = checked_tensor_element_count(output.shape);
    if (!output_count.ok() || output_count.value() % out_features != 0) {
        return invalid_operation(operation, "Linear output element count is invalid");
    }
    const std::uint64_t rows = output_count.value() / out_features;
    constexpr std::uint64_t kTile = 32;
    if (implementation == CpuMatMulImplementation::reference) {
        for (std::uint64_t row = 0; row < rows; ++row) {
            for (std::uint64_t output_feature = 0; output_feature < out_features;
                 ++output_feature) {
                float sum = 0.0F;
                for (std::uint64_t input_feature = 0; input_feature < in_features;
                     ++input_feature) {
                    sum += input.data[row * in_features + input_feature] *
                           weight.data[output_feature * in_features + input_feature];
                }
                output.data[row * out_features + output_feature] = sum;
            }
        }
    } else {
        std::fill_n(output.data, static_cast<std::size_t>(output_count.value()), 0.0F);
        for (std::uint64_t row_tile = 0; row_tile < rows; row_tile += kTile) {
            const std::uint64_t row_end = std::min(row_tile + kTile, rows);
            for (std::uint64_t output_tile = 0; output_tile < out_features; output_tile += kTile) {
                const std::uint64_t output_end = std::min(output_tile + kTile, out_features);
                for (std::uint64_t input_tile = 0; input_tile < in_features; input_tile += kTile) {
                    const std::uint64_t input_end = std::min(input_tile + kTile, in_features);
                    for (std::uint64_t row = row_tile; row < row_end; ++row) {
                        for (std::uint64_t output_feature = output_tile;
                             output_feature < output_end; ++output_feature) {
                            float sum = output.data[row * out_features + output_feature];
                            for (std::uint64_t input_feature = input_tile;
                                 input_feature < input_end; ++input_feature) {
                                sum += input.data[row * in_features + input_feature] *
                                       weight.data[output_feature * in_features + input_feature];
                            }
                            output.data[row * out_features + output_feature] = sum;
                        }
                    }
                }
            }
        }
    }
    if (inputs.size() == 3) {
        return apply_fused_bias_gelu(operation, inputs, output);
    }
    return Status::ok_status();
}

Status matmul(const Operation& operation, const std::vector<ConstTensorView>& inputs,
              TensorView output, const CpuMatMulImplementation implementation) {
    if (inputs.size() != 2 && inputs.size() != 3) {
        return invalid_operation(operation, "MatMul requires two inputs or a fused bias input");
    }
    const Status status = implementation == CpuMatMulImplementation::reference
                              ? reference_matmul(operation, inputs[0], inputs[1], output)
                              : tiled_matmul(operation, inputs[0], inputs[1], output);
    if (!status.ok()) {
        return status;
    }
    if (inputs.size() == 3) {
        return apply_fused_bias_gelu(operation, inputs, output);
    }
    return Status::ok_status();
}

Status rms_norm(const Operation& operation, const std::vector<ConstTensorView>& inputs,
                TensorView output) {
    if (inputs.size() != 2 || inputs[0].shape.empty() || inputs[1].shape.size() != 1 ||
        inputs[0].shape != output.shape || inputs[1].shape[0] != inputs[0].shape.back()) {
        return invalid_operation(operation, "RMSNorm shape contract is invalid");
    }
    const auto epsilon = operation.attributes().find("epsilon");
    if (epsilon == operation.attributes().end() || !epsilon->is_number()) {
        return invalid_operation(operation, "RMSNorm requires a numeric epsilon");
    }
    const float epsilon_value = epsilon->get<float>();
    if (!std::isfinite(epsilon_value) || epsilon_value <= 0.0F) {
        return invalid_operation(operation, "RMSNorm epsilon must be finite and positive");
    }
    const std::uint64_t width = static_cast<std::uint64_t>(inputs[0].shape.back());
    const auto count = checked_tensor_element_count(output.shape);
    if (!count.ok()) {
        return count.status();
    }
    const std::uint64_t rows = count.value() / width;
    for (std::uint64_t row = 0; row < rows; ++row) {
        float mean_square = 0.0F;
        for (std::uint64_t column = 0; column < width; ++column) {
            const float value = inputs[0].data[row * width + column];
            mean_square += value * value;
        }
        mean_square /= static_cast<float>(width);
        const float inverse_rms = 1.0F / std::sqrt(mean_square + epsilon_value);
        for (std::uint64_t column = 0; column < width; ++column) {
            output.data[row * width + column] =
                inputs[0].data[row * width + column] * inverse_rms * inputs[1].data[column];
        }
    }
    return Status::ok_status();
}

Status gelu(const Operation& operation, const std::vector<ConstTensorView>& inputs,
            TensorView output) {
    if (inputs.size() != 1 || inputs[0].shape != output.shape) {
        return invalid_operation(operation, "GELU requires equal input and output shapes");
    }
    const auto approximation = operation.attributes().find("approximate");
    if (approximation != operation.attributes().end() &&
        (!approximation->is_string() || approximation->get<std::string>() != "none")) {
        return Status::error(StatusCode::unsupported,
                             "operation " + operation.id() +
                                 " supports only exact GELU approximation='none'");
    }
    const auto count = checked_tensor_element_count(output.shape);
    if (!count.ok()) {
        return count.status();
    }
    for (std::uint64_t index = 0; index < count.value(); ++index) {
        output.data[index] = exact_gelu(inputs[0].data[index]);
    }
    return Status::ok_status();
}

Status softmax(const Operation& operation, const std::vector<ConstTensorView>& inputs,
               TensorView output) {
    if (inputs.size() != 1 || inputs[0].shape != output.shape || output.shape.empty()) {
        return invalid_operation(operation, "Softmax shape contract is invalid");
    }
    const auto axis_attribute = operation.attributes().find("axis");
    if (axis_attribute == operation.attributes().end() ||
        (!axis_attribute->is_number_integer() && !axis_attribute->is_number_unsigned())) {
        return invalid_operation(operation, "Softmax requires an integer axis");
    }
    std::int64_t axis = axis_attribute->get<std::int64_t>();
    const auto rank = static_cast<std::int64_t>(output.shape.size());
    if (axis < 0) {
        axis += rank;
    }
    if (axis < 0 || axis >= rank) {
        return invalid_operation(operation, "Softmax axis is outside the tensor rank");
    }
    const auto count = checked_tensor_element_count(output.shape);
    if (!count.ok()) {
        return count.status();
    }
    std::uint64_t inner = 1;
    for (std::size_t index = static_cast<std::size_t>(axis) + 1; index < output.shape.size();
         ++index) {
        inner *= static_cast<std::uint64_t>(output.shape[index]);
    }
    const std::uint64_t axis_size = static_cast<std::uint64_t>(output.shape[axis]);
    const std::uint64_t outer = count.value() / (axis_size * inner);
    for (std::uint64_t outer_index = 0; outer_index < outer; ++outer_index) {
        for (std::uint64_t inner_index = 0; inner_index < inner; ++inner_index) {
            const std::uint64_t base = outer_index * axis_size * inner + inner_index;
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::uint64_t axis_index = 0; axis_index < axis_size; ++axis_index) {
                maximum = std::max(maximum, inputs[0].data[base + axis_index * inner]);
            }
            float sum = 0.0F;
            for (std::uint64_t axis_index = 0; axis_index < axis_size; ++axis_index) {
                const float exponential =
                    std::exp(inputs[0].data[base + axis_index * inner] - maximum);
                output.data[base + axis_index * inner] = exponential;
                sum += exponential;
            }
            for (std::uint64_t axis_index = 0; axis_index < axis_size; ++axis_index) {
                output.data[base + axis_index * inner] /= sum;
            }
        }
    }
    return Status::ok_status();
}

Status reshape(const Operation& operation, const std::vector<ConstTensorView>& inputs,
               TensorView output) {
    if (inputs.size() != 1) {
        return invalid_operation(operation, "Reshape requires one input");
    }
    const auto input_count = checked_tensor_element_count(inputs[0].shape);
    const auto output_count = checked_tensor_element_count(output.shape);
    if (!input_count.ok() || !output_count.ok() || input_count.value() != output_count.value()) {
        return invalid_operation(operation, "Reshape must preserve element count");
    }
    std::copy_n(inputs[0].data, static_cast<std::size_t>(input_count.value()), output.data);
    return Status::ok_status();
}

Status transpose(const Operation& operation, const std::vector<ConstTensorView>& inputs,
                 TensorView output) {
    if (inputs.size() != 1 || inputs[0].shape.size() != output.shape.size()) {
        return invalid_operation(operation, "Transpose rank contract is invalid");
    }
    const auto permutation_attribute = operation.attributes().find("permutation");
    if (permutation_attribute == operation.attributes().end() ||
        !permutation_attribute->is_array() ||
        permutation_attribute->size() != output.shape.size()) {
        return invalid_operation(operation, "Transpose requires a rank-matched permutation");
    }
    std::vector<std::size_t> permutation;
    std::unordered_set<std::size_t> seen;
    for (const auto& item : *permutation_attribute) {
        if (!item.is_number_integer() && !item.is_number_unsigned()) {
            return invalid_operation(operation, "Transpose permutation must contain integers");
        }
        const auto axis = item.get<std::uint64_t>();
        if (axis >= output.shape.size() || !seen.insert(static_cast<std::size_t>(axis)).second) {
            return invalid_operation(operation, "Transpose permutation is invalid");
        }
        permutation.push_back(static_cast<std::size_t>(axis));
    }
    for (std::size_t axis = 0; axis < output.shape.size(); ++axis) {
        if (output.shape[axis] != inputs[0].shape[permutation[axis]]) {
            return invalid_operation(operation, "Transpose output shape is inconsistent");
        }
    }
    const auto input_strides = row_major_strides(inputs[0].shape);
    const auto output_strides = row_major_strides(output.shape);
    const auto count = checked_tensor_element_count(output.shape);
    if (!input_strides.ok() || !output_strides.ok() || !count.ok()) {
        return Status::error(StatusCode::overflow, "Transpose index calculation overflowed");
    }
    for (std::uint64_t output_index = 0; output_index < count.value(); ++output_index) {
        std::uint64_t input_index = 0;
        for (std::size_t output_axis = 0; output_axis < output.shape.size(); ++output_axis) {
            const std::uint64_t coordinate = (output_index / output_strides.value()[output_axis]) %
                                             static_cast<std::uint64_t>(output.shape[output_axis]);
            input_index += coordinate * input_strides.value()[permutation[output_axis]];
        }
        output.data[output_index] = inputs[0].data[input_index];
    }
    return Status::ok_status();
}

Status causal_mask(const Operation& operation, const std::vector<ConstTensorView>& inputs,
                   TensorView output) {
    if (inputs.size() != 1 || inputs[0].shape != output.shape || output.shape.size() < 2 ||
        output.shape[output.shape.size() - 2] != output.shape.back()) {
        return invalid_operation(operation, "CausalMask requires square final dimensions");
    }
    const auto masked_value = operation.attributes().find("masked_value");
    if (masked_value == operation.attributes().end() || !masked_value->is_string() ||
        masked_value->get<std::string>() != "-inf") {
        return Status::error(StatusCode::unsupported,
                             "operation " + operation.id() + " supports only masked_value='-inf'");
    }
    std::int64_t diagonal = 0;
    const auto diagonal_attribute = operation.attributes().find("diagonal");
    if (diagonal_attribute != operation.attributes().end()) {
        if (!diagonal_attribute->is_number_integer() && !diagonal_attribute->is_number_unsigned()) {
            return invalid_operation(operation, "CausalMask diagonal must be an integer");
        }
        diagonal = diagonal_attribute->get<std::int64_t>();
    }
    const std::uint64_t query = static_cast<std::uint64_t>(output.shape[output.shape.size() - 2]);
    const std::uint64_t key = static_cast<std::uint64_t>(output.shape.back());
    const auto count = checked_tensor_element_count(output.shape);
    if (!count.ok()) {
        return count.status();
    }
    const std::uint64_t matrices = count.value() / (query * key);
    for (std::uint64_t matrix = 0; matrix < matrices; ++matrix) {
        const std::uint64_t base = matrix * query * key;
        for (std::uint64_t row = 0; row < query; ++row) {
            for (std::uint64_t column = 0; column < key; ++column) {
                const auto relative_column = static_cast<std::int64_t>(column);
                const auto boundary = static_cast<std::int64_t>(row) + diagonal;
                output.data[base + row * key + column] =
                    relative_column > boundary ? -std::numeric_limits<float>::infinity()
                                               : inputs[0].data[base + row * key + column];
            }
        }
    }
    return Status::ok_status();
}

} // namespace

CpuBackend::CpuBackend(const CpuMatMulImplementation matmul_implementation)
    : matmul_implementation_(matmul_implementation) {}

std::string_view CpuBackend::name() const noexcept { return "cpu"; }

Status CpuBackend::execute(const Operation& operation, const std::vector<ConstTensorView>& inputs,
                           TensorView output) const {
    const Status output_status = validate_view(operation, output, "output");
    if (!output_status.ok()) {
        return output_status;
    }
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        const Status input_status =
            validate_view(operation, inputs[index], "input " + std::to_string(index));
        if (!input_status.ok()) {
            return input_status;
        }
    }
    switch (operation.type()) {
    case OperationType::mat_mul:
        return matmul(operation, inputs, output, matmul_implementation_);
    case OperationType::linear:
        return linear(operation, inputs, output, matmul_implementation_);
    case OperationType::add:
        return elementwise_binary(operation, inputs, output,
                                  [](const float left, const float right) { return left + right; });
    case OperationType::multiply:
        return elementwise_binary(operation, inputs, output,
                                  [](const float left, const float right) { return left * right; });
    case OperationType::divide:
        return elementwise_binary(operation, inputs, output,
                                  [](const float left, const float right) { return left / right; });
    case OperationType::rms_norm:
        return rms_norm(operation, inputs, output);
    case OperationType::gelu:
        return gelu(operation, inputs, output);
    case OperationType::softmax:
        return softmax(operation, inputs, output);
    case OperationType::reshape:
        return reshape(operation, inputs, output);
    case OperationType::transpose:
        return transpose(operation, inputs, output);
    case OperationType::causal_mask:
        return causal_mask(operation, inputs, output);
    case OperationType::input:
    case OperationType::parameter:
    case OperationType::constant:
        return invalid_operation(operation, "declaration operations are handled by RuntimeSession");
    }
    return Status::error(StatusCode::unsupported, "unsupported CPU operation");
}

Result<std::unique_ptr<Backend>>
BackendRegistry::create(const std::string_view backend_name,
                        const CpuMatMulImplementation matmul_implementation) {
    if (backend_name == "cpu") {
        std::unique_ptr<Backend> backend = std::make_unique<CpuBackend>(matmul_implementation);
        return backend;
    }
#if FORGEIR_CUDA_COMPILED
    if (backend_name == "cuda") {
        std::unique_ptr<Backend> backend = std::make_unique<CudaBackend>();
        return backend;
    }
#endif
    return Status::error(StatusCode::unsupported, "backend registry has no implementation named '" +
                                                      std::string(backend_name) + "'");
}

} // namespace forgeir
