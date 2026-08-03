#include "forgeir/runtime/runtime_session.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <limits>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "forgeir/core/sha256.hpp"
#include "forgeir/ir/graph_loader.hpp"
#include "forgeir/passes/verification.hpp"

namespace forgeir {
namespace {

struct RuntimeValue {
    const float* data{nullptr};
    std::vector<std::int64_t> shape;
    bool contiguous{true};
};

const Value* find_value(const Graph& graph, const std::string& value_id) {
    for (const Value& value : graph.values()) {
        if (value.id() == value_id) {
            return &value;
        }
    }
    return nullptr;
}

const Operation* find_operation(const Graph& graph, const std::string& operation_id) {
    for (const Operation& operation : graph.operations()) {
        if (operation.id() == operation_id) {
            return &operation;
        }
    }
    return nullptr;
}

const TensorLifetime* find_lifetime(const MemoryPlan& plan, const std::string& value_id) {
    for (const TensorLifetime& lifetime : plan.tensors) {
        if (lifetime.value_id == value_id) {
            return &lifetime;
        }
    }
    return nullptr;
}

Status validate_external_tensor(const Value& value, const ExternalTensor& tensor,
                                const std::string& role) {
    if (value.descriptor().data_type() != DataType::float32) {
        return Status::error(StatusCode::unsupported,
                             role + " " + value.id() + " does not use float32");
    }
    if (!tensor.contiguous) {
        return Status::error(StatusCode::unsupported,
                             role + " " + value.id() + " must be C-contiguous");
    }
    if (tensor.shape != value.descriptor().shape().dimensions()) {
        return Status::error(StatusCode::invalid_argument,
                             role + " " + value.id() + " shape does not match the graph");
    }
    const auto expected_bytes = value.descriptor().byte_size();
    if (!expected_bytes.ok()) {
        return expected_bytes.status();
    }
    if (tensor.byte_size != expected_bytes.value()) {
        return Status::error(StatusCode::invalid_argument,
                             role + " " + value.id() + " byte size does not match the graph");
    }
    if (tensor.data == nullptr && tensor.byte_size != 0) {
        return Status::error(StatusCode::invalid_argument,
                             role + " " + value.id() + " has a null data address");
    }
    return Status::ok_status();
}

Status validate_parameter_hash(const Operation& operation, const ExternalTensor& tensor) {
    const auto expected = operation.attributes().find("content_sha256");
    if (expected == operation.attributes().end() || !expected->is_string() ||
        !is_lowercase_sha256(expected->get<std::string>())) {
        return Status::error(StatusCode::failed_precondition, "parameter operation " +
                                                                  operation.id() +
                                                                  " has no valid content SHA-256");
    }
    if (tensor.byte_size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return Status::error(StatusCode::overflow, "parameter operation " + operation.id() +
                                                       " byte size exceeds host size_t capacity");
    }
    const auto bytes = std::string_view(reinterpret_cast<const char*>(tensor.data),
                                        static_cast<std::size_t>(tensor.byte_size));
    if (sha256(bytes) != expected->get<std::string>()) {
        return Status::error(StatusCode::failed_precondition,
                             "parameter operation " + operation.id() +
                                 " content SHA-256 does not match the graph");
    }
    return Status::ok_status();
}

Result<HostTensor> copy_tensor(const RuntimeValue& value) {
    const auto count = checked_tensor_element_count(value.shape);
    if (!count.ok()) {
        return count.status();
    }
    if (count.value() > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return Status::error(StatusCode::overflow, "tensor copy exceeds host size_t capacity");
    }
    HostTensor result;
    result.shape = value.shape;
    result.values.assign(value.data, value.data + static_cast<std::size_t>(count.value()));
    return result;
}

std::string trace_kernel(const Operation& operation) {
    const auto fused = operation.attributes().find("fused_activation");
    if (fused != operation.attributes().end() && fused->is_string() &&
        fused->get<std::string>() == "GELU") {
        return "cpu.FusedBiasGELU";
    }
    if (operation.type() == OperationType::mat_mul || operation.type() == OperationType::linear) {
        return "cpu.tiled_reference_order";
    }
    return "cpu." + std::string(to_string(operation.type()));
}

} // namespace

RuntimeSession::RuntimeSession(Graph graph, MemoryPlan memory_plan, TensorStorage arena,
                               std::unique_ptr<Backend> backend)
    : graph_(std::move(graph)), memory_plan_(std::move(memory_plan)), arena_(std::move(arena)),
      backend_(std::move(backend)) {}

Result<std::shared_ptr<RuntimeSession>> RuntimeSession::load(const std::string& graph_path,
                                                             const std::string_view backend_name) {
    auto graph_result = GraphLoader::load_from_file(graph_path);
    if (!graph_result.ok()) {
        return graph_result.status();
    }
    Graph graph = graph_result.take_value();
    const VerificationReport verification = verify_graph(graph);
    if (!verification.success()) {
        return Status::error(StatusCode::failed_precondition,
                             "graph failed semantic verification before CPU execution: " +
                                 verification.pipeline.status.message());
    }
    for (const Value& value : graph.values()) {
        if (value.descriptor().data_type() != DataType::float32) {
            return Status::error(StatusCode::unsupported,
                                 "CPU runtime supports only float32; value " + value.id() +
                                     " declares " +
                                     std::string(to_string(value.descriptor().data_type())));
        }
    }
    MemoryPlannerOptions planner_options;
    planner_options.backend = std::string(backend_name);
    planner_options.alignment_bytes = kDefaultCpuTensorAlignment;
    auto plan_result = plan_memory(graph, planner_options);
    if (!plan_result.ok()) {
        return plan_result.status();
    }
    MemoryPlan plan = plan_result.take_value();
    auto arena_result = TensorStorage::allocate(plan.arena_size_bytes, plan.alignment_bytes);
    if (!arena_result.ok()) {
        return arena_result.status();
    }
    auto backend_result = BackendRegistry::create(backend_name, CpuMatMulImplementation::tiled);
    if (!backend_result.ok()) {
        return backend_result.status();
    }
    auto session = std::shared_ptr<RuntimeSession>(new RuntimeSession(
        std::move(graph), std::move(plan), arena_result.take_value(), backend_result.take_value()));
    return session;
}

Status RuntimeSession::execute(const std::unordered_map<std::string, ExternalTensor>& inputs,
                               const std::unordered_map<std::string, ExternalTensor>& parameters,
                               const std::vector<std::string>& capture_value_ids) {
    trace_.clear();
    captured_values_.clear();
    executed_ = false;

    std::unordered_set<std::string> captures(capture_value_ids.begin(), capture_value_ids.end());
    captures.insert(graph_.output_ids().begin(), graph_.output_ids().end());
    for (const std::string& capture_id : captures) {
        if (find_value(graph_, capture_id) == nullptr) {
            return Status::error(StatusCode::not_found,
                                 "requested capture value is absent from graph: " + capture_id);
        }
    }
    for (const auto& [input_id, tensor] : inputs) {
        static_cast<void>(tensor);
        if (std::find(graph_.input_ids().begin(), graph_.input_ids().end(), input_id) ==
            graph_.input_ids().end()) {
            return Status::error(StatusCode::invalid_argument,
                                 "unexpected graph input tensor: " + input_id);
        }
    }

    std::unordered_map<std::string, RuntimeValue> values;
    values.reserve(graph_.values().size());
    std::unordered_map<std::string, TensorStorage> constants;
    constants.reserve(graph_.values().size());

    for (const ScheduledOperation& scheduled : memory_plan_.schedule.operations) {
        const Operation* operation = find_operation(graph_, scheduled.operation_id);
        if (operation == nullptr || operation->output_ids().size() != 1) {
            return Status::error(StatusCode::failed_precondition,
                                 "schedule operation " + scheduled.operation_id +
                                     " is absent or does not have one output");
        }
        const std::string& output_id = operation->output_ids()[0];
        const Value* output_metadata = find_value(graph_, output_id);
        if (output_metadata == nullptr) {
            return Status::error(StatusCode::failed_precondition,
                                 "operation output is absent from graph metadata: " + output_id);
        }
        const TensorLifetime* output_lifetime = find_lifetime(memory_plan_, output_id);
        if (output_lifetime == nullptr) {
            return Status::error(StatusCode::failed_precondition,
                                 "operation output has no memory-plan lifetime: " + output_id);
        }

        const auto start = std::chrono::steady_clock::now();
        RuntimeValue output_value;
        std::string kernel;
        if (operation->type() == OperationType::input) {
            const auto input = inputs.find(output_id);
            if (input == inputs.end()) {
                return Status::error(StatusCode::not_found,
                                     "missing external graph input: " + output_id);
            }
            const Status validation =
                validate_external_tensor(*output_metadata, input->second, "input");
            if (!validation.ok()) {
                return validation;
            }
            output_value = {input->second.data, input->second.shape, input->second.contiguous};
            kernel = "external.input";
        } else if (operation->type() == OperationType::parameter) {
            const auto archive_key = operation->attributes().find("archive_key");
            if (archive_key == operation->attributes().end() || !archive_key->is_string()) {
                return Status::error(StatusCode::failed_precondition, "parameter operation " +
                                                                          operation->id() +
                                                                          " has no archive_key");
            }
            const std::string key = archive_key->get<std::string>();
            const auto parameter = parameters.find(key);
            if (parameter == parameters.end()) {
                return Status::error(StatusCode::not_found,
                                     "missing external parameter archive key: " + key);
            }
            const Status validation =
                validate_external_tensor(*output_metadata, parameter->second, "parameter");
            if (!validation.ok()) {
                return validation;
            }
            const Status hash_status = validate_parameter_hash(*operation, parameter->second);
            if (!hash_status.ok()) {
                return hash_status;
            }
            output_value = {parameter->second.data, parameter->second.shape,
                            parameter->second.contiguous};
            kernel = "external.immutable_parameter";
        } else if (operation->type() == OperationType::constant) {
            const auto literal = operation->attributes().find("value");
            if (literal == operation->attributes().end() || !literal->is_number_float() ||
                !output_metadata->descriptor().shape().dimensions().empty()) {
                return Status::error(StatusCode::unsupported,
                                     "CPU runtime supports float32 scalar Constant operations");
            }
            auto storage = TensorStorage::allocate(sizeof(float), memory_plan_.alignment_bytes);
            if (!storage.ok()) {
                return storage.status();
            }
            auto* constant_data = reinterpret_cast<float*>(storage.value().data());
            constant_data[0] = literal->get<float>();
            constants.emplace(output_id, storage.take_value());
            output_value = {constant_data, {}, true};
            kernel = "external.immutable_constant";
        } else {
            if (!output_lifetime->arena_offset.has_value()) {
                return Status::error(StatusCode::failed_precondition,
                                     "computed output has no arena offset: " + output_id);
            }
            if (output_lifetime->arena_offset.value() > arena_.size_bytes() ||
                output_lifetime->byte_size >
                    arena_.size_bytes() - output_lifetime->arena_offset.value()) {
                return Status::error(StatusCode::overflow,
                                     "computed output exceeds the planned arena: " + output_id);
            }
            auto* output_data = reinterpret_cast<float*>(
                arena_.data() + static_cast<std::size_t>(output_lifetime->arena_offset.value()));
            std::vector<ConstTensorView> operands;
            operands.reserve(operation->input_ids().size());
            for (const std::string& input_id : operation->input_ids()) {
                const auto input = values.find(input_id);
                if (input == values.end()) {
                    return Status::error(StatusCode::failed_precondition,
                                         "operation " + operation->id() +
                                             " cannot resolve runtime input " + input_id);
                }
                operands.push_back(
                    {input->second.data, input->second.shape, input->second.contiguous});
            }
            output_value = {output_data, output_metadata->descriptor().shape().dimensions(), true};
            const Status execution_status = backend_->execute(
                *operation, operands,
                {output_data, output_metadata->descriptor().shape().dimensions(), true});
            if (!execution_status.ok()) {
                return execution_status;
            }
            kernel = trace_kernel(*operation);
        }
        values[output_id] = output_value;
        const auto end = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double, std::micro>(end - start).count();
        trace_.push_back(ExecutionTraceRecord{
            operation->id(), std::string(to_string(operation->type())), std::move(kernel),
            output_metadata->descriptor().shape().dimensions(), elapsed,
            output_lifetime->arena_offset.has_value(), output_lifetime->arena_offset.value_or(0)});

        if (captures.find(output_id) != captures.end()) {
            auto copied = copy_tensor(output_value);
            if (!copied.ok()) {
                return copied.status();
            }
            captured_values_[output_id] = copied.take_value();
        }
    }
    executed_ = true;
    return Status::ok_status();
}

Result<std::unordered_map<std::string, HostTensor>> RuntimeSession::get_outputs() const {
    if (!executed_) {
        return Status::error(StatusCode::failed_precondition,
                             "runtime outputs are unavailable before successful execution");
    }
    std::unordered_map<std::string, HostTensor> outputs;
    for (const std::string& output_id : graph_.output_ids()) {
        const auto output = captured_values_.find(output_id);
        if (output == captured_values_.end()) {
            return Status::error(StatusCode::internal,
                                 "successful execution did not retain output " + output_id);
        }
        outputs.emplace(output_id, output->second);
    }
    return outputs;
}

Result<HostTensor> RuntimeSession::get_value(const std::string& value_id) const {
    if (!executed_) {
        return Status::error(StatusCode::failed_precondition,
                             "runtime values are unavailable before successful execution");
    }
    const auto value = captured_values_.find(value_id);
    if (value == captured_values_.end()) {
        return Status::error(StatusCode::not_found,
                             "value was not retained; request it through capture_values: " +
                                 value_id);
    }
    return value->second;
}

const std::vector<ExecutionTraceRecord>& RuntimeSession::trace() const noexcept { return trace_; }

const Graph& RuntimeSession::graph() const noexcept { return graph_; }

const MemoryPlan& RuntimeSession::memory_plan() const noexcept { return memory_plan_; }

} // namespace forgeir
