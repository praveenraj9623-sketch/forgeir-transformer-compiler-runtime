#include "forgeir/ir/data_type.hpp"

#include <string>

namespace forgeir {

Result<DataType> parse_data_type(const std::string_view name) {
    if (name == "bool") {
        return DataType::boolean;
    }
    if (name == "float32") {
        return DataType::float32;
    }
    if (name == "int64") {
        return DataType::int64;
    }
    return Status::error(StatusCode::unsupported, "unsupported dtype: " + std::string(name));
}

std::string_view to_string(const DataType data_type) noexcept {
    switch (data_type) {
    case DataType::boolean:
        return "bool";
    case DataType::float32:
        return "float32";
    case DataType::int64:
        return "int64";
    }
    return "invalid";
}

std::uint64_t data_type_byte_width(const DataType data_type) noexcept {
    switch (data_type) {
    case DataType::boolean:
        return 1;
    case DataType::float32:
        return 4;
    case DataType::int64:
        return 8;
    }
    return 0;
}

} // namespace forgeir
