#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace forgeir {

enum class StatusCode : int {
    success = 0,
    invalid_argument = 2,
    unsupported = 3,
    not_found = 4,
    parse_error = 5,
    failed_precondition = 6,
    overflow = 7,
    internal = 8,
};

class Status {
  public:
    Status() = default;
    Status(StatusCode code, std::string message) : code_(code), message_(std::move(message)) {
        if (code_ == StatusCode::success && !message_.empty()) {
            throw std::invalid_argument("a successful status cannot contain an error message");
        }
    }

    [[nodiscard]] static Status ok_status() { return {}; }
    [[nodiscard]] static Status error(StatusCode code, std::string message) {
        if (code == StatusCode::success) {
            throw std::invalid_argument("an error status must use a non-success code");
        }
        return {code, std::move(message)};
    }

    [[nodiscard]] bool ok() const noexcept { return code_ == StatusCode::success; }
    [[nodiscard]] StatusCode code() const noexcept { return code_; }
    [[nodiscard]] const std::string& message() const noexcept { return message_; }

  private:
    StatusCode code_{StatusCode::success};
    std::string message_;
};

template <typename T> class Result {
  public:
    Result(T value) : status_(Status::ok_status()), value_(std::move(value)) {}
    Result(Status status) : status_(std::move(status)) {
        if (status_.ok()) {
            throw std::invalid_argument("a result without a value must contain an error status");
        }
    }

    [[nodiscard]] bool ok() const noexcept { return status_.ok(); }
    [[nodiscard]] const Status& status() const noexcept { return status_; }

    [[nodiscard]] const T& value() const& {
        if (!value_.has_value()) {
            throw std::logic_error("cannot access the value of a failed result");
        }
        return *value_;
    }

    [[nodiscard]] T& value() & {
        if (!value_.has_value()) {
            throw std::logic_error("cannot access the value of a failed result");
        }
        return *value_;
    }

    [[nodiscard]] T take_value() {
        if (!value_.has_value()) {
            throw std::logic_error("cannot take the value of a failed result");
        }
        return std::move(*value_);
    }

  private:
    Status status_;
    std::optional<T> value_;
};

} // namespace forgeir
