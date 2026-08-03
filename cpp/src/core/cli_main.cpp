#include <iostream>
#include <string_view>

#include "forgeir/core/build_info.hpp"
#include "forgeir/core/status.hpp"
#include "forgeir/core/version.hpp"

namespace {

void print_usage() { std::cerr << "Usage: forgeir_cli --version | doctor\n"; }

} // namespace

int main(int argc, char* argv[]) {
    if (argc != 2) {
        print_usage();
        return static_cast<int>(forgeir::StatusCode::invalid_argument);
    }

    const std::string_view command{argv[1]};
    if (command == "--version") {
        std::cout << "ForgeIR " << forgeir::version() << '\n';
        return static_cast<int>(forgeir::StatusCode::success);
    }
    if (command == "doctor") {
        std::cout << forgeir::build_info_json(forgeir::current_build_info()) << '\n';
        return static_cast<int>(forgeir::StatusCode::success);
    }

    print_usage();
    return static_cast<int>(forgeir::StatusCode::invalid_argument);
}
