cmake_minimum_required(VERSION 3.28)

project(ForgeIR VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

option(FORGEIR_ENABLE_CUDA "Compile the optional CUDA integration" OFF)
option(FORGEIR_ENABLE_HIP "Compile the optional HIP integration" OFF)
option(FORGEIR_ENABLE_MLIR "Compile the optional MLIR integration" OFF)

foreach(feature IN ITEMS CUDA HIP MLIR)
    if(FORGEIR_ENABLE_${feature})
        message(FATAL_ERROR
            "FORGEIR_ENABLE_${feature}=ON is unsupported in Milestone 1; "
            "the ${feature} integration has not been implemented.")
    endif()
endforeach()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ForgeIRDependencies)
include(ForgeIRWarnings)

set(Python_FIND_VIRTUALENV ONLY)
find_package(Python 3.11 EXACT REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    RESULT_VARIABLE FORGEIR_PYBIND11_RESULT
    OUTPUT_VARIABLE FORGEIR_PYBIND11_CMAKE_DIR
    ERROR_VARIABLE FORGEIR_PYBIND11_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT FORGEIR_PYBIND11_RESULT EQUAL 0 OR FORGEIR_PYBIND11_CMAKE_DIR STREQUAL "")
    message(FATAL_ERROR
        "Unable to discover pybind11 from ${Python_EXECUTABLE} with "
        "'python -m pybind11 --cmakedir': ${FORGEIR_PYBIND11_ERROR}")
endif()
set(pybind11_DIR "${FORGEIR_PYBIND11_CMAKE_DIR}" CACHE PATH
    "pybind11 CMake package discovered from the active Python environment" FORCE)
find_package(pybind11 CONFIG REQUIRED PATHS "${pybind11_DIR}" NO_DEFAULT_PATH)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/build_config.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/forgeir/core/build_config.hpp"
    @ONLY
)

add_library(forgeir_core STATIC
    cpp/src/core/build_info.cpp
    cpp/src/core/version.cpp
)
add_library(ForgeIR::core ALIAS forgeir_core)
target_include_directories(forgeir_core
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/include>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>"
)
target_link_libraries(forgeir_core PRIVATE nlohmann_json::nlohmann_json)
set_target_properties(forgeir_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
forgeir_enable_strict_warnings(forgeir_core)

add_executable(forgeir_cli cpp/src/core/cli_main.cpp)
target_link_libraries(forgeir_cli PRIVATE ForgeIR::core)
forgeir_enable_strict_warnings(forgeir_cli)

pybind11_add_module(forgeir_py MODULE bindings/python_module.cpp)
target_link_libraries(forgeir_py PRIVATE ForgeIR::core)
forgeir_enable_strict_warnings(forgeir_py)

include(CTest)
if(BUILD_TESTING)
    add_executable(forgeir_tests tests/cpp/core_smoke_test.cpp)
    target_link_libraries(forgeir_tests
        PRIVATE
            ForgeIR::core
            GTest::gtest_main
            nlohmann_json::nlohmann_json
    )
    forgeir_enable_strict_warnings(forgeir_tests)

    include(GoogleTest)
    gtest_discover_tests(forgeir_tests DISCOVERY_TIMEOUT 30)

    add_test(NAME forgeir_cli_version COMMAND forgeir_cli --version)
    set_tests_properties(forgeir_cli_version PROPERTIES
        PASS_REGULAR_EXPRESSION "^ForgeIR ${PROJECT_VERSION}"
    )

    add_test(NAME forgeir_cli_doctor COMMAND forgeir_cli doctor)
    set_tests_properties(forgeir_cli_doctor PROPERTIES
        PASS_REGULAR_EXPRESSION "\"forgeir_version\""
    )

    add_test(
        NAME forgeir_python_binding
        COMMAND "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/ctest_binding_smoke.py"
            "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
            "${CMAKE_CURRENT_SOURCE_DIR}/python"
            "${PROJECT_VERSION}"
    )
endif()
