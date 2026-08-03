include_guard(GLOBAL)

function(forgeir_require_ci_for_fetch dependency_name)
    if(NOT DEFINED ENV{CI}
       OR "$ENV{CI}" STREQUAL ""
       OR "$ENV{CI}" STREQUAL "0"
       OR "$ENV{CI}" STREQUAL "false")
        message(FATAL_ERROR
            "${dependency_name} was not found under third_party. "
            "Network fallback is disabled outside CI.")
    endif()
endfunction()

set(JSON_BuildTests OFF CACHE BOOL "Do not build nlohmann/json tests" FORCE)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann_json/CMakeLists.txt")
    add_subdirectory(
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann_json"
        "${CMAKE_CURRENT_BINARY_DIR}/_deps/nlohmann_json-build"
        EXCLUDE_FROM_ALL
        SYSTEM
    )
else()
    forgeir_require_ci_for_fetch("nlohmann/json v3.12.0")
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY "https://github.com/nlohmann/json.git"
        GIT_TAG "55f93686c01528224f448c19128836e7df245f72" # v3.12.0
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

set(BUILD_GMOCK OFF CACHE BOOL "Do not build GoogleMock" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "Do not install GoogleTest" FORCE)
if(MSVC)
    set(gtest_force_shared_crt ON CACHE BOOL "Use the shared MSVC runtime" FORCE)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest/CMakeLists.txt")
    add_subdirectory(
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest"
        "${CMAKE_CURRENT_BINARY_DIR}/_deps/googletest-build"
        EXCLUDE_FROM_ALL
        SYSTEM
    )
else()
    forgeir_require_ci_for_fetch("GoogleTest v1.17.0")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY "https://github.com/google/googletest.git"
        GIT_TAG "52eb8108c5bdec04579160ae17225d66034bd723" # v1.17.0
    )
    FetchContent_MakeAvailable(googletest)
endif()

foreach(third_party_target IN ITEMS nlohmann_json gtest gtest_main)
    if(TARGET ${third_party_target})
        set_property(TARGET ${third_party_target} PROPERTY SYSTEM TRUE)
    endif()
endforeach()
