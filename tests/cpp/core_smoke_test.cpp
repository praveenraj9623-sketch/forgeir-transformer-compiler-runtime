#include <string>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "forgeir/core/build_config.hpp"
#include "forgeir/core/build_info.hpp"
#include "forgeir/core/version.hpp"

TEST(CoreVersionSmoke, MatchesConfiguredSemanticVersion) { EXPECT_EQ(forgeir::version(), "0.1.0"); }

TEST(EnvironmentDiagnosticSmoke, HasStableSchema) {
    const forgeir::BuildInfo info = forgeir::current_build_info();
    const nlohmann::json diagnostic = nlohmann::json::parse(forgeir::build_info_json(info));

    EXPECT_EQ(diagnostic.at("forgeir_version"), "0.1.0");
    EXPECT_TRUE(diagnostic.at("compiler").is_string());
    EXPECT_FALSE(diagnostic.at("compiler").get<std::string>().empty());
    EXPECT_TRUE(diagnostic.at("build_type").is_string());
    EXPECT_TRUE(diagnostic.at("operating_system").is_string());
    EXPECT_EQ(diagnostic.at("cpp_standard"), "C++17");
    EXPECT_TRUE(diagnostic.at("python_version").is_null());

    const nlohmann::json& features = diagnostic.at("features");
    EXPECT_TRUE(features.at("cuda").is_boolean());
    EXPECT_TRUE(features.at("hip").is_boolean());
    EXPECT_TRUE(features.at("mlir").is_boolean());
    EXPECT_EQ(features.at("cuda").get<bool>(), FORGEIR_CUDA_COMPILED != 0);
    EXPECT_EQ(features.at("hip").get<bool>(), FORGEIR_HIP_COMPILED != 0);
    EXPECT_EQ(features.at("mlir").get<bool>(), FORGEIR_MLIR_COMPILED != 0);
}
