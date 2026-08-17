#include <filesystem>
#include <string>
#include <unordered_map>

#include <gtest/gtest.h>

#include "forgeir/runtime/benchmark.hpp"
#include "forgeir/runtime/runtime_session.hpp"

namespace {

std::shared_ptr<forgeir::RuntimeSession> load_test_session() {
    const std::filesystem::path graph_path = std::filesystem::path(FORGEIR_SOURCE_DIR) / "tests" /
                                             "fixtures" / "tiny_transformer_block_v1_o2.graph.json";
    auto session = forgeir::RuntimeSession::load(graph_path.string(), "cpu");
    if (!session.ok()) {
        return nullptr;
    }
    return session.take_value();
}

} // namespace

TEST(RuntimeBenchmark, RejectsZeroMeasuredIterationsBeforeExecution) {
    const auto session = load_test_session();
    ASSERT_NE(session, nullptr);
    forgeir::RuntimeBenchmarkOptions options;
    options.warmup_count = 0;
    options.measured_iteration_count = 0;
    const auto result = forgeir::benchmark_runtime(*session, {}, {}, options);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), forgeir::StatusCode::invalid_argument);
}

TEST(RuntimeBenchmark, PropagatesWarmupExecutionFailureWithContext) {
    const auto session = load_test_session();
    ASSERT_NE(session, nullptr);
    forgeir::RuntimeBenchmarkOptions options;
    options.warmup_count = 1;
    options.measured_iteration_count = 1;
    const auto result = forgeir::benchmark_runtime(*session, {}, {}, options);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), forgeir::StatusCode::not_found);
    EXPECT_NE(result.status().message().find("benchmark warm-up iteration 0 failed"),
              std::string::npos);
}
