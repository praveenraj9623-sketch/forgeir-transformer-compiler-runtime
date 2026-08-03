#include "forgeir/passes/pass_manager.hpp"

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

#include "forgeir/passes/graph_verifier_pass.hpp"

namespace forgeir {
namespace {

void append_result(PassManagerResult& aggregate, const std::string_view pass_name,
                   const std::string_view stage, PassResult result) {
    std::size_t nodes_added = 0;
    std::size_t nodes_removed = 0;
    for (RewriteRecord& rewrite : result.rewrites) {
        rewrite.pass_name = std::string(pass_name);
        rewrite.stage = std::string(stage);
        nodes_added += rewrite.nodes_added.size();
        nodes_removed += rewrite.nodes_removed.size();
    }
    aggregate.executions.push_back(
        PassExecutionRecord{std::string(pass_name), std::string(stage), result.status.ok(),
                            result.changed, result.diagnostics.size(), nodes_added, nodes_removed});
    aggregate.changed = aggregate.changed || result.changed;
    aggregate.diagnostics.insert(aggregate.diagnostics.end(),
                                 std::make_move_iterator(result.diagnostics.begin()),
                                 std::make_move_iterator(result.diagnostics.end()));
    aggregate.rewrites.insert(aggregate.rewrites.end(),
                              std::make_move_iterator(result.rewrites.begin()),
                              std::make_move_iterator(result.rewrites.end()));
    if (!result.status.ok()) {
        aggregate.status = std::move(result.status);
    }
}

} // namespace

std::size_t PassManagerResult::error_count() const noexcept {
    return static_cast<std::size_t>(
        std::count_if(diagnostics.begin(), diagnostics.end(), [](const Diagnostic& diagnostic) {
            return diagnostic.severity == DiagnosticSeverity::error;
        }));
}

std::size_t PassManagerResult::warning_count() const noexcept {
    return static_cast<std::size_t>(
        std::count_if(diagnostics.begin(), diagnostics.end(), [](const Diagnostic& diagnostic) {
            return diagnostic.severity == DiagnosticSeverity::warning;
        }));
}

void PassManager::add_pass(std::unique_ptr<Pass> pass) {
    if (!pass) {
        throw std::invalid_argument("PassManager cannot accept a null pass");
    }
    passes_.push_back(std::move(pass));
}

PassManagerResult PassManager::run(Graph& graph) const {
    PassManagerResult result;
    GraphVerifierPass verifier;

    for (const std::unique_ptr<Pass>& pass : passes_) {
        PassResult pre_verification = verifier.run(graph);
        append_result(result, pass->name(), "pre_verification", std::move(pre_verification));
        if (!result.status.ok()) {
            return result;
        }

        PassResult pass_result = pass->run(graph);
        append_result(result, pass->name(), "pass", std::move(pass_result));
        if (!result.status.ok()) {
            return result;
        }

        PassResult post_verification = verifier.run(graph);
        append_result(result, pass->name(), "post_verification", std::move(post_verification));
        if (!result.status.ok()) {
            return result;
        }
    }
    return result;
}

} // namespace forgeir
