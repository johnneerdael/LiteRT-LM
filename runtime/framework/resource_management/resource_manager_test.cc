// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/framework/resource_management/resource_manager.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/executor_stats.h"
#include "runtime/executor/fake_llm_executor.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/framework/resource_management/context_handler/context_handler.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::FieldsAre;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

// A fake executor that reports executor stats and counts stat resets.
class StatsFakeLlmExecutor : public FakeLlmExecutor {
 public:
  explicit StatsFakeLlmExecutor(int* num_resets)
      : FakeLlmExecutor(/*vocab_size=*/16, /*prefill_tokens_set=*/{},
                        /*decode_tokens_set=*/{}),
        num_resets_(num_resets) {
    stats_.AccumulatePlainStep(absl::Milliseconds(10));
  }

  std::optional<ExecutorStats> GetExecutorStats() const override {
    return stats_;
  }
  void ResetExecutorStats() override { ++*num_resets_; }

 private:
  ExecutorStats stats_{.module_name = std::string(kLlmModuleName)};
  int* num_resets_;
};

// Tasks::Decode receives the `LockedLlmExecutor` wrapper rather than the
// underlying executor, so the stats APIs must be forwarded through it.
TEST(ResourceManagerTest, AcquiredExecutorForwardsExecutorStats) {
  int num_resets = 0;
  ASSERT_OK_AND_ASSIGN(
      auto resource_manager,
      ResourceManager::Create(
          /*model_resources=*/nullptr,
          std::make_unique<StatsFakeLlmExecutor>(&num_resets),
          /*vision_executor_settings=*/nullptr,
          /*audio_executor_settings=*/nullptr, /*litert_env=*/nullptr));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LlmExecutor> executor,
                       resource_manager->AcquireExecutor());

  std::optional<ExecutorStats> stats = executor->GetExecutorStats();
  ASSERT_TRUE(stats.has_value());
  EXPECT_EQ(stats->module_name, kLlmModuleName);
  EXPECT_EQ(stats->plain_steps(), 1);

  executor->ResetExecutorStats();
  EXPECT_EQ(num_resets, 1);
}

// A fake executor that reports a given draft verification support and accepts
// every draft that it verifies.
class DraftVerificationFakeLlmExecutor : public FakeLlmExecutor {
 public:
  explicit DraftVerificationFakeLlmExecutor(absl::Status support)
      : FakeLlmExecutor(/*vocab_size=*/16, /*prefill_tokens_set=*/{},
                        /*decode_tokens_set=*/{}),
        support_(std::move(support)) {}

  absl::Status CheckDraftVerificationSupport() const override {
    return support_;
  }

  absl::StatusOr<DraftVerificationResult> VerifyDraft(
      absl::Span<const int> draft_token_ids,
      const ExecutorDecodeParams& decode_params) override {
    return DraftVerificationResult{
        .tokens =
            std::vector<int>(draft_token_ids.begin(), draft_token_ids.end()),
        .num_accepted = static_cast<int>(draft_token_ids.size())};
  }

 private:
  absl::Status support_;
};

TEST(ResourceManagerTest, AcquiredExecutorForwardsDraftVerificationSupport) {
  for (const absl::Status& support :
       {absl::OkStatus(), absl::FailedPreconditionError("No verify.")}) {
    SCOPED_TRACE(support.ToString());
    ASSERT_OK_AND_ASSIGN(
        auto resource_manager,
        ResourceManager::Create(
            /*model_resources=*/nullptr,
            std::make_unique<DraftVerificationFakeLlmExecutor>(support),
            /*vision_executor_settings=*/nullptr,
            /*audio_executor_settings=*/nullptr, /*litert_env=*/nullptr));
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<LlmExecutor> executor,
                         resource_manager->AcquireExecutor());
    EXPECT_EQ(executor->CheckDraftVerificationSupport(), support);
  }
}

TEST(ResourceManagerTest, AcquiredExecutorForwardsDraftVerification) {
  ASSERT_OK_AND_ASSIGN(
      auto resource_manager,
      ResourceManager::Create(
          /*model_resources=*/nullptr,
          std::make_unique<DraftVerificationFakeLlmExecutor>(absl::OkStatus()),
          /*vision_executor_settings=*/nullptr,
          /*audio_executor_settings=*/nullptr, /*litert_env=*/nullptr));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LlmExecutor> executor,
                       resource_manager->AcquireExecutor());
  EXPECT_THAT(executor->VerifyDraft({3, 1, 4}, ExecutorDecodeParams()),
              IsOkAndHolds(FieldsAre(ElementsAre(3, 1, 4), 3)));
}

TEST(ResourceManagerTest, ExecutorsDontSupportDraftVerificationByDefault) {
  FakeLlmExecutor executor(/*vocab_size=*/16, /*prefill_tokens_set=*/{},
                           /*decode_tokens_set=*/{});
  // The defaults of `LlmExecutor`.
  EXPECT_THAT(executor.LlmExecutor::CheckDraftVerificationSupport(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(
      executor.LlmExecutor::VerifyDraft({3, 1, 4}, ExecutorDecodeParams()),
      StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(ResourceManagerTest, ExecutorsDontVerifyDraftsByDefault) {
  FakeLlmExecutor executor(/*vocab_size=*/16, /*prefill_tokens_set=*/{},
                           /*decode_tokens_set=*/{});
  executor.SetDraftVerificationSupport(absl::OkStatus());
  // The default of `LlmExecutor` for an executor that claims support.
  EXPECT_THAT(
      executor.LlmExecutor::VerifyDraft({3, 1, 4}, ExecutorDecodeParams()),
      StatusIs(absl::StatusCode::kUnimplemented));
}

// A fake executor that holds the runtime config and state of the context that
// it runs, like the LiteRT executors, and counts the contexts that it clones,
// which copy the state (e.g. the KV cache).
class ContextFakeLlmExecutor : public FakeLlmExecutor {
 public:
  explicit ContextFakeLlmExecutor(int* num_clones)
      : FakeLlmExecutor(/*vocab_size=*/16, /*prefill_tokens_set=*/{},
                        /*decode_tokens_set=*/{}),
        num_clones_(num_clones) {}

  absl::StatusOr<std::unique_ptr<LlmContext>> CreateNewContext(
      std::optional<uint32_t> lora_id,
      RuntimeConfig runtime_config) const override {
    return std::make_unique<LlmContext>(
        std::make_unique<FakeProcessedContext>(),
        std::make_unique<RuntimeConfig>(std::move(runtime_config)),
        std::make_unique<RuntimeState>());
  }

  absl::StatusOr<std::unique_ptr<LlmContext>> CloneContext() const override {
    ++*num_clones_;
    return std::make_unique<LlmContext>(
        std::make_unique<FakeProcessedContext>(),
        std::make_unique<RuntimeConfig>(runtime_config_),
        std::make_unique<RuntimeState>(runtime_state_));
  }

  absl::Status RestoreContext(
      std::unique_ptr<LlmContext> llm_context) override {
    ABSL_ASSIGN_OR_RETURN(auto runtime_config,
                          llm_context->RetrieveRuntimeConfig());
    ABSL_ASSIGN_OR_RETURN(auto runtime_state,
                          llm_context->RetrieveRuntimeState());
    runtime_config_ = *runtime_config;
    runtime_state_ = *runtime_state;
    return absl::OkStatus();
  }

  absl::StatusOr<RuntimeConfig> GetRuntimeConfig() const override {
    return runtime_config_;
  }
  absl::Status UpdateRuntimeConfig(
      const RuntimeConfig& runtime_config) override {
    runtime_config_ = runtime_config;
    return absl::OkStatus();
  }
  absl::StatusOr<RuntimeState> GetRuntimeState() const override {
    return runtime_state_;
  }
  absl::Status UpdateRuntimeState(const RuntimeState& runtime_state) override {
    runtime_state_ = runtime_state;
    return absl::OkStatus();
  }
  absl::StatusOr<int> GetCurrentStep() const override {
    return runtime_state_.current_step;
  }
  absl::Status SetCurrentStep(int current_step) override {
    runtime_state_.current_step = current_step;
    return absl::OkStatus();
  }

 private:
  int* num_clones_;
  RuntimeConfig runtime_config_;
  RuntimeState runtime_state_;
};

// Like the NPU executor: it can't report its runtime config or state.
class StatelessContextFakeLlmExecutor : public ContextFakeLlmExecutor {
 public:
  using ContextFakeLlmExecutor::ContextFakeLlmExecutor;
  absl::StatusOr<RuntimeConfig> GetRuntimeConfig() const override {
    return absl::UnimplementedError("GetRuntimeConfig");
  }
  absl::StatusOr<RuntimeState> GetRuntimeState() const override {
    return absl::UnimplementedError("GetRuntimeState");
  }
};

class ResourceManagerContextSwitchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(
        resource_manager_,
        ResourceManager::Create(
            /*model_resources=*/nullptr,
            std::make_unique<ContextFakeLlmExecutor>(&num_clones_),
            /*vision_executor_settings=*/nullptr,
            /*audio_executor_settings=*/nullptr, /*litert_env=*/nullptr));
  }

  absl::StatusOr<std::shared_ptr<ContextHandler>> CreateHandler() {
    return resource_manager_->CreateContextHandler(
        SessionConfig::CreateDefault());
  }

  // Makes `handler` the current handler and moves it to `step`.
  absl::Status RunTo(const std::shared_ptr<ContextHandler>& handler,
                     int step) {
    ABSL_ASSIGN_OR_RETURN(
        auto executor,
        resource_manager_->AcquireExecutorWithContextHandler(handler));
    return executor->SetCurrentStep(step);
  }

  absl::StatusOr<int> CurrentStepOf(
      const std::shared_ptr<ContextHandler>& handler) {
    ABSL_ASSIGN_OR_RETURN(
        auto executor,
        resource_manager_->AcquireExecutorWithContextHandler(handler));
    return executor->GetCurrentStep();
  }

  int num_clones_ = 0;
  std::unique_ptr<ResourceManager> resource_manager_;
};

TEST_F(ResourceManagerContextSwitchTest, SavesTheContextOfAHandlerInProgress) {
  ASSERT_OK_AND_ASSIGN(auto first, CreateHandler());
  ASSERT_OK_AND_ASSIGN(auto second, CreateHandler());
  ASSERT_OK(RunTo(first, 5));
  ASSERT_OK(RunTo(second, 3));
  EXPECT_EQ(num_clones_, 1);
  EXPECT_THAT(CurrentStepOf(first), IsOkAndHolds(5));
  EXPECT_EQ(num_clones_, 2);
  EXPECT_THAT(CurrentStepOf(second), IsOkAndHolds(3));
}

TEST_F(ResourceManagerContextSwitchTest,
       DoesntCopyTheStateOfAHandlerAtStepZero) {
  ASSERT_OK_AND_ASSIGN(auto first, CreateHandler());
  ASSERT_OK_AND_ASSIGN(auto second, CreateHandler());
  ASSERT_OK(RunTo(first, 5));
  ASSERT_OK(RunTo(first, 0));
  ASSERT_OK(RunTo(second, 3));
  EXPECT_EQ(num_clones_, 0);
  EXPECT_THAT(CurrentStepOf(first), IsOkAndHolds(0));
  EXPECT_EQ(num_clones_, 1);
  EXPECT_THAT(CurrentStepOf(second), IsOkAndHolds(3));
}

TEST_F(ResourceManagerContextSwitchTest,
       DoesntSaveTheContextOfAReleasedHandler) {
  ASSERT_OK_AND_ASSIGN(auto first, CreateHandler());
  ASSERT_OK_AND_ASSIGN(auto second, CreateHandler());
  ASSERT_OK(RunTo(first, 5));
  ASSERT_OK(RunTo(second, 3));
  ASSERT_EQ(num_clones_, 1);
  resource_manager_->ReleaseContextHandler(second);
  second.reset();
  EXPECT_THAT(CurrentStepOf(first), IsOkAndHolds(5));
  EXPECT_EQ(num_clones_, 1);
}

TEST_F(ResourceManagerContextSwitchTest,
       SavesTheContextOfAReleasedHandlerThatSharesIt) {
  ASSERT_OK_AND_ASSIGN(auto first, CreateHandler());
  ASSERT_OK(RunTo(first, 5));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<ContextHandler> clone,
                       resource_manager_->CloneContextHandler(first));
  ASSERT_OK_AND_ASSIGN(auto other, CreateHandler());
  // The clone shares the processed context of `first`, which needs it.
  ASSERT_OK(RunTo(clone, 0));
  resource_manager_->ReleaseContextHandler(clone);
  clone.reset();
  const int num_clones = num_clones_;
  ASSERT_OK(RunTo(other, 3));
  EXPECT_EQ(num_clones_, num_clones + 1);
  EXPECT_THAT(CurrentStepOf(first), IsOkAndHolds(5));
}

TEST_F(ResourceManagerContextSwitchTest, ReleasingAnotherHandlerKeepsTheCurrent) {
  ASSERT_OK_AND_ASSIGN(auto first, CreateHandler());
  ASSERT_OK_AND_ASSIGN(auto second, CreateHandler());
  ASSERT_OK(RunTo(second, 3));
  ASSERT_OK(RunTo(first, 5));
  resource_manager_->ReleaseContextHandler(second);
  second.reset();
  EXPECT_THAT(CurrentStepOf(first), IsOkAndHolds(5));
  ASSERT_OK_AND_ASSIGN(auto third, CreateHandler());
  const int num_clones = num_clones_;
  ASSERT_OK(RunTo(third, 1));
  EXPECT_EQ(num_clones_, num_clones + 1);
  EXPECT_THAT(CurrentStepOf(first), IsOkAndHolds(5));
}

TEST(ResourceManagerContextSwitchFallbackTest,
     CopiesTheStateWhenTheExecutorCantReportIt) {
  int num_clones = 0;
  ASSERT_OK_AND_ASSIGN(
      auto resource_manager,
      ResourceManager::Create(
          /*model_resources=*/nullptr,
          std::make_unique<StatelessContextFakeLlmExecutor>(&num_clones),
          /*vision_executor_settings=*/nullptr,
          /*audio_executor_settings=*/nullptr, /*litert_env=*/nullptr));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<ContextHandler> first,
                       resource_manager->CreateContextHandler(
                           SessionConfig::CreateDefault()));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<ContextHandler> second,
                       resource_manager->CreateContextHandler(
                           SessionConfig::CreateDefault()));
  ASSERT_OK(
      resource_manager->AcquireExecutorWithContextHandler(first).status());
  // `first` is at step 0, but the executor can't say so: the switch saves a
  // copy, as before, instead of failing.
  ASSERT_OK_AND_ASSIGN(
      auto executor,
      resource_manager->AcquireExecutorWithContextHandler(second));
  EXPECT_EQ(num_clones, 1);
  ASSERT_OK(executor->SetCurrentStep(3));
}

}  // namespace
}  // namespace litert::lm
