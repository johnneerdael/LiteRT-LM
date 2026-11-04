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

// A fake executor whose state, like the LiteRT executors' state buffers, moves
// into the context that TakeContext() returns. Its next restore can be made to
// fail.
class MovingStateFakeLlmExecutor : public FakeLlmExecutor {
 public:
  MovingStateFakeLlmExecutor()
      : FakeLlmExecutor(/*vocab_size=*/16, /*prefill_tokens_set=*/{},
                        /*decode_tokens_set=*/{}) {}

  void FailNextRestore() { fail_next_restore_ = true; }

  absl::StatusOr<std::unique_ptr<LlmContext>> CreateNewContext(
      std::optional<uint32_t> lora_id,
      RuntimeConfig runtime_config) const override {
    return std::make_unique<LlmContext>(
        std::make_unique<FakeProcessedContext>(),
        std::make_unique<RuntimeConfig>(std::move(runtime_config)),
        std::make_unique<RuntimeState>());
  }
  absl::StatusOr<std::unique_ptr<LlmContext>> CloneContext() const override {
    RuntimeState runtime_state;
    runtime_state.current_step = step_.value_or(0);
    return std::make_unique<LlmContext>(
        std::make_unique<FakeProcessedContext>(),
        std::make_unique<RuntimeConfig>(),
        std::make_unique<RuntimeState>(runtime_state));
  }
  absl::StatusOr<std::unique_ptr<LlmContext>> TakeContext() override {
    ABSL_ASSIGN_OR_RETURN(auto context, CloneContext());
    step_.reset();  // The state now belongs to `context`.
    return context;
  }
  absl::Status RestoreContext(
      std::unique_ptr<LlmContext> llm_context) override {
    if (fail_next_restore_) {
      fail_next_restore_ = false;
      return absl::InternalError("Restore failed.");
    }
    step_ = llm_context->runtime_state().current_step;
    return absl::OkStatus();
  }
  absl::StatusOr<int> GetCurrentStep() const override {
    if (!step_.has_value()) {
      return absl::FailedPreconditionError("The executor has no state.");
    }
    return *step_;
  }
  absl::Status SetCurrentStep(int current_step) override {
    step_ = current_step;
    return absl::OkStatus();
  }

 private:
  std::optional<int> step_ = 0;
  bool fail_next_restore_ = false;
};

TEST(ResourceManagerTest, AFailedRestoreKeepsTheSavedContextRestorable) {
  auto executor = std::make_unique<MovingStateFakeLlmExecutor>();
  MovingStateFakeLlmExecutor* fake = executor.get();
  ASSERT_OK_AND_ASSIGN(
      auto resource_manager,
      ResourceManager::Create(/*model_resources=*/nullptr, std::move(executor),
                              /*vision_executor_settings=*/nullptr,
                              /*audio_executor_settings=*/nullptr,
                              /*litert_env=*/nullptr));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<ContextHandler> first,
                       resource_manager->CreateContextHandler(
                           SessionConfig::CreateDefault()));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<ContextHandler> second,
                       resource_manager->CreateContextHandler(
                           SessionConfig::CreateDefault()));
  {
    ASSERT_OK_AND_ASSIGN(
        auto locked,
        resource_manager->AcquireExecutorWithContextHandler(first));
    ASSERT_OK(locked->SetCurrentStep(5));
  }
  fake->FailNextRestore();
  EXPECT_FALSE(
      resource_manager->AcquireExecutorWithContextHandler(second).ok());
  // `first` was saved before the failed restore took the executor's state:
  // switching back restores it.
  ASSERT_OK_AND_ASSIGN(
      auto locked, resource_manager->AcquireExecutorWithContextHandler(first));
  EXPECT_THAT(locked->GetCurrentStep(), IsOkAndHolds(5));
}

}  // namespace
}  // namespace litert::lm
