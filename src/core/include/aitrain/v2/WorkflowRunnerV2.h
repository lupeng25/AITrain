#pragma once

#include "aitrain/core/Cancellation.h"
#include "aitrain/v2/StorageV2.h"

#include <functional>

namespace aitrain::v2 {

struct WorkflowStepExecutionResultV2 final {
    WorkflowStepState state = WorkflowStepState::Failed;
    ArtifactId outputArtifactId;
    Failure failure;
};

struct WorkflowRunExecutionResultV2 final {
    WorkflowStepState state = WorkflowStepState::Pending;
    ArtifactId finalOutputArtifactId;
    Failure failure;
};

// 异步执行宿主使用的派发结果。hasStep 为 true 时，调用方负责执行 step，
// 并在收到真实后端终态后调用 completeStep()；为 false 时工作流已到达终态。
struct WorkflowStepDispatchV2 final {
    bool hasStep = false;
    WorkflowStepSnapshotV2 step;
    WorkflowRunExecutionResultV2 result;
};

using WorkflowStepExecutorV2 = std::function<WorkflowStepExecutionResultV2(
    const WorkflowStepSnapshotV2& step,
    const aitrain::CancellationCallback& cancellation)>;

// V2 Workflow Runner 只做顺序编排和状态收口：实际后端由步骤执行器负责，
// 并且只能返回已提交 Artifact，禁止把临时路径传递给下游。
class WorkflowRunnerV2 final {
public:
    explicit WorkflowRunnerV2(StorageV2* storage);

    bool run(const WorkflowRunId& workflowRunId,
        const WorkflowStepExecutorV2& executor,
        WorkflowRunExecutionResultV2* result,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {});

    // 将下一个待执行步骤原子地变为 Running。它是 GUI/Worker 等异步执行宿主的
    // 唯一派发入口，避免宿主直接写入步骤状态。
    bool beginNextStep(const WorkflowRunId& workflowRunId,
        WorkflowStepDispatchV2* dispatch,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {});
    // 收口已派发步骤的真实后端结果；成功时自动派发下一个步骤，失败/取消时跳过后续步骤。
    bool completeStep(const WorkflowRunId& workflowRunId,
        const WorkflowStepId& workflowStepId,
        const WorkflowStepExecutionResultV2& execution,
        WorkflowStepDispatchV2* dispatch,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {});

private:
    bool skipPendingSteps(const QVector<WorkflowStepSnapshotV2>& steps,
        int firstIndex,
        QString* error);

    StorageV2* storage_ = nullptr;
};

} // namespace aitrain::v2
