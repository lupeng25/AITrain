#pragma once

#include "aitrain/core/Cancellation.h"
#include "aitrain/storage/ProjectStore.h"

#include <functional>

namespace aitrain {

struct WorkflowStepExecutionResult final {
    WorkflowStepState state = WorkflowStepState::Failed;
    ArtifactId outputArtifactId;
    Failure failure;
};

struct WorkflowRunExecutionResult final {
    WorkflowStepState state = WorkflowStepState::Pending;
    ArtifactId finalOutputArtifactId;
    Failure failure;
};

// 异步执行宿主使用的派发结果。hasStep 为 true 时，调用方负责执行 step，
// 并在收到真实后端终态后调用 completeStep()；为 false 时工作流已到达终态。
struct WorkflowStepDispatch final {
    bool hasStep = false;
    WorkflowStepSnapshot step;
    WorkflowRunExecutionResult result;
};

using WorkflowStepExecutor = std::function<WorkflowStepExecutionResult(
    const WorkflowStepSnapshot& step,
    const aitrain::CancellationCallback& cancellation)>;

//  Workflow Runner 只做顺序编排和状态收口：实际后端由步骤执行器负责，
// 并且只能返回已提交 Artifact，禁止把临时路径传递给下游。
class WorkflowRunner final {
public:
    explicit WorkflowRunner(ProjectStore* storage);

    bool run(const WorkflowRunId& workflowRunId,
        const WorkflowStepExecutor& executor,
        WorkflowRunExecutionResult* result,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {});

    // 将下一个待执行步骤原子地变为 Running。它是 GUI/Worker 等异步执行宿主的
    // 唯一派发入口，避免宿主直接写入步骤状态。
    bool beginNextStep(const WorkflowRunId& workflowRunId,
        WorkflowStepDispatch* dispatch,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {});
    // 收口已派发步骤的真实后端结果；成功时自动派发下一个步骤，失败/取消时跳过后续步骤。
    bool completeStep(const WorkflowRunId& workflowRunId,
        const WorkflowStepId& workflowStepId,
        const WorkflowStepExecutionResult& execution,
        WorkflowStepDispatch* dispatch,
        QString* error = nullptr,
        const aitrain::CancellationCallback& cancellation = {});

private:
    bool skipPendingSteps(const QVector<WorkflowStepSnapshot>& steps,
        int firstIndex,
        QString* error);

    ProjectStore* storage_ = nullptr;
};

} // namespace aitrain
