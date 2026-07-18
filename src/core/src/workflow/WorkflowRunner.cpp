#include "aitrain/workflow/WorkflowRunner.h"

namespace aitrain {
namespace {

Failure failureFor(WorkflowStepState state, const QString& message)
{
    if (state == WorkflowStepState::Canceled) {
        return {FailureCode::Canceled, message,
            QStringLiteral("确认取消原因；如需继续，请重新发起该工作流。"),
            QDateTime::currentDateTimeUtc()};
    }
    return {FailureCode::InternalError, message,
        QStringLiteral("检查工作流步骤与 Evidence 后重试。"),
        QDateTime::currentDateTimeUtc()};
}

} // namespace

WorkflowRunner::WorkflowRunner(ProjectStore* storage)
    : storage_(storage)
{
}

bool WorkflowRunner::skipPendingSteps(const QVector<WorkflowStepSnapshot>& steps,
    int firstIndex,
    QString* error)
{
    for (int index = firstIndex; index < steps.size(); ++index) {
        const WorkflowStepSnapshot& step = steps.at(index);
        if (step.state != WorkflowStepState::Pending) continue;
        if (!storage_->transitionWorkflowStep(step.id, WorkflowStepState::Pending,
                WorkflowStepState::Skipped, {}, {}, error)) {
            return false;
        }
    }
    return true;
}

bool WorkflowRunner::run(const WorkflowRunId& workflowRunId,
    const WorkflowStepExecutor& executor,
    WorkflowRunExecutionResult* result,
    QString* error,
    const aitrain::CancellationCallback& cancellation)
{
    if (!storage_ || !storage_->isOpen() || !workflowRunId.isValid() || !executor || !result) {
        if (error) *error = QStringLiteral("运行  Workflow 需要已打开存储、有效运行 ID、执行器和结果对象。");
        return false;
    }
    WorkflowStepDispatch dispatch;
    if (!beginNextStep(workflowRunId, &dispatch, error, cancellation)) return false;
    while (dispatch.hasStep) {
        const WorkflowStepExecutionResult execution = executor(dispatch.step, cancellation);
        if (!completeStep(workflowRunId, dispatch.step.id, execution, &dispatch, error, cancellation)) return false;
    }
    *result = dispatch.result;
    return true;
}

bool WorkflowRunner::beginNextStep(const WorkflowRunId& workflowRunId,
    WorkflowStepDispatch* dispatch,
    QString* error,
    const aitrain::CancellationCallback& cancellation)
{
    if (!storage_ || !storage_->isOpen() || !workflowRunId.isValid() || !dispatch) {
        if (error) *error = QStringLiteral("派发  Workflow 步骤需要已打开存储、有效运行 ID 和输出对象。");
        return false;
    }
    *dispatch = {};
    WorkflowRunSnapshot workflow;
    if (!storage_->workflowRun(workflowRunId, &workflow, error)) return false;
    const QVector<WorkflowStepSnapshot> steps = storage_->workflowSteps(workflowRunId, error);
    if (steps.isEmpty()) {
        if (error && error->isEmpty()) *error = QStringLiteral(" Workflow 不包含步骤。");
        return false;
    }

    ArtifactId previousOutput;
    for (int index = 0; index < steps.size(); ++index) {
        WorkflowStepSnapshot step = steps.at(index);
        if (step.state == WorkflowStepState::Succeeded) {
            if (!step.outputArtifactId.isValid()) {
                if (error) *error = QStringLiteral("已成功的工作流步骤缺少输出 Artifact。");
                return false;
            }
            previousOutput = step.outputArtifactId;
            dispatch->result.finalOutputArtifactId = previousOutput;
            continue;
        }
        if (step.state == WorkflowStepState::Running) {
            if (error) *error = QStringLiteral("工作流已存在正在执行的步骤，不可重复派发。");
            return false;
        }
        if (step.state == WorkflowStepState::Failed || step.state == WorkflowStepState::Canceled) {
            if (!storage_->terminalizeWorkflowStepAndSkipSuccessors(step.id, step.state,
                    step.state, step.failure, error)) {
                return false;
            }
            dispatch->result.state = step.state;
            dispatch->result.failure = step.failure;
            return true;
        }
        if (step.state == WorkflowStepState::Skipped) {
            dispatch->result.state = WorkflowStepState::Failed;
            dispatch->result.failure = failureFor(WorkflowStepState::Failed,
                QStringLiteral("工作流包含已跳过步骤，需从最后一个成功步骤重新规划。"));
            return true;
        }
        if (step.inputArtifactId.isValid() && previousOutput.isValid() && step.inputArtifactId != previousOutput) {
            const Failure failure = {FailureCode::ArtifactIncompatible,
                QStringLiteral("工作流步骤输入 Artifact 与上一步输出不一致。"), {}, QDateTime::currentDateTimeUtc()};
            if (!storage_->terminalizeWorkflowStepAndSkipSuccessors(step.id,
                    WorkflowStepState::Pending, WorkflowStepState::Failed, failure, error)) return false;
            dispatch->result.state = WorkflowStepState::Failed;
            dispatch->result.failure = failure;
            return true;
        }
        if (!step.inputArtifactId.isValid() && previousOutput.isValid()) {
            if (!storage_->bindWorkflowStepInput(step.id, previousOutput, error)) return false;
            step.inputArtifactId = previousOutput;
        }
        if (aitrain::isCancellationRequested(cancellation)) {
            const Failure failure = failureFor(WorkflowStepState::Canceled,
                QStringLiteral("Workflow 在步骤启动前收到取消请求。"));
            if (!storage_->terminalizeWorkflowStepAndSkipSuccessors(step.id,
                    WorkflowStepState::Pending, WorkflowStepState::Canceled, failure, error)) return false;
            dispatch->result.state = WorkflowStepState::Canceled;
            dispatch->result.failure = failure;
            return true;
        }
        if (!storage_->transitionWorkflowStep(step.id, WorkflowStepState::Pending,
                WorkflowStepState::Running, {}, {}, error)) return false;
        dispatch->hasStep = true;
        dispatch->step = step;
        dispatch->step.state = WorkflowStepState::Running;
        dispatch->result.state = WorkflowStepState::Running;
        return true;
    }
    dispatch->result.state = WorkflowStepState::Succeeded;
    return true;
}

bool WorkflowRunner::completeStep(const WorkflowRunId& workflowRunId,
    const WorkflowStepId& workflowStepId,
    const WorkflowStepExecutionResult& execution,
    WorkflowStepDispatch* dispatch,
    QString* error,
    const aitrain::CancellationCallback& cancellation)
{
    if (!storage_ || !storage_->isOpen() || !workflowRunId.isValid()
        || !workflowStepId.isValid() || !dispatch) {
        if (error) *error = QStringLiteral("收口  Workflow 步骤需要已打开存储、有效运行/步骤 ID 和输出对象。");
        return false;
    }
    const QVector<WorkflowStepSnapshot> steps = storage_->workflowSteps(workflowRunId, error);
    int index = -1;
    for (int candidate = 0; candidate < steps.size(); ++candidate) {
        if (steps.at(candidate).id == workflowStepId) {
            index = candidate;
            break;
        }
    }
    if (index < 0 || steps.at(index).state != WorkflowStepState::Running) {
        if (error) *error = QStringLiteral("待收口的 Workflow 步骤不存在或不处于 Running 状态。");
        return false;
    }
    if (execution.state == WorkflowStepState::Succeeded) {
        if (!execution.outputArtifactId.isValid()) {
            const Failure failure = {FailureCode::ArtifactIncomplete,
                QStringLiteral("步骤执行器声明成功，但没有返回已提交 Artifact。"), {}, QDateTime::currentDateTimeUtc()};
            if (!storage_->terminalizeWorkflowStepAndSkipSuccessors(workflowStepId,
                    WorkflowStepState::Running, WorkflowStepState::Failed, failure, error)) return false;
            dispatch->hasStep = false;
            dispatch->result = {WorkflowStepState::Failed, {}, failure};
            return true;
        }
        if (!storage_->transitionWorkflowStep(workflowStepId, WorkflowStepState::Running,
                WorkflowStepState::Succeeded, execution.outputArtifactId, {}, error)) return false;
        return beginNextStep(workflowRunId, dispatch, error, cancellation);
    }
    const WorkflowStepState terminal = execution.state == WorkflowStepState::Canceled
        ? WorkflowStepState::Canceled : WorkflowStepState::Failed;
    Failure failure = execution.failure;
    if (!failure.isFailure()) {
        failure = failureFor(terminal, terminal == WorkflowStepState::Canceled
            ? QStringLiteral("步骤执行器已取消。")
            : QStringLiteral("步骤执行器失败。"));
    }
    if (!storage_->terminalizeWorkflowStepAndSkipSuccessors(workflowStepId,
            WorkflowStepState::Running, terminal, failure, error)) return false;
    dispatch->hasStep = false;
    dispatch->result = {terminal, {}, failure};
    return true;
}

} // namespace aitrain
