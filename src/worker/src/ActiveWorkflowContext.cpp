#include "aitrain/worker/ActiveWorkflowContext.h"

#include "aitrain/workflow/ProjectWorkspace.h"

namespace aitrain {

ActiveWorkflowContext::ActiveWorkflowContext() = default;
ActiveWorkflowContext::~ActiveWorkflowContext() = default;

bool ActiveWorkflowContext::begin(TaskCommandKind kind, QString* error)
{
    if (phase_ != ActiveWorkflowPhase::Inactive) {
        if (error) *error = QStringLiteral("Worker 已有活动任务。");
        return false;
    }
    commandKind_ = kind;
    phase_ = ActiveWorkflowPhase::Running;
    cancellationRequested_ = false;
    return true;
}

bool ActiveWorkflowContext::bind(std::unique_ptr<ProjectWorkspace> workspace,
    const TaskId& taskId, QString* error)
{
    if (phase_ == ActiveWorkflowPhase::Inactive || !workspace || !taskId.isValid()
        || workspace_ || taskId_.isValid()) {
        if (error) *error = QStringLiteral("活动任务的 Workspace/TaskId 绑定无效。");
        return false;
    }
    workspace_ = std::move(workspace);
    taskId_ = taskId;
    if (cancellationRequested_) {
        QString ignored;
        workspace_->requestTaskCancellation(taskId_, &ignored);
    }
    return true;
}

bool ActiveWorkflowContext::requestCancel()
{
    if (phase_ == ActiveWorkflowPhase::Inactive
        || phase_ == ActiveWorkflowPhase::DurableTerminal) return false;
    cancellationRequested_ = true;
    phase_ = ActiveWorkflowPhase::CancelRequested;
    if (workspace_ && taskId_.isValid()) {
        QString ignored;
        workspace_->requestTaskCancellation(taskId_, &ignored);
    }
    return true;
}

bool ActiveWorkflowContext::markDurableTerminal(QString* error)
{
    if (phase_ != ActiveWorkflowPhase::Running
        && phase_ != ActiveWorkflowPhase::CancelRequested) {
        if (error) *error = QStringLiteral("只有活动任务可以进入 DurableTerminal。");
        return false;
    }
    phase_ = ActiveWorkflowPhase::DurableTerminal;
    return true;
}

void ActiveWorkflowContext::clear()
{
    workspace_.reset();
    taskId_ = {};
    cancellationRequested_ = false;
    phase_ = ActiveWorkflowPhase::Inactive;
}

ActiveWorkflowPhase ActiveWorkflowContext::phase() const { return phase_; }
TaskCommandKind ActiveWorkflowContext::commandKind() const { return commandKind_; }
bool ActiveWorkflowContext::cancellationRequested() const { return cancellationRequested_; }
ProjectWorkspace* ActiveWorkflowContext::workspace() const { return workspace_.get(); }
const TaskId& ActiveWorkflowContext::taskId() const { return taskId_; }

} // namespace aitrain
