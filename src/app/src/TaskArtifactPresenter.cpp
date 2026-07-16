#include "TaskArtifactPresenter.h"

#include <QDateTime>

namespace {

QString localTimeText(const QDateTime& value)
{
    return value.isValid()
        ? value.toLocalTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"))
        : QStringLiteral("--");
}

QString taskStateLabel(aitrain::TaskState state)
{
    switch (state) {
    case aitrain::TaskState::Created: return QStringLiteral("已创建");
    case aitrain::TaskState::Queued: return QStringLiteral("排队中");
    case aitrain::TaskState::Starting: return QStringLiteral("启动中");
    case aitrain::TaskState::Running: return QStringLiteral("运行中");
    case aitrain::TaskState::CancelRequested: return QStringLiteral("取消中");
    case aitrain::TaskState::Succeeded: return QStringLiteral("已完成");
    case aitrain::TaskState::Failed: return QStringLiteral("失败");
    case aitrain::TaskState::Canceled: return QStringLiteral("已取消");
    }
    return QStringLiteral("未知");
}

} // namespace

TaskArtifactPresenter::TaskArtifactPresenter(
    const aitrain::ProjectQueryService* queryService,
    QObject* parent)
    : QObject(parent)
    , queryService_(queryService)
{
    setObjectName(QStringLiteral("TaskArtifactPresenter"));
}

bool TaskArtifactPresenter::refresh(int limit)
{
    QString error;
    const QVector<aitrain::TaskSnapshot> tasks = queryService_
        ? queryService_->recentTasks(limit, &error)
        : QVector<aitrain::TaskSnapshot>();
    if (!error.isEmpty()) {
        taskRows_.clear();
        emit taskRowsChanged();
        fail(error);
        return false;
    }

    QVector<TaskListItem> rows;
    rows.reserve(tasks.size());
    for (const aitrain::TaskSnapshot& task : tasks) {
        TaskListItem row;
        row.taskId = task.id.toString();
        row.capabilityId = task.capabilityId;
        row.taskType = task.taskType;
        row.state = aitrain::taskStateToString(task.state);
        row.stateLabel = taskStateLabel(task.state);
        row.updatedAt = localTimeText(task.updatedAt);
        row.message = task.failure.isFailure() ? task.failure.message : QString();
        rows.append(row);
    }
    taskRows_ = rows;
    lastError_.clear();
    emit taskRowsChanged();
    return true;
}

bool TaskArtifactPresenter::selectTask(const QString& taskIdText)
{
    aitrain::TaskId taskId;
    QString error;
    if (!aitrain::TaskId::parse(taskIdText, &taskId, &error)) {
        clearSelection();
        fail(error);
        return false;
    }

    aitrain::TaskReadModel model;
    if (!queryService_ || !queryService_->taskDetails(taskId, &model, &error)) {
        clearSelection();
        fail(error.isEmpty() ? QStringLiteral("无法读取  任务详情。") : error);
        return false;
    }

    TaskArtifactDetails details;
    details.taskId = taskId.toString();
    for (const aitrain::ArtifactSnapshot& artifact : model.artifacts) {
        if (artifact.files.isEmpty()) {
            ArtifactFileItem row;
            row.artifactId = artifact.id.toString();
            row.kind = artifact.kind;
            row.createdAt = localTimeText(artifact.createdAt);
            details.artifacts.append(row);
            continue;
        }
        for (const aitrain::ArtifactFileSnapshot& file : artifact.files) {
            ArtifactFileItem row;
            row.artifactId = artifact.id.toString();
            row.kind = artifact.kind;
            row.relativePath = file.relativePath;
            row.sha256 = file.sha256;
            row.byteCount = file.byteCount;
            row.createdAt = localTimeText(artifact.createdAt);
            details.artifacts.append(row);
        }
    }
    for (const aitrain::MetricSnapshot& metric : model.metrics) {
        details.metrics.append({metric.name, metric.value, localTimeText(metric.occurredAt)});
    }
    for (const aitrain::WorkflowReadModel& workflow : model.workflows) {
        for (const aitrain::WorkflowStepSnapshot& step : workflow.steps) {
            details.workflowSteps.append({workflow.run.id.toString(), workflow.run.templateId,
                step.ordinal, step.kind, aitrain::workflowStepStateToString(step.state),
                step.backend, step.outputArtifactId.toString()});
        }
    }

    details.summary = QStringLiteral(" 任务 %1：%2 / %3 / %4，%5 个已提交产物文件，%6 个指标点，%7 个工作流步骤")
        .arg(details.taskId.left(8),
            model.task.taskType.isEmpty() ? QStringLiteral("未记录类型") : model.task.taskType,
            model.task.capabilityId.isEmpty() ? QStringLiteral("未记录能力") : model.task.capabilityId,
            taskStateLabel(model.task.state))
        .arg(details.artifacts.size())
        .arg(details.metrics.size())
        .arg(details.workflowSteps.size());
    if (model.task.failure.isFailure()) {
        details.summary.append(QStringLiteral("\n失败摘要：%1\n建议：%2")
            .arg(model.task.failure.message, model.task.failure.suggestedAction));
    }

    details_ = details;
    lastError_.clear();
    emit detailsChanged();
    return true;
}

void TaskArtifactPresenter::clearSelection()
{
    details_ = TaskArtifactDetails();
    emit detailsChanged();
}

int TaskArtifactPresenter::taskCount() const { return taskRows_.size(); }
QString TaskArtifactPresenter::selectedTaskId() const { return details_.taskId; }
int TaskArtifactPresenter::artifactCount() const { return details_.artifacts.size(); }
int TaskArtifactPresenter::metricCount() const { return details_.metrics.size(); }
int TaskArtifactPresenter::workflowStepCount() const { return details_.workflowSteps.size(); }
QString TaskArtifactPresenter::lastError() const { return lastError_; }
const QVector<TaskListItem>& TaskArtifactPresenter::taskRows() const { return taskRows_; }
const TaskArtifactDetails& TaskArtifactPresenter::details() const { return details_; }

void TaskArtifactPresenter::fail(const QString& error)
{
    lastError_ = error;
    emit queryFailed(error);
}
