#include "TaskArtifactPresenterV2.h"

#include <QDateTime>

namespace {

QString localTimeText(const QDateTime& value)
{
    return value.isValid()
        ? value.toLocalTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"))
        : QStringLiteral("--");
}

QString taskStateLabel(aitrain::v2::TaskState state)
{
    switch (state) {
    case aitrain::v2::TaskState::Created: return QStringLiteral("已创建");
    case aitrain::v2::TaskState::Queued: return QStringLiteral("排队中");
    case aitrain::v2::TaskState::Starting: return QStringLiteral("启动中");
    case aitrain::v2::TaskState::Running: return QStringLiteral("运行中");
    case aitrain::v2::TaskState::CancelRequested: return QStringLiteral("取消中");
    case aitrain::v2::TaskState::Succeeded: return QStringLiteral("已完成");
    case aitrain::v2::TaskState::Failed: return QStringLiteral("失败");
    case aitrain::v2::TaskState::Canceled: return QStringLiteral("已取消");
    }
    return QStringLiteral("未知");
}

} // namespace

TaskArtifactPresenterV2::TaskArtifactPresenterV2(
    const aitrain::v2::ProjectQueryServiceV2* queryService,
    QObject* parent)
    : QObject(parent)
    , queryService_(queryService)
{
    setObjectName(QStringLiteral("TaskArtifactPresenterV2"));
}

bool TaskArtifactPresenterV2::refresh(int limit)
{
    QString error;
    const QVector<aitrain::v2::TaskSnapshot> tasks = queryService_
        ? queryService_->recentTasks(limit, &error)
        : QVector<aitrain::v2::TaskSnapshot>();
    if (!error.isEmpty()) {
        taskRows_.clear();
        emit taskRowsChanged();
        fail(error);
        return false;
    }

    QVector<TaskListItemV2> rows;
    rows.reserve(tasks.size());
    for (const aitrain::v2::TaskSnapshot& task : tasks) {
        TaskListItemV2 row;
        row.taskId = task.id.toString();
        row.capabilityId = task.capabilityId;
        row.taskType = task.taskType;
        row.state = aitrain::v2::taskStateToString(task.state);
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

bool TaskArtifactPresenterV2::selectTask(const QString& taskIdText)
{
    aitrain::v2::TaskId taskId;
    QString error;
    if (!aitrain::v2::TaskId::parse(taskIdText, &taskId, &error)) {
        clearSelection();
        fail(error);
        return false;
    }

    aitrain::v2::TaskReadModelV2 model;
    if (!queryService_ || !queryService_->taskDetails(taskId, &model, &error)) {
        clearSelection();
        fail(error.isEmpty() ? QStringLiteral("无法读取 V2 任务详情。") : error);
        return false;
    }

    TaskArtifactDetailsV2 details;
    details.taskId = taskId.toString();
    for (const aitrain::v2::ArtifactSnapshotV2& artifact : model.artifacts) {
        if (artifact.files.isEmpty()) {
            ArtifactFileItemV2 row;
            row.artifactId = artifact.id.toString();
            row.kind = artifact.kind;
            row.createdAt = localTimeText(artifact.createdAt);
            details.artifacts.append(row);
            continue;
        }
        for (const aitrain::v2::ArtifactFileSnapshot& file : artifact.files) {
            ArtifactFileItemV2 row;
            row.artifactId = artifact.id.toString();
            row.kind = artifact.kind;
            row.relativePath = file.relativePath;
            row.sha256 = file.sha256;
            row.byteCount = file.byteCount;
            row.createdAt = localTimeText(artifact.createdAt);
            details.artifacts.append(row);
        }
    }
    for (const aitrain::v2::MetricSnapshotV2& metric : model.metrics) {
        details.metrics.append({metric.name, metric.value, localTimeText(metric.occurredAt)});
    }
    for (const aitrain::v2::WorkflowReadModelV2& workflow : model.workflows) {
        for (const aitrain::v2::WorkflowStepSnapshotV2& step : workflow.steps) {
            details.workflowSteps.append({workflow.run.id.toString(), workflow.run.templateId,
                step.ordinal, step.kind, aitrain::v2::workflowStepStateToString(step.state),
                step.backend, step.outputArtifactId.toString()});
        }
    }

    details.summary = QStringLiteral("V2 任务 %1：%2 / %3 / %4，%5 个已提交产物文件，%6 个指标点，%7 个工作流步骤")
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

void TaskArtifactPresenterV2::clearSelection()
{
    details_ = TaskArtifactDetailsV2();
    emit detailsChanged();
}

int TaskArtifactPresenterV2::taskCount() const { return taskRows_.size(); }
QString TaskArtifactPresenterV2::selectedTaskId() const { return details_.taskId; }
int TaskArtifactPresenterV2::artifactCount() const { return details_.artifacts.size(); }
int TaskArtifactPresenterV2::metricCount() const { return details_.metrics.size(); }
int TaskArtifactPresenterV2::workflowStepCount() const { return details_.workflowSteps.size(); }
QString TaskArtifactPresenterV2::lastError() const { return lastError_; }
const QVector<TaskListItemV2>& TaskArtifactPresenterV2::taskRows() const { return taskRows_; }
const TaskArtifactDetailsV2& TaskArtifactPresenterV2::details() const { return details_; }

void TaskArtifactPresenterV2::fail(const QString& error)
{
    lastError_ = error;
    emit queryFailed(error);
}
