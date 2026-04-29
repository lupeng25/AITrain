#include "aitrain/core/TaskModels.h"

#include <QJsonValue>

namespace aitrain {

QString taskKindToString(TaskKind kind)
{
    switch (kind) {
    case TaskKind::Train: return QStringLiteral("train");
    case TaskKind::Validate: return QStringLiteral("validate");
    case TaskKind::Export: return QStringLiteral("export");
    case TaskKind::Infer: return QStringLiteral("infer");
    }
    return QStringLiteral("train");
}

TaskKind taskKindFromString(const QString& value)
{
    if (value == QStringLiteral("validate")) return TaskKind::Validate;
    if (value == QStringLiteral("export")) return TaskKind::Export;
    if (value == QStringLiteral("infer")) return TaskKind::Infer;
    return TaskKind::Train;
}

QString taskStateToString(TaskState state)
{
    switch (state) {
    case TaskState::Queued: return QStringLiteral("queued");
    case TaskState::Running: return QStringLiteral("running");
    case TaskState::Paused: return QStringLiteral("paused");
    case TaskState::Completed: return QStringLiteral("completed");
    case TaskState::Failed: return QStringLiteral("failed");
    case TaskState::Canceled: return QStringLiteral("canceled");
    }
    return QStringLiteral("queued");
}

TaskState taskStateFromString(const QString& value)
{
    if (value == QStringLiteral("running")) return TaskState::Running;
    if (value == QStringLiteral("paused")) return TaskState::Paused;
    if (value == QStringLiteral("completed")) return TaskState::Completed;
    if (value == QStringLiteral("failed")) return TaskState::Failed;
    if (value == QStringLiteral("canceled")) return TaskState::Canceled;
    return TaskState::Queued;
}

bool isTerminalTaskState(TaskState state)
{
    return state == TaskState::Completed
        || state == TaskState::Failed
        || state == TaskState::Canceled;
}

bool isValidTaskStateTransition(TaskState from, TaskState to)
{
    if (from == to) {
        return true;
    }

    switch (from) {
    case TaskState::Queued:
        return to == TaskState::Running || to == TaskState::Failed || to == TaskState::Canceled;
    case TaskState::Running:
        return to == TaskState::Paused
            || to == TaskState::Completed
            || to == TaskState::Failed
            || to == TaskState::Canceled;
    case TaskState::Paused:
        return to == TaskState::Running || to == TaskState::Failed || to == TaskState::Canceled;
    case TaskState::Completed:
    case TaskState::Failed:
    case TaskState::Canceled:
        return false;
    }
    return false;
}

QJsonObject TrainingRequest::toJson() const
{
    QJsonObject object;
    object.insert(QStringLiteral("taskId"), taskId);
    object.insert(QStringLiteral("projectPath"), projectPath);
    object.insert(QStringLiteral("pluginId"), pluginId);
    object.insert(QStringLiteral("taskType"), taskType);
    object.insert(QStringLiteral("datasetPath"), datasetPath);
    object.insert(QStringLiteral("outputPath"), outputPath);
    object.insert(QStringLiteral("parameters"), parameters);
    return object;
}

TrainingRequest TrainingRequest::fromJson(const QJsonObject& object)
{
    TrainingRequest request;
    request.taskId = object.value(QStringLiteral("taskId")).toString();
    request.projectPath = object.value(QStringLiteral("projectPath")).toString();
    request.pluginId = object.value(QStringLiteral("pluginId")).toString();
    request.taskType = object.value(QStringLiteral("taskType")).toString();
    request.datasetPath = object.value(QStringLiteral("datasetPath")).toString();
    request.outputPath = object.value(QStringLiteral("outputPath")).toString();
    request.parameters = object.value(QStringLiteral("parameters")).toObject();
    return request;
}

} // namespace aitrain
