#pragma once

#include <QDateTime>
#include <QJsonObject>
#include <QString>

namespace aitrain {

enum class TaskKind {
    Train,
    Validate,
    Export,
    Infer
};

enum class TaskState {
    Queued,
    Running,
    Paused,
    Completed,
    Failed,
    Canceled
};

QString taskKindToString(TaskKind kind);
TaskKind taskKindFromString(const QString& value);

QString taskStateToString(TaskState state);
TaskState taskStateFromString(const QString& value);
bool isTerminalTaskState(TaskState state);
bool isValidTaskStateTransition(TaskState from, TaskState to);

struct TrainingRequest {
    QString taskId;
    QString projectPath;
    QString pluginId;
    QString taskType;
    QString datasetPath;
    QString outputPath;
    QJsonObject parameters;

    QJsonObject toJson() const;
    static TrainingRequest fromJson(const QJsonObject& object);
};

struct TaskRecord {
    QString id;
    QString projectName;
    QString pluginId;
    QString taskType;
    TaskKind kind = TaskKind::Train;
    TaskState state = TaskState::Queued;
    QString workDir;
    QString message;
    QDateTime createdAt;
    QDateTime updatedAt;
    QDateTime startedAt;
    QDateTime finishedAt;
};

struct MetricPoint {
    QString taskId;
    QString name;
    double value = 0.0;
    int step = 0;
    int epoch = 0;
    QDateTime createdAt;
};

struct ArtifactRecord {
    QString taskId;
    QString kind;
    QString path;
    QString message;
    QDateTime createdAt;
};

struct EnvironmentCheckRecord {
    QString name;
    QString status;
    QString message;
    QString detailsJson;
    QDateTime checkedAt;
};

} // namespace aitrain
