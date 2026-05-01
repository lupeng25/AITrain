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
    int id = 0;
    QString taskId;
    QString name;
    double value = 0.0;
    int step = 0;
    int epoch = 0;
    QDateTime createdAt;
};

struct ArtifactRecord {
    int id = 0;
    QString taskId;
    QString kind;
    QString path;
    QString message;
    QDateTime createdAt;
};

struct ExportRecord {
    int id = 0;
    QString taskId;
    QString sourceCheckpointPath;
    QString format;
    QString path;
    QString configJson;
    QString inputShapeJson;
    QString outputShapeJson;
    QDateTime createdAt;
};

struct EnvironmentCheckRecord {
    QString name;
    QString status;
    QString message;
    QString detailsJson;
    QDateTime checkedAt;
};

struct DatasetRecord {
    int id = 0;
    QString name;
    QString format;
    QString rootPath;
    QString validationStatus;
    int sampleCount = 0;
    QString lastReportJson;
    QDateTime createdAt;
    QDateTime updatedAt;
    QDateTime lastValidatedAt;
};

struct DatasetVersionRecord {
    int id = 0;
    int datasetId = 0;
    QString version;
    QString rootPath;
    QString metadataJson;
    QDateTime createdAt;
};

} // namespace aitrain
