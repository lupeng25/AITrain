#pragma once

#include <QDateTime>
#include <QJsonObject>
#include <QString>

namespace aitrain {

enum class TaskKind {
    Train,
    Validate,
    Export,
    Infer,
    Evaluate,
    Benchmark,
    Curate,
    Snapshot,
    Pipeline,
    Report
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

struct ExperimentRecord {
    int id = 0;
    QString name;
    QString taskType;
    int datasetId = 0;
    QString notes;
    QString tagsJson;
    QDateTime createdAt;
    QDateTime updatedAt;
};

struct ExperimentRunRecord {
    int id = 0;
    int experimentId = 0;
    QString taskId;
    QString trainingBackend;
    QString modelPreset;
    int datasetSnapshotId = 0;
    QString requestJson;
    QString environmentJson;
    QString bestMetricsJson;
    QString artifactSummaryJson;
    QDateTime createdAt;
    QDateTime updatedAt;
};

struct DatasetSnapshotRecord {
    int id = 0;
    int datasetId = 0;
    QString name;
    QString rootPath;
    QString manifestPath;
    QString contentHash;
    int fileCount = 0;
    qint64 totalBytes = 0;
    QString metadataJson;
    QDateTime createdAt;
};

struct ModelVersionRecord {
    int id = 0;
    QString modelName;
    QString version;
    QString sourceTaskId;
    int experimentRunId = 0;
    int datasetSnapshotId = 0;
    QString checkpointPath;
    QString onnxPath;
    QString tensorRtEnginePath;
    int evaluationReportId = 0;
    QString status;
    QString notes;
    QString metricsJson;
    QDateTime createdAt;
    QDateTime updatedAt;
};

struct EvaluationReportRecord {
    int id = 0;
    QString taskId;
    QString modelPath;
    QString taskType;
    int datasetSnapshotId = 0;
    QString reportPath;
    QString summaryJson;
    QDateTime createdAt;
};

struct PipelineRunRecord {
    int id = 0;
    QString name;
    QString templateId;
    QString taskIdsJson;
    QString state;
    QString summaryJson;
    QDateTime createdAt;
    QDateTime updatedAt;
};

} // namespace aitrain
