#pragma once

#include "aitrain/domain/DomainTypes.h"

#include <QJsonObject>
#include <QMetaType>
#include <QString>

#include <variant>

namespace aitrain {
namespace worker_protocol {

// TaskCommand 是 GUI/Worker 控制面上的唯一业务命令模型。Envelope 负责
// requestId/taskId/sequence，命令对象只描述具体 Workflow 的参数。
struct TaskCommandContext final {
    TaskId taskId;
    QString projectRoot;
};

struct EnvironmentCheckCommand final {
    TaskCommandContext context;
};

struct DatasetSplitCommand final {
    TaskCommandContext context;
    QString sourceDatasetId;
    QString sourceDatasetVersionId;
    QString sourceSnapshotId;
    QString sourceSnapshotArtifactId;
    QString targetDatasetId;
    QString targetDatasetName;
    QJsonObject options;
};

struct DatasetConversionCommand final {
    TaskCommandContext context;
    QString sourcePath;
    QString sourceFormat;
    QString targetFormat;
    QString targetDatasetId;
    QString targetDatasetName;
    QJsonObject options;
};

struct DataQualityCommand final {
    TaskCommandContext context;
    QString datasetId;
    QString datasetVersionId;
    QString snapshotId;
    QString snapshotArtifactId;
    QJsonObject options;
};

struct AnnotationSessionCreateCommand final {
    TaskCommandContext context;
    QString repairManifestArtifactId;
    QString workingDirectory;
    QJsonObject toolSummary;
    QJsonObject options;
};

struct AnnotationSessionSyncCommand final {
    TaskCommandContext context;
    QString sessionArtifactId;
    QString workingDirectory;
    QJsonObject options;
};

struct DatasetSnapshotImportCommand final {
    TaskCommandContext context;
    QString sourcePath;
    QString sourceFormat;
    QString targetDatasetId;
    QString targetDatasetName;
    QJsonObject options;
};

struct OcrOfficialReportImportCommand final {
    TaskCommandContext context;
    QJsonObject det;
    QJsonObject rec;
    QJsonObject system;
    QString acceptanceCohortId;
    QString customerDomainId;
    QString evidenceClass;
};

struct OcrAcceptanceCommand final {
    TaskCommandContext context;
    QString detReportArtifactId;
    QString recReportArtifactId;
    QString systemReportArtifactId;
    QJsonObject thresholds;
};

struct DiagnosticsCommand final {
    TaskCommandContext context;
    QJsonObject options;
};

struct ExternalAcceptanceEvidenceImportCommand final {
    TaskCommandContext context;
    QString sourcePath;
};

struct RuntimeDeliveryCommand final {
    TaskCommandContext context;
    QString modelPackageId;
    QString runtimeRoute;
    QString sampleDatasetId;
    QString sampleDatasetVersionId;
    QString sampleSnapshotId;
    QString sampleSnapshotArtifactId;
    QString sampleRelativePath;
    QJsonObject options;
};

struct ModelImportCommand final {
    TaskCommandContext context;
    QString sourceFilePath;
    QJsonObject manifestDraft;
};

struct TrainingCommand final {
    TaskCommandContext context;
    QString datasetId;
    QString datasetVersionId;
    QString snapshotId;
    QString snapshotArtifactId;
    QString capabilityId;
    QString taskType;
    QString trainingBackend;
    QString deploymentSampleRelativePath;
    QJsonObject parameters;
};

using TaskCommandPayload = std::variant<
    EnvironmentCheckCommand,
    DatasetSplitCommand,
    DatasetConversionCommand,
    DataQualityCommand,
    AnnotationSessionCreateCommand,
    AnnotationSessionSyncCommand,
    DatasetSnapshotImportCommand,
    OcrOfficialReportImportCommand,
    OcrAcceptanceCommand,
    DiagnosticsCommand,
    ExternalAcceptanceEvidenceImportCommand,
    RuntimeDeliveryCommand,
    ModelImportCommand,
    TrainingCommand>;

struct TaskCommand final {
    TaskCommandPayload payload;
};

enum class TaskEventKind {
    Ready,
    Log,
    Progress,
    Metric,
    Artifact,
    Succeeded,
    Failed,
    Canceled,
    Result,
};

struct TaskEvent final {
    TaskEventKind kind = TaskEventKind::Log;
    QString resultType;
    // 事件内容由 WorkerProtocol 的 codec 按 kind 校验；该对象不是协议中的
    // generic business payload，且不会在 envelope 外再套一层名称/类型字段。
    QJsonObject details;
};

} // namespace worker_protocol
} // namespace aitrain

Q_DECLARE_METATYPE(aitrain::worker_protocol::TaskCommand)
Q_DECLARE_METATYPE(aitrain::worker_protocol::TaskEvent)
