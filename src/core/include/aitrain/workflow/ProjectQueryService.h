#pragma once

#include "aitrain/workflow/ProjectWorkspace.h"

namespace aitrain {

struct WorkflowReadModel final {
    WorkflowRunSnapshot run;
    QVector<WorkflowStepSnapshot> steps;
};

struct TaskReadModel final {
    TaskSnapshot task;
    QVector<ArtifactSnapshot> artifacts;
    QVector<MetricSnapshot> metrics;
    QVector<WorkflowReadModel> workflows;
};

// 模型库 Presenter 的只读模型。只暴露已持久化身份、lineage 与模型合同摘要，
// 不暴露 Artifact Store 位置、模型入口相对路径或任何用户文件路径。
struct ModelPackageReadModel final {
    ModelPackageId modelPackageId;
    TaskId sourceTaskId;
    SnapshotId sourceSnapshotId;
    ArtifactId sourceArtifactId;
    QString sourceArtifactSha256;
    QString modelFamily;
    QString taskType;
    QString sourceBackend;
    QString artifactFormat;
    QString decoder;
    QString exporterVersion;
    QStringList runtimeRoutes;
    QStringList limitations;
    bool verified = false;
    QDateTime createdAt;
};

// 数据集目录只暴露持久化身份和摘要，不暴露源目录或 Artifact Store 路径。
struct DatasetCatalogReadModel final {
    DatasetId datasetId;
    QString datasetFormat;
    qint64 versionCount = 0;
    qint64 snapshotCount = 0;
    DatasetVersionId latestVersionId;
    SnapshotId latestSnapshotId;
    ArtifactId latestArtifactId;
    QString latestRootHash;
    qsizetype latestFileCount = 0;
    QDateTime latestCreatedAt;
};

struct DeliveryEvidenceReadModel final {
    TaskId taskId;
    ArtifactId evidenceArtifactId;
    QString taskState;
    QString evidenceKind;
    QString runtimeStatus;
    QString producer;
    QStringList limitations;
    QDateTime observedAt;
    bool verified = false;
    // Artifact 已进入证据索引但内容无法验证时，保留一条可见的无效记录，
    // 避免单条损坏证据清空整个交付面板。
    bool valid = true;
    Failure validationFailure;
};

// 项目总览只读 DTO。底层 Snapshot 本身不包含裸 Artifact 路径或 legacy 数据。
using ProjectSummaryReadModel = ProjectSummarySnapshot;

// GUI Presenter 的只读项目边界。页面只能获得已持久化的 Task、Metric、
// committed Artifact 与 Workflow Step，不能看到 staging，也不能写状态。
class ProjectQueryService final {
public:
    explicit ProjectQueryService(const ProjectWorkspace* workspace);

    QVector<TaskSnapshot> recentTasks(int limit, QString* error = nullptr) const;
    bool taskDetails(const TaskId& taskId, TaskReadModel* result, QString* error = nullptr) const;
    bool artifactFilePreview(const ArtifactId& artifactId,
        const QString& relativePath,
        ArtifactFilePreview* result,
        qint64 maxBytes = 512 * 1024,
        QString* error = nullptr) const;
    QVector<DatasetCatalogReadModel> datasetCatalog(int limit, QString* error = nullptr) const;
    QVector<ModelPackageReadModel> modelPackages(int limit, QString* error = nullptr) const;
    bool projectSummary(ProjectSummaryReadModel* result, QString* error = nullptr) const;
    bool environmentCheckReport(const TaskId& taskId, QJsonObject* result, QString* error = nullptr) const;
    QVector<DeliveryEvidenceReadModel> deliveryEvidence(int limit, QString* error = nullptr) const;

private:
    const ProjectWorkspace* workspace_ = nullptr;
};

} // namespace aitrain
