#pragma once

#include "aitrain/workflow/ProjectWorkspace.h"

#include <functional>

class QObject;

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
    QString artifactNextCursor;
    QString metricNextCursor;
    QString workflowNextCursor;
    bool artifactsHasMore = false;
    bool metricsHasMore = false;
    bool workflowsHasMore = false;
};

// 模型库 Presenter 的只读模型。只暴露已持久化身份、lineage 与模型合同摘要，
// 不暴露 Artifact Store 位置、模型入口相对路径或任何用户文件路径。
struct ModelPackageReadModel final {
    ModelPackageId modelPackageId;
    TaskId latestValidationTaskId;
    QString latestValidationState;
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
    QString displayName;
    QString datasetFormat;
    qint64 versionCount = 0;
    qint64 snapshotCount = 0;
    DatasetVersionId latestVersionId;
    SnapshotId latestSnapshotId;
    ArtifactId latestArtifactId;
    QString latestRootHash;
    qsizetype latestFileCount = 0;
    QDateTime latestCreatedAt;
    TaskId latestSourceTaskId;
    TaskId latestQualityTaskId;
};

// 版本选择器只获得同一条记录的完整身份，不暴露快照的物理目录。
struct DatasetSnapshotReadModel final {
    DatasetId datasetId;
    DatasetVersionId datasetVersionId;
    SnapshotId snapshotId;
    ArtifactId artifactId;
    TaskId sourceTaskId;
    QString datasetFormat;
    QDateTime createdAt;
    qint64 fileCount = 0;
    TaskId latestQualityTaskId;
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

using DeliveryEvidenceCallback = std::function<void(
    bool success, Page<DeliveryEvidenceReadModel> page, QString error)>;

// 项目总览只读 DTO。底层 Snapshot 本身不包含裸 Artifact 路径或 legacy 数据。
using ProjectSummaryReadModel = ProjectSummarySnapshot;

// GUI Presenter 的只读项目边界。页面只能获得已持久化的 Task、Metric、
// committed Artifact 与 Workflow Step，不能看到 staging，也不能写状态。
class ProjectQueryService final {
public:
    explicit ProjectQueryService(const ProjectWorkspace* workspace);

    Page<TaskSnapshot> recentTasks(
        const PageRequest& request, QString* error = nullptr, const CatalogFilter& filter = {}) const;
    bool taskDetails(const TaskId& taskId, TaskReadModel* result, QString* error = nullptr) const;
    Page<ArtifactSnapshot> taskArtifacts(const TaskId& taskId,
        const PageRequest& request, QString* error = nullptr) const;
    QString projectIdentity(QString* error = nullptr) const;
    Page<ArtifactFileSnapshot> artifactFiles(const ArtifactId& artifactId,
        const PageRequest& request, QString* error = nullptr) const;
    Page<MetricSnapshot> taskMetrics(const TaskId& taskId,
        const PageRequest& request, QString* error = nullptr) const;
    Page<WorkflowReadModel> taskWorkflows(const TaskId& taskId,
        const PageRequest& request, QString* error = nullptr) const;
    bool artifactFilePreview(const ArtifactId& artifactId,
        const QString& relativePath,
        ArtifactFilePreview* result,
        qint64 maxBytes = 512 * 1024,
        QString* error = nullptr) const;
    bool artifactFilePreviewAsync(const ArtifactId& artifactId,
        const QString& relativePath,
        QObject* receiver,
        ArtifactFilePreviewCallback callback,
        qint64 maxBytes = 512 * 1024,
        QString* error = nullptr) const;
    Page<DatasetCatalogReadModel> datasetCatalog(
        const PageRequest& request, QString* error = nullptr, const CatalogFilter& filter = {}) const;
    Page<DatasetSnapshotReadModel> datasetSnapshots(const DatasetId& datasetId,
        const PageRequest& request, QString* error = nullptr) const;
    Page<ArtifactSnapshot> artifactCatalog(const QStringList& kinds,
        const PageRequest& request, QString* error = nullptr) const;
    Page<ModelPackageReadModel> modelPackages(
        const PageRequest& request, QString* error = nullptr, const CatalogFilter& filter = {}) const;
    bool projectSummary(ProjectSummaryReadModel* result, QString* error = nullptr) const;
    bool environmentCheckReport(const TaskId& taskId, QJsonObject* result, QString* error = nullptr) const;
    Page<DeliveryEvidenceReadModel> deliveryEvidence(
        const PageRequest& request, QString* error = nullptr, const CatalogFilter& filter = {}) const;
    // 当前线程只读取候选 Task/Artifact 元数据；证据文件的读取、SHA-256
    // 复验和 JSON/schema 校验均在线程池执行，完成后回到 receiver 线程。
    bool deliveryEvidenceAsync(const PageRequest& request,
        QObject* receiver,
        DeliveryEvidenceCallback callback,
        QString* error = nullptr, const CatalogFilter& filter = {}) const;

private:
    const ProjectWorkspace* workspace_ = nullptr;
};

} // namespace aitrain
