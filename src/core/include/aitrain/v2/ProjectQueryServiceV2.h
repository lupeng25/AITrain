#pragma once

#include "aitrain/v2/ProjectWorkspaceV2.h"

namespace aitrain::v2 {

struct WorkflowReadModelV2 final {
    WorkflowRunSnapshotV2 run;
    QVector<WorkflowStepSnapshotV2> steps;
};

struct TaskReadModelV2 final {
    TaskSnapshot task;
    QVector<ArtifactSnapshotV2> artifacts;
    QVector<MetricSnapshotV2> metrics;
    QVector<WorkflowReadModelV2> workflows;
};

// 模型库 Presenter 的只读模型。只暴露已持久化身份、lineage 与模型合同摘要，
// 不暴露 Artifact Store 位置、模型入口相对路径或任何用户文件路径。
struct ModelPackageReadModelV2 final {
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

// 项目总览只读 DTO。底层 Snapshot 本身不包含裸 Artifact 路径或 legacy 数据。
using ProjectSummaryReadModelV2 = ProjectSummarySnapshotV2;

// GUI Presenter 的只读项目边界。页面只能获得已持久化的 Task、Metric、
// committed Artifact 与 Workflow Step，不能看到 staging，也不能写状态。
class ProjectQueryServiceV2 final {
public:
    explicit ProjectQueryServiceV2(const ProjectWorkspaceV2* workspace);

    QVector<TaskSnapshot> recentTasks(int limit, QString* error = nullptr) const;
    bool taskDetails(const TaskId& taskId, TaskReadModelV2* result, QString* error = nullptr) const;
    QVector<ModelPackageReadModelV2> modelPackages(int limit, QString* error = nullptr) const;
    bool projectSummary(ProjectSummaryReadModelV2* result, QString* error = nullptr) const;
    bool environmentCheckReport(const TaskId& taskId, QJsonObject* result, QString* error = nullptr) const;

private:
    const ProjectWorkspaceV2* workspace_ = nullptr;
};

} // namespace aitrain::v2
