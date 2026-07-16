#include "aitrain/v2/ProjectQueryServiceV2.h"

namespace aitrain::v2 {

ProjectQueryServiceV2::ProjectQueryServiceV2(const ProjectWorkspaceV2* workspace)
    : workspace_(workspace)
{
}

QVector<TaskSnapshot> ProjectQueryServiceV2::recentTasks(int limit, QString* error) const
{
    if (error) error->clear();
    if (!workspace_ || !workspace_->isOpen()) {
        if (error) *error = QStringLiteral("项目查询服务需要已打开的 V2 工作区。");
        return {};
    }
    return workspace_->tasks(limit, error);
}

bool ProjectQueryServiceV2::taskDetails(const TaskId& taskId, TaskReadModelV2* result, QString* error) const
{
    if (error) error->clear();
    if (!workspace_ || !workspace_->isOpen() || !result || !taskId.isValid()) {
        if (error) *error = QStringLiteral("查询任务详情需要已打开工作区、有效任务 ID 和输出对象。");
        return false;
    }

    TaskReadModelV2 model;
    if (!workspace_->task(taskId, &model.task, error)) return false;
    model.artifacts = workspace_->artifactsForTask(taskId, error);
    if (error && !error->isEmpty()) return false;
    model.metrics = workspace_->metricsForTask(taskId, error);
    if (error && !error->isEmpty()) return false;

    const QVector<WorkflowRunSnapshotV2> runs = workspace_->workflowRunsForTask(taskId, error);
    if (error && !error->isEmpty()) return false;
    for (const WorkflowRunSnapshotV2& run : runs) {
        WorkflowReadModelV2 workflow;
        workflow.run = run;
        workflow.steps = workspace_->workflowSteps(run.id, error);
        if (error && !error->isEmpty()) return false;
        model.workflows.append(workflow);
    }
    *result = model;
    return true;
}

QVector<ModelPackageReadModelV2> ProjectQueryServiceV2::modelPackages(
    int limit, QString* error) const
{
    if (error) error->clear();
    if (!workspace_ || !workspace_->isOpen()) {
        if (error) *error = QStringLiteral("查询模型包需要已打开的 V2 工作区。");
        return {};
    }

    const QVector<ModelPackageSnapshotV2> snapshots = workspace_->modelPackages(limit, error);
    if (error && !error->isEmpty()) return {};

    QVector<ModelPackageReadModelV2> models;
    models.reserve(snapshots.size());
    for (const ModelPackageSnapshotV2& snapshot : snapshots) {
        const ModelManifestV2& manifest = snapshot.manifest;
        ModelPackageReadModelV2 model;
        model.modelPackageId = manifest.modelPackageId;
        model.sourceTaskId = manifest.sourceTaskId;
        model.sourceSnapshotId = manifest.sourceSnapshotId;
        model.sourceArtifactId = snapshot.sourceArtifactId;
        model.sourceArtifactSha256 = manifest.sourceArtifactSha256;
        model.modelFamily = manifest.modelFamily;
        model.taskType = manifest.taskType;
        model.sourceBackend = manifest.sourceBackend;
        model.artifactFormat = manifest.artifactFormat;
        model.decoder = manifest.decoder;
        model.exporterVersion = manifest.exporterVersion;
        model.runtimeRoutes = manifest.runtimeRoutes;
        model.limitations = manifest.limitations;
        model.verified = manifest.verified;
        model.createdAt = snapshot.createdAt;
        models.append(model);
    }
    return models;
}

bool ProjectQueryServiceV2::projectSummary(ProjectSummaryReadModelV2* result, QString* error) const
{
    if (error) error->clear();
    if (!workspace_ || !workspace_->isOpen() || !result) {
        if (error) *error = QStringLiteral("查询项目汇总需要已打开的 V2 工作区和输出对象。");
        return false;
    }
    return workspace_->projectSummary(result, error);
}

bool ProjectQueryServiceV2::environmentCheckReport(
    const TaskId& taskId, QJsonObject* result, QString* error) const
{
    if (error) error->clear();
    if (!workspace_ || !workspace_->isOpen() || !result || !taskId.isValid()) {
        if (error) *error = QStringLiteral("查询环境检查报告需要已打开工作区、有效任务 ID 和输出对象。");
        return false;
    }
    return workspace_->environmentCheckReportForTask(taskId, result, error);
}

} // namespace aitrain::v2
