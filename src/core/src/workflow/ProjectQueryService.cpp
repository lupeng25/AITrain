#include "aitrain/workflow/ProjectQueryService.h"

#include "aitrain/workflow/EvidenceBundle.h"

#include <QJsonDocument>

#include <algorithm>

namespace aitrain {

ProjectQueryService::ProjectQueryService(const ProjectWorkspace* workspace)
    : workspace_(workspace)
{
}

QVector<TaskSnapshot> ProjectQueryService::recentTasks(int limit, QString* error) const
{
    if (error) error->clear();
    if (!workspace_ || !workspace_->isOpen()) {
        if (error) *error = QStringLiteral("项目查询服务需要已打开的  工作区。");
        return {};
    }
    return workspace_->tasks(limit, error);
}

bool ProjectQueryService::taskDetails(const TaskId& taskId, TaskReadModel* result, QString* error) const
{
    if (error) error->clear();
    if (!workspace_ || !workspace_->isOpen() || !result || !taskId.isValid()) {
        if (error) *error = QStringLiteral("查询任务详情需要已打开工作区、有效任务 ID 和输出对象。");
        return false;
    }

    TaskReadModel model;
    if (!workspace_->task(taskId, &model.task, error)) return false;
    model.artifacts = workspace_->artifactsForTask(taskId, error);
    if (error && !error->isEmpty()) return false;
    model.metrics = workspace_->metricsForTask(taskId, error);
    if (error && !error->isEmpty()) return false;

    const QVector<WorkflowRunSnapshot> runs = workspace_->workflowRunsForTask(taskId, error);
    if (error && !error->isEmpty()) return false;
    for (const WorkflowRunSnapshot& run : runs) {
        WorkflowReadModel workflow;
        workflow.run = run;
        workflow.steps = workspace_->workflowSteps(run.id, error);
        if (error && !error->isEmpty()) return false;
        model.workflows.append(workflow);
    }
    *result = model;
    return true;
}

bool ProjectQueryService::artifactFilePreview(const ArtifactId& artifactId,
    const QString& relativePath,
    ArtifactFilePreview* result,
    qint64 maxBytes,
    QString* error) const
{
    if (error) error->clear();
    if (!workspace_ || !workspace_->isOpen() || !result) {
        if (error) *error = QStringLiteral("Artifact 预览查询需要已打开工作区和输出对象。");
        return false;
    }
    return workspace_->readCommittedArtifactFile(artifactId, relativePath, result, maxBytes, error);
}

QVector<DatasetCatalogReadModel> ProjectQueryService::datasetCatalog(
    int limit, QString* error) const
{
    if (error) error->clear();
    if (!workspace_ || !workspace_->isOpen()) {
        if (error) *error = QStringLiteral("查询数据集目录需要已打开的工作区。");
        return {};
    }

    const QVector<DatasetCatalogItem> items = workspace_->datasets(limit, error);
    if (error && !error->isEmpty()) return {};

    QVector<DatasetCatalogReadModel> models;
    models.reserve(items.size());
    for (const DatasetCatalogItem& item : items) {
        DatasetCatalogReadModel model;
        model.datasetId = item.datasetId;
        model.datasetFormat = item.datasetFormat;
        model.versionCount = item.versionCount;
        model.snapshotCount = item.snapshotCount;
        model.latestVersionId = item.latestVersionId;
        model.latestSnapshotId = item.latestSnapshotId;
        model.latestArtifactId = item.latestArtifactId;
        model.latestRootHash = item.latestRootHash;
        model.latestFileCount = item.latestFileCount;
        model.latestCreatedAt = item.latestCreatedAt;
        models.append(model);
    }
    return models;
}

QVector<ModelPackageReadModel> ProjectQueryService::modelPackages(
    int limit, QString* error) const
{
    if (error) error->clear();
    if (!workspace_ || !workspace_->isOpen()) {
        if (error) *error = QStringLiteral("查询模型包需要已打开的  工作区。");
        return {};
    }

    const QVector<ModelPackageSnapshot> snapshots = workspace_->modelPackages(limit, error);
    if (error && !error->isEmpty()) return {};

    QVector<ModelPackageReadModel> models;
    models.reserve(snapshots.size());
    for (const ModelPackageSnapshot& snapshot : snapshots) {
        const ModelManifest& manifest = snapshot.manifest;
        ModelPackageReadModel model;
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

bool ProjectQueryService::projectSummary(ProjectSummaryReadModel* result, QString* error) const
{
    if (error) error->clear();
    if (!workspace_ || !workspace_->isOpen() || !result) {
        if (error) *error = QStringLiteral("查询项目汇总需要已打开的  工作区和输出对象。");
        return false;
    }
    return workspace_->projectSummary(result, error);
}

bool ProjectQueryService::environmentCheckReport(
    const TaskId& taskId, QJsonObject* result, QString* error) const
{
    if (error) error->clear();
    if (!workspace_ || !workspace_->isOpen() || !result || !taskId.isValid()) {
        if (error) *error = QStringLiteral("查询环境检查报告需要已打开工作区、有效任务 ID 和输出对象。");
        return false;
    }
    return workspace_->environmentCheckReportForTask(taskId, result, error);
}

QVector<DeliveryEvidenceReadModel> ProjectQueryService::deliveryEvidence(
    int limit, QString* error) const
{
    if (error) error->clear();
    if (!workspace_ || !workspace_->isOpen() || limit <= 0) {
        if (error) *error = QStringLiteral("查询交付证据需要已打开工作区和正数 limit。");
        return {};
    }

    const QVector<DeliveryEvidenceCandidate> candidates =
        workspace_->deliveryEvidenceCandidates(limit, error);
    if (error && !error->isEmpty()) return {};

    QVector<DeliveryEvidenceReadModel> result;
    const auto appendInvalid = [&result](const DeliveryEvidenceCandidate& candidate,
        const QString& message) {
        DeliveryEvidenceReadModel model;
        model.taskId = candidate.task.id;
        model.evidenceArtifactId = candidate.artifact.id;
        model.taskState = taskStateToString(candidate.task.state);
        model.evidenceKind = candidate.artifact.kind;
        model.runtimeStatus = QStringLiteral("invalid");
        model.observedAt = candidate.artifact.createdAt;
        model.verified = false;
        model.valid = false;
        model.validationFailure = Failure{
            FailureCode::ArtifactIncomplete,
            message,
            QStringLiteral("重新生成或重新导入该证据 Artifact。"),
            QDateTime::currentDateTimeUtc()};
        model.limitations.append(message);
        result.append(model);
    };

    for (const DeliveryEvidenceCandidate& candidate : candidates) {
        const ArtifactSnapshot& artifact = candidate.artifact;
        const QString requiredFile = artifact.kind == QStringLiteral("external_acceptance_evidence")
            ? QStringLiteral("acceptance.json") : QStringLiteral("evidence.json");
        const auto fileIt = std::find_if(artifact.files.cbegin(), artifact.files.cend(),
            [&requiredFile](const ArtifactFileSnapshot& file) {
                return file.relativePath == requiredFile;
            });
        if (fileIt == artifact.files.cend()) {
            appendInvalid(candidate, QStringLiteral("证据 Artifact 缺少 %1：%2")
                .arg(requiredFile, artifact.id.toString()));
            continue;
        }

        ArtifactFilePreview preview;
        QString rowError;
        const qint64 maxBytes = artifact.kind == QStringLiteral("external_acceptance_evidence")
            ? 1024 * 1024 : 512 * 1024;
        if (!workspace_->readCommittedArtifactFile(artifact, fileIt->relativePath,
                &preview, maxBytes, &rowError)) {
            appendInvalid(candidate, QStringLiteral("证据 Artifact 文件无法读取或校验：%1")
                .arg(rowError));
            continue;
        }

        QJsonParseError parseError;
        const QJsonDocument document = QJsonDocument::fromJson(preview.content, &parseError);
        if (artifact.kind == QStringLiteral("external_acceptance_evidence")) {
            if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
                appendInvalid(candidate, QStringLiteral("外部验收 Artifact JSON 无效：%1")
                    .arg(artifact.id.toString()));
                continue;
            }
            const QJsonObject object = document.object();
            const QStringList allowed = {QStringLiteral("schemaVersion"), QStringLiteral("kind"),
                QStringLiteral("evidenceKind"), QStringLiteral("status"), QStringLiteral("producer"),
                QStringLiteral("observedAt"), QStringLiteral("message"), QStringLiteral("limitations")};
            QString schemaError;
            for (const QString& key : object.keys()) {
                if (!allowed.contains(key)) {
                    schemaError = QStringLiteral("外部验收 Artifact 包含未知字段：%1").arg(key);
                    break;
                }
            }
            const QString evidenceKind = object.value(QStringLiteral("evidenceKind")).toString().trimmed();
            const QString status = object.value(QStringLiteral("status")).toString().trimmed();
            const QString producer = object.value(QStringLiteral("producer")).toString().trimmed();
            const QDateTime observedAt = QDateTime::fromString(
                object.value(QStringLiteral("observedAt")).toString(), Qt::ISODate);
            const bool limitationsValid = !object.contains(QStringLiteral("limitations"))
                || object.value(QStringLiteral("limitations")).isArray();
            if (schemaError.isEmpty() && (object.value(QStringLiteral("schemaVersion")).toInt(-1) != 1
                || object.value(QStringLiteral("kind")).toString()
                    != QStringLiteral("aitrain_external_acceptance_evidence")
                || evidenceKind.isEmpty() || producer.isEmpty() || !observedAt.isValid()
                || !limitationsValid)) {
                schemaError = QStringLiteral("外部验收 Artifact schema 校验失败：%1").arg(artifact.id.toString());
            }
            if (schemaError.isEmpty() && !QStringList{QStringLiteral("passed"), QStringLiteral("failed"),
                    QStringLiteral("blocked"), QStringLiteral("collected"),
                    QStringLiteral("imported")}.contains(status)) {
                schemaError = QStringLiteral("外部验收 Artifact status 无效：%1").arg(artifact.id.toString());
            }
            if (!schemaError.isEmpty()) {
                appendInvalid(candidate, schemaError);
                continue;
            }
            DeliveryEvidenceReadModel model;
            model.taskId = candidate.task.id;
            model.evidenceArtifactId = artifact.id;
            model.taskState = taskStateToString(candidate.task.state);
            model.evidenceKind = evidenceKind;
            model.runtimeStatus = status;
            model.producer = producer;
            model.observedAt = observedAt.toUTC();
            model.verified = false;
            for (const QJsonValue& value : object.value(QStringLiteral("limitations")).toArray()) {
                if (!value.isString()) {
                    schemaError = QStringLiteral("外部验收 Artifact limitations 无效：%1")
                        .arg(artifact.id.toString());
                    break;
                }
                model.limitations.append(value.toString());
            }
            if (!schemaError.isEmpty()) {
                appendInvalid(candidate, schemaError);
                continue;
            }
            model.limitations.append(QStringLiteral("外部证据未经过 AITrain 内部生产验收证明，verified=false。"));
            result.append(model);
            continue;
        }

        EvidenceBundle bundle;
        QString decodeError;
        if (parseError.error != QJsonParseError::NoError || !document.isObject()
            || !decodeEvidenceBundle(document.object(), &bundle, &decodeError)) {
            appendInvalid(candidate, QStringLiteral("Evidence Artifact 无法验证：%1")
                .arg(decodeError.isEmpty() ? QStringLiteral("JSON 无效") : decodeError));
            continue;
        }
        DeliveryEvidenceReadModel model;
        model.taskId = candidate.task.id;
        model.evidenceArtifactId = artifact.id;
        model.taskState = taskStateToString(bundle.task.state);
        model.evidenceKind = QStringLiteral("aitrain_evidence_bundle");
        model.runtimeStatus = bundle.runtimeStatus.value(QStringLiteral("status")).toString();
        if (model.runtimeStatus.isEmpty()) {
            model.runtimeStatus = bundle.runtimeStatus.value(QStringLiteral("runtimeStatus")).toString();
        }
        model.producer = bundle.backendEnvironment.value(QStringLiteral("producer")).toString();
        if (model.producer.isEmpty()) model.producer = bundle.projectIdentity;
        model.limitations = bundle.limitations;
        model.observedAt = bundle.createdAt;
        model.verified = bundle.task.state == TaskState::Succeeded
            && bundle.task.failure.code == FailureCode::None;
        result.append(model);
    }
    return result;
}

} // namespace aitrain
