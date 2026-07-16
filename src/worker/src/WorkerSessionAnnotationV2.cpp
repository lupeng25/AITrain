#include "WorkerSession.h"

#include "aitrain/core/WorkerProtocol.h"

#include <QDateTime>
#include <QFileInfo>
#include <QJsonObject>

namespace wp = aitrain::worker_protocol;

namespace {

QString annotationStatusText(aitrain::v2::AnnotationSyncStatusV2 status)
{
    switch (status) {
    case aitrain::v2::AnnotationSyncStatusV2::Inspected: return QStringLiteral("inspected");
    case aitrain::v2::AnnotationSyncStatusV2::ChangesDetected: return QStringLiteral("changes_detected");
    case aitrain::v2::AnnotationSyncStatusV2::NoChanges: return QStringLiteral("no_changes");
    case aitrain::v2::AnnotationSyncStatusV2::InvalidSession: return QStringLiteral("invalid_session");
    case aitrain::v2::AnnotationSyncStatusV2::Conflict: return QStringLiteral("conflict");
    case aitrain::v2::AnnotationSyncStatusV2::Canceled: return QStringLiteral("canceled");
    }
    return QStringLiteral("invalid_session");
}

aitrain::v2::Failure workerFailure(const QString& message)
{
    return {aitrain::v2::FailureCode::InternalError, message,
        QStringLiteral("检查项目工作区与 Artifact 输入后重新执行。"), QDateTime::currentDateTimeUtc()};
}

} // namespace

void WorkerSession::createAnnotationSessionV2(const QJsonObject& payload)
{
    if (running_ || annotationWorkspaceV2_) {
        fail(QStringLiteral("Worker 已有运行任务，不能并发创建标注会话。"));
        return;
    }
    const QString taskIdText = payload.value(wp::field::taskId()).toString().trimmed();
    const QString projectRoot = payload.value(QStringLiteral("projectRoot")).toString().trimmed();
    const QString repairArtifactText = payload.value(QStringLiteral("repairManifestArtifactId")).toString().trimmed();
    const QString workingDirectory = payload.value(QStringLiteral("workingDirectory")).toString().trimmed();
    const QJsonObject toolSummary = payload.value(QStringLiteral("toolSummary")).toObject();
    const QJsonObject options = payload.value(wp::field::options()).toObject();
    QString error;
    aitrain::v2::ArtifactId repairArtifactId;
    if (!aitrain::v2::TaskId::parse(taskIdText, &annotationTaskIdV2_, &error)
        || annotationTaskIdV2_ != controlTaskId_
        || !aitrain::v2::ArtifactId::parse(repairArtifactText, &repairArtifactId, &error)
        || projectRoot.isEmpty() || workingDirectory.isEmpty() || !QFileInfo(projectRoot).isDir()) {
        annotationTaskIdV2_ = {};
        fail(QStringLiteral("创建标注会话需要有效项目、Repair ArtifactId、独立工作目录和一致的 Protocol V2 TaskId。"));
        return;
    }

    annotationWorkspaceV2_ = std::make_unique<aitrain::v2::ProjectWorkspaceV2>();
    if (!annotationWorkspaceV2_->open(projectRoot, &error)) {
        annotationWorkspaceV2_.reset();
        annotationTaskIdV2_ = {};
        fail(QStringLiteral("无法打开 Annotation Session V2 工作区：%1").arg(error));
        return;
    }
    aitrain::v2::TaskSnapshot task;
    if (!annotationWorkspaceV2_->startTask(annotationTaskIdV2_,
            QStringLiteral("dataset.annotation.session.v2"), QStringLiteral("annotation_session_create"),
            &task, &error)) {
        annotationWorkspaceV2_.reset();
        annotationTaskIdV2_ = {};
        fail(QStringLiteral("无法创建 Annotation Session V2 根任务：%1").arg(error));
        return;
    }

    activeTaskId_ = taskIdText;
    canceled_ = false;
    running_ = true;
    annotationRunningV2_ = true;
    send(wp::event::progress(), QJsonObject{{wp::field::taskId(), taskIdText},
        {QStringLiteral("percent"), 0},
        {wp::field::message(), QStringLiteral("Annotation Session V2 正在校验 Repair Artifact 并准备受控工作副本。")}});

    aitrain::v2::AnnotationSessionCreateRequestV2 request;
    request.repairManifestArtifactId = repairArtifactId;
    request.workingDirectory = workingDirectory;
    request.toolParameters = toolSummary;
    if (!options.isEmpty()) request.toolParameters.insert(QStringLiteral("options"), options);
    aitrain::v2::AnnotationSessionCreateResultV2 result;
    const bool executed = annotationWorkspaceV2_->createAnnotationSession(annotationTaskIdV2_, request,
        &result, &error, pollingCancellationCallback(0));
    annotationRunningV2_ = false;

    if (!executed) {
        aitrain::v2::TaskSnapshot stored;
        if (annotationWorkspaceV2_->task(annotationTaskIdV2_, &stored, nullptr)
            && !aitrain::v2::isTerminalTaskState(stored.state)) {
            annotationWorkspaceV2_->finalizeTask(annotationTaskIdV2_, aitrain::v2::TaskState::Failed,
                workerFailure(error.isEmpty() ? QStringLiteral("Annotation Session V2 执行失败。") : error), nullptr);
        }
        annotationWorkspaceV2_.reset();
        annotationTaskIdV2_ = {};
        running_ = false;
        failWithDetails(QStringLiteral("Annotation Session V2 执行失败：%1").arg(error),
            QStringLiteral("annotation_session_v2_execution_failed"));
        return;
    }

    QJsonObject response{{wp::field::taskId(), taskIdText},
        {QStringLiteral("workflowRunId"), result.workflowRunId.toString()},
        {QStringLiteral("state"), aitrain::v2::taskStateToString(result.terminalState)},
        {QStringLiteral("status"), annotationStatusText(result.status)},
        {QStringLiteral("sessionArtifactId"), result.sessionArtifactId.toString()},
        {QStringLiteral("evidenceArtifactId"), result.evidenceArtifactId.toString()}};
    send(wp::event::annotationSessionV2(), response);

    const aitrain::v2::TaskState terminalState = result.terminalState;
    annotationWorkspaceV2_.reset();
    annotationTaskIdV2_ = {};
    running_ = false;
    if (terminalState == aitrain::v2::TaskState::Canceled) {
        sendCanceledAndFinish(taskIdText, QStringLiteral("Annotation Session V2 已取消。"));
    } else if (terminalState == aitrain::v2::TaskState::Failed) {
        failWithDetails(QStringLiteral("Annotation Session V2 创建失败。"),
            QStringLiteral("annotation_session_v2_failed"), response);
    } else {
        send(wp::event::completed(), QJsonObject{{wp::field::taskId(), taskIdText},
            {wp::field::message(), QStringLiteral("Annotation Session V2 创建完成。")}});
        finishSession();
    }
}

void WorkerSession::syncAnnotationSessionV2(const QJsonObject& payload)
{
    if (running_ || annotationWorkspaceV2_) {
        fail(QStringLiteral("Worker 已有运行任务，不能并发同步标注会话。"));
        return;
    }
    const QString taskIdText = payload.value(wp::field::taskId()).toString().trimmed();
    const QString projectRoot = payload.value(QStringLiteral("projectRoot")).toString().trimmed();
    const QString sessionArtifactText = payload.value(QStringLiteral("sessionArtifactId")).toString().trimmed();
    const QString workingDirectory = payload.value(QStringLiteral("workingDirectory")).toString().trimmed();
    const QJsonObject options = payload.value(wp::field::options()).toObject();
    Q_UNUSED(options);
    QString error;
    aitrain::v2::ArtifactId sessionArtifactId;
    if (!aitrain::v2::TaskId::parse(taskIdText, &annotationTaskIdV2_, &error)
        || annotationTaskIdV2_ != controlTaskId_
        || !aitrain::v2::ArtifactId::parse(sessionArtifactText, &sessionArtifactId, &error)
        || projectRoot.isEmpty() || workingDirectory.isEmpty() || !QFileInfo(projectRoot).isDir()) {
        annotationTaskIdV2_ = {};
        fail(QStringLiteral("同步标注会话需要有效项目、Session ArtifactId、独立工作目录和一致的 Protocol V2 TaskId。"));
        return;
    }

    annotationWorkspaceV2_ = std::make_unique<aitrain::v2::ProjectWorkspaceV2>();
    if (!annotationWorkspaceV2_->open(projectRoot, &error)) {
        annotationWorkspaceV2_.reset();
        annotationTaskIdV2_ = {};
        fail(QStringLiteral("无法打开 Annotation Sync V2 工作区：%1").arg(error));
        return;
    }
    aitrain::v2::TaskSnapshot task;
    if (!annotationWorkspaceV2_->startTask(annotationTaskIdV2_,
            QStringLiteral("dataset.annotation.sync.v2"), QStringLiteral("annotation_session_sync"),
            &task, &error)) {
        annotationWorkspaceV2_.reset();
        annotationTaskIdV2_ = {};
        fail(QStringLiteral("无法创建 Annotation Sync V2 根任务：%1").arg(error));
        return;
    }

    activeTaskId_ = taskIdText;
    canceled_ = false;
    running_ = true;
    annotationRunningV2_ = true;
    send(wp::event::progress(), QJsonObject{{wp::field::taskId(), taskIdText},
        {QStringLiteral("percent"), 0},
        {wp::field::message(), QStringLiteral("Annotation Sync V2 正在重验基线、白名单、文件集合和哈希。")}});

    aitrain::v2::AnnotationSessionSyncRequestV2 request;
    request.sessionArtifactId = sessionArtifactId;
    request.workingDirectory = workingDirectory;
    aitrain::v2::AnnotationSessionSyncResultV2 result;
    const bool executed = annotationWorkspaceV2_->syncAnnotationSession(annotationTaskIdV2_, request,
        &result, &error, pollingCancellationCallback(0));
    annotationRunningV2_ = false;

    if (!executed) {
        aitrain::v2::TaskSnapshot stored;
        if (annotationWorkspaceV2_->task(annotationTaskIdV2_, &stored, nullptr)
            && !aitrain::v2::isTerminalTaskState(stored.state)) {
            annotationWorkspaceV2_->finalizeTask(annotationTaskIdV2_, aitrain::v2::TaskState::Failed,
                workerFailure(error.isEmpty() ? QStringLiteral("Annotation Sync V2 执行失败。") : error), nullptr);
        }
        annotationWorkspaceV2_.reset();
        annotationTaskIdV2_ = {};
        running_ = false;
        failWithDetails(QStringLiteral("Annotation Sync V2 执行失败：%1").arg(error),
            QStringLiteral("annotation_sync_v2_execution_failed"));
        return;
    }

    QJsonObject response{{wp::field::taskId(), taskIdText},
        {QStringLiteral("workflowRunId"), result.workflowRunId.toString()},
        {QStringLiteral("state"), aitrain::v2::taskStateToString(result.terminalState)},
        {QStringLiteral("status"), annotationStatusText(result.status)},
        {QStringLiteral("inspectionArtifactId"), result.inspectionArtifactId.toString()},
        {QStringLiteral("changesArtifactId"), result.changesArtifactId.toString()},
        {QStringLiteral("syncReportArtifactId"), result.syncReportArtifactId.toString()},
        {QStringLiteral("evidenceArtifactId"), result.evidenceArtifactId.toString()},
        {QStringLiteral("newSnapshotId"), result.datasetSnapshot.id.toString()},
        {QStringLiteral("newDatasetVersionId"), result.datasetSnapshot.datasetVersionId.toString()},
        {QStringLiteral("newSnapshotArtifactId"), result.datasetSnapshot.artifactId.toString()},
        {QStringLiteral("newDatasetVersionCreated"), result.datasetSnapshot.id.isValid()}};
    send(wp::event::annotationSyncV2(), response);

    const aitrain::v2::TaskState terminalState = result.terminalState;
    annotationWorkspaceV2_.reset();
    annotationTaskIdV2_ = {};
    running_ = false;
    if (terminalState == aitrain::v2::TaskState::Canceled) {
        sendCanceledAndFinish(taskIdText, QStringLiteral("Annotation Sync V2 已取消。"));
    } else if (terminalState == aitrain::v2::TaskState::Failed) {
        failWithDetails(QStringLiteral("Annotation Sync V2 失败或存在冲突。"),
            QStringLiteral("annotation_sync_v2_failed"), response);
    } else {
        send(wp::event::completed(), QJsonObject{{wp::field::taskId(), taskIdText},
            {wp::field::message(), QStringLiteral("Annotation Sync V2 完成。")}});
        finishSession();
    }
}
