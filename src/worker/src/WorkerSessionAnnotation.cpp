#include "WorkerSession.h"

#include "aitrain/core/WorkerProtocol.h"

#include <QDateTime>
#include <QFileInfo>
#include <QJsonObject>

namespace wp = aitrain::worker_protocol;

namespace {

QString annotationStatusText(aitrain::AnnotationSyncStatus status)
{
    switch (status) {
    case aitrain::AnnotationSyncStatus::Inspected: return QStringLiteral("inspected");
    case aitrain::AnnotationSyncStatus::ChangesDetected: return QStringLiteral("changes_detected");
    case aitrain::AnnotationSyncStatus::NoChanges: return QStringLiteral("no_changes");
    case aitrain::AnnotationSyncStatus::InvalidSession: return QStringLiteral("invalid_session");
    case aitrain::AnnotationSyncStatus::Conflict: return QStringLiteral("conflict");
    case aitrain::AnnotationSyncStatus::Canceled: return QStringLiteral("canceled");
    }
    return QStringLiteral("invalid_session");
}

aitrain::Failure workerFailure(const QString& message)
{
    return {aitrain::FailureCode::InternalError, message,
        QStringLiteral("检查项目工作区与 Artifact 输入后重新执行。"), QDateTime::currentDateTimeUtc()};
}

} // namespace

void WorkerSession::createAnnotationSession(const QJsonObject& payload)
{
    if (running_ || annotationWorkspace_) {
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
    aitrain::ArtifactId repairArtifactId;
    if (!aitrain::TaskId::parse(taskIdText, &annotationTaskId_, &error)
        || annotationTaskId_ != controlTaskId_
        || !aitrain::ArtifactId::parse(repairArtifactText, &repairArtifactId, &error)
        || projectRoot.isEmpty() || workingDirectory.isEmpty() || !QFileInfo(projectRoot).isDir()) {
        annotationTaskId_ = {};
        fail(QStringLiteral("创建标注会话需要有效项目、Repair ArtifactId、独立工作目录和一致的 Protocol  TaskId。"));
        return;
    }

    annotationWorkspace_ = std::make_unique<aitrain::ProjectWorkspace>();
    if (!annotationWorkspace_->open(projectRoot, &error)) {
        annotationWorkspace_.reset();
        annotationTaskId_ = {};
        fail(QStringLiteral("无法打开 Annotation Session  工作区：%1").arg(error));
        return;
    }
    aitrain::TaskSnapshot task;
    if (!annotationWorkspace_->startTask(annotationTaskId_,
            QStringLiteral("dataset.annotation.session"), QStringLiteral("annotation_session_create"),
            &task, &error)) {
        annotationWorkspace_.reset();
        annotationTaskId_ = {};
        fail(QStringLiteral("无法创建 Annotation Session  根任务：%1").arg(error));
        return;
    }

    activeTaskId_ = taskIdText;
    canceled_ = false;
    running_ = true;
    annotationRunning_ = true;
    send(wp::event::progress(), QJsonObject{{wp::field::taskId(), taskIdText},
        {QStringLiteral("percent"), 0},
        {wp::field::message(), QStringLiteral("Annotation Session  正在校验 Repair Artifact 并准备受控工作副本。")}});

    aitrain::AnnotationSessionCreateRequest request;
    request.repairManifestArtifactId = repairArtifactId;
    request.workingDirectory = workingDirectory;
    request.toolParameters = toolSummary;
    if (!options.isEmpty()) request.toolParameters.insert(QStringLiteral("options"), options);
    aitrain::AnnotationSessionCreateResult result;
    const bool executed = annotationWorkspace_->createAnnotationSession(annotationTaskId_, request,
        &result, &error, pollingCancellationCallback(0));
    annotationRunning_ = false;

    if (!executed) {
        aitrain::TaskSnapshot stored;
        if (annotationWorkspace_->task(annotationTaskId_, &stored, nullptr)
            && !aitrain::isTerminalTaskState(stored.state)) {
            annotationWorkspace_->finalizeTask(annotationTaskId_, aitrain::TaskState::Failed,
                workerFailure(error.isEmpty() ? QStringLiteral("Annotation Session  执行失败。") : error), nullptr);
        }
        annotationWorkspace_.reset();
        annotationTaskId_ = {};
        running_ = false;
        failWithDetails(QStringLiteral("Annotation Session  执行失败：%1").arg(error),
            QStringLiteral("annotation_session_execution_failed"));
        return;
    }

    QJsonObject response{{wp::field::taskId(), taskIdText},
        {QStringLiteral("workflowRunId"), result.workflowRunId.toString()},
        {QStringLiteral("state"), aitrain::taskStateToString(result.terminalState)},
        {QStringLiteral("status"), annotationStatusText(result.status)},
        {QStringLiteral("sessionArtifactId"), result.sessionArtifactId.toString()},
        {QStringLiteral("evidenceArtifactId"), result.evidenceArtifactId.toString()}};
    send(wp::event::annotationSession(), response);

    const aitrain::TaskState terminalState = result.terminalState;
    annotationWorkspace_.reset();
    annotationTaskId_ = {};
    running_ = false;
    if (terminalState == aitrain::TaskState::Canceled) {
        sendCanceledAndFinish(taskIdText, QStringLiteral("Annotation Session  已取消。"));
    } else if (terminalState == aitrain::TaskState::Failed) {
        failWithDetails(QStringLiteral("Annotation Session  创建失败。"),
            QStringLiteral("annotation_session_failed"), response);
    } else {
        send(wp::event::completed(), QJsonObject{{wp::field::taskId(), taskIdText},
            {wp::field::message(), QStringLiteral("Annotation Session  创建完成。")}});
        finishSession();
    }
}

void WorkerSession::syncAnnotationSession(const QJsonObject& payload)
{
    if (running_ || annotationWorkspace_) {
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
    aitrain::ArtifactId sessionArtifactId;
    if (!aitrain::TaskId::parse(taskIdText, &annotationTaskId_, &error)
        || annotationTaskId_ != controlTaskId_
        || !aitrain::ArtifactId::parse(sessionArtifactText, &sessionArtifactId, &error)
        || projectRoot.isEmpty() || workingDirectory.isEmpty() || !QFileInfo(projectRoot).isDir()) {
        annotationTaskId_ = {};
        fail(QStringLiteral("同步标注会话需要有效项目、Session ArtifactId、独立工作目录和一致的 Protocol  TaskId。"));
        return;
    }

    annotationWorkspace_ = std::make_unique<aitrain::ProjectWorkspace>();
    if (!annotationWorkspace_->open(projectRoot, &error)) {
        annotationWorkspace_.reset();
        annotationTaskId_ = {};
        fail(QStringLiteral("无法打开 Annotation Sync  工作区：%1").arg(error));
        return;
    }
    aitrain::TaskSnapshot task;
    if (!annotationWorkspace_->startTask(annotationTaskId_,
            QStringLiteral("dataset.annotation.sync"), QStringLiteral("annotation_session_sync"),
            &task, &error)) {
        annotationWorkspace_.reset();
        annotationTaskId_ = {};
        fail(QStringLiteral("无法创建 Annotation Sync  根任务：%1").arg(error));
        return;
    }

    activeTaskId_ = taskIdText;
    canceled_ = false;
    running_ = true;
    annotationRunning_ = true;
    send(wp::event::progress(), QJsonObject{{wp::field::taskId(), taskIdText},
        {QStringLiteral("percent"), 0},
        {wp::field::message(), QStringLiteral("Annotation Sync  正在重验基线、白名单、文件集合和哈希。")}});

    aitrain::AnnotationSessionSyncRequest request;
    request.sessionArtifactId = sessionArtifactId;
    request.workingDirectory = workingDirectory;
    aitrain::AnnotationSessionSyncResult result;
    const bool executed = annotationWorkspace_->syncAnnotationSession(annotationTaskId_, request,
        &result, &error, pollingCancellationCallback(0));
    annotationRunning_ = false;

    if (!executed) {
        aitrain::TaskSnapshot stored;
        if (annotationWorkspace_->task(annotationTaskId_, &stored, nullptr)
            && !aitrain::isTerminalTaskState(stored.state)) {
            annotationWorkspace_->finalizeTask(annotationTaskId_, aitrain::TaskState::Failed,
                workerFailure(error.isEmpty() ? QStringLiteral("Annotation Sync  执行失败。") : error), nullptr);
        }
        annotationWorkspace_.reset();
        annotationTaskId_ = {};
        running_ = false;
        failWithDetails(QStringLiteral("Annotation Sync  执行失败：%1").arg(error),
            QStringLiteral("annotation_sync_execution_failed"));
        return;
    }

    QJsonObject response{{wp::field::taskId(), taskIdText},
        {QStringLiteral("workflowRunId"), result.workflowRunId.toString()},
        {QStringLiteral("state"), aitrain::taskStateToString(result.terminalState)},
        {QStringLiteral("status"), annotationStatusText(result.status)},
        {QStringLiteral("inspectionArtifactId"), result.inspectionArtifactId.toString()},
        {QStringLiteral("changesArtifactId"), result.changesArtifactId.toString()},
        {QStringLiteral("syncReportArtifactId"), result.syncReportArtifactId.toString()},
        {QStringLiteral("evidenceArtifactId"), result.evidenceArtifactId.toString()},
        {QStringLiteral("newSnapshotId"), result.datasetSnapshot.id.toString()},
        {QStringLiteral("newDatasetVersionId"), result.datasetSnapshot.datasetVersionId.toString()},
        {QStringLiteral("newSnapshotArtifactId"), result.datasetSnapshot.artifactId.toString()},
        {QStringLiteral("newDatasetVersionCreated"), result.datasetSnapshot.id.isValid()}};
    send(wp::event::annotationSync(), response);

    const aitrain::TaskState terminalState = result.terminalState;
    annotationWorkspace_.reset();
    annotationTaskId_ = {};
    running_ = false;
    if (terminalState == aitrain::TaskState::Canceled) {
        sendCanceledAndFinish(taskIdText, QStringLiteral("Annotation Sync  已取消。"));
    } else if (terminalState == aitrain::TaskState::Failed) {
        failWithDetails(QStringLiteral("Annotation Sync  失败或存在冲突。"),
            QStringLiteral("annotation_sync_failed"), response);
    } else {
        send(wp::event::completed(), QJsonObject{{wp::field::taskId(), taskIdText},
            {wp::field::message(), QStringLiteral("Annotation Sync  完成。")}});
        finishSession();
    }
}
