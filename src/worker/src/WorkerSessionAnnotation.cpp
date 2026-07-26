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

void WorkerSession::createAnnotationSession(const wp::AnnotationSessionCreateCommand& command)
{
    const QString taskIdText = command.context.taskId.toString();
    const QString projectRoot = command.context.projectRoot.trimmed();
    const QString repairArtifactText = command.repairManifestArtifactId.trimmed();
    const QString workingDirectory = command.workingDirectory.trimmed();
    const QJsonObject toolSummary = command.toolSummary;
    const QJsonObject options = command.options;
    QString error;
    aitrain::ArtifactId repairArtifactId;
    const aitrain::TaskId taskId = command.context.taskId;
    if (!taskId.isValid() || taskId != controlTaskId_
        || !aitrain::ArtifactId::parse(repairArtifactText, &repairArtifactId, &error)
        || projectRoot.isEmpty() || workingDirectory.isEmpty() || !QFileInfo(projectRoot).isDir()) {
        fail(QStringLiteral("创建标注会话需要有效项目、Repair ArtifactId、独立工作目录和一致的 Protocol  TaskId。"));
        return;
    }

    auto workspace = std::make_unique<aitrain::ProjectWorkspace>();
    if (!workspace->openForWorkerChild(projectRoot, &error)) {
        fail(QStringLiteral("无法打开 Annotation Session  工作区：%1").arg(error));
        return;
    }
    aitrain::TaskSnapshot task;
    if (!workspace->startTask(taskId,
            QStringLiteral("dataset.annotation.session"), QStringLiteral("annotation_session_create"),
            &task, &error)) {
        fail(QStringLiteral("无法创建 Annotation Session  根任务：%1").arg(error));
        return;
    }
    if (!activeWorkflow_.bind(std::move(workspace), taskId, &error)) {
        fail(QStringLiteral("无法绑定 Annotation Session 活动任务：%1").arg(error));
        return;
    }
    auto* const activeWorkspace = activeWorkflow_.workspace();

    activeTaskId_ = taskIdText;
    send(wp::event::progress(), QJsonObject{{wp::field::taskId(), taskIdText},
        {QStringLiteral("percent"), 0},
        {wp::field::message(), QStringLiteral("Annotation Session  正在校验 Repair Artifact 并准备受控工作副本。")}});

    aitrain::AnnotationSessionCreateRequest request;
    request.repairManifestArtifactId = repairArtifactId;
    request.workingDirectory = workingDirectory;
    request.toolParameters = toolSummary;
    if (!options.isEmpty()) request.toolParameters.insert(QStringLiteral("options"), options);
    aitrain::AnnotationSessionCreateResult result;
    const bool executed = activeWorkspace->createAnnotationSession(taskId, request,
        &result, &error, pollingCancellationCallback(0));

    if (!executed) {
        aitrain::TaskSnapshot stored;
        if (activeWorkspace->task(taskId, &stored, nullptr)
            && !aitrain::isTerminalTaskState(stored.state)) {
            activeWorkspace->finalizeTask(taskId, aitrain::TaskState::Failed,
                workerFailure(error.isEmpty() ? QStringLiteral("Annotation Session  执行失败。") : error), nullptr);
        }
        publishPersistedTerminal(taskId);
        return;
    }

    QJsonObject response{{wp::field::taskId(), taskIdText},
        {QStringLiteral("workflowRunId"), result.workflowRunId.toString()},
        {QStringLiteral("state"), aitrain::taskStateToString(result.terminalState)},
        {QStringLiteral("status"), annotationStatusText(result.status)},
        {QStringLiteral("sessionArtifactId"), result.sessionArtifactId.toString()},
        {QStringLiteral("evidenceArtifactId"), result.evidenceArtifactId.toString()}};
    send(wp::event::annotationSession(), response);

    publishPersistedTerminal(taskId,
        QStringLiteral("Annotation Session  创建完成。"));
}

void WorkerSession::syncAnnotationSession(const wp::AnnotationSessionSyncCommand& command)
{
    const QString taskIdText = command.context.taskId.toString();
    const QString projectRoot = command.context.projectRoot.trimmed();
    const QString sessionArtifactText = command.sessionArtifactId.trimmed();
    const QString workingDirectory = command.workingDirectory.trimmed();
    const QJsonObject options = command.options;
    Q_UNUSED(options);
    QString error;
    aitrain::ArtifactId sessionArtifactId;
    const aitrain::TaskId taskId = command.context.taskId;
    if (!taskId.isValid() || taskId != controlTaskId_
        || !aitrain::ArtifactId::parse(sessionArtifactText, &sessionArtifactId, &error)
        || projectRoot.isEmpty() || workingDirectory.isEmpty() || !QFileInfo(projectRoot).isDir()) {
        fail(QStringLiteral("同步标注会话需要有效项目、Session ArtifactId、独立工作目录和一致的 Protocol  TaskId。"));
        return;
    }

    auto workspace = std::make_unique<aitrain::ProjectWorkspace>();
    if (!workspace->openForWorkerChild(projectRoot, &error)) {
        fail(QStringLiteral("无法打开 Annotation Sync  工作区：%1").arg(error));
        return;
    }
    aitrain::TaskSnapshot task;
    if (!workspace->startTask(taskId,
            QStringLiteral("dataset.annotation.sync"), QStringLiteral("annotation_session_sync"),
            &task, &error)) {
        fail(QStringLiteral("无法创建 Annotation Sync  根任务：%1").arg(error));
        return;
    }
    if (!activeWorkflow_.bind(std::move(workspace), taskId, &error)) {
        fail(QStringLiteral("无法绑定 Annotation Sync 活动任务：%1").arg(error));
        return;
    }
    auto* const activeWorkspace = activeWorkflow_.workspace();

    activeTaskId_ = taskIdText;
    send(wp::event::progress(), QJsonObject{{wp::field::taskId(), taskIdText},
        {QStringLiteral("percent"), 0},
        {wp::field::message(), QStringLiteral("Annotation Sync  正在重验基线、白名单、文件集合和哈希。")}});

    aitrain::AnnotationSessionSyncRequest request;
    request.sessionArtifactId = sessionArtifactId;
    request.workingDirectory = workingDirectory;
    aitrain::AnnotationSessionSyncResult result;
    const bool executed = activeWorkspace->syncAnnotationSession(taskId, request,
        &result, &error, pollingCancellationCallback(0));

    if (!executed) {
        aitrain::TaskSnapshot stored;
        if (activeWorkspace->task(taskId, &stored, nullptr)
            && !aitrain::isTerminalTaskState(stored.state)) {
            activeWorkspace->finalizeTask(taskId, aitrain::TaskState::Failed,
                workerFailure(error.isEmpty() ? QStringLiteral("Annotation Sync  执行失败。") : error), nullptr);
        }
        publishPersistedTerminal(taskId);
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

    publishPersistedTerminal(taskId, QStringLiteral("Annotation Sync  完成。"));
}
