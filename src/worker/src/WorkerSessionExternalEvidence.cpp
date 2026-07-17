#include "WorkerSession.h"

#include "aitrain/core/WorkerProtocol.h"

#include <QDateTime>
#include <QFileInfo>

namespace wp = aitrain::worker_protocol;

void WorkerSession::importExternalAcceptanceEvidence(
    const wp::ExternalAcceptanceEvidenceImportCommand& command)
{
    if (running_) {
        fail(QStringLiteral("Worker 已有运行任务，不能并发导入外部验收证据。"));
        return;
    }
    const QString taskIdText = command.context.taskId.toString();
    const QString projectRoot = command.context.projectRoot.trimmed();
    const QString sourcePath = command.sourcePath.trimmed();
    QString error;
    aitrain::TaskId externalTaskId;
    if (!aitrain::TaskId::parse(taskIdText, &externalTaskId, &error)
        || externalTaskId != controlTaskId_
        || projectRoot.isEmpty() || !QFileInfo(projectRoot).isDir()
        || sourcePath.isEmpty()) {
        activeTaskId_.clear();
        fail(QStringLiteral("外部验收证据请求缺少有效项目、TaskId 或 sourcePath：%1").arg(error));
        return;
    }
    activeTaskId_ = taskIdText;

    auto workspace = std::make_unique<aitrain::ProjectWorkspace>();
    if (!workspace->open(projectRoot, &error)) {
        fail(QStringLiteral("无法打开外部验收证据工作区：%1").arg(error));
        return;
    }
    aitrain::TaskSnapshot task;
    if (!workspace->startTask(externalTaskId, QStringLiteral("delivery.external_acceptance"),
            QStringLiteral("external_acceptance_evidence"), &task, &error)) {
        fail(QStringLiteral("无法创建外部验收证据任务：%1").arg(error));
        return;
    }

    running_ = true;
    send(wp::event::progress(), QJsonObject{
        {wp::field::taskId(), taskIdText},
        {QStringLiteral("percent"), 10},
        {wp::field::message(), QStringLiteral("正在严格校验外部验收证据。")}});

    aitrain::ExternalAcceptanceEvidenceImportRequest request;
    request.sourcePath = sourcePath;
    aitrain::ExternalAcceptanceEvidenceImportResult result;
    const bool imported = workspace->importExternalAcceptanceEvidence(
        externalTaskId, request, &result, &error, pollingCancellationCallback(0));
    if (!imported) {
        aitrain::Failure failure;
        failure.code = aitrain::FailureCode::InvalidRequest;
        failure.message = error.isEmpty() ? QStringLiteral("外部验收证据校验失败。") : error;
        failure.suggestedAction = QStringLiteral("修正 evidence schema 后重新导入；外部证据不会自动被视为 verified。" );
        failure.occurredAt = QDateTime::currentDateTimeUtc();
        workspace->finalizeTask(externalTaskId, aitrain::TaskState::Failed, failure, nullptr);
        running_ = false;
        failWithDetails(failure.message, QStringLiteral("external_acceptance_evidence_rejected"),
            QJsonObject{{wp::field::taskId(), taskIdText}});
        activeTaskId_.clear();
        return;
    }

    aitrain::Failure noFailure;
    if (!workspace->finalizeTask(externalTaskId, aitrain::TaskState::Succeeded, noFailure, &error)) {
        running_ = false;
        failWithDetails(QStringLiteral("外部验收证据任务无法收口：%1").arg(error),
            QStringLiteral("external_acceptance_evidence_finalize_failed"));
        activeTaskId_.clear();
        return;
    }
    send(wp::event::progress(), QJsonObject{
        {wp::field::taskId(), taskIdText},
        {QStringLiteral("percent"), 100},
        {wp::field::message(), QStringLiteral("外部验收证据已提交；仅记录事实，不自动通过验收。")}});
    send(wp::event::externalAcceptanceEvidenceImported(), QJsonObject{
        {wp::field::taskId(), taskIdText},
        {QStringLiteral("evidenceArtifactId"), result.evidenceArtifactId.toString()},
        {QStringLiteral("evidenceKind"), result.evidenceKind},
        {wp::field::status(), result.status},
        {QStringLiteral("producer"), result.producer},
        {QStringLiteral("observedAt"), result.observedAt.toString(Qt::ISODateWithMs)},
        {QStringLiteral("summary"), result.summary},
        {wp::field::message(), QStringLiteral("外部验收证据已提交；仅记录事实，不自动通过验收。")}});
    running_ = false;
    send(wp::event::completed(), QJsonObject{
        {wp::field::taskId(), taskIdText},
        {wp::field::message(), QStringLiteral("外部验收证据导入完成。")}});
    activeTaskId_.clear();
    finishSession();
}
