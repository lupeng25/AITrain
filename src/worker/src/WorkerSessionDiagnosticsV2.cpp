#include "WorkerSession.h"

#include "aitrain/core/WorkerProtocol.h"

#include <QDateTime>
#include <QFileInfo>

namespace wp = aitrain::worker_protocol;

void WorkerSession::runDiagnosticsWorkflowV2(const QJsonObject& payload)
{
    if (running_ || diagnosticsWorkspaceV2_) {
        fail(QStringLiteral("Worker 已有运行任务，不能并发运行 Diagnostics Bundle V2。"));
        return;
    }
    const QString taskIdText = payload.value(wp::field::taskId()).toString().trimmed();
    const QString projectRoot = payload.value(QStringLiteral("projectRoot")).toString().trimmed();
    aitrain::v2::DiagnosticsWorkflowRequestV2 request;
    request.options = payload.value(wp::field::options()).toObject();
    QString error;
    if (!aitrain::v2::TaskId::parse(taskIdText, &diagnosticsTaskIdV2_, &error)
        || diagnosticsTaskIdV2_ != controlTaskId_
        || projectRoot.isEmpty() || !QFileInfo(projectRoot).isDir()) {
        diagnosticsTaskIdV2_ = {};
        fail(QStringLiteral("Diagnostics Bundle V2 请求缺少有效项目或 TaskId：%1").arg(error));
        return;
    }

    diagnosticsWorkspaceV2_ = std::make_unique<aitrain::v2::ProjectWorkspaceV2>();
    if (!diagnosticsWorkspaceV2_->open(projectRoot, &error)) {
        diagnosticsWorkspaceV2_.reset();
        diagnosticsTaskIdV2_ = {};
        fail(QStringLiteral("无法打开 Diagnostics Bundle V2 工作区：%1").arg(error));
        return;
    }
    aitrain::v2::TaskSnapshot task;
    if (!diagnosticsWorkspaceV2_->startTask(diagnosticsTaskIdV2_,
            QStringLiteral("diagnostics.bundle.v2"), QStringLiteral("diagnostics"),
            &task, &error)) {
        diagnosticsWorkspaceV2_.reset();
        diagnosticsTaskIdV2_ = {};
        fail(QStringLiteral("无法创建 Diagnostics Bundle V2 根任务：%1").arg(error));
        return;
    }
    activeTaskId_ = taskIdText;
    canceled_ = false;
    running_ = true;
    diagnosticsRunningV2_ = true;
    send(wp::event::progress(), QJsonObject{{wp::field::taskId(), taskIdText},
        {QStringLiteral("percent"), 0},
        {wp::field::message(), QStringLiteral("Diagnostics Bundle V2 正在采集受限事实。")}});

    aitrain::v2::DiagnosticsWorkflowResultV2 result;
    const bool executed = diagnosticsWorkspaceV2_->runDiagnosticsWorkflow(
        diagnosticsTaskIdV2_, request, &result, &error, pollingCancellationCallback(0));
    diagnosticsRunningV2_ = false;
    if (!executed) {
        aitrain::v2::TaskSnapshot stored;
        if (diagnosticsWorkspaceV2_->task(diagnosticsTaskIdV2_, &stored, nullptr)
            && !aitrain::v2::isTerminalTaskState(stored.state)) {
            aitrain::v2::Failure failure;
            failure.code = aitrain::v2::FailureCode::InternalError;
            failure.message = error.isEmpty() ? QStringLiteral("Diagnostics Bundle V2 执行失败。") : error;
            failure.suggestedAction = QStringLiteral("检查 V2 工作区和 Evidence 后重试。");
            failure.occurredAt = QDateTime::currentDateTimeUtc();
            diagnosticsWorkspaceV2_->finalizeTask(diagnosticsTaskIdV2_,
                aitrain::v2::TaskState::Failed, failure, nullptr);
        }
        diagnosticsWorkspaceV2_.reset();
        diagnosticsTaskIdV2_ = {};
        running_ = false;
        failWithDetails(QStringLiteral("Diagnostics Bundle V2 执行失败：%1").arg(error),
            QStringLiteral("diagnostics_v2_execution_failed"));
        return;
    }

    QJsonObject response{{wp::field::taskId(), taskIdText},
        {QStringLiteral("workflowRunId"), result.workflowRunId.toString()},
        {QStringLiteral("state"), aitrain::v2::taskStateToString(result.terminalState)},
        {QStringLiteral("diagnosticsArtifactId"), result.diagnosticsArtifactId.toString()},
        {QStringLiteral("evidenceArtifactId"), result.evidenceArtifactId.toString()},
        {QStringLiteral("summary"), result.summary}};
    if (result.failure.isFailure()) {
        response.insert(QStringLiteral("failureCode"),
            aitrain::v2::failureCodeToString(result.failure.code));
        response.insert(wp::field::message(), result.failure.message);
    } else {
        response.insert(wp::field::message(), QStringLiteral("Diagnostics Bundle V2 已提交。"));
    }
    send(wp::event::diagnosticsWorkflowV2(), response);

    const aitrain::v2::TaskState finalState = result.terminalState;
    const QString terminalMessage = response.value(wp::field::message()).toString();
    diagnosticsWorkspaceV2_.reset();
    diagnosticsTaskIdV2_ = {};
    running_ = false;
    if (finalState == aitrain::v2::TaskState::Canceled) {
        sendCanceledAndFinish(taskIdText, terminalMessage);
    } else if (finalState == aitrain::v2::TaskState::Failed) {
        failWithDetails(terminalMessage,
            response.value(QStringLiteral("failureCode")).toString(QStringLiteral("diagnostics_v2_failed")), response);
    } else {
        send(wp::event::completed(), QJsonObject{{wp::field::taskId(), taskIdText},
            {wp::field::message(), QStringLiteral("Diagnostics Bundle V2 completed")}});
        finishSession();
    }
}
