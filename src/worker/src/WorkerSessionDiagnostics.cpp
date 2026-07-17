#include "WorkerSession.h"

#include "aitrain/core/WorkerProtocol.h"

#include <QDateTime>
#include <QFileInfo>

namespace wp = aitrain::worker_protocol;

void WorkerSession::runDiagnosticsWorkflow(const wp::DiagnosticsCommand& command)
{
    if (running_ || diagnosticsWorkspace_) {
        fail(QStringLiteral("Worker 已有运行任务，不能并发运行 Diagnostics Bundle 。"));
        return;
    }
    const QString taskIdText = command.context.taskId.toString();
    const QString projectRoot = command.context.projectRoot.trimmed();
    aitrain::DiagnosticsWorkflowRequest request;
    request.options = command.options;
    QString error;
    if (!aitrain::TaskId::parse(taskIdText, &diagnosticsTaskId_, &error)
        || diagnosticsTaskId_ != controlTaskId_
        || projectRoot.isEmpty() || !QFileInfo(projectRoot).isDir()) {
        diagnosticsTaskId_ = {};
        fail(QStringLiteral("Diagnostics Bundle  请求缺少有效项目或 TaskId：%1").arg(error));
        return;
    }

    diagnosticsWorkspace_ = std::make_unique<aitrain::ProjectWorkspace>();
    if (!diagnosticsWorkspace_->open(projectRoot, &error)) {
        diagnosticsWorkspace_.reset();
        diagnosticsTaskId_ = {};
        fail(QStringLiteral("无法打开 Diagnostics Bundle  工作区：%1").arg(error));
        return;
    }
    aitrain::TaskSnapshot task;
    if (!diagnosticsWorkspace_->startTask(diagnosticsTaskId_,
            QStringLiteral("diagnostics.bundle"), QStringLiteral("diagnostics"),
            &task, &error)) {
        diagnosticsWorkspace_.reset();
        diagnosticsTaskId_ = {};
        fail(QStringLiteral("无法创建 Diagnostics Bundle  根任务：%1").arg(error));
        return;
    }
    activeTaskId_ = taskIdText;
    canceled_ = false;
    running_ = true;
    diagnosticsRunning_ = true;
    send(wp::event::progress(), QJsonObject{{wp::field::taskId(), taskIdText},
        {QStringLiteral("percent"), 0},
        {wp::field::message(), QStringLiteral("Diagnostics Bundle  正在采集受限事实。")}});

    aitrain::DiagnosticsWorkflowResult result;
    const bool executed = diagnosticsWorkspace_->runDiagnosticsWorkflow(
        diagnosticsTaskId_, request, &result, &error, pollingCancellationCallback(0));
    diagnosticsRunning_ = false;
    if (!executed) {
        aitrain::TaskSnapshot stored;
        if (diagnosticsWorkspace_->task(diagnosticsTaskId_, &stored, nullptr)
            && !aitrain::isTerminalTaskState(stored.state)) {
            aitrain::Failure failure;
            failure.code = aitrain::FailureCode::InternalError;
            failure.message = error.isEmpty() ? QStringLiteral("Diagnostics Bundle  执行失败。") : error;
            failure.suggestedAction = QStringLiteral("检查  工作区和 Evidence 后重试。");
            failure.occurredAt = QDateTime::currentDateTimeUtc();
            diagnosticsWorkspace_->finalizeTask(diagnosticsTaskId_,
                aitrain::TaskState::Failed, failure, nullptr);
        }
        diagnosticsWorkspace_.reset();
        diagnosticsTaskId_ = {};
        running_ = false;
        failWithDetails(QStringLiteral("Diagnostics Bundle  执行失败：%1").arg(error),
            QStringLiteral("diagnostics_execution_failed"));
        return;
    }

    QJsonObject response{{wp::field::taskId(), taskIdText},
        {QStringLiteral("workflowRunId"), result.workflowRunId.toString()},
        {QStringLiteral("state"), aitrain::taskStateToString(result.terminalState)},
        {QStringLiteral("diagnosticsArtifactId"), result.diagnosticsArtifactId.toString()},
        {QStringLiteral("evidenceArtifactId"), result.evidenceArtifactId.toString()},
        {QStringLiteral("summary"), result.summary}};
    if (result.failure.isFailure()) {
        response.insert(QStringLiteral("failureCode"),
            aitrain::failureCodeToString(result.failure.code));
        response.insert(wp::field::message(), result.failure.message);
    } else {
        response.insert(wp::field::message(), QStringLiteral("Diagnostics Bundle  已提交。"));
    }
    send(wp::event::diagnosticsWorkflow(), response);

    const aitrain::TaskState finalState = result.terminalState;
    const QString terminalMessage = response.value(wp::field::message()).toString();
    diagnosticsWorkspace_.reset();
    diagnosticsTaskId_ = {};
    running_ = false;
    if (finalState == aitrain::TaskState::Canceled) {
        sendCanceledAndFinish(taskIdText, terminalMessage);
    } else if (finalState == aitrain::TaskState::Failed) {
        failWithDetails(terminalMessage,
            response.value(QStringLiteral("failureCode")).toString(QStringLiteral("diagnostics_failed")), response);
    } else {
        send(wp::event::completed(), QJsonObject{{wp::field::taskId(), taskIdText},
            {wp::field::message(), QStringLiteral("Diagnostics Bundle  completed")}});
        finishSession();
    }
}
