#include "WorkerSession.h"

#include "aitrain/core/WorkerProtocol.h"

#include <QDateTime>
#include <QFileInfo>

namespace wp = aitrain::worker_protocol;

void WorkerSession::runDiagnosticsWorkflow(const wp::DiagnosticsCommand& command)
{
    const QString taskIdText = command.context.taskId.toString();
    const QString projectRoot = command.context.projectRoot.trimmed();
    aitrain::DiagnosticsWorkflowRequest request;
    request.options = command.options;
    QString error;
    const aitrain::TaskId taskId = command.context.taskId;
    if (!taskId.isValid() || taskId != controlTaskId_
        || projectRoot.isEmpty() || !QFileInfo(projectRoot).isDir()) {
        fail(QStringLiteral("Diagnostics Bundle  请求缺少有效项目或 TaskId：%1").arg(error));
        return;
    }

    auto workspace = std::make_unique<aitrain::ProjectWorkspace>();
    if (!workspace->openForWorkerChild(projectRoot, &error)) {
        fail(QStringLiteral("无法打开 Diagnostics Bundle  工作区：%1").arg(error));
        return;
    }
    aitrain::TaskSnapshot task;
    if (!workspace->startTask(taskId,
            QStringLiteral("diagnostics.bundle"), QStringLiteral("diagnostics"),
            &task, &error)) {
        fail(QStringLiteral("无法创建 Diagnostics Bundle  根任务：%1").arg(error));
        return;
    }
    if (!activeWorkflow_.bind(std::move(workspace), taskId, &error)) {
        fail(QStringLiteral("无法绑定 Diagnostics Bundle 活动任务：%1").arg(error));
        return;
    }
    auto* const activeWorkspace = activeWorkflow_.workspace();
    activeTaskId_ = taskIdText;
    send(wp::event::progress(), QJsonObject{{wp::field::taskId(), taskIdText},
        {QStringLiteral("percent"), 0},
        {wp::field::message(), QStringLiteral("Diagnostics Bundle  正在采集受限事实。")}});

    aitrain::DiagnosticsWorkflowResult result;
    const bool executed = activeWorkspace->runDiagnosticsWorkflow(
        taskId, request, &result, &error, pollingCancellationCallback(0));
    if (!executed) {
        aitrain::TaskSnapshot stored;
        if (activeWorkspace->task(taskId, &stored, nullptr)
            && !aitrain::isTerminalTaskState(stored.state)) {
            aitrain::Failure failure;
            failure.code = aitrain::FailureCode::InternalError;
            failure.message = error.isEmpty() ? QStringLiteral("Diagnostics Bundle  执行失败。") : error;
            failure.suggestedAction = QStringLiteral("检查  工作区和 Evidence 后重试。");
            failure.occurredAt = QDateTime::currentDateTimeUtc();
            activeWorkspace->finalizeTask(taskId,
                aitrain::TaskState::Failed, failure, nullptr);
        }
        publishPersistedTerminal(taskId);
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

    publishPersistedTerminal(taskId, QStringLiteral("Diagnostics Bundle  completed"));
}
