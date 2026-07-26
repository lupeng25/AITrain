#include "WorkerSession.h"

#include "aitrain/core/WorkerProtocol.h"

#include <QDateTime>
#include <QFileInfo>
#include <QJsonObject>

namespace wp = aitrain::worker_protocol;

void WorkerSession::runDataQualityWorkflow(const wp::DataQualityCommand& command)
{

    const aitrain::TaskId taskId = command.context.taskId;
    const QString taskIdText = taskId.toString();
    const QString projectRoot = command.context.projectRoot.trimmed();
    aitrain::DataQualityWorkflowRequest request;
    QString error;
    if (!taskId.isValid()) {
        error = QStringLiteral("TaskId 无效。");
    }
    if (!taskId.isValid() || taskId != controlTaskId_
        || projectRoot.isEmpty() || !QFileInfo(projectRoot).isDir()
        || !aitrain::DatasetId::parse(command.datasetId,
            &request.datasetId, &error)
        || !aitrain::DatasetVersionId::parse(command.datasetVersionId,
            &request.datasetVersionId, &error)
        || !aitrain::SnapshotId::parse(command.snapshotId,
            &request.snapshotId, &error)
        || !aitrain::ArtifactId::parse(command.snapshotArtifactId,
            &request.snapshotArtifactId, &error)) {
        fail(QStringLiteral("Data Quality  只接受有效项目和完整登记身份：%1").arg(error));
        return;
    }
    request.options = command.options;

    auto workspace = std::make_unique<aitrain::ProjectWorkspace>();
    if (!workspace->openForWorkerChild(projectRoot, &error)) {
        fail(QStringLiteral("无法打开 Data Quality  工作区：%1").arg(error));
        return;
    }
    aitrain::TaskSnapshot task;
    if (!workspace->startTask(taskId,
            QStringLiteral("dataset.quality"), QStringLiteral("dataset_quality"),
            &task, &error)) {
        fail(QStringLiteral("无法创建 Data Quality  根任务：%1").arg(error));
        return;
    }
    if (!activeWorkflow_.bind(std::move(workspace), taskId, &error)) {
        fail(QStringLiteral("无法绑定 Data Quality 活动任务：%1").arg(error));
        return;
    }
    auto* const activeWorkspace = activeWorkflow_.workspace();

    activeTaskId_ = taskIdText;
    send(wp::event::progress(), QJsonObject{{wp::field::taskId(), taskIdText},
        {QStringLiteral("percent"), 0},
        {wp::field::message(), QStringLiteral("Data Quality  正在校验快照身份、分析质量并生成受控 Artifact。")}});

    aitrain::DataQualityWorkflowResult result;
    const bool executed = activeWorkspace->runDataQualityWorkflow(
        taskId, request, &result, &error, pollingCancellationCallback(0));
    if (!executed) {
        aitrain::TaskSnapshot stored;
        if (activeWorkspace->task(taskId, &stored, nullptr)
            && !aitrain::isTerminalTaskState(stored.state)) {
            aitrain::Failure failure;
            failure.code = aitrain::FailureCode::InternalError;
            failure.message = error.isEmpty() ? QStringLiteral("Data Quality  执行失败。") : error;
            failure.suggestedAction = QStringLiteral("检查  Snapshot 身份和 Artifact 完整性后重试。");
            failure.occurredAt = QDateTime::currentDateTimeUtc();
            activeWorkspace->finalizeTask(taskId,
                aitrain::TaskState::Failed, failure, nullptr);
        }
        publishPersistedTerminal(taskId);
        return;
    }

    aitrain::TaskSnapshot stored;
    activeWorkspace->task(taskId, &stored, nullptr);
    QJsonObject response{{wp::field::taskId(), taskIdText},
        {QStringLiteral("workflowRunId"), result.workflowRunId.toString()},
        {QStringLiteral("state"), aitrain::taskStateToString(result.terminalState)},
        {QStringLiteral("snapshotValidationArtifactId"), result.snapshotValidationArtifactId.toString()},
        {QStringLiteral("qualityAnalysisArtifactId"), result.qualityAnalysisArtifactId.toString()},
        {QStringLiteral("repairManifestArtifactId"), result.repairManifestArtifactId.toString()},
        {QStringLiteral("qualityReportArtifactId"), result.qualityReportArtifactId.toString()},
        {QStringLiteral("evidenceArtifactId"), result.evidenceArtifactId.toString()},
        {QStringLiteral("summary"), result.summary}};
    if (stored.failure.isFailure()) {
        response.insert(QStringLiteral("failureCode"),
            aitrain::failureCodeToString(stored.failure.code));
        response.insert(wp::field::message(), stored.failure.message);
    } else {
        response.insert(wp::field::message(), QStringLiteral("Data Quality  四步工作流已完成。"));
    }
    send(wp::event::dataQualityWorkflow(), response);

    publishPersistedTerminal(taskId, QStringLiteral("Data Quality  completed"));
}
