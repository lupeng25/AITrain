#include "WorkerSession.h"

#include "aitrain/core/WorkerProtocol.h"

#include <QDateTime>
#include <QFileInfo>
#include <QJsonObject>

namespace wp = aitrain::worker_protocol;

void WorkerSession::runDataQualityWorkflow(const wp::DataQualityCommand& command)
{
    if (running_ || dataQualityWorkspace_) {
        fail(QStringLiteral("Worker 已有运行任务，不能并发运行 Data Quality 。"));
        return;
    }

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
        dataQualityTaskId_ = {};
        fail(QStringLiteral("Data Quality  只接受有效项目和完整登记身份：%1").arg(error));
        return;
    }
    dataQualityTaskId_ = taskId;
    request.options = command.options;

    dataQualityWorkspace_ = std::make_unique<aitrain::ProjectWorkspace>();
    if (!dataQualityWorkspace_->open(projectRoot, &error)) {
        dataQualityWorkspace_.reset();
        dataQualityTaskId_ = {};
        fail(QStringLiteral("无法打开 Data Quality  工作区：%1").arg(error));
        return;
    }
    aitrain::TaskSnapshot task;
    if (!dataQualityWorkspace_->startTask(dataQualityTaskId_,
            QStringLiteral("dataset.quality"), QStringLiteral("dataset_quality"),
            &task, &error)) {
        dataQualityWorkspace_.reset();
        dataQualityTaskId_ = {};
        fail(QStringLiteral("无法创建 Data Quality  根任务：%1").arg(error));
        return;
    }

    activeTaskId_ = taskIdText;
    canceled_ = false;
    running_ = true;
    dataQualityRunning_ = true;
    send(wp::event::progress(), QJsonObject{{wp::field::taskId(), taskIdText},
        {QStringLiteral("percent"), 0},
        {wp::field::message(), QStringLiteral("Data Quality  正在校验快照身份、分析质量并生成受控 Artifact。")}});

    aitrain::DataQualityWorkflowResult result;
    const bool executed = dataQualityWorkspace_->runDataQualityWorkflow(
        dataQualityTaskId_, request, &result, &error, pollingCancellationCallback(0));
    dataQualityRunning_ = false;
    if (!executed) {
        aitrain::TaskSnapshot stored;
        if (dataQualityWorkspace_->task(dataQualityTaskId_, &stored, nullptr)
            && !aitrain::isTerminalTaskState(stored.state)) {
            aitrain::Failure failure;
            failure.code = aitrain::FailureCode::InternalError;
            failure.message = error.isEmpty() ? QStringLiteral("Data Quality  执行失败。") : error;
            failure.suggestedAction = QStringLiteral("检查  Snapshot 身份和 Artifact 完整性后重试。");
            failure.occurredAt = QDateTime::currentDateTimeUtc();
            dataQualityWorkspace_->finalizeTask(dataQualityTaskId_,
                aitrain::TaskState::Failed, failure, nullptr);
        }
        dataQualityWorkspace_.reset();
        dataQualityTaskId_ = {};
        running_ = false;
        failWithDetails(QStringLiteral("Data Quality  执行失败：%1").arg(error),
            QStringLiteral("data_quality_execution_failed"));
        return;
    }

    aitrain::TaskSnapshot stored;
    dataQualityWorkspace_->task(dataQualityTaskId_, &stored, nullptr);
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

    const aitrain::TaskState terminalState = result.terminalState;
    const QString terminalMessage = response.value(wp::field::message()).toString();
    dataQualityWorkspace_.reset();
    dataQualityTaskId_ = {};
    running_ = false;
    if (terminalState == aitrain::TaskState::Canceled) {
        sendCanceledAndFinish(taskIdText, terminalMessage);
    } else if (terminalState == aitrain::TaskState::Failed) {
        failWithDetails(terminalMessage,
            response.value(QStringLiteral("failureCode")).toString(
                QStringLiteral("data_quality_failed")), response);
    } else {
        send(wp::event::completed(), QJsonObject{{wp::field::taskId(), taskIdText},
            {wp::field::message(), QStringLiteral("Data Quality  completed")}});
        finishSession();
    }
}
