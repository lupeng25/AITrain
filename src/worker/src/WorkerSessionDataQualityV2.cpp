#include "WorkerSession.h"

#include "aitrain/core/WorkerProtocol.h"

#include <QDateTime>
#include <QFileInfo>
#include <QJsonObject>

namespace wp = aitrain::worker_protocol;

void WorkerSession::runDataQualityWorkflowV2(const QJsonObject& payload)
{
    if (running_ || dataQualityWorkspaceV2_) {
        fail(QStringLiteral("Worker 已有运行任务，不能并发运行 Data Quality V2。"));
        return;
    }

    const QString taskIdText = payload.value(wp::field::taskId()).toString().trimmed();
    const QString projectRoot = payload.value(QStringLiteral("projectRoot")).toString().trimmed();
    aitrain::v2::DataQualityWorkflowRequestV2 request;
    QString error;
    if (!aitrain::v2::TaskId::parse(taskIdText, &dataQualityTaskIdV2_, &error)
        || dataQualityTaskIdV2_ != controlTaskId_
        || projectRoot.isEmpty() || !QFileInfo(projectRoot).isDir()
        || !aitrain::v2::DatasetId::parse(payload.value(QStringLiteral("datasetId")).toString(),
            &request.datasetId, &error)
        || !aitrain::v2::DatasetVersionId::parse(payload.value(QStringLiteral("datasetVersionId")).toString(),
            &request.datasetVersionId, &error)
        || !aitrain::v2::SnapshotId::parse(payload.value(QStringLiteral("snapshotId")).toString(),
            &request.snapshotId, &error)
        || !aitrain::v2::ArtifactId::parse(payload.value(QStringLiteral("snapshotArtifactId")).toString(),
            &request.snapshotArtifactId, &error)) {
        dataQualityTaskIdV2_ = {};
        fail(QStringLiteral("Data Quality V2 只接受有效项目和完整登记身份：%1").arg(error));
        return;
    }
    request.options = payload.value(wp::field::options()).toObject();

    dataQualityWorkspaceV2_ = std::make_unique<aitrain::v2::ProjectWorkspaceV2>();
    if (!dataQualityWorkspaceV2_->open(projectRoot, &error)) {
        dataQualityWorkspaceV2_.reset();
        dataQualityTaskIdV2_ = {};
        fail(QStringLiteral("无法打开 Data Quality V2 工作区：%1").arg(error));
        return;
    }
    aitrain::v2::TaskSnapshot task;
    if (!dataQualityWorkspaceV2_->startTask(dataQualityTaskIdV2_,
            QStringLiteral("dataset.quality.v2"), QStringLiteral("dataset_quality"),
            &task, &error)) {
        dataQualityWorkspaceV2_.reset();
        dataQualityTaskIdV2_ = {};
        fail(QStringLiteral("无法创建 Data Quality V2 根任务：%1").arg(error));
        return;
    }

    activeTaskId_ = taskIdText;
    canceled_ = false;
    running_ = true;
    dataQualityRunningV2_ = true;
    send(wp::event::progress(), QJsonObject{{wp::field::taskId(), taskIdText},
        {QStringLiteral("percent"), 0},
        {wp::field::message(), QStringLiteral("Data Quality V2 正在校验快照身份、分析质量并生成受控 Artifact。")}});

    aitrain::v2::DataQualityWorkflowResultV2 result;
    const bool executed = dataQualityWorkspaceV2_->runDataQualityWorkflow(
        dataQualityTaskIdV2_, request, &result, &error, pollingCancellationCallback(0));
    dataQualityRunningV2_ = false;
    if (!executed) {
        aitrain::v2::TaskSnapshot stored;
        if (dataQualityWorkspaceV2_->task(dataQualityTaskIdV2_, &stored, nullptr)
            && !aitrain::v2::isTerminalTaskState(stored.state)) {
            aitrain::v2::Failure failure;
            failure.code = aitrain::v2::FailureCode::InternalError;
            failure.message = error.isEmpty() ? QStringLiteral("Data Quality V2 执行失败。") : error;
            failure.suggestedAction = QStringLiteral("检查 V2 Snapshot 身份和 Artifact 完整性后重试。");
            failure.occurredAt = QDateTime::currentDateTimeUtc();
            dataQualityWorkspaceV2_->finalizeTask(dataQualityTaskIdV2_,
                aitrain::v2::TaskState::Failed, failure, nullptr);
        }
        dataQualityWorkspaceV2_.reset();
        dataQualityTaskIdV2_ = {};
        running_ = false;
        failWithDetails(QStringLiteral("Data Quality V2 执行失败：%1").arg(error),
            QStringLiteral("data_quality_v2_execution_failed"));
        return;
    }

    aitrain::v2::TaskSnapshot stored;
    dataQualityWorkspaceV2_->task(dataQualityTaskIdV2_, &stored, nullptr);
    QJsonObject response{{wp::field::taskId(), taskIdText},
        {QStringLiteral("workflowRunId"), result.workflowRunId.toString()},
        {QStringLiteral("state"), aitrain::v2::taskStateToString(result.terminalState)},
        {QStringLiteral("snapshotValidationArtifactId"), result.snapshotValidationArtifactId.toString()},
        {QStringLiteral("qualityAnalysisArtifactId"), result.qualityAnalysisArtifactId.toString()},
        {QStringLiteral("repairManifestArtifactId"), result.repairManifestArtifactId.toString()},
        {QStringLiteral("qualityReportArtifactId"), result.qualityReportArtifactId.toString()},
        {QStringLiteral("evidenceArtifactId"), result.evidenceArtifactId.toString()},
        {QStringLiteral("summary"), result.summary}};
    if (stored.failure.isFailure()) {
        response.insert(QStringLiteral("failureCode"),
            aitrain::v2::failureCodeToString(stored.failure.code));
        response.insert(wp::field::message(), stored.failure.message);
    } else {
        response.insert(wp::field::message(), QStringLiteral("Data Quality V2 四步工作流已完成。"));
    }
    send(wp::event::dataQualityWorkflowV2(), response);

    const aitrain::v2::TaskState terminalState = result.terminalState;
    const QString terminalMessage = response.value(wp::field::message()).toString();
    dataQualityWorkspaceV2_.reset();
    dataQualityTaskIdV2_ = {};
    running_ = false;
    if (terminalState == aitrain::v2::TaskState::Canceled) {
        sendCanceledAndFinish(taskIdText, terminalMessage);
    } else if (terminalState == aitrain::v2::TaskState::Failed) {
        failWithDetails(terminalMessage,
            response.value(QStringLiteral("failureCode")).toString(
                QStringLiteral("data_quality_v2_failed")), response);
    } else {
        send(wp::event::completed(), QJsonObject{{wp::field::taskId(), taskIdText},
            {wp::field::message(), QStringLiteral("Data Quality V2 completed")}});
        finishSession();
    }
}
