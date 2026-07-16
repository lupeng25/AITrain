#include "WorkerSession.h"

#include "aitrain/core/WorkerProtocol.h"

#include <QDateTime>
#include <QFileInfo>
#include <QJsonObject>

namespace wp = aitrain::worker_protocol;

namespace {

bool parseArtifactId(const QJsonObject& payload, const QString& key,
    aitrain::v2::ArtifactId* value, QString* error)
{
    return aitrain::v2::ArtifactId::parse(payload.value(key).toString().trimmed(), value, error);
}

bool parseSnapshotReference(const QJsonObject& object,
    aitrain::v2::OcrOfficialReportImportSourceV2* source, QString* error)
{
    source->reportPath = object.value(QStringLiteral("reportPath")).toString().trimmed();
    const QString snapshotId = object.value(QStringLiteral("snapshotId")).toString().trimmed();
    const QString artifactId = object.value(QStringLiteral("snapshotArtifactId")).toString().trimmed();
    if (source->reportPath.isEmpty() || (snapshotId.isEmpty() && artifactId.isEmpty())) {
        if (error) *error = QStringLiteral("报告路径与 SnapshotId/Snapshot ArtifactId 至少一项不能为空。");
        return false;
    }
    if (!snapshotId.isEmpty()
        && !aitrain::v2::SnapshotId::parse(snapshotId, &source->datasetSnapshotId, error)) return false;
    if (!artifactId.isEmpty()
        && !aitrain::v2::ArtifactId::parse(artifactId, &source->datasetSnapshotArtifactId, error)) return false;
    return true;
}

aitrain::v2::Failure normalizedFailure(const aitrain::v2::Failure& source,
    const QString& fallback)
{
    aitrain::v2::Failure failure = source;
    if (!failure.isFailure()) failure.code = aitrain::v2::FailureCode::InternalError;
    if (failure.message.trimmed().isEmpty()) failure.message = fallback;
    if (failure.suggestedAction.trimmed().isEmpty()) {
        failure.suggestedAction = QStringLiteral("检查 V2 ArtifactId、Snapshot lineage 和官方报告合同后重新执行。");
    }
    if (!failure.occurredAt.isValid()) failure.occurredAt = QDateTime::currentDateTimeUtc();
    return failure;
}

} // namespace

void WorkerSession::importOcrOfficialReportsV2(const QJsonObject& payload)
{
    if (running_ || ocrAcceptanceWorkspaceV2_) {
        fail(QStringLiteral("Worker 已有运行任务，不能并发导入 OCR 官方报告。"));
        return;
    }
    const QString taskIdText = payload.value(wp::field::taskId()).toString().trimmed();
    const QString projectRoot = payload.value(QStringLiteral("projectRoot")).toString().trimmed();
    QString error;
    aitrain::v2::OcrOfficialReportImportRequestV2 request;
    if (!aitrain::v2::TaskId::parse(taskIdText, &ocrAcceptanceTaskIdV2_, &error)
        || ocrAcceptanceTaskIdV2_ != controlTaskId_
        || projectRoot.isEmpty() || !QFileInfo(projectRoot).isDir()
        || !parseSnapshotReference(payload.value(QStringLiteral("det")).toObject(), &request.det, &error)
        || !parseSnapshotReference(payload.value(QStringLiteral("rec")).toObject(), &request.rec, &error)
        || !parseSnapshotReference(payload.value(QStringLiteral("system")).toObject(), &request.system, &error)) {
        ocrAcceptanceTaskIdV2_ = {};
        fail(QStringLiteral("OCR 官方报告受控导入请求无效：%1").arg(error));
        return;
    }
    request.acceptanceCohortId = payload.value(QStringLiteral("acceptanceCohortId")).toString().trimmed();
    request.customerDomainId = payload.value(QStringLiteral("customerDomainId")).toString().trimmed();
    request.evidenceClass = payload.value(QStringLiteral("evidenceClass")).toString().trimmed();

    ocrAcceptanceWorkspaceV2_ = std::make_unique<aitrain::v2::ProjectWorkspaceV2>();
    if (!ocrAcceptanceWorkspaceV2_->open(projectRoot, &error)) {
        ocrAcceptanceWorkspaceV2_.reset();
        ocrAcceptanceTaskIdV2_ = {};
        fail(QStringLiteral("无法打开 OCR Acceptance V2 工作区：%1").arg(error));
        return;
    }
    aitrain::v2::TaskSnapshot task;
    if (!ocrAcceptanceWorkspaceV2_->startTask(ocrAcceptanceTaskIdV2_,
            QStringLiteral("paddleocr.official.report.import.v2"),
            QStringLiteral("ocr_official_report_import"), &task, &error)) {
        ocrAcceptanceWorkspaceV2_.reset();
        ocrAcceptanceTaskIdV2_ = {};
        fail(QStringLiteral("无法启动 OCR 官方报告导入根任务：%1").arg(error));
        return;
    }

    activeTaskId_ = taskIdText;
    canceled_ = false;
    running_ = true;
    ocrAcceptanceRunningV2_ = true;
    send(wp::event::progress(), QJsonObject{{wp::field::taskId(), taskIdText},
        {QStringLiteral("percent"), 0},
        {wp::field::message(), QStringLiteral("正在重验三个 Snapshot 与 PaddleOCR 官方报告合同并原子打包。")}});

    aitrain::v2::OcrOfficialReportImportResultV2 result;
    const bool executed = ocrAcceptanceWorkspaceV2_->importOcrOfficialReports(
        ocrAcceptanceTaskIdV2_, request, &result, &error, pollingCancellationCallback(0));
    ocrAcceptanceRunningV2_ = false;
    aitrain::v2::Failure failure = normalizedFailure(result.failure,
        error.isEmpty() ? QStringLiteral("OCR 官方报告受控导入失败。") : error);
    aitrain::v2::TaskState terminalState = executed
        ? aitrain::v2::TaskState::Succeeded
        : (failure.code == aitrain::v2::FailureCode::Canceled
            ? aitrain::v2::TaskState::Canceled : aitrain::v2::TaskState::Failed);
    if (!ocrAcceptanceWorkspaceV2_->finalizeTask(ocrAcceptanceTaskIdV2_, terminalState,
            executed ? aitrain::v2::Failure{} : failure, &error)) {
        result = {};
        failure = normalizedFailure({}, QStringLiteral("OCR 导入根任务终态持久化失败：%1").arg(error));
        result.failure = failure;
        terminalState = aitrain::v2::TaskState::Failed;
    }

    QJsonObject response{{wp::field::taskId(), taskIdText},
        {QStringLiteral("state"), aitrain::v2::taskStateToString(terminalState)},
        {QStringLiteral("detReportArtifactId"), result.detReportArtifactId.toString()},
        {QStringLiteral("recReportArtifactId"), result.recReportArtifactId.toString()},
        {QStringLiteral("systemReportArtifactId"), result.systemReportArtifactId.toString()},
        {QStringLiteral("evidenceArtifactId"), result.evidenceArtifactId.toString()}};
    if (terminalState != aitrain::v2::TaskState::Succeeded) {
        response.insert(QStringLiteral("failureCode"), aitrain::v2::failureCodeToString(failure.code));
        response.insert(wp::field::message(), failure.message);
    } else {
        response.insert(wp::field::message(), QStringLiteral("三份 OCR 官方报告已完成受控导入。"));
    }
    send(wp::event::ocrOfficialReportsImportedV2(), response);

    ocrAcceptanceWorkspaceV2_.reset();
    ocrAcceptanceTaskIdV2_ = {};
    running_ = false;
    if (terminalState == aitrain::v2::TaskState::Canceled) {
        sendCanceledAndFinish(taskIdText, failure.message);
    } else if (terminalState == aitrain::v2::TaskState::Failed) {
        failWithDetails(failure.message, aitrain::v2::failureCodeToString(failure.code), response);
    } else {
        send(wp::event::completed(), QJsonObject{{wp::field::taskId(), taskIdText},
            {wp::field::message(), QStringLiteral("OCR official reports import completed")}});
        finishSession();
    }
}

void WorkerSession::runOcrAcceptanceWorkflowV2(const QJsonObject& payload)
{
    if (running_ || ocrAcceptanceWorkspaceV2_) {
        fail(QStringLiteral("Worker 已有运行任务，不能并发运行 OCR Acceptance V2。"));
        return;
    }
    const QString taskIdText = payload.value(wp::field::taskId()).toString().trimmed();
    const QString projectRoot = payload.value(QStringLiteral("projectRoot")).toString().trimmed();
    const QJsonObject thresholds = payload.value(QStringLiteral("thresholds")).toObject();
    QString error;
    aitrain::v2::OcrAcceptanceWorkflowRequestV2 request;
    if (!aitrain::v2::TaskId::parse(taskIdText, &ocrAcceptanceTaskIdV2_, &error)
        || ocrAcceptanceTaskIdV2_ != controlTaskId_
        || projectRoot.isEmpty() || !QFileInfo(projectRoot).isDir()
        || !parseArtifactId(payload, QStringLiteral("detReportArtifactId"), &request.detReportArtifactId, &error)
        || !parseArtifactId(payload, QStringLiteral("recReportArtifactId"), &request.recReportArtifactId, &error)
        || !parseArtifactId(payload, QStringLiteral("systemReportArtifactId"), &request.systemReportArtifactId, &error)) {
        ocrAcceptanceTaskIdV2_ = {};
        fail(QStringLiteral("OCR Acceptance V2 只接受有效项目和三个官方报告 ArtifactId：%1").arg(error));
        return;
    }
    request.minimumDetSamples = thresholds.value(QStringLiteral("minimumDetSamples")).toInt(1);
    request.minimumRecSamples = thresholds.value(QStringLiteral("minimumRecSamples")).toInt(1);
    request.minimumSystemSamples = thresholds.value(QStringLiteral("minimumSystemSamples")).toInt(1);
    request.minimumDetHmean = thresholds.value(QStringLiteral("minimumDetHmean")).toDouble(0.50);
    request.minimumRecAccuracy = thresholds.value(QStringLiteral("minimumRecAccuracy")).toDouble(0.70);
    request.maximumRecCer = thresholds.value(QStringLiteral("maximumRecCer")).toDouble(0.30);
    request.minimumSystemAccuracy = thresholds.value(QStringLiteral("minimumSystemAccuracy")).toDouble(0.70);

    ocrAcceptanceWorkspaceV2_ = std::make_unique<aitrain::v2::ProjectWorkspaceV2>();
    if (!ocrAcceptanceWorkspaceV2_->open(projectRoot, &error)) {
        ocrAcceptanceWorkspaceV2_.reset();
        ocrAcceptanceTaskIdV2_ = {};
        fail(QStringLiteral("无法打开 OCR Acceptance V2 工作区：%1").arg(error));
        return;
    }
    aitrain::v2::TaskSnapshot task;
    if (!ocrAcceptanceWorkspaceV2_->startTask(ocrAcceptanceTaskIdV2_,
            QStringLiteral("paddleocr.acceptance.v2"), QStringLiteral("ocr_acceptance"),
            &task, &error)) {
        ocrAcceptanceWorkspaceV2_.reset();
        ocrAcceptanceTaskIdV2_ = {};
        fail(QStringLiteral("无法启动 OCR Acceptance V2 根任务：%1").arg(error));
        return;
    }
    activeTaskId_ = taskIdText;
    canceled_ = false;
    running_ = true;
    ocrAcceptanceRunningV2_ = true;
    send(wp::event::progress(), QJsonObject{{wp::field::taskId(), taskIdText},
        {QStringLiteral("percent"), 0},
        {wp::field::message(), QStringLiteral("OCR Acceptance V2 四步 ArtifactId-only 工作流已启动。")}});

    aitrain::v2::OcrAcceptanceWorkflowResultV2 result;
    const bool executed = ocrAcceptanceWorkspaceV2_->runOcrAcceptanceWorkflow(
        ocrAcceptanceTaskIdV2_, request, &result, &error, pollingCancellationCallback(0));
    ocrAcceptanceRunningV2_ = false;
    if (!executed) {
        aitrain::v2::TaskSnapshot stored;
        if (ocrAcceptanceWorkspaceV2_->task(ocrAcceptanceTaskIdV2_, &stored, nullptr)
            && !aitrain::v2::isTerminalTaskState(stored.state)) {
            ocrAcceptanceWorkspaceV2_->finalizeTask(ocrAcceptanceTaskIdV2_,
                aitrain::v2::TaskState::Failed,
                normalizedFailure({}, error.isEmpty() ? QStringLiteral("OCR Acceptance V2 持久化失败。") : error), nullptr);
        }
        ocrAcceptanceWorkspaceV2_.reset();
        ocrAcceptanceTaskIdV2_ = {};
        running_ = false;
        failWithDetails(QStringLiteral("OCR Acceptance V2 执行失败：%1").arg(error),
            QStringLiteral("ocr_acceptance_v2_execution_failed"));
        return;
    }

    QJsonObject response{{wp::field::taskId(), taskIdText},
        {QStringLiteral("workflowRunId"), result.workflowRunId.toString()},
        {QStringLiteral("state"), aitrain::v2::taskStateToString(result.terminalState)},
        {QStringLiteral("resolvedEvidenceArtifactId"), result.resolvedEvidenceArtifactId.toString()},
        {QStringLiteral("officialReportValidationArtifactId"), result.officialReportValidationArtifactId.toString()},
        {QStringLiteral("thresholdEvaluationArtifactId"), result.thresholdEvaluationArtifactId.toString()},
        {QStringLiteral("acceptanceReportArtifactId"), result.acceptanceReportArtifactId.toString()},
        {QStringLiteral("evidenceArtifactId"), result.evidenceArtifactId.toString()},
        {QStringLiteral("productionAccepted"), result.productionAccepted}};
    if (result.failure.isFailure()) {
        response.insert(QStringLiteral("failureCode"), aitrain::v2::failureCodeToString(result.failure.code));
        response.insert(wp::field::message(), result.failure.message);
    } else {
        response.insert(wp::field::message(), QStringLiteral("OCR Acceptance V2 已完成 customer_domain 验收。"));
    }
    send(wp::event::ocrAcceptanceWorkflowV2(), response);
    const aitrain::v2::TaskState terminalState = result.terminalState;
    ocrAcceptanceWorkspaceV2_.reset();
    ocrAcceptanceTaskIdV2_ = {};
    running_ = false;
    if (terminalState == aitrain::v2::TaskState::Canceled) {
        sendCanceledAndFinish(taskIdText, result.failure.message);
    } else if (terminalState == aitrain::v2::TaskState::Failed) {
        failWithDetails(result.failure.message,
            aitrain::v2::failureCodeToString(result.failure.code), response);
    } else {
        send(wp::event::completed(), QJsonObject{{wp::field::taskId(), taskIdText},
            {wp::field::message(), QStringLiteral("OCR Acceptance V2 completed")}});
        finishSession();
    }
}
