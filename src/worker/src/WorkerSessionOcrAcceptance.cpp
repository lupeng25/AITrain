#include "WorkerSession.h"

#include "aitrain/core/WorkerProtocol.h"

#include <QDateTime>
#include <QFileInfo>
#include <QJsonObject>

namespace wp = aitrain::worker_protocol;

namespace {

bool parseArtifactId(const QString& text, aitrain::ArtifactId* value, QString* error)
{
    return aitrain::ArtifactId::parse(text.trimmed(), value, error);
}

bool parseSnapshotReference(const QJsonObject& object,
    aitrain::OcrOfficialReportImportSource* source, QString* error)
{
    source->reportPath = object.value(QStringLiteral("reportPath")).toString().trimmed();
    const QString snapshotId = object.value(QStringLiteral("snapshotId")).toString().trimmed();
    const QString artifactId = object.value(QStringLiteral("snapshotArtifactId")).toString().trimmed();
    if (source->reportPath.isEmpty() || (snapshotId.isEmpty() && artifactId.isEmpty())) {
        if (error) *error = QStringLiteral("报告路径与 SnapshotId/Snapshot ArtifactId 至少一项不能为空。");
        return false;
    }
    if (!snapshotId.isEmpty()
        && !aitrain::SnapshotId::parse(snapshotId, &source->datasetSnapshotId, error)) return false;
    if (!artifactId.isEmpty()
        && !aitrain::ArtifactId::parse(artifactId, &source->datasetSnapshotArtifactId, error)) return false;
    return true;
}

aitrain::Failure normalizedFailure(const aitrain::Failure& source,
    const QString& fallback)
{
    aitrain::Failure failure = source;
    if (!failure.isFailure()) failure.code = aitrain::FailureCode::InternalError;
    if (failure.message.trimmed().isEmpty()) failure.message = fallback;
    if (failure.suggestedAction.trimmed().isEmpty()) {
        failure.suggestedAction = QStringLiteral("检查  ArtifactId、Snapshot lineage 和官方报告合同后重新执行。");
    }
    if (!failure.occurredAt.isValid()) failure.occurredAt = QDateTime::currentDateTimeUtc();
    return failure;
}

} // namespace

void WorkerSession::importOcrOfficialReports(const wp::OcrOfficialReportImportCommand& command)
{
    const QString taskIdText = command.context.taskId.toString();
    const QString projectRoot = command.context.projectRoot.trimmed();
    QString error;
    aitrain::OcrOfficialReportImportRequest request;
    const aitrain::TaskId taskId = command.context.taskId;
    if (!taskId.isValid() || taskId != controlTaskId_
        || projectRoot.isEmpty() || !QFileInfo(projectRoot).isDir()
        || !parseSnapshotReference(command.det, &request.det, &error)
        || !parseSnapshotReference(command.rec, &request.rec, &error)
        || !parseSnapshotReference(command.system, &request.system, &error)) {
        fail(QStringLiteral("OCR 官方报告受控导入请求无效：%1").arg(error));
        return;
    }
    request.acceptanceCohortId = command.acceptanceCohortId.trimmed();
    request.customerDomainId = command.customerDomainId.trimmed();
    request.evidenceClass = command.evidenceClass.trimmed();

    auto workspace = std::make_unique<aitrain::ProjectWorkspace>();
    if (!workspace->openForWorkerChild(projectRoot, &error)) {
        fail(QStringLiteral("无法打开 OCR Acceptance  工作区：%1").arg(error));
        return;
    }
    aitrain::TaskSnapshot task;
    if (!workspace->startTask(taskId,
            QStringLiteral("paddleocr.official.report.import"),
            QStringLiteral("ocr_official_report_import"), &task, &error)) {
        fail(QStringLiteral("无法启动 OCR 官方报告导入根任务：%1").arg(error));
        return;
    }
    if (!activeWorkflow_.bind(std::move(workspace), taskId, &error)) {
        fail(QStringLiteral("无法绑定 OCR 官方报告导入活动任务：%1").arg(error));
        return;
    }
    auto* const activeWorkspace = activeWorkflow_.workspace();

    activeTaskId_ = taskIdText;
    send(wp::event::progress(), QJsonObject{{wp::field::taskId(), taskIdText},
        {QStringLiteral("percent"), 0},
        {wp::field::message(), QStringLiteral("正在重验三个 Snapshot 与 PaddleOCR 官方报告合同并原子打包。")}});

    aitrain::OcrOfficialReportImportResult result;
    const bool executed = activeWorkspace->importOcrOfficialReports(
        taskId, request, &result, &error, pollingCancellationCallback(0));
    aitrain::Failure failure = normalizedFailure(result.failure,
        error.isEmpty() ? QStringLiteral("OCR 官方报告受控导入失败。") : error);
    aitrain::TaskState terminalState = executed
        ? aitrain::TaskState::Succeeded
        : (failure.code == aitrain::FailureCode::Canceled
            ? aitrain::TaskState::Canceled : aitrain::TaskState::Failed);
    if (!activeWorkspace->finalizeTask(taskId, terminalState,
            executed ? aitrain::Failure{} : failure, &error)) {
        result = {};
        failure = normalizedFailure({}, QStringLiteral("OCR 导入根任务终态持久化失败：%1").arg(error));
        result.failure = failure;
        terminalState = aitrain::TaskState::Failed;
    }

    QJsonObject response{{wp::field::taskId(), taskIdText},
        {QStringLiteral("state"), aitrain::taskStateToString(terminalState)},
        {QStringLiteral("detReportArtifactId"), result.detReportArtifactId.toString()},
        {QStringLiteral("recReportArtifactId"), result.recReportArtifactId.toString()},
        {QStringLiteral("systemReportArtifactId"), result.systemReportArtifactId.toString()},
        {QStringLiteral("evidenceArtifactId"), result.evidenceArtifactId.toString()}};
    if (terminalState != aitrain::TaskState::Succeeded) {
        response.insert(QStringLiteral("failureCode"), aitrain::failureCodeToString(failure.code));
        response.insert(wp::field::message(), failure.message);
    } else {
        response.insert(wp::field::message(), QStringLiteral("三份 OCR 官方报告已完成受控导入。"));
    }
    send(wp::event::ocrOfficialReportsImported(), response);
    publishPersistedTerminal(taskId,
        QStringLiteral("OCR official reports import completed"));
}

void WorkerSession::runOcrAcceptanceWorkflow(const wp::OcrAcceptanceCommand& command)
{
    const QString taskIdText = command.context.taskId.toString();
    const QString projectRoot = command.context.projectRoot.trimmed();
    const QJsonObject thresholds = command.thresholds;
    QString error;
    aitrain::OcrAcceptanceWorkflowRequest request;
    const aitrain::TaskId taskId = command.context.taskId;
    if (!taskId.isValid() || taskId != controlTaskId_
        || projectRoot.isEmpty() || !QFileInfo(projectRoot).isDir()
        || !parseArtifactId(command.detReportArtifactId, &request.detReportArtifactId, &error)
        || !parseArtifactId(command.recReportArtifactId, &request.recReportArtifactId, &error)
        || !parseArtifactId(command.systemReportArtifactId, &request.systemReportArtifactId, &error)) {
        fail(QStringLiteral("OCR Acceptance  只接受有效项目和三个官方报告 ArtifactId：%1").arg(error));
        return;
    }
    request.minimumDetSamples = thresholds.value(QStringLiteral("minimumDetSamples")).toInt(1);
    request.minimumRecSamples = thresholds.value(QStringLiteral("minimumRecSamples")).toInt(1);
    request.minimumSystemSamples = thresholds.value(QStringLiteral("minimumSystemSamples")).toInt(1);
    request.minimumDetHmean = thresholds.value(QStringLiteral("minimumDetHmean")).toDouble(0.50);
    request.minimumRecAccuracy = thresholds.value(QStringLiteral("minimumRecAccuracy")).toDouble(0.70);
    request.maximumRecCer = thresholds.value(QStringLiteral("maximumRecCer")).toDouble(0.30);
    request.minimumSystemAccuracy = thresholds.value(QStringLiteral("minimumSystemAccuracy")).toDouble(0.70);

    auto workspace = std::make_unique<aitrain::ProjectWorkspace>();
    if (!workspace->openForWorkerChild(projectRoot, &error)) {
        fail(QStringLiteral("无法打开 OCR Acceptance  工作区：%1").arg(error));
        return;
    }
    aitrain::TaskSnapshot task;
    if (!workspace->startTask(taskId,
            QStringLiteral("paddleocr.acceptance"), QStringLiteral("ocr_acceptance"),
            &task, &error)) {
        fail(QStringLiteral("无法启动 OCR Acceptance  根任务：%1").arg(error));
        return;
    }
    if (!activeWorkflow_.bind(std::move(workspace), taskId, &error)) {
        fail(QStringLiteral("无法绑定 OCR Acceptance 活动任务：%1").arg(error));
        return;
    }
    auto* const activeWorkspace = activeWorkflow_.workspace();
    activeTaskId_ = taskIdText;
    send(wp::event::progress(), QJsonObject{{wp::field::taskId(), taskIdText},
        {QStringLiteral("percent"), 0},
        {wp::field::message(), QStringLiteral("OCR Acceptance  四步 ArtifactId-only 工作流已启动。")}});

    aitrain::OcrAcceptanceWorkflowResult result;
    const bool executed = activeWorkspace->runOcrAcceptanceWorkflow(
        taskId, request, &result, &error, pollingCancellationCallback(0));
    if (!executed) {
        aitrain::TaskSnapshot stored;
        if (activeWorkspace->task(taskId, &stored, nullptr)
            && !aitrain::isTerminalTaskState(stored.state)) {
            activeWorkspace->finalizeTask(taskId,
                aitrain::TaskState::Failed,
                normalizedFailure({}, error.isEmpty() ? QStringLiteral("OCR Acceptance  持久化失败。") : error), nullptr);
        }
        publishPersistedTerminal(taskId);
        return;
    }

    QJsonObject response{{wp::field::taskId(), taskIdText},
        {QStringLiteral("workflowRunId"), result.workflowRunId.toString()},
        {QStringLiteral("state"), aitrain::taskStateToString(result.terminalState)},
        {QStringLiteral("resolvedEvidenceArtifactId"), result.resolvedEvidenceArtifactId.toString()},
        {QStringLiteral("officialReportValidationArtifactId"), result.officialReportValidationArtifactId.toString()},
        {QStringLiteral("thresholdEvaluationArtifactId"), result.thresholdEvaluationArtifactId.toString()},
        {QStringLiteral("acceptanceReportArtifactId"), result.acceptanceReportArtifactId.toString()},
        {QStringLiteral("evidenceArtifactId"), result.evidenceArtifactId.toString()},
        {QStringLiteral("productionAccepted"), result.productionAccepted}};
    if (result.failure.isFailure()) {
        response.insert(QStringLiteral("failureCode"), aitrain::failureCodeToString(result.failure.code));
        response.insert(wp::field::message(), result.failure.message);
    } else {
        response.insert(wp::field::message(), QStringLiteral("OCR Acceptance  已完成 customer_domain 验收。"));
    }
    send(wp::event::ocrAcceptanceWorkflow(), response);
    publishPersistedTerminal(taskId, QStringLiteral("OCR Acceptance  completed"));
}
