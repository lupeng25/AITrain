#include "aitrain/workflow/ProjectWorkspace.h"
#include "aitrain/artifact/ArtifactStore.h"

#include <QCryptographicHash>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QSaveFile>
#include <QSet>

#include <cmath>

namespace aitrain {
namespace {

const QStringList kSteps{
    QStringLiteral("ResolveEvidence"),
    QStringLiteral("ValidateOfficialReports"),
    QStringLiteral("EvaluateThresholds"),
    QStringLiteral("RenderAcceptanceReport")};

struct ReportContract final {
    QString component;
    ArtifactId artifactId;
    QString artifactKind;
    QString reportFileName;
    QString backend;
    QString modelFamily;
    QString mode;
};

struct ResolvedReport final {
    ReportContract contract;
    ArtifactSnapshot artifact;
    QJsonObject report;
    QJsonObject lineage;
    QString reportSha256;
};

Failure failure(FailureCode code, const QString& stableCode, const QString& details,
    const QString& action = QStringLiteral("检查已提交官方报告 Artifact、lineage 和验收参数后重新执行。"))
{
    return {code, QStringLiteral("%1:%2").arg(stableCode, details), action,
        QDateTime::currentDateTimeUtc()};
}

WorkflowStepExecutionResult failed(const Failure& value)
{
    return {value.code == FailureCode::Canceled ? WorkflowStepState::Canceled
                                                 : WorkflowStepState::Failed,
        {}, value};
}

bool writeBytes(const QString& path, const QByteArray& bytes, QString* error)
{
    if (!QDir().mkpath(QFileInfo(path).absolutePath())) {
        if (error) *error = QStringLiteral("无法创建 OCR 验收暂存目录。 ");
        return false;
    }
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly) || file.write(bytes) != bytes.size() || !file.commit()) {
        if (error) *error = QStringLiteral("无法写入 OCR 验收暂存文件：%1").arg(path);
        return false;
    }
    return true;
}

bool writeJson(const QString& path, const QJsonObject& object, QString* error)
{
    return writeBytes(path, QJsonDocument(object).toJson(QJsonDocument::Indented), error);
}

QString hashBytes(const QByteArray& bytes)
{
    return QString::fromLatin1(QCryptographicHash::hash(bytes, QCryptographicHash::Sha256).toHex());
}

bool readAndHashFile(const QString& path, QByteArray* bytes, QString* sha256, qint64* byteCount,
    QString* error)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        if (error) *error = QStringLiteral("无法读取已提交 OCR 报告 Artifact 文件：%1").arg(path);
        return false;
    }
    QCryptographicHash hash(QCryptographicHash::Sha256);
    QByteArray content;
    while (!file.atEnd()) {
        const QByteArray block = file.read(1024 * 1024);
        if (block.isEmpty() && file.error() != QFileDevice::NoError) {
            if (error) *error = QStringLiteral("读取已提交 OCR 报告 Artifact 文件失败：%1").arg(path);
            return false;
        }
        hash.addData(block);
        if (bytes) content.append(block);
    }
    if (bytes) *bytes = content;
    if (sha256) *sha256 = QString::fromLatin1(hash.result().toHex());
    if (byteCount) *byteCount = file.size();
    return true;
}

bool safeRelativePath(const QString& relativePath)
{
    const QString normalized = QDir::fromNativeSeparators(relativePath.trimmed());
    return !normalized.isEmpty() && !QDir::isAbsolutePath(normalized)
        && normalized == QDir::cleanPath(normalized)
        && normalized != QStringLiteral("..")
        && !normalized.startsWith(QStringLiteral("../"));
}

bool readJsonObject(const QByteArray& bytes, QJsonObject* result, QString* error)
{
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(bytes, &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        if (error) *error = QStringLiteral("OCR 官方报告不是有效 JSON 对象：%1").arg(parseError.errorString());
        return false;
    }
    *result = document.object();
    return true;
}

bool resolveReport(ProjectStore* storage, const ArtifactStore* store,
    const ReportContract& contract, ResolvedReport* result, Failure* problem)
{
    ArtifactSnapshot artifact;
    QString lookupError;
    if (!storage->artifact(contract.artifactId, &artifact, &lookupError)) {
        *problem = failure(FailureCode::ArtifactIncomplete, QStringLiteral("ocr_acceptance.report_missing"),
            QStringLiteral("%1 报告 Artifact 不存在或未提交").arg(contract.component));
        return false;
    }
    if (artifact.kind != contract.artifactKind) {
        *problem = failure(FailureCode::ArtifactIncompatible, QStringLiteral("ocr_acceptance.artifact_kind_invalid"),
            QStringLiteral("%1 Artifact kind 应为 %2，实际为 %3")
                .arg(contract.component, contract.artifactKind, artifact.kind));
        return false;
    }

    const QString artifactRoot = store->artifactPath(contract.artifactId);
    const QFileInfo rootInfo(artifactRoot);
    if (!rootInfo.exists() || !rootInfo.isDir() || rootInfo.isSymLink()) {
        *problem = failure(FailureCode::ArtifactIncomplete, QStringLiteral("ocr_acceptance.report_missing"),
            QStringLiteral("%1 Artifact 目录不存在或不安全").arg(contract.component));
        return false;
    }

    QHash<QString, QByteArray> contents;
    QHash<QString, QString> hashes;
    QSet<QString> declaredPaths;
    for (const ArtifactFileSnapshot& declared : artifact.files) {
        if (!safeRelativePath(declared.relativePath)) {
            *problem = failure(FailureCode::ArtifactIncompatible, QStringLiteral("ocr_acceptance.artifact_path_invalid"),
                QStringLiteral("%1 Artifact 包含不安全相对路径").arg(contract.component));
            return false;
        }
        const QString absolutePath = QDir(artifactRoot).filePath(declared.relativePath);
        const QFileInfo info(absolutePath);
        const QString canonicalRoot = rootInfo.canonicalFilePath();
        const QString canonicalFile = info.canonicalFilePath();
        if (!info.exists() || !info.isFile() || info.isSymLink() || canonicalRoot.isEmpty()
            || canonicalFile.isEmpty()
            || (!canonicalFile.startsWith(canonicalRoot + QLatin1Char('/'), Qt::CaseInsensitive)
                && !canonicalFile.startsWith(canonicalRoot + QLatin1Char('\\'), Qt::CaseInsensitive))) {
            *problem = failure(FailureCode::ArtifactIncomplete, QStringLiteral("ocr_acceptance.report_missing"),
                QStringLiteral("%1 Artifact 文件缺失：%2").arg(contract.component, declared.relativePath));
            return false;
        }
        QByteArray bytes;
        QString actualHash;
        qint64 actualBytes = 0;
        QString readError;
        if (!readAndHashFile(absolutePath, &bytes, &actualHash, &actualBytes, &readError)
            || actualHash != declared.sha256 || actualBytes != declared.byteCount) {
            *problem = failure(FailureCode::ArtifactIncompatible, QStringLiteral("ocr_acceptance.report_tampered"),
                QStringLiteral("%1 Artifact 文件哈希或长度与提交记录不一致：%2")
                    .arg(contract.component, declared.relativePath));
            return false;
        }
        contents.insert(declared.relativePath, bytes);
        hashes.insert(declared.relativePath, actualHash);
        declaredPaths.insert(QDir::fromNativeSeparators(declared.relativePath));
    }
    QSet<QString> actualPaths;
    QDirIterator iterator(artifactRoot, QDir::Files, QDirIterator::Subdirectories);
    while (iterator.hasNext()) {
        const QString absolutePath = iterator.next();
        const QString relativePath = QDir::fromNativeSeparators(QDir(artifactRoot).relativeFilePath(absolutePath));
        if (relativePath == QStringLiteral("manifest.json")) continue;
        actualPaths.insert(relativePath);
    }
    if (actualPaths != declaredPaths) {
        *problem = failure(FailureCode::ArtifactIncompatible, QStringLiteral("ocr_acceptance.report_tampered"),
            QStringLiteral("%1 Artifact 磁盘文件集合与提交记录不一致").arg(contract.component));
        return false;
    }
    const QString lineagePath = QStringLiteral("lineage/official_report_lineage.json");
    if (!contents.contains(contract.reportFileName)
        || !contents.contains(lineagePath)) {
        *problem = failure(FailureCode::ArtifactIncomplete, QStringLiteral("ocr_acceptance.report_missing"),
            QStringLiteral("%1 Artifact 缺少官方报告或 lineage 文件").arg(contract.component));
        return false;
    }

    QJsonObject report;
    QJsonObject lineage;
    QString jsonError;
    if (!readJsonObject(contents.value(contract.reportFileName), &report, &jsonError)
        || !readJsonObject(contents.value(lineagePath), &lineage, &jsonError)) {
        *problem = failure(FailureCode::ArtifactIncompatible, QStringLiteral("ocr_acceptance.report_schema_invalid"),
            QStringLiteral("%1：%2").arg(contract.component, jsonError));
        return false;
    }
    if (lineage.value(QStringLiteral("schemaVersion")).toInt(-1) != 1
        || lineage.value(QStringLiteral("kind")).toString() != QStringLiteral("paddleocr_official_report_lineage")
        || lineage.value(QStringLiteral("component")).toString() != contract.component
        || lineage.value(QStringLiteral("reportRelativePath")).toString() != contract.reportFileName
        || lineage.value(QStringLiteral("reportSha256")).toString() != hashes.value(contract.reportFileName)) {
        *problem = failure(FailureCode::ArtifactIncompatible, QStringLiteral("ocr_acceptance.lineage_invalid"),
            QStringLiteral("%1 lineage schema、组件或报告哈希不一致").arg(contract.component));
        return false;
    }
    result->contract = contract;
    result->artifact = artifact;
    result->report = report;
    result->lineage = lineage;
    result->reportSha256 = hashes.value(contract.reportFileName);
    return true;
}

bool finiteMetric(const QJsonObject& metrics, const QString& name, double* value)
{
    const QJsonValue item = metrics.value(name);
    if (!item.isDouble() || !std::isfinite(item.toDouble())) return false;
    *value = item.toDouble();
    return true;
}

Failure validateReportSchema(const ResolvedReport& resolved)
{
    const QJsonObject& report = resolved.report;
    const ReportContract& contract = resolved.contract;
    if (!report.value(QStringLiteral("ok")).isBool() || !report.value(QStringLiteral("ok")).toBool()
        || report.value(QStringLiteral("backend")).toString() != contract.backend
        || report.value(QStringLiteral("framework")).toString() != QStringLiteral("PaddleOCR official tools")
        || report.value(QStringLiteral("modelFamily")).toString() != contract.modelFamily
        || report.value(QStringLiteral("mode")).toString() != contract.mode
        || !report.value(QStringLiteral("metrics")).isObject()) {
        return failure(FailureCode::ArtifactIncompatible, QStringLiteral("ocr_acceptance.report_schema_invalid"),
            QStringLiteral("%1 报告缺少成功状态、官方 backend/framework/modelFamily/mode/metrics 合同")
                .arg(contract.component));
    }
    const QJsonObject metrics = report.value(QStringLiteral("metrics")).toObject();
    double sampleCount = 0.0;
    if (!finiteMetric(metrics, QStringLiteral("sampleCount"), &sampleCount)
        || sampleCount < 0.0 || std::floor(sampleCount) != sampleCount) {
        return failure(FailureCode::ArtifactIncompatible, QStringLiteral("ocr_acceptance.report_schema_invalid"),
            QStringLiteral("%1 报告 metrics.sampleCount 缺失或不是非负整数").arg(contract.component));
    }
    const auto probability = [](double value) { return value >= 0.0 && value <= 1.0; };
    double primary = 0.0;
    double secondary = 0.0;
    const bool componentMetricsValid = contract.component == QStringLiteral("det")
        ? finiteMetric(metrics, QStringLiteral("hmean"), &primary) && probability(primary)
        : (contract.component == QStringLiteral("rec")
                ? finiteMetric(metrics, QStringLiteral("accuracy"), &primary)
                    && finiteMetric(metrics, QStringLiteral("cer"), &secondary)
                    && probability(primary) && probability(secondary)
                : finiteMetric(metrics, QStringLiteral("accuracy"), &primary) && probability(primary));
    if (!componentMetricsValid) {
        return failure(FailureCode::ArtifactIncompatible, QStringLiteral("ocr_acceptance.report_schema_invalid"),
            QStringLiteral("%1 报告缺少验收所需官方指标").arg(contract.component));
    }
    return {};
}

bool validThresholds(const OcrAcceptanceWorkflowRequest& request)
{
    const auto probability = [](double value) { return std::isfinite(value) && value >= 0.0 && value <= 1.0; };
    return request.minimumDetSamples > 0 && request.minimumRecSamples > 0
        && request.minimumSystemSamples > 0 && probability(request.minimumDetHmean)
        && probability(request.minimumRecAccuracy) && probability(request.maximumRecCer)
        && probability(request.minimumSystemAccuracy);
}

TaskState taskState(WorkflowStepState state)
{
    if (state == WorkflowStepState::Succeeded) return TaskState::Succeeded;
    if (state == WorkflowStepState::Canceled) return TaskState::Canceled;
    return TaskState::Failed;
}

} // namespace

bool ProjectWorkspace::runOcrAcceptanceWorkflow(const TaskId& taskId,
    const OcrAcceptanceWorkflowRequest& request,
    OcrAcceptanceWorkflowResult* result,
    QString* error,
    const aitrain::CancellationCallback& cancellation)
{
    if (error) error->clear();
    if (!isOpen() || !artifactStore_ || !taskId.isValid() || !result
        || !request.detReportArtifactId.isValid() || !request.recReportArtifactId.isValid()
        || !request.systemReportArtifactId.isValid() || !validThresholds(request)) {
        if (error) *error = QStringLiteral("OCR Acceptance Workflow 需要运行中任务、三个已提交官方报告 ArtifactId 和有效阈值。 ");
        return false;
    }
    *result = {};
    TaskSnapshot rootTask;
    if (!storage_.task(taskId, &rootTask, error) || rootTask.state != TaskState::Running) {
        if (error && error->isEmpty()) *error = QStringLiteral("OCR Acceptance Workflow 只能绑定运行中的根任务。 ");
        return false;
    }

    const QVector<ReportContract> contracts{
        {QStringLiteral("det"), request.detReportArtifactId,
            QStringLiteral("paddleocr_det_official_report"),
            QStringLiteral("report/paddleocr_official_det_report.json"),
            QStringLiteral("paddleocr_det_official"), QStringLiteral("ocr_detection"),
            QStringLiteral("officialEvaluate")},
        {QStringLiteral("rec"), request.recReportArtifactId,
            QStringLiteral("paddleocr_rec_official_report"),
            QStringLiteral("report/paddleocr_official_rec_report.json"),
            QStringLiteral("paddleocr_rec_official"), QStringLiteral("ocr_recognition"),
            QStringLiteral("officialEvaluate")},
        {QStringLiteral("system"), request.systemReportArtifactId,
            QStringLiteral("paddleocr_system_official_report"),
            QStringLiteral("report/paddleocr_official_system_report.json"),
            QStringLiteral("paddleocr_system_official"), QStringLiteral("ocr"),
            QStringLiteral("officialSystemPredict")}};

    WorkflowRunSnapshot workflow;
    workflow.id = WorkflowRunId::create();
    workflow.taskId = taskId;
    workflow.templateId = QStringLiteral("ocr-acceptance");
    workflow.terminalPolicy = WorkflowTerminalPolicy::EvidenceRequired;
    const QJsonObject parameters{
        {QStringLiteral("detReportArtifactId"), request.detReportArtifactId.toString()},
        {QStringLiteral("recReportArtifactId"), request.recReportArtifactId.toString()},
        {QStringLiteral("systemReportArtifactId"), request.systemReportArtifactId.toString()},
        {QStringLiteral("minimumDetSamples"), request.minimumDetSamples},
        {QStringLiteral("minimumRecSamples"), request.minimumRecSamples},
        {QStringLiteral("minimumSystemSamples"), request.minimumSystemSamples},
        {QStringLiteral("minimumDetHmean"), request.minimumDetHmean},
        {QStringLiteral("minimumRecAccuracy"), request.minimumRecAccuracy},
        {QStringLiteral("maximumRecCer"), request.maximumRecCer},
        {QStringLiteral("minimumSystemAccuracy"), request.minimumSystemAccuracy}};
    QVector<WorkflowStepSnapshot> steps;
    for (int ordinal = 0; ordinal < kSteps.size(); ++ordinal) {
        WorkflowStepSnapshot step;
        step.id = WorkflowStepId::create();
        step.workflowRunId = workflow.id;
        step.ordinal = ordinal;
        step.kind = kSteps.at(ordinal);
        step.backend = ordinal == 3 ? QStringLiteral("evidence_renderer")
                                    : QStringLiteral("paddleocr_official_acceptance");
        step.parameterSummary = parameters;
        steps.append(step);
    }
    if (!storage_.createWorkflowRun(workflow, steps, error)) return false;
    result->workflowRunId = workflow.id;

    QVector<ResolvedReport> resolved;
    QJsonObject thresholdFacts;
    QJsonObject acceptanceFacts;
    const QString stagingRoot = runtimeStagingPath(taskId);
    const auto commitReport = [&](const QString& directoryName, const QString& fileName,
                                  const QString& artifactKind, const QJsonObject& document,
                                  ArtifactId* output, QString* stepError) -> bool {
        const QString path = QDir(stagingRoot).filePath(
            QStringLiteral("ocr-acceptance/%1/%2").arg(directoryName, fileName));
        if (!writeJson(path, document, stepError)) return false;
        RuntimeArtifactCandidate candidate;
        candidate.kind = QFileInfo(fileName).completeBaseName();
        candidate.sourcePath = path;
        RuntimeArtifactBundle bundle;
        if (!commitRuntimeArtifacts(taskId, artifactKind, {candidate}, &bundle, stepError)) return false;
        *output = bundle.artifactId;
        return true;
    };

    WorkflowRunner runner(&storage_);
    WorkflowRunExecutionResult execution;
    const bool ran = runner.run(workflow.id,
        [&](const WorkflowStepSnapshot& step,
            const aitrain::CancellationCallback& stepCancellation) -> WorkflowStepExecutionResult {
            if (aitrain::isCancellationRequested(stepCancellation)) {
                return failed(failure(FailureCode::Canceled, QStringLiteral("ocr_acceptance.canceled"),
                    QStringLiteral("步骤 %1 执行前收到取消请求").arg(step.kind),
                    QStringLiteral("确认取消原因后使用新任务重新执行。")));
            }
            QString stepError;
            ArtifactId output;
            if (step.kind == QStringLiteral("ResolveEvidence")) {
                resolved.clear();
                QJsonArray reports;
                for (const ReportContract& contract : contracts) {
                    ResolvedReport report;
                    Failure problem;
                    if (!resolveReport(&storage_, artifactStore_.get(), contract, &report, &problem)) {
                        return failed(problem);
                    }
                    resolved.append(report);
                    reports.append(QJsonObject{{QStringLiteral("component"), contract.component},
                        {QStringLiteral("artifactId"), contract.artifactId.toString()},
                        {QStringLiteral("artifactKind"), contract.artifactKind},
                        {QStringLiteral("reportSha256"), report.reportSha256}});
                }
                const QJsonObject document{{QStringLiteral("schemaVersion"), 1},
                    {QStringLiteral("kind"), QStringLiteral("ocr_acceptance_resolved_evidence")},
                    {QStringLiteral("reports"), reports}};
                if (!commitReport(QStringLiteral("01-resolve-evidence"), QStringLiteral("resolved_evidence.json"),
                        QStringLiteral("ocr_acceptance_resolved_evidence"), document, &output, &stepError)) {
                    return failed(failure(FailureCode::ArtifactIncomplete,
                        QStringLiteral("ocr_acceptance.output_commit_failed"), stepError));
                }
                result->resolvedEvidenceArtifactId = output;
                return {WorkflowStepState::Succeeded, output, {}};
            }
            if (step.kind == QStringLiteral("ValidateOfficialReports")) {
                if (resolved.size() != 3) {
                    return failed(failure(FailureCode::InternalError,
                        QStringLiteral("ocr_acceptance.resolve_state_missing"), QStringLiteral("报告解析状态不完整")));
                }
                const QString cohort = resolved.first().lineage.value(QStringLiteral("acceptanceCohortId")).toString().trimmed();
                const QString domain = resolved.first().lineage.value(QStringLiteral("customerDomainId")).toString().trimmed();
                for (const ResolvedReport& report : resolved) {
                    const Failure schemaFailure = validateReportSchema(report);
                    if (schemaFailure.isFailure()) return failed(schemaFailure);
                    const QJsonObject& lineage = report.lineage;
                    const QString evidenceClass = lineage.value(QStringLiteral("evidenceClass")).toString();
                    if (evidenceClass != QStringLiteral("customer_domain")) {
                        return failed(failure(FailureCode::InvalidDataset,
                            QStringLiteral("ocr_acceptance.customer_domain_evidence_required"),
                            QStringLiteral("%1 报告证据分类为 %2，不能生成 production accepted")
                                .arg(report.contract.component, evidenceClass)));
                    }
                    if (cohort.isEmpty() || domain.isEmpty()
                        || lineage.value(QStringLiteral("acceptanceCohortId")).toString() != cohort
                        || lineage.value(QStringLiteral("customerDomainId")).toString() != domain
                        || lineage.value(QStringLiteral("datasetFingerprint")).toString().trimmed().isEmpty()) {
                        return failed(failure(FailureCode::ArtifactIncompatible,
                            QStringLiteral("ocr_acceptance.lineage_mismatch"),
                            QStringLiteral("Det/Rec/System 的验收批次、客户域或数据集指纹不完整/不一致")));
                    }
                }
                const QJsonObject upstream = resolved.at(2).lineage.value(QStringLiteral("upstream")).toObject();
                if (upstream.value(QStringLiteral("detReportSha256")).toString() != resolved.at(0).reportSha256
                    || upstream.value(QStringLiteral("recReportSha256")).toString() != resolved.at(1).reportSha256) {
                    return failed(failure(FailureCode::ArtifactIncompatible,
                        QStringLiteral("ocr_acceptance.lineage_mismatch"),
                        QStringLiteral("System 报告未绑定当前 Det/Rec 官方报告哈希")));
                }
                QJsonArray identities;
                for (const ResolvedReport& report : resolved) {
                    identities.append(QJsonObject{{QStringLiteral("component"), report.contract.component},
                        {QStringLiteral("backend"), report.contract.backend},
                        {QStringLiteral("reportSha256"), report.reportSha256}});
                }
                const QJsonObject document{{QStringLiteral("schemaVersion"), 1},
                    {QStringLiteral("kind"), QStringLiteral("ocr_acceptance_official_report_validation")},
                    {QStringLiteral("acceptanceCohortId"), cohort},
                    {QStringLiteral("customerDomainId"), domain},
                    {QStringLiteral("evidenceClass"), QStringLiteral("customer_domain")},
                    {QStringLiteral("officialReports"), identities}};
                if (!commitReport(QStringLiteral("02-validate-official-reports"),
                        QStringLiteral("official_report_validation.json"),
                        QStringLiteral("ocr_acceptance_official_report_validation"), document, &output, &stepError)) {
                    return failed(failure(FailureCode::ArtifactIncomplete,
                        QStringLiteral("ocr_acceptance.output_commit_failed"), stepError));
                }
                result->officialReportValidationArtifactId = output;
                return {WorkflowStepState::Succeeded, output, {}};
            }
            if (step.kind == QStringLiteral("EvaluateThresholds")) {
                const QJsonObject det = resolved.at(0).report.value(QStringLiteral("metrics")).toObject();
                const QJsonObject rec = resolved.at(1).report.value(QStringLiteral("metrics")).toObject();
                const QJsonObject system = resolved.at(2).report.value(QStringLiteral("metrics")).toObject();
                const int detSamples = det.value(QStringLiteral("sampleCount")).toInt();
                const int recSamples = rec.value(QStringLiteral("sampleCount")).toInt();
                const int systemSamples = system.value(QStringLiteral("sampleCount")).toInt();
                const double detHmean = det.value(QStringLiteral("hmean")).toDouble();
                const double recAccuracy = rec.value(QStringLiteral("accuracy")).toDouble();
                const double recCer = rec.value(QStringLiteral("cer")).toDouble();
                const double systemAccuracy = system.value(QStringLiteral("accuracy")).toDouble();
                thresholdFacts = {{QStringLiteral("detSamples"), detSamples},
                    {QStringLiteral("recSamples"), recSamples},
                    {QStringLiteral("systemSamples"), systemSamples},
                    {QStringLiteral("detHmean"), detHmean},
                    {QStringLiteral("recAccuracy"), recAccuracy},
                    {QStringLiteral("recCer"), recCer},
                    {QStringLiteral("systemAccuracy"), systemAccuracy},
                    {QStringLiteral("productionAccepted"), false}};
                if (detSamples < request.minimumDetSamples || recSamples < request.minimumRecSamples
                    || systemSamples < request.minimumSystemSamples) {
                    return failed(failure(FailureCode::InvalidDataset,
                        QStringLiteral("ocr_acceptance.sample_count_insufficient"),
                        QStringLiteral("样本数 Det=%1/%2、Rec=%3/%4、System=%5/%6")
                            .arg(detSamples).arg(request.minimumDetSamples).arg(recSamples)
                            .arg(request.minimumRecSamples).arg(systemSamples).arg(request.minimumSystemSamples)));
                }
                if (detHmean < request.minimumDetHmean || recAccuracy < request.minimumRecAccuracy
                    || recCer > request.maximumRecCer || systemAccuracy < request.minimumSystemAccuracy) {
                    return failed(failure(FailureCode::InvalidDataset,
                        QStringLiteral("ocr_acceptance.threshold_not_met"),
                        QStringLiteral("指标 Det hmean=%1/%2、Rec accuracy=%3/%4、Rec CER=%5/%6、System accuracy=%7/%8")
                            .arg(detHmean).arg(request.minimumDetHmean).arg(recAccuracy)
                            .arg(request.minimumRecAccuracy).arg(recCer).arg(request.maximumRecCer)
                            .arg(systemAccuracy).arg(request.minimumSystemAccuracy)));
                }
                thresholdFacts.insert(QStringLiteral("productionAccepted"), true);
                const QJsonObject document{{QStringLiteral("schemaVersion"), 1},
                    {QStringLiteral("kind"), QStringLiteral("ocr_acceptance_threshold_evaluation")},
                    {QStringLiteral("observed"), thresholdFacts},
                    {QStringLiteral("thresholds"), parameters}};
                if (!commitReport(QStringLiteral("03-evaluate-thresholds"),
                        QStringLiteral("threshold_evaluation.json"),
                        QStringLiteral("ocr_acceptance_threshold_evaluation"), document, &output, &stepError)) {
                    return failed(failure(FailureCode::ArtifactIncomplete,
                        QStringLiteral("ocr_acceptance.output_commit_failed"), stepError));
                }
                result->thresholdEvaluationArtifactId = output;
                return {WorkflowStepState::Succeeded, output, {}};
            }
            if (step.kind == QStringLiteral("RenderAcceptanceReport")) {
                QJsonArray limitations;
                limitations.append(QStringLiteral("本结论仅覆盖该客户域验收批次和已提交的 PaddleOCR Det/Rec/System 官方报告。"));
                limitations.append(QStringLiteral("它不证明其他客户域、Clean Windows、TensorRT、NCNN、SMP、Anomaly 或 OBB 验收通过。"));
                acceptanceFacts = QJsonObject{};
                acceptanceFacts.insert(QStringLiteral("schemaVersion"), 1);
                acceptanceFacts.insert(QStringLiteral("kind"), QStringLiteral("ocr_acceptance_report"));
                acceptanceFacts.insert(QStringLiteral("status"), QStringLiteral("accepted"));
                acceptanceFacts.insert(QStringLiteral("productionAccepted"), true);
                acceptanceFacts.insert(QStringLiteral("evidenceClass"), QStringLiteral("customer_domain"));
                acceptanceFacts.insert(QStringLiteral("detReportArtifactId"), request.detReportArtifactId.toString());
                acceptanceFacts.insert(QStringLiteral("recReportArtifactId"), request.recReportArtifactId.toString());
                acceptanceFacts.insert(QStringLiteral("systemReportArtifactId"), request.systemReportArtifactId.toString());
                acceptanceFacts.insert(QStringLiteral("metrics"), thresholdFacts);
                acceptanceFacts.insert(QStringLiteral("limitations"), limitations);
                const QString jsonPath = QDir(stagingRoot).filePath(
                    QStringLiteral("ocr-acceptance/04-render-report/ocr_acceptance_report.json"));
                const QString markdownPath = QDir(stagingRoot).filePath(
                    QStringLiteral("ocr-acceptance/04-render-report/ocr_acceptance_report.md"));
                const QString markdown = QStringLiteral("# OCR 客户域验收报告 \n\n"
                    "- 结论：已接受\n- 证据分类：customer_domain\n"
                    "- Det 报告 Artifact：`%1`\n- Rec 报告 Artifact：`%2`\n"
                    "- System 报告 Artifact：`%3`\n\n"
                    "本结论只覆盖当前验收批次，不代表其他客户域或其他运行时/算法验收通过。\n")
                    .arg(request.detReportArtifactId.toString(), request.recReportArtifactId.toString(),
                        request.systemReportArtifactId.toString());
                if (!writeJson(jsonPath, acceptanceFacts, &stepError)
                    || !writeBytes(markdownPath, markdown.toUtf8(), &stepError)) {
                    return failed(failure(FailureCode::ArtifactIncomplete,
                        QStringLiteral("ocr_acceptance.output_commit_failed"), stepError));
                }
                QVector<RuntimeArtifactCandidate> candidates;
                candidates.append({QStringLiteral("acceptance_report_json"), jsonPath});
                candidates.append({QStringLiteral("acceptance_report_markdown"), markdownPath});
                RuntimeArtifactBundle bundle;
                if (!commitRuntimeArtifacts(taskId, QStringLiteral("ocr_acceptance_report"),
                        candidates, &bundle, &stepError)) {
                    return failed(failure(FailureCode::ArtifactIncomplete,
                        QStringLiteral("ocr_acceptance.output_commit_failed"), stepError));
                }
                result->acceptanceReportArtifactId = bundle.artifactId;
                return {WorkflowStepState::Succeeded, bundle.artifactId, {}};
            }
            return failed(failure(FailureCode::InternalError,
                QStringLiteral("ocr_acceptance.unknown_step"), step.kind));
        }, &execution, error, cancellation);
    if (!ran) {
        QString ignored;
        cleanupRuntimeStaging(taskId, &ignored);
        return false;
    }

    QString cleanupError;
    TaskState terminalState = taskState(execution.state);
    Failure terminalFailure = execution.failure;
    if (!cleanupRuntimeStaging(taskId, &cleanupError)) {
        terminalState = TaskState::Failed;
        terminalFailure = failure(FailureCode::ArtifactIncomplete,
            QStringLiteral("ocr_acceptance.staging_cleanup_failed"), cleanupError);
    }
    if (terminalState == TaskState::Canceled) {
        TaskSnapshot current;
        if (!storage_.task(taskId, &current, error)) return false;
        if (current.state == TaskState::Running && !taskCoordinator_->requestCancellation(taskId, error)) return false;
    }
    if (terminalState == TaskState::Succeeded) {
        terminalFailure = {};
    } else {
        terminalFailure.code = terminalState == TaskState::Canceled
            ? FailureCode::Canceled
            : (terminalFailure.isFailure() && terminalFailure.code != FailureCode::Canceled
                    ? terminalFailure.code : FailureCode::InternalError);
        if (terminalFailure.message.trimmed().isEmpty()) {
            terminalFailure.message = QStringLiteral("ocr_acceptance.terminal_failure_missing:工作流未返回失败说明");
        }
        if (terminalFailure.suggestedAction.trimmed().isEmpty()) {
            terminalFailure.suggestedAction = terminalState == TaskState::Canceled
                ? QStringLiteral("确认取消原因后使用新任务重新执行。")
                : QStringLiteral("检查官方报告 Artifact、lineage 和失败步骤后重新执行。");
        }
        if (!terminalFailure.occurredAt.isValid()) {
            terminalFailure.occurredAt = QDateTime::currentDateTimeUtc();
        }
    }
    if (!storage_.sealWorkflowTerminalization(workflow.id, terminalState, terminalFailure,
            QDateTime::currentDateTimeUtc(), error)) return false;

    EvidenceBundle evidence;
    if (!buildWorkflowEvidenceBundle(workflow.id, &evidence, error)) return false;
    evidence.evaluation = {{QStringLiteral("kind"), QStringLiteral("ocr_acceptance")},
        {QStringLiteral("productionAccepted"), terminalState == TaskState::Succeeded},
        {QStringLiteral("thresholds"), parameters},
        {QStringLiteral("observed"), thresholdFacts}};
    evidence.runtimeStatus.insert(QStringLiteral("officialOnly"), true);
    evidence.runtimeStatus.insert(QStringLiteral("detBackend"), QStringLiteral("paddleocr_det_official"));
    evidence.runtimeStatus.insert(QStringLiteral("recBackend"), QStringLiteral("paddleocr_rec_official"));
    evidence.runtimeStatus.insert(QStringLiteral("systemBackend"), QStringLiteral("paddleocr_system_official"));
    evidence.limitations.append(QStringLiteral("OCR Acceptance  只消费已提交的 PaddleOCR Det/Rec/System 官方报告 ArtifactId，不接受裸报告路径。"));
    evidence.limitations.append(QStringLiteral("public、generated 或 smoke 证据不得生成 production accepted；客户域结论仅覆盖当前验收批次。"));
    EvidenceArtifactBundle evidenceArtifact;
    if (!commitEvidenceBundle(evidence, &evidenceArtifact, error)) {
        const Failure evidenceFailure = failure(FailureCode::ArtifactIncomplete,
            QStringLiteral("ocr_acceptance.evidence_commit_failed"),
            error && !error->isEmpty() ? *error : QStringLiteral("Evidence Bundle 提交失败"));
        QString ignored;
        recordWorkflowEvidenceFailure(workflow.id, evidenceFailure, &ignored);
        return false;
    }
    if (!closeWorkflowTerminalization(workflow.id, error)) return false;

    result->terminalState = terminalState;
    result->evidenceArtifactId = evidenceArtifact.artifactId;
    result->productionAccepted = terminalState == TaskState::Succeeded;
    result->failure = terminalFailure;
    return true;
}

} // namespace aitrain
