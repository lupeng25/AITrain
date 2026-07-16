#include "aitrain/workflow/ProjectWorkspace.h"
#include "aitrain/artifact/ArtifactStore.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QSaveFile>
#include <QSet>
#include <QStringList>

#include <algorithm>
#include <cmath>

namespace aitrain {
namespace {

struct VerifiedOcrSnapshot final {
    DatasetSnapshotRecord record;
    QString rootPath;
    QJsonObject manifest;
    int sampleCount = 0;
};

struct PreparedOcrReport final {
    QString component;
    QString artifactKind;
    QString normalizedFileName;
    QString rawFileName;
    QByteArray rawBytes;
    QByteArray normalizedBytes;
    QByteArray lineageBytes;
    QString normalizedSha256;
};

bool canceled(const aitrain::CancellationCallback& callback)
{
    return aitrain::isCancellationRequested(callback);
}

QString sha256(const QByteArray& bytes)
{
    return QString::fromLatin1(QCryptographicHash::hash(bytes, QCryptographicHash::Sha256).toHex());
}

Failure importFailure(FailureCode code, const QString& stableCode, const QString& details)
{
    return {code, QStringLiteral("%1:%2").arg(stableCode, details),
        code == FailureCode::Canceled
            ? QStringLiteral("确认取消原因后重新选择三份官方报告执行导入。")
            : QStringLiteral("检查原始官方报告、已提交 Snapshot 及客户域元数据后重新导入。"),
        QDateTime::currentDateTimeUtc()};
}

bool failImport(OcrOfficialReportImportResult* result, FailureCode code,
    const QString& stableCode, const QString& details, QString* error)
{
    result->failure = importFailure(code, stableCode, details);
    if (error) *error = result->failure.message;
    return false;
}

bool readFile(const QString& path, QByteArray* bytes, QString* hash,
    const aitrain::CancellationCallback& cancellation, QString* error)
{
    const QFileInfo info(path);
    if (!info.exists() || !info.isFile() || info.isSymLink()) {
        if (error) *error = QStringLiteral("文件不存在、不是普通文件或为符号链接：%1").arg(path);
        return false;
    }
    QFile file(info.absoluteFilePath());
    if (!file.open(QIODevice::ReadOnly)) {
        if (error) *error = QStringLiteral("无法读取文件：%1").arg(path);
        return false;
    }
    QCryptographicHash digest(QCryptographicHash::Sha256);
    QByteArray content;
    while (!file.atEnd()) {
        if (canceled(cancellation)) {
            if (error) *error = QStringLiteral("ocr_report_import.canceled");
            return false;
        }
        const QByteArray block = file.read(1024 * 1024);
        if (block.isEmpty() && file.error() != QFileDevice::NoError) {
            if (error) *error = QStringLiteral("读取文件失败：%1").arg(path);
            return false;
        }
        digest.addData(block);
        if (bytes) content.append(block);
    }
    if (bytes) *bytes = content;
    if (hash) *hash = QString::fromLatin1(digest.result().toHex());
    return true;
}

bool parseObject(const QByteArray& bytes, QJsonObject* object, QString* error)
{
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(bytes, &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        if (error) *error = QStringLiteral("JSON 不是有效对象：%1").arg(parseError.errorString());
        return false;
    }
    *object = document.object();
    return true;
}

bool safeRelative(const QString& path)
{
    const QString normalized = QDir::fromNativeSeparators(path.trimmed());
    return !normalized.isEmpty() && !QDir::isAbsolutePath(normalized)
        && normalized == QDir::cleanPath(normalized)
        && normalized != QStringLiteral("..")
        && !normalized.startsWith(QStringLiteral("../"));
}

int countLabelSamples(const QString& format, const QString& relativePath, const QByteArray& bytes)
{
    const QString name = QFileInfo(relativePath).fileName();
    const bool selected = format == QStringLiteral("paddleocr_det")
        ? name.startsWith(QStringLiteral("det_gt")) && name.endsWith(QStringLiteral(".txt"))
        : name.startsWith(QStringLiteral("rec_gt")) && name.endsWith(QStringLiteral(".txt"));
    if (!selected) return 0;
    int count = 0;
    for (const QByteArray& line : bytes.split('\n')) {
        const QByteArray row = line.trimmed();
        if (!row.isEmpty() && row.indexOf('\t') > 0) ++count;
    }
    return count;
}

bool resolveSnapshot(ProjectStore* storage, const ArtifactStore* store,
    const OcrOfficialReportImportSource& source, const QString& component,
    VerifiedOcrSnapshot* result, const aitrain::CancellationCallback& cancellation,
    QString* error)
{
    DatasetSnapshotRecord record;
    if (source.datasetSnapshotId.isValid()) {
        if (!storage->datasetSnapshot(source.datasetSnapshotId, &record, error)) return false;
    } else if (source.datasetSnapshotArtifactId.isValid()) {
        if (!storage->datasetSnapshotForArtifact(source.datasetSnapshotArtifactId, &record, error)) return false;
    } else {
        if (error) *error = QStringLiteral("%1 缺少 SnapshotId 或 Snapshot ArtifactId").arg(component);
        return false;
    }
    if (source.datasetSnapshotArtifactId.isValid()
        && record.artifactId != source.datasetSnapshotArtifactId) {
        if (error) *error = QStringLiteral("%1 的 SnapshotId 与 ArtifactId 不一致").arg(component);
        return false;
    }
    if (record.datasetFormat != QStringLiteral("paddleocr_det")
        && record.datasetFormat != QStringLiteral("paddleocr_rec")) {
        if (error) *error = QStringLiteral("%1 Snapshot 格式不是 PaddleOCR Det/Rec：%2")
            .arg(component, record.datasetFormat);
        return false;
    }
    if (component == QStringLiteral("det") && record.datasetFormat != QStringLiteral("paddleocr_det")) {
        if (error) *error = QStringLiteral("Det 报告必须绑定 paddleocr_det Snapshot");
        return false;
    }
    if (component == QStringLiteral("rec") && record.datasetFormat != QStringLiteral("paddleocr_rec")) {
        if (error) *error = QStringLiteral("Rec 报告必须绑定 paddleocr_rec Snapshot");
        return false;
    }

    ArtifactSnapshot artifact;
    if (!storage->artifact(record.artifactId, &artifact, error)
        || artifact.kind != QStringLiteral("dataset_snapshot")) {
        if (error && error->isEmpty()) *error = QStringLiteral("%1 Snapshot Artifact 无效").arg(component);
        return false;
    }
    const auto manifestFile = std::find_if(artifact.files.cbegin(), artifact.files.cend(),
        [](const ArtifactFileSnapshot& file) {
            return file.relativePath == QStringLiteral("dataset_snapshot.json");
        });
    const QString artifactRoot = store->artifactPath(artifact.id);
    QByteArray manifestBytes;
    QString manifestHash;
    if (manifestFile == artifact.files.cend()
        || !readFile(QDir(artifactRoot).filePath(QStringLiteral("dataset_snapshot.json")),
            &manifestBytes, &manifestHash, cancellation, error)
        || manifestHash != manifestFile->sha256 || manifestHash != record.manifestSha256
        || manifestBytes.size() != manifestFile->byteCount) {
        if (error && error->isEmpty()) *error = QStringLiteral("%1 Snapshot manifest 完整性校验失败").arg(component);
        return false;
    }
    QJsonObject manifest;
    if (!parseObject(manifestBytes, &manifest, error)
        || manifest.value(QStringLiteral("schemaVersion")).toInt(-1) != 2
        || !manifest.value(QStringLiteral("complete")).toBool()
        || manifest.value(QStringLiteral("snapshotId")).toString() != record.id.toString()
        || manifest.value(QStringLiteral("datasetFormat")).toString() != record.datasetFormat
        || manifest.value(QStringLiteral("rootHash")).toString() != record.rootHash) {
        if (error && error->isEmpty()) *error = QStringLiteral("%1 Snapshot manifest 合同无效").arg(component);
        return false;
    }
    const QString rootPath = store->artifactPath(record.artifactId);
    if (rootPath.isEmpty()) {
        if (error) *error = QStringLiteral("%1 Snapshot Artifact 路径无效").arg(component);
        return false;
    }
    const QDir root(rootPath);
    if (!root.exists()) {
        if (error) *error = QStringLiteral("%1 Snapshot 源 locator 当前不可用").arg(component);
        return false;
    }
    int sampleCount = 0;
    int fileCount = 0;
    for (const QJsonValue& value : manifest.value(QStringLiteral("files")).toArray()) {
        if (canceled(cancellation)) {
            if (error) *error = QStringLiteral("ocr_report_import.canceled");
            return false;
        }
        const QJsonObject declared = value.toObject();
        const QString relative = declared.value(QStringLiteral("relativePath")).toString();
        QByteArray bytes;
        QString actualHash;
        if (!safeRelative(relative)
            || !readFile(root.filePath(relative), &bytes, &actualHash, cancellation, error)
            || actualHash != declared.value(QStringLiteral("sha256")).toString()
            || bytes.size() != declared.value(QStringLiteral("bytes")).toString().toLongLong()) {
            if (error && error->isEmpty()) *error = QStringLiteral("%1 Snapshot 源文件已变化：%2").arg(component, relative);
            return false;
        }
        ++fileCount;
        sampleCount += countLabelSamples(record.datasetFormat, relative, bytes);
    }
    if (fileCount != record.fileCount || sampleCount <= 0) {
        if (error) *error = QStringLiteral("%1 Snapshot 文件数或可验证标签样本数无效").arg(component);
        return false;
    }
    result->record = record;
    result->rootPath = rootPath;
    result->manifest = manifest;
    result->sampleCount = sampleCount;
    return true;
}

bool finiteMetric(const QJsonObject& metrics, const QString& name, double* value)
{
    const QJsonValue item = metrics.value(name);
    if (!item.isDouble() || !std::isfinite(item.toDouble())) return false;
    *value = item.toDouble();
    return true;
}

bool reportBindsSnapshot(const QJsonObject& report, const VerifiedOcrSnapshot& snapshot,
    const aitrain::CancellationCallback& cancellation)
{
    if (report.value(QStringLiteral("datasetSnapshotId")).toString() == snapshot.record.id.toString()) {
        return true;
    }
    const QString manifestPath = report.value(QStringLiteral("datasetSnapshotManifest")).toString().trimmed();
    if (manifestPath.isEmpty()) return false;
    QByteArray bytes;
    QString hash;
    QString ignored;
    if (!readFile(manifestPath, &bytes, &hash, cancellation, &ignored)
        || hash != snapshot.record.manifestSha256) return false;
    QJsonObject manifest;
    return parseObject(bytes, &manifest, &ignored)
        && manifest.value(QStringLiteral("snapshotId")).toString() == snapshot.record.id.toString();
}

bool prepareReport(const QString& component, const OcrOfficialReportImportSource& source,
    const VerifiedOcrSnapshot& snapshot, const QString& cohort, const QString& domain,
    const QString& evidenceClass, const QString& detHash, const QString& recHash,
    PreparedOcrReport* prepared, const aitrain::CancellationCallback& cancellation,
    Failure* problem)
{
    QByteArray rawBytes;
    QString rawHash;
    QString readError;
    if (!readFile(source.reportPath, &rawBytes, &rawHash, cancellation, &readError)) {
        *problem = importFailure(canceled(cancellation) ? FailureCode::Canceled : FailureCode::ArtifactIncomplete,
            canceled(cancellation) ? QStringLiteral("ocr_report_import.canceled")
                                   : QStringLiteral("ocr_report_import.report_unreadable"),
            QStringLiteral("%1：%2").arg(component, readError));
        return false;
    }
    QJsonObject raw;
    if (!parseObject(rawBytes, &raw, &readError)) {
        *problem = importFailure(FailureCode::ArtifactIncompatible,
            QStringLiteral("ocr_report_import.invalid_evidence"),
            QStringLiteral("%1 原始报告 JSON 无效：%2").arg(component, readError));
        return false;
    }
    const QJsonObject metrics = raw.value(QStringLiteral("metrics")).toObject();
    double primary = 0.0;
    double secondary = 0.0;
    const bool probability = component == QStringLiteral("det")
        ? finiteMetric(metrics, QStringLiteral("hmean"), &primary)
        : (finiteMetric(metrics, QStringLiteral("accuracy"), &primary)
            || (component == QStringLiteral("rec")
                && finiteMetric(metrics, QStringLiteral("acc"), &primary)));
    const bool componentMetrics = component != QStringLiteral("rec")
        || finiteMetric(metrics, QStringLiteral("cer"), &secondary);
    const auto inRange = [](double value) { return value >= 0.0 && value <= 1.0; };

    QString sourceBackend;
    QString sourceMode;
    QString normalizedBackend;
    QString modelFamily;
    if (component == QStringLiteral("det")) {
        sourceBackend = QStringLiteral("paddleocr_det_official_eval");
        sourceMode = QStringLiteral("officialEvaluate");
        normalizedBackend = QStringLiteral("paddleocr_det_official");
        modelFamily = QStringLiteral("ocr_detection");
        if (raw.value(QStringLiteral("schemaVersion")).toInt(-1) != 2
            || raw.value(QStringLiteral("backend")).toString() != sourceBackend
            || raw.value(QStringLiteral("taskType")).toString() != modelFamily
            || raw.value(QStringLiteral("component")).toString() != component) {
            *problem = importFailure(FailureCode::ArtifactIncompatible,
                QStringLiteral("ocr_report_import.invalid_evidence"),
                QStringLiteral("Det 报告不是当前 PaddleOCR 官方 evaluator schema/backend"));
            return false;
        }
    } else if (component == QStringLiteral("rec")) {
        sourceBackend = QStringLiteral("paddleocr_rec_official_eval");
        sourceMode = QStringLiteral("officialEvaluate");
        normalizedBackend = QStringLiteral("paddleocr_rec_official");
        modelFamily = QStringLiteral("ocr_recognition");
        if (raw.value(QStringLiteral("schemaVersion")).toInt(-1) != 2
            || raw.value(QStringLiteral("backend")).toString() != sourceBackend
            || raw.value(QStringLiteral("taskType")).toString() != modelFamily
            || raw.value(QStringLiteral("component")).toString() != component) {
            *problem = importFailure(FailureCode::ArtifactIncompatible,
                QStringLiteral("ocr_report_import.invalid_evidence"),
                QStringLiteral("Rec 报告不是当前 PaddleOCR 官方 evaluator schema/backend"));
            return false;
        }
    } else {
        sourceBackend = QStringLiteral("paddleocr_system_official");
        sourceMode = QStringLiteral("officialSystemPredict");
        normalizedBackend = sourceBackend;
        modelFamily = QStringLiteral("ocr");
        if (!raw.value(QStringLiteral("ok")).toBool()
            || raw.value(QStringLiteral("backend")).toString() != sourceBackend
            || raw.value(QStringLiteral("framework")).toString() != QStringLiteral("PaddleOCR official tools")
            || raw.value(QStringLiteral("mode")).toString() != sourceMode) {
            *problem = importFailure(FailureCode::ArtifactIncompatible,
                QStringLiteral("ocr_report_import.invalid_evidence"),
                QStringLiteral("System 报告缺少成功状态或官方 backend/framework/mode"));
            return false;
        }
        if (!finiteMetric(metrics, QStringLiteral("accuracy"), &primary)) {
            *problem = importFailure(FailureCode::BackendUnsupported,
                QStringLiteral("ocr_report_import.system_accuracy_unsupported"),
                QStringLiteral("当前 System 官方报告没有可验证 accuracy，不能生成验收 Artifact"));
            return false;
        }
        if (raw.value(QStringLiteral("predictionCount")).isDouble()
            && raw.value(QStringLiteral("predictionCount")).toInt(-1) != snapshot.sampleCount) {
            *problem = importFailure(FailureCode::ArtifactIncompatible,
                QStringLiteral("ocr_report_import.sample_count_mismatch"),
                QStringLiteral("System predictionCount 与 committed Snapshot 样本数不一致"));
            return false;
        }
    }
    if (!probability || !componentMetrics || !inRange(primary)
        || (component == QStringLiteral("rec") && !inRange(secondary))) {
        *problem = importFailure(FailureCode::ArtifactIncompatible,
            QStringLiteral("ocr_report_import.invalid_evidence"),
            QStringLiteral("%1 报告缺少验收所需的真实官方指标").arg(component));
        return false;
    }
    if (!reportBindsSnapshot(raw, snapshot, cancellation)) {
        *problem = importFailure(FailureCode::ArtifactIncompatible,
            QStringLiteral("ocr_report_import.snapshot_lineage_invalid"),
            QStringLiteral("%1 报告不能绑定到所选 committed Snapshot").arg(component));
        return false;
    }

    QJsonObject normalizedMetrics{{QStringLiteral("sampleCount"), snapshot.sampleCount}};
    if (component == QStringLiteral("det")) normalizedMetrics.insert(QStringLiteral("hmean"), primary);
    else normalizedMetrics.insert(QStringLiteral("accuracy"), primary);
    if (component == QStringLiteral("rec")) normalizedMetrics.insert(QStringLiteral("cer"), secondary);
    const QJsonObject normalized{{QStringLiteral("ok"), true},
        {QStringLiteral("backend"), normalizedBackend},
        {QStringLiteral("framework"), QStringLiteral("PaddleOCR official tools")},
        {QStringLiteral("modelFamily"), modelFamily},
        {QStringLiteral("mode"), sourceMode},
        {QStringLiteral("metrics"), normalizedMetrics},
        {QStringLiteral("sourceReportSha256"), rawHash},
        {QStringLiteral("sourceBackend"), sourceBackend},
        {QStringLiteral("datasetSnapshotId"), snapshot.record.id.toString()}};
    const QByteArray normalizedBytes = QJsonDocument(normalized).toJson(QJsonDocument::Indented);
    const QString normalizedHash = sha256(normalizedBytes);
    QJsonObject normalization{{QStringLiteral("sourceBackend"), sourceBackend},
        {QStringLiteral("sourceReportSha256"), rawHash},
        {QStringLiteral("sampleCountSource"), QStringLiteral("committed_dataset_snapshot_labels")},
        {QStringLiteral("identityDerivation"), component == QStringLiteral("system")
                ? QStringLiteral("raw_report_fields")
                : QStringLiteral("registered_official_adapter_backend_contract")}};
    QJsonObject lineage{{QStringLiteral("schemaVersion"), 1},
        {QStringLiteral("kind"), QStringLiteral("paddleocr_official_report_lineage")},
        {QStringLiteral("component"), component},
        {QStringLiteral("reportRelativePath"), QStringLiteral("report/%1").arg(
            component == QStringLiteral("det") ? QStringLiteral("paddleocr_official_det_report.json")
            : component == QStringLiteral("rec") ? QStringLiteral("paddleocr_official_rec_report.json")
                                                   : QStringLiteral("paddleocr_official_system_report.json"))},
        {QStringLiteral("reportSha256"), normalizedHash},
        {QStringLiteral("evidenceClass"), evidenceClass},
        {QStringLiteral("acceptanceCohortId"), cohort},
        {QStringLiteral("customerDomainId"), domain},
        {QStringLiteral("datasetFingerprint"), snapshot.record.rootHash},
        {QStringLiteral("datasetSnapshotId"), snapshot.record.id.toString()},
        {QStringLiteral("datasetSnapshotArtifactId"), snapshot.record.artifactId.toString()},
        {QStringLiteral("normalization"), normalization}};
    if (component == QStringLiteral("system")) {
        lineage.insert(QStringLiteral("upstream"), QJsonObject{
            {QStringLiteral("detReportSha256"), detHash},
            {QStringLiteral("recReportSha256"), recHash}});
    }
    prepared->component = component;
    prepared->artifactKind = QStringLiteral("paddleocr_%1_official_report").arg(component);
    prepared->normalizedFileName = component == QStringLiteral("det")
        ? QStringLiteral("paddleocr_official_det_report.json")
        : component == QStringLiteral("rec") ? QStringLiteral("paddleocr_official_rec_report.json")
                                              : QStringLiteral("paddleocr_official_system_report.json");
    prepared->rawFileName = QFileInfo(source.reportPath).fileName();
    prepared->rawBytes = rawBytes;
    prepared->normalizedBytes = normalizedBytes;
    prepared->lineageBytes = QJsonDocument(lineage).toJson(QJsonDocument::Indented);
    prepared->normalizedSha256 = normalizedHash;
    return true;
}

bool writeBytes(const QString& path, const QByteArray& bytes, QString* error)
{
    if (!QDir().mkpath(QFileInfo(path).absolutePath())) {
        if (error) *error = QStringLiteral("无法创建 OCR 报告打包暂存目录：%1").arg(path);
        return false;
    }
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly) || file.write(bytes) != bytes.size() || !file.commit()) {
        if (error) *error = QStringLiteral("无法写入 OCR 报告打包暂存文件：%1").arg(path);
        return false;
    }
    return true;
}

} // namespace

bool ProjectWorkspace::importOcrOfficialReports(const TaskId& taskId,
    const OcrOfficialReportImportRequest& request,
    OcrOfficialReportImportResult* result,
    QString* error,
    const aitrain::CancellationCallback& cancellation)
{
    if (!result) {
        if (error) *error = QStringLiteral("OCR 官方报告导入需要输出对象。");
        return false;
    }
    *result = {};
    if (!isOpen() || !taskId.isValid()) {
        return failImport(result, FailureCode::InvalidRequest,
            QStringLiteral("ocr_report_import.invalid_request"),
            QStringLiteral("工作区未打开或 TaskId 无效"), error);
    }
    TaskSnapshot task;
    if (!storage_.task(taskId, &task, error) || task.state != TaskState::Running) {
        return failImport(result, FailureCode::InvalidRequest,
            QStringLiteral("ocr_report_import.task_not_running"),
            QStringLiteral("只能由运行中的任务导入报告"), error);
    }
    const QString cohort = request.acceptanceCohortId.trimmed();
    const QString domain = request.customerDomainId.trimmed();
    const QString evidenceClass = request.evidenceClass.trimmed();
    const QSet<QString> allowedEvidenceClasses{QStringLiteral("customer_domain"),
        QStringLiteral("public"), QStringLiteral("generated"), QStringLiteral("smoke")};
    if (cohort.isEmpty() || domain.isEmpty() || !allowedEvidenceClasses.contains(evidenceClass)
        || request.det.reportPath.trimmed().isEmpty() || request.rec.reportPath.trimmed().isEmpty()
        || request.system.reportPath.trimmed().isEmpty()) {
        return failImport(result, FailureCode::InvalidRequest,
            QStringLiteral("ocr_report_import.invalid_request"),
            QStringLiteral("报告路径、验收批次、客户域或证据分类无效"), error);
    }
    if (canceled(cancellation)) {
        return failImport(result, FailureCode::Canceled,
            QStringLiteral("ocr_report_import.canceled"), QStringLiteral("导入前收到取消请求"), error);
    }

    VerifiedOcrSnapshot detSnapshot;
    VerifiedOcrSnapshot recSnapshot;
    VerifiedOcrSnapshot systemSnapshot;
    QString validationError;
    if (!resolveSnapshot(&storage_, artifactStore_.get(), request.det, QStringLiteral("det"),
            &detSnapshot, cancellation, &validationError)
        || !resolveSnapshot(&storage_, artifactStore_.get(), request.rec, QStringLiteral("rec"),
            &recSnapshot, cancellation, &validationError)
        || !resolveSnapshot(&storage_, artifactStore_.get(), request.system, QStringLiteral("system"),
            &systemSnapshot, cancellation, &validationError)) {
        return failImport(result, canceled(cancellation) ? FailureCode::Canceled : FailureCode::ArtifactIncompatible,
            canceled(cancellation) ? QStringLiteral("ocr_report_import.canceled")
                                   : QStringLiteral("ocr_report_import.snapshot_invalid"),
            validationError, error);
    }

    PreparedOcrReport det;
    PreparedOcrReport rec;
    PreparedOcrReport system;
    Failure problem;
    if (!prepareReport(QStringLiteral("det"), request.det, detSnapshot, cohort, domain,
            evidenceClass, {}, {}, &det, cancellation, &problem)
        || !prepareReport(QStringLiteral("rec"), request.rec, recSnapshot, cohort, domain,
            evidenceClass, {}, {}, &rec, cancellation, &problem)
        || !prepareReport(QStringLiteral("system"), request.system, systemSnapshot, cohort, domain,
            evidenceClass, det.normalizedSha256, rec.normalizedSha256, &system, cancellation, &problem)) {
        result->failure = problem;
        if (error) *error = problem.message;
        return false;
    }

    const QString stagingRoot = QDir(runtimeStagingPath(taskId)).filePath(QStringLiteral("ocr-report-import"));
    QVector<PreparedOcrReport> prepared{det, rec, system};
    struct Staged final { PreparedOcrReport report; QVector<RuntimeArtifactCandidate> candidates; };
    QVector<Staged> staged;
    for (const PreparedOcrReport& report : prepared) {
        if (canceled(cancellation)) {
            cleanupRuntimeStaging(taskId, nullptr);
            return failImport(result, FailureCode::Canceled,
                QStringLiteral("ocr_report_import.canceled"), QStringLiteral("提交前收到取消请求"), error);
        }
        const QString base = QDir(stagingRoot).filePath(report.component);
        const QString normalizedPath = QDir(base).filePath(report.normalizedFileName);
        const QString rawPath = QDir(base).filePath(QStringLiteral("raw_%1").arg(report.rawFileName));
        const QString lineagePath = QDir(base).filePath(QStringLiteral("official_report_lineage.json"));
        if (!writeBytes(normalizedPath, report.normalizedBytes, &validationError)
            || !writeBytes(rawPath, report.rawBytes, &validationError)
            || !writeBytes(lineagePath, report.lineageBytes, &validationError)) {
            cleanupRuntimeStaging(taskId, nullptr);
            return failImport(result, FailureCode::ArtifactIncomplete,
                QStringLiteral("ocr_report_import.staging_failed"), validationError, error);
        }
        staged.append({report, {{QStringLiteral("report"), normalizedPath},
            {QStringLiteral("raw_report"), rawPath}, {QStringLiteral("lineage"), lineagePath}}});
    }

    QVector<ArtifactId> committed;
    const auto compensate = [&]() -> bool {
        bool complete = true;
        QStringList failures;
        for (auto it = committed.crbegin(); it != committed.crend(); ++it) {
            QString discardError;
            if (!artifactStore_->discardCommitted(*it, &storage_, &discardError)) {
                complete = false;
                failures.append(discardError);
            }
        }
        QString cleanupError;
        if (!cleanupRuntimeStaging(taskId, &cleanupError)) {
            complete = false;
            failures.append(cleanupError);
        }
        if (!complete && error) *error = failures.join(QStringLiteral("；"));
        return complete;
    };
    for (int index = 0; index < staged.size(); ++index) {
        if (canceled(cancellation)) {
            const bool compensated = compensate();
            result->detReportArtifactId = {};
            result->recReportArtifactId = {};
            result->systemReportArtifactId = {};
            return failImport(result, compensated ? FailureCode::Canceled : FailureCode::ArtifactIncomplete,
                compensated ? QStringLiteral("ocr_report_import.canceled")
                            : QStringLiteral("ocr_report_import.compensation_failed"),
                compensated ? QStringLiteral("提交过程中收到取消请求，已完整补偿")
                            : QStringLiteral("提交取消后的 Artifact 补偿不完整"), error);
        }
        RuntimeArtifactBundle bundle;
        if (!commitRuntimeArtifacts(taskId, staged.at(index).report.artifactKind,
                staged.at(index).candidates, &bundle, &validationError)) {
            const bool compensated = compensate();
            result->detReportArtifactId = {};
            result->recReportArtifactId = {};
            result->systemReportArtifactId = {};
            return failImport(result, FailureCode::ArtifactIncomplete,
                compensated ? QStringLiteral("ocr_report_import.commit_failed")
                            : QStringLiteral("ocr_report_import.compensation_failed"),
                compensated ? validationError
                            : QStringLiteral("提交失败且已提交 Artifact 补偿不完整"), error);
        }
        committed.append(bundle.artifactId);
        if (index == 0) result->detReportArtifactId = bundle.artifactId;
        else if (index == 1) result->recReportArtifactId = bundle.artifactId;
        else result->systemReportArtifactId = bundle.artifactId;
    }
    const QJsonObject importEvidence{
        {QStringLiteral("schemaVersion"), 1},
        {QStringLiteral("kind"), QStringLiteral("ocr_official_report_import_evidence")},
        {QStringLiteral("acceptanceCohortId"), cohort},
        {QStringLiteral("customerDomainId"), domain},
        {QStringLiteral("evidenceClass"), evidenceClass},
        {QStringLiteral("detReportArtifactId"), result->detReportArtifactId.toString()},
        {QStringLiteral("recReportArtifactId"), result->recReportArtifactId.toString()},
        {QStringLiteral("systemReportArtifactId"), result->systemReportArtifactId.toString()},
        {QStringLiteral("officialOnly"), true},
        {QStringLiteral("limitations"), QJsonArray{
            QStringLiteral("受控导入只证明三份官方报告及其 Snapshot lineage 已完成校验和原子打包。"),
            QStringLiteral("只有后续 OCR Acceptance  的 customer_domain 四步验收成功才能生成 production accepted。")}}};
    const QString evidencePath = QDir(stagingRoot).filePath(QStringLiteral("import_evidence.json"));
    if (!writeBytes(evidencePath, QJsonDocument(importEvidence).toJson(QJsonDocument::Indented),
            &validationError)) {
        const bool compensated = compensate();
        *result = {};
        return failImport(result, FailureCode::ArtifactIncomplete,
            compensated ? QStringLiteral("ocr_report_import.evidence_staging_failed")
                        : QStringLiteral("ocr_report_import.compensation_failed"),
            validationError, error);
    }
    RuntimeArtifactBundle importEvidenceBundle;
    if (!commitRuntimeArtifacts(taskId, QStringLiteral("ocr_official_report_import_evidence"),
            {{QStringLiteral("import_evidence"), evidencePath}}, &importEvidenceBundle,
            &validationError)) {
        const bool compensated = compensate();
        *result = {};
        return failImport(result, FailureCode::ArtifactIncomplete,
            compensated ? QStringLiteral("ocr_report_import.evidence_commit_failed")
                        : QStringLiteral("ocr_report_import.compensation_failed"),
            validationError, error);
    }
    committed.append(importEvidenceBundle.artifactId);
    result->evidenceArtifactId = importEvidenceBundle.artifactId;
    if (!cleanupRuntimeStaging(taskId, &validationError)) {
        const bool compensated = compensate();
        *result = {};
        return failImport(result, FailureCode::ArtifactIncomplete,
            compensated ? QStringLiteral("ocr_report_import.cleanup_failed")
                        : QStringLiteral("ocr_report_import.compensation_failed"),
            validationError, error);
    }
    return true;
}

} // namespace aitrain
