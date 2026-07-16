#include "aitrain/v2/ProjectWorkspaceV2.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QTemporaryDir>
#include <QTest>

namespace {

bool writeBytes(const QString& path, const QByteArray& bytes)
{
    if (!QDir().mkpath(QFileInfo(path).absolutePath())) return false;
    QFile file(path);
    return file.open(QIODevice::WriteOnly | QIODevice::Truncate)
        && file.write(bytes) == bytes.size();
}

QByteArray jsonBytes(const QJsonObject& object)
{
    return QJsonDocument(object).toJson(QJsonDocument::Indented);
}

QString sha256(const QByteArray& bytes)
{
    return QString::fromLatin1(QCryptographicHash::hash(bytes, QCryptographicHash::Sha256).toHex());
}

aitrain::v2::TaskId startTask(aitrain::v2::ProjectWorkspaceV2* workspace,
    const QString& capability, QString* error)
{
    const aitrain::v2::TaskId id = aitrain::v2::TaskId::create();
    aitrain::v2::TaskSnapshot task;
    if (!workspace->startTask(id, capability, capability, &task, error)) return {};
    return id;
}

struct ReportSpec final {
    QString component;
    QString artifactKind;
    QString fileName;
    QString backend;
    QString modelFamily;
    QString mode;
    QJsonObject metrics;
};

aitrain::v2::ArtifactId commitReport(aitrain::v2::ProjectWorkspaceV2* workspace,
    const ReportSpec& spec, const QString& cohort, const QString& domain,
    const QString& evidenceClass, const QJsonObject& upstream, QString* reportHash,
    QString* error)
{
    const aitrain::v2::TaskId taskId = startTask(workspace, QStringLiteral("test.official.report"), error);
    if (!taskId.isValid()) return {};
    const QJsonObject report{{QStringLiteral("ok"), true},
        {QStringLiteral("backend"), spec.backend},
        {QStringLiteral("framework"), QStringLiteral("PaddleOCR official tools")},
        {QStringLiteral("modelFamily"), spec.modelFamily},
        {QStringLiteral("mode"), spec.mode},
        {QStringLiteral("metrics"), spec.metrics}};
    const QByteArray reportContent = jsonBytes(report);
    const QString hash = sha256(reportContent);
    QJsonObject lineage{{QStringLiteral("schemaVersion"), 1},
        {QStringLiteral("kind"), QStringLiteral("paddleocr_official_report_lineage")},
        {QStringLiteral("component"), spec.component},
        {QStringLiteral("reportRelativePath"), QStringLiteral("report/%1").arg(spec.fileName)},
        {QStringLiteral("reportSha256"), hash},
        {QStringLiteral("evidenceClass"), evidenceClass},
        {QStringLiteral("acceptanceCohortId"), cohort},
        {QStringLiteral("customerDomainId"), domain},
        {QStringLiteral("datasetFingerprint"), QStringLiteral("dataset-%1").arg(spec.component)}};
    if (!upstream.isEmpty()) lineage.insert(QStringLiteral("upstream"), upstream);
    const QString staging = workspace->runtimeStagingPath(taskId);
    const QString reportPath = QDir(staging).filePath(QStringLiteral("source/%1").arg(spec.fileName));
    const QString lineagePath = QDir(staging).filePath(QStringLiteral("source/official_report_lineage.json"));
    if (!writeBytes(reportPath, reportContent) || !writeBytes(lineagePath, jsonBytes(lineage))) return {};
    aitrain::v2::RuntimeArtifactBundleV2 bundle;
    const QVector<aitrain::v2::RuntimeArtifactCandidateV2> candidates{
        {QStringLiteral("report"), reportPath}, {QStringLiteral("lineage"), lineagePath}};
    if (!workspace->commitRuntimeArtifacts(taskId, spec.artifactKind, candidates, &bundle, error)
        || !workspace->cleanupRuntimeStaging(taskId, error)
        || !workspace->finalizeTask(taskId, aitrain::v2::TaskState::Succeeded, {}, error)) return {};
    if (reportHash) *reportHash = hash;
    return bundle.artifactId;
}

struct Fixture final {
    aitrain::v2::ArtifactId det;
    aitrain::v2::ArtifactId rec;
    aitrain::v2::ArtifactId system;
};

Fixture commitFixture(aitrain::v2::ProjectWorkspaceV2* workspace,
    const QString& evidenceClass, int samples, double detHmean, double recAccuracy,
    double recCer, double systemAccuracy, QString* error)
{
    const QString cohort = QStringLiteral("customer-batch-2026-07");
    const QString domain = QStringLiteral("customer-line-a");
    QString detHash;
    QString recHash;
    Fixture result;
    result.det = commitReport(workspace,
        {QStringLiteral("det"), QStringLiteral("paddleocr_det_official_report_v2"),
            QStringLiteral("paddleocr_official_det_report.json"),
            QStringLiteral("paddleocr_det_official"), QStringLiteral("ocr_detection"),
            QStringLiteral("officialEvaluate"),
            {{QStringLiteral("sampleCount"), samples}, {QStringLiteral("hmean"), detHmean}}},
        cohort, domain, evidenceClass, {}, &detHash, error);
    result.rec = commitReport(workspace,
        {QStringLiteral("rec"), QStringLiteral("paddleocr_rec_official_report_v2"),
            QStringLiteral("paddleocr_official_rec_report.json"),
            QStringLiteral("paddleocr_rec_official"), QStringLiteral("ocr_recognition"),
            QStringLiteral("officialEvaluate"),
            {{QStringLiteral("sampleCount"), samples}, {QStringLiteral("accuracy"), recAccuracy},
                {QStringLiteral("cer"), recCer}}},
        cohort, domain, evidenceClass, {}, &recHash, error);
    result.system = commitReport(workspace,
        {QStringLiteral("system"), QStringLiteral("paddleocr_system_official_report_v2"),
            QStringLiteral("paddleocr_official_system_report.json"),
            QStringLiteral("paddleocr_system_official"), QStringLiteral("ocr"),
            QStringLiteral("officialSystemPredict"),
            {{QStringLiteral("sampleCount"), samples}, {QStringLiteral("accuracy"), systemAccuracy}}},
        cohort, domain, evidenceClass,
        {{QStringLiteral("detReportSha256"), detHash}, {QStringLiteral("recReportSha256"), recHash}},
        nullptr, error);
    return result;
}

bool run(aitrain::v2::ProjectWorkspaceV2* workspace,
    const Fixture& fixture, aitrain::v2::OcrAcceptanceWorkflowResultV2* result,
    QString* error, const aitrain::CancellationCallback& cancellation = {})
{
    const aitrain::v2::TaskId taskId = startTask(workspace, QStringLiteral("ocr.acceptance.v2"), error);
    if (!taskId.isValid() || !result) return false;
    aitrain::v2::OcrAcceptanceWorkflowRequestV2 request;
    request.detReportArtifactId = fixture.det;
    request.recReportArtifactId = fixture.rec;
    request.systemReportArtifactId = fixture.system;
    request.minimumDetSamples = 10;
    request.minimumRecSamples = 10;
    request.minimumSystemSamples = 10;
    return workspace->runOcrAcceptanceWorkflow(taskId, request, result, error, cancellation);
}

QString artifactFile(const aitrain::v2::ProjectWorkspaceV2& workspace,
    const aitrain::v2::ArtifactId& artifactId, const QString& relativePath)
{
    return QDir(workspace.workspacePath()).filePath(
        QStringLiteral("artifact-store/artifacts/%1/%2").arg(artifactId.toString(), relativePath));
}

} // namespace

class V2OcrAcceptanceTests final : public QObject {
    Q_OBJECT

private slots:
    void customerOfficialReportsProduceFourStepsAndEvidence();
    void missingReportFailsPreciselyWithEvidence();
    void tamperedReportFailsPreciselyWithEvidence();
    void insufficientSamplesFailPrecisely();
    void thresholdsNotMetFailPrecisely();
    void publicEvidenceCannotBecomeProductionAccepted();
    void cancellationHasUniqueTerminalAndEvidence();
};

void V2OcrAcceptanceTests::customerOfficialReportsProduceFourStepsAndEvidence()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::ProjectWorkspaceV2 workspace;
    QString error;
    QVERIFY2(workspace.open(directory.filePath(QStringLiteral("project")), &error), qPrintable(error));
    const Fixture fixture = commitFixture(&workspace, QStringLiteral("customer_domain"), 20,
        0.88, 0.92, 0.08, 0.86, &error);
    QVERIFY2(fixture.det.isValid() && fixture.rec.isValid() && fixture.system.isValid(), qPrintable(error));
    aitrain::v2::OcrAcceptanceWorkflowResultV2 result;
    QVERIFY2(run(&workspace, fixture, &result, &error), qPrintable(error));
    QVERIFY2(result.workflowRunId.isValid(), qPrintable(error));
    QCOMPARE(result.terminalState, aitrain::v2::TaskState::Succeeded);
    QVERIFY(result.productionAccepted);
    QVERIFY(result.resolvedEvidenceArtifactId.isValid());
    QVERIFY(result.officialReportValidationArtifactId.isValid());
    QVERIFY(result.thresholdEvaluationArtifactId.isValid());
    QVERIFY(result.acceptanceReportArtifactId.isValid());
    QVERIFY(result.evidenceArtifactId.isValid());
    const auto steps = workspace.workflowSteps(result.workflowRunId, &error);
    QCOMPARE(steps.size(), 4);
    for (const auto& step : steps) {
        QCOMPARE(step.state, aitrain::v2::WorkflowStepState::Succeeded);
        QVERIFY(step.outputArtifactId.isValid());
    }
    const QString evidenceRoot = artifactFile(workspace, result.evidenceArtifactId, QString());
    QVERIFY(QFileInfo::exists(QDir(evidenceRoot).filePath(QStringLiteral("evidence.json"))));
    QVERIFY(QFileInfo::exists(QDir(evidenceRoot).filePath(QStringLiteral("evidence.md"))));
    QVERIFY(QFileInfo::exists(QDir(evidenceRoot).filePath(QStringLiteral("evidence.html"))));
    QVERIFY(QFileInfo::exists(QDir(evidenceRoot).filePath(QStringLiteral("model_card.json"))));
}

void V2OcrAcceptanceTests::missingReportFailsPreciselyWithEvidence()
{
    QTemporaryDir directory;
    aitrain::v2::ProjectWorkspaceV2 workspace;
    QString error;
    QVERIFY(workspace.open(directory.filePath(QStringLiteral("project")), &error));
    Fixture fixture = commitFixture(&workspace, QStringLiteral("customer_domain"), 20, 0.9, 0.9, 0.1, 0.9, &error);
    fixture.system = aitrain::v2::ArtifactId::create();
    aitrain::v2::OcrAcceptanceWorkflowResultV2 result;
    QVERIFY2(run(&workspace, fixture, &result, &error), qPrintable(error));
    QCOMPARE(result.terminalState, aitrain::v2::TaskState::Failed);
    QVERIFY(result.evidenceArtifactId.isValid());
    QVERIFY(result.failure.message.startsWith(QStringLiteral("ocr_acceptance.report_missing:")));
    QVERIFY(!result.productionAccepted);
}

void V2OcrAcceptanceTests::tamperedReportFailsPreciselyWithEvidence()
{
    QTemporaryDir directory;
    aitrain::v2::ProjectWorkspaceV2 workspace;
    QString error;
    QVERIFY(workspace.open(directory.filePath(QStringLiteral("project")), &error));
    const Fixture fixture = commitFixture(&workspace, QStringLiteral("customer_domain"), 20, 0.9, 0.9, 0.1, 0.9, &error);
    QVERIFY(writeBytes(artifactFile(workspace, fixture.rec,
        QStringLiteral("report/paddleocr_official_rec_report.json")), QByteArray("{}\n")));
    aitrain::v2::OcrAcceptanceWorkflowResultV2 result;
    QVERIFY2(run(&workspace, fixture, &result, &error), qPrintable(error));
    QCOMPARE(result.terminalState, aitrain::v2::TaskState::Failed);
    QVERIFY(result.evidenceArtifactId.isValid());
    QVERIFY(result.failure.message.startsWith(QStringLiteral("ocr_acceptance.report_tampered:")));
}

void V2OcrAcceptanceTests::insufficientSamplesFailPrecisely()
{
    QTemporaryDir directory;
    aitrain::v2::ProjectWorkspaceV2 workspace;
    QString error;
    QVERIFY(workspace.open(directory.filePath(QStringLiteral("project")), &error));
    const Fixture fixture = commitFixture(&workspace, QStringLiteral("customer_domain"), 3, 0.9, 0.9, 0.1, 0.9, &error);
    aitrain::v2::OcrAcceptanceWorkflowResultV2 result;
    QVERIFY2(run(&workspace, fixture, &result, &error), qPrintable(error));
    QCOMPARE(result.terminalState, aitrain::v2::TaskState::Failed);
    QVERIFY(result.evidenceArtifactId.isValid());
    QVERIFY(result.failure.message.startsWith(QStringLiteral("ocr_acceptance.sample_count_insufficient:")));
    QVERIFY(!result.acceptanceReportArtifactId.isValid());
}

void V2OcrAcceptanceTests::thresholdsNotMetFailPrecisely()
{
    QTemporaryDir directory;
    aitrain::v2::ProjectWorkspaceV2 workspace;
    QString error;
    QVERIFY(workspace.open(directory.filePath(QStringLiteral("project")), &error));
    const Fixture fixture = commitFixture(&workspace, QStringLiteral("customer_domain"), 20, 0.4, 0.6, 0.4, 0.5, &error);
    aitrain::v2::OcrAcceptanceWorkflowResultV2 result;
    QVERIFY2(run(&workspace, fixture, &result, &error), qPrintable(error));
    QCOMPARE(result.terminalState, aitrain::v2::TaskState::Failed);
    QVERIFY(result.evidenceArtifactId.isValid());
    QVERIFY(result.failure.message.startsWith(QStringLiteral("ocr_acceptance.threshold_not_met:")));
}

void V2OcrAcceptanceTests::publicEvidenceCannotBecomeProductionAccepted()
{
    QTemporaryDir directory;
    aitrain::v2::ProjectWorkspaceV2 workspace;
    QString error;
    QVERIFY(workspace.open(directory.filePath(QStringLiteral("project")), &error));
    const Fixture fixture = commitFixture(&workspace, QStringLiteral("public"), 20, 0.9, 0.9, 0.1, 0.9, &error);
    aitrain::v2::OcrAcceptanceWorkflowResultV2 result;
    QVERIFY2(run(&workspace, fixture, &result, &error), qPrintable(error));
    QCOMPARE(result.terminalState, aitrain::v2::TaskState::Failed);
    QVERIFY(result.evidenceArtifactId.isValid());
    QVERIFY(result.failure.message.startsWith(
        QStringLiteral("ocr_acceptance.customer_domain_evidence_required:")));
    QVERIFY(!result.productionAccepted);
}

void V2OcrAcceptanceTests::cancellationHasUniqueTerminalAndEvidence()
{
    QTemporaryDir directory;
    aitrain::v2::ProjectWorkspaceV2 workspace;
    QString error;
    QVERIFY(workspace.open(directory.filePath(QStringLiteral("project")), &error));
    const Fixture fixture = commitFixture(&workspace, QStringLiteral("customer_domain"), 20, 0.9, 0.9, 0.1, 0.9, &error);
    aitrain::v2::OcrAcceptanceWorkflowResultV2 result;
    QVERIFY2(run(&workspace, fixture, &result, &error, [] { return true; }), qPrintable(error));
    QCOMPARE(result.terminalState, aitrain::v2::TaskState::Canceled);
    QVERIFY(result.evidenceArtifactId.isValid());
    QCOMPARE(result.failure.code, aitrain::v2::FailureCode::Canceled);
    QVERIFY(!result.productionAccepted);
}

QTEST_MAIN(V2OcrAcceptanceTests)
#include "tst_v2_ocr_acceptance.moc"
