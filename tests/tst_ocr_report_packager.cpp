#include "aitrain/workflow/ProjectWorkspace.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
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

bool writeJson(const QString& path, const QJsonObject& object)
{
    return writeBytes(path, QJsonDocument(object).toJson(QJsonDocument::Indented));
}

aitrain::TaskId startTask(aitrain::ProjectWorkspace* workspace,
    const QString& type, QString* error)
{
    const aitrain::TaskId id = aitrain::TaskId::create();
    aitrain::TaskSnapshot task;
    return workspace->startTask(id, type, type, &task, error) ? id : aitrain::TaskId{};
}

aitrain::DatasetSnapshotArtifactBundle commitSnapshot(
    aitrain::ProjectWorkspace* workspace, const QString& root,
    const QString& format, QString* error)
{
    const aitrain::TaskId taskId = startTask(workspace, QStringLiteral("test.snapshot"), error);
    aitrain::DatasetSnapshotCommitRequest request;
    request.datasetRoot = root;
    request.datasetFormat = format;
    request.driverId = QStringLiteral("test.%1.driver").arg(format);
    request.driverVersion = QStringLiteral("2");
    aitrain::DatasetSnapshotArtifactBundle result;
    if (!taskId.isValid() || !workspace->commitDatasetSnapshot(taskId, request, &result, error)
        || !workspace->finalizeTask(taskId, aitrain::TaskState::Succeeded, {}, error)) return {};
    return result;
}

struct Fixture final {
    aitrain::DatasetSnapshotArtifactBundle detSnapshot;
    aitrain::DatasetSnapshotArtifactBundle recSnapshot;
    QString detReport;
    QString recReport;
    QString systemReport;
};

Fixture createFixture(const QTemporaryDir& directory,
    aitrain::ProjectWorkspace* workspace, bool includeSystemAccuracy,
    QString* error)
{
    const QString detRoot = directory.filePath(QStringLiteral("det-dataset"));
    const QString recRoot = directory.filePath(QStringLiteral("rec-dataset"));
    writeBytes(QDir(detRoot).filePath(QStringLiteral("images/a.jpg")), QByteArray("det-a"));
    writeBytes(QDir(detRoot).filePath(QStringLiteral("images/b.jpg")), QByteArray("det-b"));
    writeBytes(QDir(detRoot).filePath(QStringLiteral("det_gt_train.txt")),
        QByteArray("images/a.jpg\t[{\"transcription\":\"A\",\"points\":[[0,0],[2,0],[2,2],[0,2]]}]\n"
                   "images/b.jpg\t[{\"transcription\":\"B\",\"points\":[[0,0],[2,0],[2,2],[0,2]]}]\n"));
    writeBytes(QDir(recRoot).filePath(QStringLiteral("images/a.jpg")), QByteArray("rec-a"));
    writeBytes(QDir(recRoot).filePath(QStringLiteral("images/b.jpg")), QByteArray("rec-b"));
    writeBytes(QDir(recRoot).filePath(QStringLiteral("rec_gt_train.txt")),
        QByteArray("images/a.jpg\tA\nimages/b.jpg\tB\n"));

    Fixture fixture;
    fixture.detSnapshot = commitSnapshot(workspace, detRoot, QStringLiteral("paddleocr_det"), error);
    fixture.recSnapshot = commitSnapshot(workspace, recRoot, QStringLiteral("paddleocr_rec"), error);
    fixture.detReport = directory.filePath(QStringLiteral("raw/det_evaluation_report.json"));
    fixture.recReport = directory.filePath(QStringLiteral("raw/rec_evaluation_report.json"));
    fixture.systemReport = directory.filePath(QStringLiteral("raw/system_report.json"));
    writeJson(fixture.detReport, {{QStringLiteral("schemaVersion"), 2},
        {QStringLiteral("backend"), QStringLiteral("paddleocr_det_official_eval")},
        {QStringLiteral("taskType"), QStringLiteral("ocr_detection")},
        {QStringLiteral("component"), QStringLiteral("det")},
        {QStringLiteral("datasetSnapshotManifest"), fixture.detSnapshot.manifestPath},
        {QStringLiteral("metrics"), QJsonObject{{QStringLiteral("hmean"), 0.88}}}});
    writeJson(fixture.recReport, {{QStringLiteral("schemaVersion"), 2},
        {QStringLiteral("backend"), QStringLiteral("paddleocr_rec_official_eval")},
        {QStringLiteral("taskType"), QStringLiteral("ocr_recognition")},
        {QStringLiteral("component"), QStringLiteral("rec")},
        {QStringLiteral("datasetSnapshotManifest"), fixture.recSnapshot.manifestPath},
        {QStringLiteral("metrics"), QJsonObject{{QStringLiteral("accuracy"), 0.92},
            {QStringLiteral("cer"), 0.08}}}});
    QJsonObject systemMetrics;
    if (includeSystemAccuracy) systemMetrics.insert(QStringLiteral("accuracy"), 0.86);
    writeJson(fixture.systemReport, {{QStringLiteral("ok"), true},
        {QStringLiteral("backend"), QStringLiteral("paddleocr_system_official")},
        {QStringLiteral("framework"), QStringLiteral("PaddleOCR official tools")},
        {QStringLiteral("mode"), QStringLiteral("officialSystemPredict")},
        {QStringLiteral("datasetSnapshotManifest"), fixture.detSnapshot.manifestPath},
        {QStringLiteral("predictionCount"), 2},
        {QStringLiteral("metrics"), systemMetrics}});
    return fixture;
}

aitrain::OcrOfficialReportImportRequest requestFor(const Fixture& fixture)
{
    aitrain::OcrOfficialReportImportRequest request;
    request.det.reportPath = fixture.detReport;
    request.det.datasetSnapshotId = fixture.detSnapshot.snapshot.id;
    request.rec.reportPath = fixture.recReport;
    request.rec.datasetSnapshotArtifactId = fixture.recSnapshot.snapshot.artifactId;
    request.system.reportPath = fixture.systemReport;
    request.system.datasetSnapshotId = fixture.detSnapshot.snapshot.id;
    request.acceptanceCohortId = QStringLiteral("customer-batch-2026-07");
    request.customerDomainId = QStringLiteral("customer-line-a");
    request.evidenceClass = QStringLiteral("customer_domain");
    return request;
}

QString artifactRoot(const aitrain::ProjectWorkspace& workspace,
    const aitrain::ArtifactId& id)
{
    return QDir(workspace.workspacePath()).filePath(
        QStringLiteral("artifacts/committed/%1").arg(id.toString()));
}

} // namespace

class OcrReportPackagerTests final : public QObject {
    Q_OBJECT

private slots:
    void packagesThreeVerifiedReportsAndSnapshotDerivedCounts();
    void missingSystemAccuracyIsUnsupportedAndCommitsNothing();
    void changedSnapshotSourceIsInvalidEvidenceAndCommitsNothing();
    void cancellationCommitsNothing();
};

void OcrReportPackagerTests::packagesThreeVerifiedReportsAndSnapshotDerivedCounts()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY2(workspace.open(directory.filePath(QStringLiteral("project")), &error), qPrintable(error));
    const Fixture fixture = createFixture(directory, &workspace, true, &error);
    QVERIFY2(fixture.detSnapshot.snapshot.id.isValid() && fixture.recSnapshot.snapshot.id.isValid(), qPrintable(error));
    const aitrain::TaskId taskId = startTask(&workspace, QStringLiteral("ocr.report.import"), &error);
    aitrain::OcrOfficialReportImportResult result;
    QVERIFY2(workspace.importOcrOfficialReports(taskId, requestFor(fixture), &result, &error), qPrintable(error));
    QVERIFY(result.detReportArtifactId.isValid());
    QVERIFY(result.recReportArtifactId.isValid());
    QVERIFY(result.systemReportArtifactId.isValid());
    QVERIFY(result.evidenceArtifactId.isValid());
    QCOMPARE(workspace.artifactsForTask(taskId, &error).size(), 4);

    QFile normalized(QDir(artifactRoot(workspace, result.detReportArtifactId))
        .filePath(QStringLiteral("report/paddleocr_official_det_report.json")));
    QVERIFY(normalized.open(QIODevice::ReadOnly));
    const QJsonObject report = QJsonDocument::fromJson(normalized.readAll()).object();
    QCOMPARE(report.value(QStringLiteral("mode")).toString(), QStringLiteral("officialEvaluate"));
    QCOMPARE(report.value(QStringLiteral("metrics")).toObject().value(QStringLiteral("sampleCount")).toInt(), 2);
    QVERIFY(QFileInfo::exists(QDir(artifactRoot(workspace, result.detReportArtifactId))
        .filePath(QStringLiteral("raw_report/raw_det_evaluation_report.json"))));
    QVERIFY(QFileInfo::exists(QDir(artifactRoot(workspace, result.systemReportArtifactId))
        .filePath(QStringLiteral("lineage/official_report_lineage.json"))));
}

void OcrReportPackagerTests::missingSystemAccuracyIsUnsupportedAndCommitsNothing()
{
    QTemporaryDir directory;
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY(workspace.open(directory.filePath(QStringLiteral("project")), &error));
    const Fixture fixture = createFixture(directory, &workspace, false, &error);
    const aitrain::TaskId taskId = startTask(&workspace, QStringLiteral("ocr.report.import"), &error);
    aitrain::OcrOfficialReportImportResult result;
    QVERIFY(!workspace.importOcrOfficialReports(taskId, requestFor(fixture), &result, &error));
    QCOMPARE(result.failure.code, aitrain::FailureCode::BackendUnsupported);
    QVERIFY(result.failure.message.startsWith(QStringLiteral("ocr_report_import.system_accuracy_unsupported:")));
    QCOMPARE(workspace.artifactsForTask(taskId, &error).size(), 0);
}

void OcrReportPackagerTests::changedSnapshotSourceIsInvalidEvidenceAndCommitsNothing()
{
    QTemporaryDir directory;
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY(workspace.open(directory.filePath(QStringLiteral("project")), &error));
    const Fixture fixture = createFixture(directory, &workspace, true, &error);
    QVERIFY(writeBytes(QDir(fixture.detSnapshot.snapshot.rootPath)
        .filePath(QStringLiteral("images/a.jpg")), QByteArray("changed")));
    const aitrain::TaskId taskId = startTask(&workspace, QStringLiteral("ocr.report.import"), &error);
    aitrain::OcrOfficialReportImportResult result;
    QVERIFY(!workspace.importOcrOfficialReports(taskId, requestFor(fixture), &result, &error));
    QCOMPARE(result.failure.code, aitrain::FailureCode::ArtifactIncompatible);
    QVERIFY(result.failure.message.startsWith(QStringLiteral("ocr_report_import.snapshot_invalid:")));
    QCOMPARE(workspace.artifactsForTask(taskId, &error).size(), 0);
}

void OcrReportPackagerTests::cancellationCommitsNothing()
{
    QTemporaryDir directory;
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY(workspace.open(directory.filePath(QStringLiteral("project")), &error));
    const Fixture fixture = createFixture(directory, &workspace, true, &error);
    const aitrain::TaskId taskId = startTask(&workspace, QStringLiteral("ocr.report.import"), &error);
    aitrain::OcrOfficialReportImportResult result;
    QVERIFY(!workspace.importOcrOfficialReports(taskId, requestFor(fixture), &result, &error, [] { return true; }));
    QCOMPARE(result.failure.code, aitrain::FailureCode::Canceled);
    QCOMPARE(workspace.artifactsForTask(taskId, &error).size(), 0);
}

QTEST_MAIN(OcrReportPackagerTests)
#include "tst_ocr_report_packager.moc"
