#include "aitrain/workflow/ProjectWorkspace.h"
#include "aitrain/storage/ProjectStore.h"

#include <QDir>
#include <QFile>
#include <QImage>
#include <QJsonDocument>
#include <QSet>
#include <QTemporaryDir>
#include <QTest>

namespace {

bool writeBytes(const QString& path, const QByteArray& bytes)
{
    if (!QDir().mkpath(QFileInfo(path).absolutePath())) return false;
    QFile file(path);
    return file.open(QIODevice::WriteOnly | QIODevice::Truncate) && file.write(bytes) == bytes.size();
}

bool writeImage(const QString& path)
{
    if (!QDir().mkpath(QFileInfo(path).absolutePath())) return false;
    QImage image(16, 12, QImage::Format_RGB32);
    image.fill(Qt::white);
    return image.save(path);
}

bool writeMask(const QString& path, bool foreground)
{
    if (!QDir().mkpath(QFileInfo(path).absolutePath())) return false;
    QImage mask(16, 12, QImage::Format_Grayscale8);
    mask.fill(0);
    if (foreground) {
        for (int y = 3; y < 9; ++y) {
            uchar* row = mask.scanLine(y);
            for (int x = 4; x < 12; ++x) row[x] = 1;
        }
    }
    return mask.save(path);
}

QJsonObject readJson(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) return {};
    return QJsonDocument::fromJson(file.readAll()).object();
}

QString artifactFile(const aitrain::ProjectWorkspace& workspace,
    const aitrain::ArtifactId& artifactId, const QString& relative)
{
    return QDir(workspace.workspacePath()).filePath(
        QStringLiteral("artifacts/artifacts/%1/%2").arg(artifactId.toString(), relative));
}

aitrain::DatasetSnapshotArtifactBundle commitSnapshot(
    aitrain::ProjectWorkspace* workspace,
    const QString& root,
    const QString& format,
    QString* error)
{
    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::TaskSnapshot task;
    if (!workspace->startTask(taskId, QStringLiteral("dataset.snapshot"), QStringLiteral("dataset_snapshot"), &task, error)) return {};
    aitrain::DatasetSnapshotCommitRequest request;
    request.datasetRoot = root;
    request.datasetFormat = format;
    request.driverId = format;
    request.driverVersion = QStringLiteral("2.0");
    request.options.classDefinitions = QJsonArray{QJsonObject{{QStringLiteral("id"), 0}, {QStringLiteral("name"), QStringLiteral("item")}}};
    aitrain::DatasetSnapshotArtifactBundle snapshot;
    if (!workspace->commitDatasetSnapshot(taskId, request, &snapshot, error)
        || !workspace->finalizeTask(taskId, aitrain::TaskState::Succeeded, {}, error)) return {};
    return snapshot;
}

aitrain::TaskId startQualityTask(aitrain::ProjectWorkspace* workspace, QString* error)
{
    const aitrain::TaskId id = aitrain::TaskId::create();
    aitrain::TaskSnapshot task;
    if (!workspace->startTask(id, QStringLiteral("dataset.quality"), QStringLiteral("dataset_quality"), &task, error)) return {};
    return id;
}

bool issueHasStableFields(const QJsonObject& issue)
{
    const QString severity = issue.value(QStringLiteral("severity")).toString();
    return !issue.value(QStringLiteral("code")).toString().isEmpty()
        && (severity == QStringLiteral("info") || severity == QStringLiteral("warning") || severity == QStringLiteral("error"))
        && !issue.value(QStringLiteral("sampleRelativePath")).toString().isEmpty()
        && !QDir::isAbsolutePath(issue.value(QStringLiteral("sampleRelativePath")).toString())
        && !issue.value(QStringLiteral("sourceRelativePath")).toString().isEmpty();
}

bool hasQualityArtifact(const aitrain::ProjectWorkspace& workspace,
    const aitrain::TaskId& taskId, QString* error)
{
    const auto artifacts = workspace.artifactsForTask(taskId, error);
    for (const auto& artifact : artifacts) {
        if (artifact.kind.startsWith(QStringLiteral("dataset_snapshot_validation"))
            || artifact.kind.startsWith(QStringLiteral("dataset_quality_analysis"))
            || artifact.kind.startsWith(QStringLiteral("dataset_repair_manifest"))
            || artifact.kind.startsWith(QStringLiteral("dataset_quality_report"))) return true;
    }
    return false;
}

} // namespace

class DataQualityTests final : public QObject {
    Q_OBJECT

private slots:
    void yoloDetectionProducesFourArtifactsAndEvidence();
    void paddleOcrRecUsesStableIssueContract();
    void remainingDriverFormatsUseStableConservativeRules_data();
    void remainingDriverFormatsUseStableConservativeRules();
    void changedSnapshotSourceFailsWithEvidence();
    void cancellationHasUniqueCanceledTerminalAndEvidence();
};

void DataQualityTests::yoloDetectionProducesFourArtifactsAndEvidence()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString root = directory.filePath(QStringLiteral("yolo 数据"));
    QVERIFY(writeBytes(QDir(root).filePath(QStringLiteral("data.yaml")), "path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: [item]\n"));
    QVERIFY(writeImage(QDir(root).filePath(QStringLiteral("images/train/a.png"))));
    QVERIFY(writeImage(QDir(root).filePath(QStringLiteral("images/val/b.png"))));
    QVERIFY(writeBytes(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), "0 0.5 0.5 0.05 0.05\n"));
    QVERIFY(writeBytes(QDir(root).filePath(QStringLiteral("labels/val/b.txt")), "0 0.5 0.5 0.5 0.5\n"));
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY2(workspace.open(directory.filePath(QStringLiteral("project")), &error), qPrintable(error));
    const auto snapshot = commitSnapshot(&workspace, root, QStringLiteral("yolo_detection"), &error);
    QVERIFY2(snapshot.snapshot.id.isValid(), qPrintable(error));
    const QByteArray originalLabel = QFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt"))).exists()
        ? QByteArray("0 0.5 0.5 0.05 0.05\n") : QByteArray();
    const auto taskId = startQualityTask(&workspace, &error);
    QVERIFY(taskId.isValid());
    aitrain::DataQualityWorkflowRequest request;
    request.snapshotId = snapshot.snapshot.id;
    aitrain::DataQualityWorkflowResult result;
    QVERIFY2(workspace.runDataQualityWorkflow(taskId, request, &result, &error), qPrintable(error));
    QCOMPARE(result.terminalState, aitrain::TaskState::Succeeded);
    QVERIFY(result.snapshotValidationArtifactId.isValid());
    QVERIFY(result.qualityAnalysisArtifactId.isValid());
    QVERIFY(result.repairManifestArtifactId.isValid());
    QVERIFY(result.qualityReportArtifactId.isValid());
    QVERIFY(result.evidenceArtifactId.isValid());
    const QJsonObject analysis = readJson(artifactFile(workspace, result.qualityAnalysisArtifactId,
        QStringLiteral("quality_analysis.json")));
    const QJsonArray issues = analysis.value(QStringLiteral("issues")).toArray();
    QCOMPARE(issues.size(), 1);
    QCOMPARE(issues.first().toObject().value(QStringLiteral("code")).toString(),
        QStringLiteral("quality.yolo_detection.bbox_too_small"));
    QVERIFY(issueHasStableFields(issues.first().toObject()));
    const QJsonObject repair = readJson(artifactFile(workspace, result.repairManifestArtifactId,
        QStringLiteral("repair_manifest.json")));
    QVERIFY(!repair.value(QStringLiteral("mutatesSource")).toBool(true));
    QVERIFY(QFileInfo::exists(artifactFile(workspace, result.repairManifestArtifactId,
        QStringLiteral("xanylabeling_review_manifest.json"))));
    QFile label(QDir(root).filePath(QStringLiteral("labels/train/a.txt")));
    QVERIFY(label.open(QIODevice::ReadOnly));
    QCOMPARE(label.readAll(), originalLabel);
    aitrain::TaskSnapshot persisted;
    QVERIFY(workspace.task(taskId, &persisted, &error));
    QCOMPARE(persisted.state, aitrain::TaskState::Succeeded);
    aitrain::ProjectStore storage;
    QVERIFY2(storage.open(QDir(workspace.workspacePath()).filePath(QStringLiteral("project.sqlite")), &error),
        qPrintable(error));
    aitrain::WorkflowInputBinding input;
    QVERIFY2(storage.workflowInput(result.workflowRunId, QStringLiteral("dataset_snapshot"), &input, &error),
        qPrintable(error));
    QCOMPARE(input.workflowRunId, result.workflowRunId);
    QCOMPARE(input.sourceArtifactId, snapshot.snapshot.artifactId);
    QCOMPARE(input.sourceTaskId, snapshot.snapshot.taskId);
    QCOMPARE(input.sourceArtifactKind, QStringLiteral("dataset_snapshot"));
    QCOMPARE(input.datasetId, snapshot.snapshot.datasetId);
    QCOMPARE(input.datasetVersionId, snapshot.snapshot.datasetVersionId);
    QCOMPARE(input.datasetSnapshotId, snapshot.snapshot.id);
    QCOMPARE(input.manifestSha256, snapshot.snapshot.manifestSha256);
    QCOMPARE(input.rootHash, snapshot.snapshot.rootHash);
    const QJsonObject evidence = readJson(artifactFile(workspace, result.evidenceArtifactId,
        QStringLiteral("evidence.json")));
    const QJsonArray externalInputs = evidence.value(QStringLiteral("externalInputs")).toArray();
    QCOMPARE(externalInputs.size(), 1);
    const QJsonObject external = externalInputs.first().toObject();
    QCOMPARE(external.value(QStringLiteral("role")).toString(), QStringLiteral("dataset_snapshot"));
    QCOMPARE(external.value(QStringLiteral("producerTaskId")).toString(), snapshot.snapshot.taskId.toString());
    QCOMPARE(external.value(QStringLiteral("artifactId")).toString(), snapshot.snapshot.artifactId.toString());
    QCOMPARE(external.value(QStringLiteral("datasetId")).toString(), snapshot.snapshot.datasetId.toString());
    QCOMPARE(external.value(QStringLiteral("datasetVersionId")).toString(), snapshot.snapshot.datasetVersionId.toString());
    QCOMPARE(external.value(QStringLiteral("datasetSnapshotId")).toString(), snapshot.snapshot.id.toString());
    QCOMPARE(external.value(QStringLiteral("manifestSha256")).toString(), snapshot.snapshot.manifestSha256);
    QCOMPARE(external.value(QStringLiteral("rootHash")).toString(), snapshot.snapshot.rootHash);
}

void DataQualityTests::paddleOcrRecUsesStableIssueContract()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString root = directory.filePath(QStringLiteral("ocr rec"));
    QVERIFY(writeImage(QDir(root).filePath(QStringLiteral("images/a.png"))));
    QVERIFY(writeImage(QDir(root).filePath(QStringLiteral("images/b.png"))));
    QVERIFY(writeBytes(QDir(root).filePath(QStringLiteral("dict.txt")), "a\nb\nc\n"));
    QVERIFY(writeBytes(QDir(root).filePath(QStringLiteral("rec_gt.txt")),
        "images/a.png\tabcabc\nimages/b.png\tab\n"));
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY(workspace.open(directory.filePath(QStringLiteral("project")), &error));
    const auto snapshot = commitSnapshot(&workspace, root, QStringLiteral("paddleocr_rec"), &error);
    QVERIFY2(snapshot.snapshot.id.isValid(), qPrintable(error));
    const auto taskId = startQualityTask(&workspace, &error);
    aitrain::DataQualityWorkflowRequest request;
    request.snapshotId = snapshot.snapshot.id;
    request.options.insert(QStringLiteral("maximumLabelLength"), 3);
    aitrain::DataQualityWorkflowResult result;
    QVERIFY2(workspace.runDataQualityWorkflow(taskId, request, &result, &error), qPrintable(error));
    const QJsonArray issues = readJson(artifactFile(workspace, result.qualityAnalysisArtifactId,
        QStringLiteral("quality_analysis.json"))).value(QStringLiteral("issues")).toArray();
    QCOMPARE(issues.size(), 1);
    QSet<QString> codes;
    for (const QJsonValue& value : issues) {
        QVERIFY(issueHasStableFields(value.toObject()));
        codes.insert(value.toObject().value(QStringLiteral("code")).toString());
    }
    QVERIFY(codes.contains(QStringLiteral("quality.paddleocr_rec.label_too_long")));
}

void DataQualityTests::remainingDriverFormatsUseStableConservativeRules_data()
{
    QTest::addColumn<QString>("format");
    QTest::addColumn<QString>("expectedCode");
    QTest::newRow("YOLO Segmentation") << QStringLiteral("yolo_segmentation")
        << QStringLiteral("quality.yolo_segmentation.polygon_too_small");
    QTest::newRow("YOLO OBB") << QStringLiteral("yolo_obb")
        << QStringLiteral("quality.yolo_obb.quad_too_small");
    QTest::newRow("Semantic Mask") << QStringLiteral("semantic_segmentation_mask")
        << QStringLiteral("quality.semantic_mask.no_foreground_pixels");
    QTest::newRow("Anomaly Folder") << QStringLiteral("anomaly_folder")
        << QStringLiteral("quality.anomaly_folder.good_only_dataset");
    QTest::newRow("PaddleOCR Det") << QStringLiteral("paddleocr_det")
        << QStringLiteral("quality.paddleocr_det.text_polygon_too_small");
}

void DataQualityTests::remainingDriverFormatsUseStableConservativeRules()
{
    QFETCH(QString, format);
    QFETCH(QString, expectedCode);
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString root = directory.filePath(QStringLiteral("质量数据"));
    if (format == QStringLiteral("yolo_segmentation") || format == QStringLiteral("yolo_obb")) {
        QVERIFY(writeBytes(QDir(root).filePath(QStringLiteral("data.yaml")),
            "path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: [item]\n"));
        QVERIFY(writeImage(QDir(root).filePath(QStringLiteral("images/train/a.png"))));
        QVERIFY(writeImage(QDir(root).filePath(QStringLiteral("images/val/b.png"))));
        if (format == QStringLiteral("yolo_segmentation")) {
            QVERIFY(writeBytes(QDir(root).filePath(QStringLiteral("labels/train/a.txt")),
                "0 0.10 0.10 0.15 0.10 0.10 0.15\n"));
            QVERIFY(writeBytes(QDir(root).filePath(QStringLiteral("labels/val/b.txt")),
                "0 0.10 0.10 0.90 0.10 0.50 0.90\n"));
        } else {
            QVERIFY(writeBytes(QDir(root).filePath(QStringLiteral("labels/train/a.txt")),
                "0 0.10 0.10 0.15 0.10 0.15 0.15 0.10 0.15\n"));
            QVERIFY(writeBytes(QDir(root).filePath(QStringLiteral("labels/val/b.txt")),
                "0 0.10 0.10 0.80 0.10 0.80 0.80 0.10 0.80\n"));
        }
    } else if (format == QStringLiteral("semantic_segmentation_mask")) {
        QVERIFY(writeBytes(QDir(root).filePath(QStringLiteral("classes.txt")), "background\nitem\n"));
        QVERIFY(writeImage(QDir(root).filePath(QStringLiteral("images/train/a.png"))));
        QVERIFY(writeImage(QDir(root).filePath(QStringLiteral("images/val/b.png"))));
        QVERIFY(writeMask(QDir(root).filePath(QStringLiteral("masks/train/a.png")), false));
        QVERIFY(writeMask(QDir(root).filePath(QStringLiteral("masks/val/b.png")), true));
    } else if (format == QStringLiteral("anomaly_folder")) {
        QVERIFY(writeImage(QDir(root).filePath(QStringLiteral("train/good/a.png"))));
    } else if (format == QStringLiteral("paddleocr_det")) {
        QVERIFY(writeImage(QDir(root).filePath(QStringLiteral("images/a.png"))));
        QVERIFY(writeBytes(QDir(root).filePath(QStringLiteral("det_gt.txt")),
            "images/a.png\t[{\"transcription\":\"a\",\"points\":[[1,1],[3,1],[3,3],[1,3]]}]\n"));
    }

    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY2(workspace.open(directory.filePath(QStringLiteral("project")), &error), qPrintable(error));
    const auto snapshot = commitSnapshot(&workspace, root, format, &error);
    QVERIFY2(snapshot.snapshot.id.isValid(), qPrintable(error));
    const auto taskId = startQualityTask(&workspace, &error);
    QVERIFY(taskId.isValid());
    aitrain::DataQualityWorkflowResult result;
    QVERIFY2(workspace.runDataQualityWorkflow(taskId, {snapshot.snapshot.id, {}}, &result, &error), qPrintable(error));
    QCOMPARE(result.terminalState, aitrain::TaskState::Succeeded);
    QVERIFY(result.snapshotValidationArtifactId.isValid());
    QVERIFY(result.qualityAnalysisArtifactId.isValid());
    QVERIFY(result.repairManifestArtifactId.isValid());
    QVERIFY(result.qualityReportArtifactId.isValid());
    QVERIFY(result.evidenceArtifactId.isValid());
    const QJsonArray issues = readJson(artifactFile(workspace, result.qualityAnalysisArtifactId,
        QStringLiteral("quality_analysis.json"))).value(QStringLiteral("issues")).toArray();
    QVERIFY2(!issues.isEmpty(), qPrintable(format));
    bool found = false;
    for (const QJsonValue& value : issues) {
        const QJsonObject issue = value.toObject();
        QVERIFY(issueHasStableFields(issue));
        if (issue.value(QStringLiteral("code")).toString() == expectedCode) found = true;
    }
    QVERIFY2(found, qPrintable(expectedCode));
    const QJsonObject repair = readJson(artifactFile(workspace, result.repairManifestArtifactId,
        QStringLiteral("repair_manifest.json")));
    QVERIFY(!repair.value(QStringLiteral("mutatesSource")).toBool(true));
}

void DataQualityTests::changedSnapshotSourceFailsWithEvidence()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString root = directory.filePath(QStringLiteral("changed"));
    QVERIFY(writeBytes(QDir(root).filePath(QStringLiteral("data.yaml")), "path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: [item]\n"));
    QVERIFY(writeImage(QDir(root).filePath(QStringLiteral("images/train/a.png"))));
    QVERIFY(writeImage(QDir(root).filePath(QStringLiteral("images/val/b.png"))));
    QVERIFY(writeBytes(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), "0 0.5 0.5 0.5 0.5\n"));
    QVERIFY(writeBytes(QDir(root).filePath(QStringLiteral("labels/val/b.txt")), "0 0.5 0.5 0.5 0.5\n"));
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY(workspace.open(directory.filePath(QStringLiteral("project")), &error));
    const auto snapshot = commitSnapshot(&workspace, root, QStringLiteral("yolo_detection"), &error);
    // 外部导入根在 Snapshot 提交后不再是业务输入；这里直接篡改 committed
    // Artifact，验证完整性门禁，而不是继续假设源目录变化会影响不可变快照。
    QVERIFY(writeBytes(QDir(snapshot.artifactPath).filePath(QStringLiteral("labels/train/a.txt")),
        "0 0.4 0.4 0.2 0.2\n"));
    const auto taskId = startQualityTask(&workspace, &error);
    aitrain::DataQualityWorkflowResult result;
    QVERIFY2(workspace.runDataQualityWorkflow(taskId, {snapshot.snapshot.id, {}}, &result, &error), qPrintable(error));
    QCOMPARE(result.terminalState, aitrain::TaskState::Failed);
    QVERIFY(result.evidenceArtifactId.isValid());
    QVERIFY(!result.snapshotValidationArtifactId.isValid());
    aitrain::TaskSnapshot task;
    QVERIFY(workspace.task(taskId, &task, &error));
    QCOMPARE(task.state, aitrain::TaskState::Failed);
    QCOMPARE(task.failure.code, aitrain::FailureCode::ArtifactIncompatible);
    QVERIFY(!hasQualityArtifact(workspace, taskId, &error));
}

void DataQualityTests::cancellationHasUniqueCanceledTerminalAndEvidence()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString root = directory.filePath(QStringLiteral("cancel"));
    QVERIFY(writeImage(QDir(root).filePath(QStringLiteral("images/a.png"))));
    QVERIFY(writeBytes(QDir(root).filePath(QStringLiteral("dict.txt")), "a\n"));
    QVERIFY(writeBytes(QDir(root).filePath(QStringLiteral("rec_gt.txt")), "images/a.png\ta\n"));
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY(workspace.open(directory.filePath(QStringLiteral("project")), &error));
    const auto snapshot = commitSnapshot(&workspace, root, QStringLiteral("paddleocr_rec"), &error);
    const auto taskId = startQualityTask(&workspace, &error);
    aitrain::DataQualityWorkflowResult result;
    QVERIFY2(workspace.runDataQualityWorkflow(taskId, {snapshot.snapshot.id, {}}, &result, &error, [] { return true; }), qPrintable(error));
    QCOMPARE(result.terminalState, aitrain::TaskState::Canceled);
    QVERIFY(result.evidenceArtifactId.isValid());
    aitrain::TaskSnapshot task;
    QVERIFY(workspace.task(taskId, &task, &error));
    QCOMPARE(task.state, aitrain::TaskState::Canceled);
    QCOMPARE(task.failure.code, aitrain::FailureCode::Canceled);
    const auto workflows = workspace.workflowRunsForTask(taskId, &error);
    QCOMPARE(workflows.size(), 1);
    QVERIFY(!hasQualityArtifact(workspace, taskId, &error));
}

QTEST_MAIN(DataQualityTests)
#include "tst_data_quality.moc"
