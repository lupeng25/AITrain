#include "aitrain/workflow/ProjectWorkspace.h"
#include "aitrain/storage/ProjectStore.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QJsonDocument>
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
    QImage image(20, 16, QImage::Format_RGB32);
    image.fill(Qt::white);
    return image.save(path);
}

bool writeColoredImage(const QString& path, QRgb color)
{
    if (!QDir().mkpath(QFileInfo(path).absolutePath())) return false;
    QImage image(20, 16, QImage::Format_RGB32);
    image.fill(color);
    return image.save(path);
}

bool writeMask(const QString& path, bool foreground)
{
    if (!QDir().mkpath(QFileInfo(path).absolutePath())) return false;
    QImage mask(20, 16, QImage::Format_Grayscale8);
    mask.fill(0);
    if (foreground) {
        for (int y = 4; y < 12; ++y) {
            uchar* line = mask.scanLine(y);
            for (int x = 5; x < 15; ++x) line[x] = 1;
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
        QStringLiteral("artifacts/committed/%1/%2").arg(artifactId.toString(), relative));
}

aitrain::TaskId startTask(aitrain::ProjectWorkspace* workspace,
    const QString& capability, const QString& type, QString* error)
{
    const aitrain::TaskId id = aitrain::TaskId::create();
    aitrain::TaskSnapshot task;
    if (!workspace->startTask(id, capability, type, &task, error)) return {};
    return id;
}

aitrain::DatasetSnapshotArtifactBundle commitSnapshot(
    aitrain::ProjectWorkspace* workspace, const QString& root,
    const QString& format, QString* error)
{
    const auto taskId = startTask(workspace, QStringLiteral("dataset.snapshot"),
        QStringLiteral("dataset_snapshot"), error);
    aitrain::DatasetSnapshotCommitRequest request;
    request.datasetRoot = root;
    request.datasetFormat = format;
    request.driverId = format;
    request.driverVersion = QStringLiteral("2.0");
    request.options.classDefinitions = QJsonArray{
        QJsonObject{{QStringLiteral("id"), 0}, {QStringLiteral("name"), QStringLiteral("item")}}};
    aitrain::DatasetSnapshotArtifactBundle result;
    if (!workspace->commitDatasetSnapshot(taskId, request, &result, error)
        || !workspace->finalizeTask(taskId, aitrain::TaskState::Succeeded, {}, error)) return {};
    return result;
}

aitrain::DataQualityWorkflowResult qualityRepair(
    aitrain::ProjectWorkspace* workspace, const aitrain::SnapshotId& snapshotId,
    const QJsonObject& options, QString* error)
{
    const auto taskId = startTask(workspace, QStringLiteral("dataset.quality"),
        QStringLiteral("dataset_quality"), error);
    aitrain::DataQualityWorkflowResult result;
    if (!workspace->runDataQualityWorkflow(taskId, {snapshotId, options}, &result, error)) return {};
    return result;
}

aitrain::AnnotationSessionCreateResult createSession(
    aitrain::ProjectWorkspace* workspace, const aitrain::ArtifactId& repairArtifactId,
    const QString& workingDirectory, QString* error)
{
    const auto taskId = startTask(workspace, QStringLiteral("dataset.annotation.session"),
        QStringLiteral("annotation_session_create"), error);
    aitrain::AnnotationSessionCreateRequest request;
    request.repairManifestArtifactId = repairArtifactId;
    request.workingDirectory = workingDirectory;
    request.toolParameters = {{QStringLiteral("toolId"), QStringLiteral("x-anylabeling")},
        {QStringLiteral("toolVersion"), QStringLiteral("2.5")},
        {QStringLiteral("executablePath"), QStringLiteral("C:/不得持久化/工具.exe")}};
    aitrain::AnnotationSessionCreateResult result;
    if (!workspace->createAnnotationSession(taskId, request, &result, error)) return {};
    return result;
}

aitrain::AnnotationSessionSyncResult syncSession(
    aitrain::ProjectWorkspace* workspace, const aitrain::ArtifactId& sessionArtifactId,
    const QString& workingDirectory, QString* error,
    const aitrain::CancellationCallback& cancellation = {})
{
    const auto taskId = startTask(workspace, QStringLiteral("dataset.annotation.sync"),
        QStringLiteral("annotation_session_sync"), error);
    aitrain::AnnotationSessionSyncResult result;
    if (!workspace->syncAnnotationSession(taskId, {sessionArtifactId, workingDirectory},
            &result, error, cancellation)) return {};
    return result;
}

struct YoloFixture final {
    QString root;
    aitrain::DatasetSnapshotArtifactBundle snapshot;
    aitrain::DataQualityWorkflowResult quality;
};

YoloFixture prepareYolo(QTemporaryDir* directory,
    aitrain::ProjectWorkspace* workspace, QString* error)
{
    YoloFixture fixture;
    fixture.root = directory->filePath(QStringLiteral("原始 yolo"));
    if (!writeBytes(QDir(fixture.root).filePath(QStringLiteral("data.yaml")),
            "path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: [item]\n")
        || !writeImage(QDir(fixture.root).filePath(QStringLiteral("images/train/a.png")))
        || !writeImage(QDir(fixture.root).filePath(QStringLiteral("images/val/b.png")))
        || !writeBytes(QDir(fixture.root).filePath(QStringLiteral("labels/train/a.txt")),
            "0 0.5 0.5 0.05 0.05\n")
        || !writeBytes(QDir(fixture.root).filePath(QStringLiteral("labels/val/b.txt")),
            "0 0.5 0.5 0.5 0.5\n")) return {};
    fixture.snapshot = commitSnapshot(workspace, fixture.root, QStringLiteral("yolo_detection"), error);
    fixture.quality = qualityRepair(workspace, fixture.snapshot.snapshot.id, {}, error);
    return fixture;
}

struct FormatFixture final {
    QString root;
    QString format;
    QString editableFile;
    QString outOfScopeFile;
    aitrain::DatasetSnapshotArtifactBundle snapshot;
    aitrain::DataQualityWorkflowResult quality;
};

FormatFixture prepareRemainingFormat(const QString& format, QTemporaryDir* directory,
    aitrain::ProjectWorkspace* workspace, QString* error)
{
    FormatFixture fixture;
    fixture.root = directory->filePath(QStringLiteral("格式数据-%1").arg(format));
    fixture.format = format;
    if (format == QStringLiteral("yolo_segmentation") || format == QStringLiteral("yolo_obb")) {
        writeBytes(QDir(fixture.root).filePath(QStringLiteral("data.yaml")),
            "path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: [item]\n");
        writeImage(QDir(fixture.root).filePath(QStringLiteral("images/train/a.png")));
        writeImage(QDir(fixture.root).filePath(QStringLiteral("images/val/b.png")));
        fixture.editableFile = QStringLiteral("labels/train/a.txt");
        fixture.outOfScopeFile = QStringLiteral("data.yaml");
        if (format == QStringLiteral("yolo_segmentation")) {
            writeBytes(QDir(fixture.root).filePath(fixture.editableFile),
                "0 0.10 0.10 0.15 0.10 0.10 0.15\n");
            writeBytes(QDir(fixture.root).filePath(QStringLiteral("labels/val/b.txt")),
                "0 0.10 0.10 0.90 0.10 0.50 0.90\n");
        } else {
            writeBytes(QDir(fixture.root).filePath(fixture.editableFile),
                "0 0.10 0.10 0.15 0.10 0.15 0.15 0.10 0.15\n");
            writeBytes(QDir(fixture.root).filePath(QStringLiteral("labels/val/b.txt")),
                "0 0.10 0.10 0.80 0.10 0.80 0.80 0.10 0.80\n");
        }
    } else if (format == QStringLiteral("semantic_segmentation_mask")) {
        writeBytes(QDir(fixture.root).filePath(QStringLiteral("classes.txt")), "background\nitem\n");
        writeImage(QDir(fixture.root).filePath(QStringLiteral("images/train/a.png")));
        writeImage(QDir(fixture.root).filePath(QStringLiteral("images/val/b.png")));
        writeMask(QDir(fixture.root).filePath(QStringLiteral("masks/train/a.png")), false);
        writeMask(QDir(fixture.root).filePath(QStringLiteral("masks/val/b.png")), true);
        fixture.editableFile = QStringLiteral("masks/train/a.png");
        fixture.outOfScopeFile = QStringLiteral("classes.txt");
    } else if (format == QStringLiteral("paddleocr_det")) {
        writeImage(QDir(fixture.root).filePath(QStringLiteral("images/a.png")));
        writeBytes(QDir(fixture.root).filePath(QStringLiteral("det_gt.txt")),
            "images/a.png\t[{\"transcription\":\"a\",\"points\":[[1,1],[3,1],[3,3],[1,3]]}]\n");
        fixture.editableFile = QStringLiteral("det_gt.txt");
        fixture.outOfScopeFile = QStringLiteral("images/a.png");
    }
    fixture.snapshot = commitSnapshot(workspace, fixture.root, format, error);
    if (fixture.snapshot.snapshot.id.isValid()) {
        fixture.quality = qualityRepair(workspace, fixture.snapshot.snapshot.id, {}, error);
    }
    return fixture;
}

bool applyValidEdit(const FormatFixture& fixture, const QString& root)
{
    const QString path = QDir(root).filePath(fixture.editableFile);
    if (fixture.format == QStringLiteral("yolo_segmentation")) {
        return writeBytes(path, "0 0.20 0.20 0.80 0.20 0.50 0.80\n");
    }
    if (fixture.format == QStringLiteral("yolo_obb")) {
        return writeBytes(path, "0 0.20 0.20 0.80 0.20 0.80 0.80 0.20 0.80\n");
    }
    if (fixture.format == QStringLiteral("semantic_segmentation_mask")) {
        return writeMask(path, true);
    }
    if (fixture.format == QStringLiteral("paddleocr_det")) {
        return writeBytes(path,
            "images/a.png\t[{\"transcription\":\"a\",\"points\":[[1,1],[10,1],[10,10],[1,10]]}]\n");
    }
    return false;
}

bool applyOutOfScopeEdit(const FormatFixture& fixture, const QString& root)
{
    const QString path = QDir(root).filePath(fixture.outOfScopeFile);
    if (fixture.format == QStringLiteral("paddleocr_det")) return writeColoredImage(path, Qt::black);
    if (fixture.format == QStringLiteral("semantic_segmentation_mask")) {
        return writeBytes(path, "background\nchanged\n");
    }
    return writeBytes(path,
        "path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: [changed]\n");
}

} // namespace

class AnnotationSessionTests final : public QObject {
    Q_OBJECT

private slots:
    void yoloAllowedChangeCreatesSelfContainedSnapshot();
    void paddleOcrRecAllowedChangeCreatesVersion();
    void noChangesDoesNotCreateVersion();
    void changedBaselineProducesConflictAndEvidence();
    void outOfScopeChangeIsRejected();
    void cancellationHasNoFormalVersionAndHasEvidence();
    void canceledCreateRemovesOnlyPreparedWorkingCopy();
    void remainingEditableFormatsCreateSelfContainedSnapshots_data();
    void remainingEditableFormatsCreateSelfContainedSnapshots();
    void remainingEditableFormatsEnforceSafety_data();
    void remainingEditableFormatsEnforceSafety();
    void anomalyFolderReviewOnlyActionIsUnsupported();
};

void AnnotationSessionTests::yoloAllowedChangeCreatesSelfContainedSnapshot()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY2(workspace.createProject(directory.filePath(QStringLiteral("project")), &error), qPrintable(error));
    const YoloFixture fixture = prepareYolo(&directory, &workspace, &error);
    QVERIFY2(fixture.quality.repairManifestArtifactId.isValid(), qPrintable(error));
    const QString work = directory.filePath(QStringLiteral("外部工作副本"));
    const auto session = createSession(&workspace, fixture.quality.repairManifestArtifactId, work, &error);
    QVERIFY2(session.sessionArtifactId.isValid(), qPrintable(error));
    QCOMPARE(session.terminalState, aitrain::TaskState::Succeeded);
    QVERIFY(session.evidenceArtifactId.isValid());
    const QString sessionPath = artifactFile(workspace, session.sessionArtifactId,
        QStringLiteral("annotation_session.json"));
    QFile sessionFile(sessionPath);
    QVERIFY(sessionFile.open(QIODevice::ReadOnly));
    const QByteArray sessionBytes = sessionFile.readAll();
    QVERIFY(!sessionBytes.contains(work.toUtf8()));
    QVERIFY(!sessionBytes.contains("不得持久化"));
    const QJsonObject sessionManifest = QJsonDocument::fromJson(sessionBytes).object();
    QCOMPARE(sessionManifest.value(QStringLiteral("sourceDatasetVersionId")).toString(),
        fixture.snapshot.snapshot.datasetVersionId.toString());
    QCOMPARE(sessionManifest.value(QStringLiteral("outputArtifactId")).toString(), session.sessionArtifactId.toString());
    aitrain::ProjectStore storage;
    QVERIFY2(storage.open(QDir(workspace.workspacePath()).filePath(QStringLiteral("project.sqlite")), &error),
        qPrintable(error));
    aitrain::WorkflowInputBinding createInput;
    QVERIFY2(storage.workflowInput(session.workflowRunId, QStringLiteral("dataset_repair_manifest"),
        &createInput, &error), qPrintable(error));
    aitrain::ArtifactSnapshot repairArtifact;
    QVERIFY2(storage.artifact(fixture.quality.repairManifestArtifactId, &repairArtifact, &error), qPrintable(error));
    QCOMPARE(createInput.sourceArtifactId, fixture.quality.repairManifestArtifactId);
    QCOMPARE(createInput.sourceTaskId, repairArtifact.taskId);
    QCOMPARE(createInput.sourceArtifactKind, QStringLiteral("dataset_repair_manifest"));
    QVERIFY(!createInput.datasetId.isValid());
    QVERIFY(!createInput.datasetVersionId.isValid());
    QVERIFY(!createInput.datasetSnapshotId.isValid());
    QVERIFY(createInput.manifestSha256.isEmpty());
    QVERIFY(createInput.rootHash.isEmpty());
    QVERIFY(writeBytes(QDir(work).filePath(QStringLiteral("labels/train/a.txt")),
        "0 0.5 0.5 0.4 0.4\n"));
    const auto sync = syncSession(&workspace, session.sessionArtifactId, work, &error);
    QVERIFY2(sync.workflowRunId.isValid(), qPrintable(error));
    QCOMPARE(sync.terminalState, aitrain::TaskState::Succeeded);
    QCOMPARE(sync.status, aitrain::AnnotationSyncStatus::ChangesDetected);
    QVERIFY(sync.datasetSnapshot.id.isValid());
    QVERIFY(sync.datasetSnapshot.datasetVersionId.isValid());
    QVERIFY(sync.datasetSnapshot.artifactId.isValid());
    QVERIFY(sync.evidenceArtifactId.isValid());
    aitrain::WorkflowInputBinding syncInput;
    QVERIFY2(storage.workflowInput(sync.workflowRunId, QStringLiteral("annotation_session"),
        &syncInput, &error), qPrintable(error));
    aitrain::ArtifactSnapshot sessionArtifact;
    QVERIFY2(storage.artifact(session.sessionArtifactId, &sessionArtifact, &error), qPrintable(error));
    QCOMPARE(syncInput.sourceArtifactId, session.sessionArtifactId);
    QCOMPARE(syncInput.sourceTaskId, sessionArtifact.taskId);
    QCOMPARE(syncInput.sourceArtifactKind, QStringLiteral("annotation_session"));
    QVERIFY(!syncInput.datasetId.isValid());
    QVERIFY(!syncInput.datasetVersionId.isValid());
    QVERIFY(!syncInput.datasetSnapshotId.isValid());
    QVERIFY(QFileInfo::exists(artifactFile(workspace, sync.datasetSnapshot.artifactId,
        QStringLiteral("labels/train/a.txt"))));
    QVERIFY(QFileInfo::exists(artifactFile(workspace, sync.datasetSnapshot.artifactId,
        QStringLiteral("images/train/a.png"))));
    const QJsonObject snapshotManifest = readJson(artifactFile(workspace, sync.datasetSnapshot.artifactId,
        QStringLiteral("dataset_snapshot.json")));
    QCOMPARE(snapshotManifest.value(QStringLiteral("sourceSnapshotId")).toString(),
        fixture.snapshot.snapshot.id.toString());
    QCOMPARE(snapshotManifest.value(QStringLiteral("annotationSessionArtifactId")).toString(),
        session.sessionArtifactId.toString());
    QFile original(QDir(fixture.root).filePath(QStringLiteral("labels/train/a.txt")));
    QVERIFY(original.open(QIODevice::ReadOnly));
    QCOMPARE(original.readAll(), QByteArray("0 0.5 0.5 0.05 0.05\n"));
}

void AnnotationSessionTests::paddleOcrRecAllowedChangeCreatesVersion()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString root = directory.filePath(QStringLiteral("ocr 原始"));
    QVERIFY(writeImage(QDir(root).filePath(QStringLiteral("images/a.png"))));
    QVERIFY(writeImage(QDir(root).filePath(QStringLiteral("images/b.png"))));
    QVERIFY(writeBytes(QDir(root).filePath(QStringLiteral("dict.txt")), "a\nb\nc\n"));
    QVERIFY(writeBytes(QDir(root).filePath(QStringLiteral("rec_gt.txt")),
        "images/a.png\tabcabc\nimages/b.png\tab\n"));
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY(workspace.createProject(directory.filePath(QStringLiteral("project")), &error));
    const auto snapshot = commitSnapshot(&workspace, root, QStringLiteral("paddleocr_rec"), &error);
    const auto quality = qualityRepair(&workspace, snapshot.snapshot.id,
        {{QStringLiteral("maximumLabelLength"), 3}}, &error);
    const QString work = directory.filePath(QStringLiteral("ocr work"));
    const auto session = createSession(&workspace, quality.repairManifestArtifactId, work, &error);
    QVERIFY2(session.sessionArtifactId.isValid(), qPrintable(error));
    QVERIFY(writeBytes(QDir(work).filePath(QStringLiteral("rec_gt.txt")),
        "images/a.png\tabc\nimages/b.png\tab\n"));
    const auto sync = syncSession(&workspace, session.sessionArtifactId, work, &error);
    QVERIFY2(sync.datasetSnapshot.id.isValid(), qPrintable(error));
    QCOMPARE(sync.status, aitrain::AnnotationSyncStatus::ChangesDetected);
    QVERIFY(sync.datasetSnapshot.datasetVersionId.isValid());
    QVERIFY(sync.evidenceArtifactId.isValid());
    QFile original(QDir(root).filePath(QStringLiteral("rec_gt.txt")));
    QVERIFY(original.open(QIODevice::ReadOnly));
    QCOMPARE(original.readAll(), QByteArray("images/a.png\tabcabc\nimages/b.png\tab\n"));
}

void AnnotationSessionTests::noChangesDoesNotCreateVersion()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY(workspace.createProject(directory.filePath(QStringLiteral("project")), &error));
    const YoloFixture fixture = prepareYolo(&directory, &workspace, &error);
    const QString work = directory.filePath(QStringLiteral("no changes"));
    const auto session = createSession(&workspace, fixture.quality.repairManifestArtifactId, work, &error);
    const auto sync = syncSession(&workspace, session.sessionArtifactId, work, &error);
    QVERIFY2(sync.workflowRunId.isValid(), qPrintable(error));
    QCOMPARE(sync.terminalState, aitrain::TaskState::Succeeded);
    QCOMPARE(sync.status, aitrain::AnnotationSyncStatus::NoChanges);
    QVERIFY(!sync.datasetSnapshot.id.isValid());
    QVERIFY(sync.syncReportArtifactId.isValid());
    QVERIFY(sync.evidenceArtifactId.isValid());
}

void AnnotationSessionTests::changedBaselineProducesConflictAndEvidence()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY(workspace.createProject(directory.filePath(QStringLiteral("project")), &error));
    const YoloFixture fixture = prepareYolo(&directory, &workspace, &error);
    const QString work = directory.filePath(QStringLiteral("conflict"));
    const auto session = createSession(&workspace, fixture.quality.repairManifestArtifactId, work, &error);
    QVERIFY(writeBytes(QDir(fixture.snapshot.artifactPath).filePath(QStringLiteral("labels/train/a.txt")),
        "0 0.4 0.4 0.3 0.3\n"));
    const auto sync = syncSession(&workspace, session.sessionArtifactId, work, &error);
    QVERIFY2(sync.workflowRunId.isValid(), qPrintable(error));
    QCOMPARE(sync.terminalState, aitrain::TaskState::Failed);
    QCOMPARE(sync.status, aitrain::AnnotationSyncStatus::Conflict);
    QVERIFY(!sync.datasetSnapshot.id.isValid());
    QVERIFY(sync.changesArtifactId.isValid());
    QVERIFY(sync.syncReportArtifactId.isValid());
    QVERIFY(sync.evidenceArtifactId.isValid());
    const QJsonObject report = readJson(artifactFile(workspace, sync.changesArtifactId,
        QStringLiteral("change_report.json")));
    QCOMPARE(report.value(QStringLiteral("status")).toString(), QStringLiteral("Conflict"));
}

void AnnotationSessionTests::outOfScopeChangeIsRejected()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY(workspace.createProject(directory.filePath(QStringLiteral("project")), &error));
    const YoloFixture fixture = prepareYolo(&directory, &workspace, &error);
    const QString work = directory.filePath(QStringLiteral("越界"));
    const auto session = createSession(&workspace, fixture.quality.repairManifestArtifactId, work, &error);
    QVERIFY(writeBytes(QDir(work).filePath(QStringLiteral("data.yaml")),
        "path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: [changed]\n"));
    const auto sync = syncSession(&workspace, session.sessionArtifactId, work, &error);
    QVERIFY2(sync.workflowRunId.isValid(), qPrintable(error));
    QCOMPARE(sync.terminalState, aitrain::TaskState::Failed);
    QCOMPARE(sync.status, aitrain::AnnotationSyncStatus::InvalidSession);
    QVERIFY(!sync.datasetSnapshot.id.isValid());
    QVERIFY(sync.evidenceArtifactId.isValid());
    const QJsonObject report = readJson(artifactFile(workspace, sync.changesArtifactId,
        QStringLiteral("change_report.json")));
    QCOMPARE(report.value(QStringLiteral("code")).toString(),
        QStringLiteral("annotation.session.change_out_of_scope"));
}

void AnnotationSessionTests::cancellationHasNoFormalVersionAndHasEvidence()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY(workspace.createProject(directory.filePath(QStringLiteral("project")), &error));
    const YoloFixture fixture = prepareYolo(&directory, &workspace, &error);
    const QString work = directory.filePath(QStringLiteral("cancel"));
    const auto session = createSession(&workspace, fixture.quality.repairManifestArtifactId, work, &error);
    const auto sync = syncSession(&workspace, session.sessionArtifactId, work, &error, [] { return true; });
    QVERIFY2(sync.workflowRunId.isValid(), qPrintable(error));
    QCOMPARE(sync.terminalState, aitrain::TaskState::Canceled);
    QCOMPARE(sync.status, aitrain::AnnotationSyncStatus::Canceled);
    QVERIFY(!sync.datasetSnapshot.id.isValid());
    QVERIFY(sync.evidenceArtifactId.isValid());
}

void AnnotationSessionTests::canceledCreateRemovesOnlyPreparedWorkingCopy()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY(workspace.createProject(directory.filePath(QStringLiteral("project")), &error));
    const YoloFixture fixture = prepareYolo(&directory, &workspace, &error);
    const QString work = directory.filePath(QStringLiteral("cancel during copy"));
    const auto taskId = startTask(&workspace, QStringLiteral("dataset.annotation.session"),
        QStringLiteral("annotation_session_create"), &error);
    aitrain::AnnotationSessionCreateRequest request;
    request.repairManifestArtifactId = fixture.quality.repairManifestArtifactId;
    request.workingDirectory = work;
    aitrain::AnnotationSessionCreateResult result;
    const auto cancelAfterDirectoryCreation = [&work] {
        return QDir(work).exists();
    };
    QVERIFY2(workspace.createAnnotationSession(taskId, request, &result, &error,
        cancelAfterDirectoryCreation), qPrintable(error));
    QCOMPARE(result.terminalState, aitrain::TaskState::Canceled);
    QVERIFY(!result.sessionArtifactId.isValid());
    QVERIFY(result.evidenceArtifactId.isValid());
    QVERIFY(!QDir(work).exists());
    QVERIFY(QFileInfo::exists(QDir(fixture.root).filePath(QStringLiteral("labels/train/a.txt"))));
}

void AnnotationSessionTests::remainingEditableFormatsCreateSelfContainedSnapshots_data()
{
    QTest::addColumn<QString>("format");
    QTest::newRow("YOLO Segmentation") << QStringLiteral("yolo_segmentation");
    QTest::newRow("YOLO OBB") << QStringLiteral("yolo_obb");
    QTest::newRow("Semantic Mask") << QStringLiteral("semantic_segmentation_mask");
    QTest::newRow("PaddleOCR Det") << QStringLiteral("paddleocr_det");
}

void AnnotationSessionTests::remainingEditableFormatsCreateSelfContainedSnapshots()
{
    QFETCH(QString, format);
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY2(workspace.createProject(directory.filePath(QStringLiteral("project")), &error), qPrintable(error));
    const FormatFixture fixture = prepareRemainingFormat(format, &directory, &workspace, &error);
    QVERIFY2(fixture.quality.repairManifestArtifactId.isValid(), qPrintable(error));
    QFile originalFile(QDir(fixture.root).filePath(fixture.editableFile));
    QVERIFY(originalFile.open(QIODevice::ReadOnly));
    const QByteArray originalBytes = originalFile.readAll();
    originalFile.close();
    const QString work = directory.filePath(QStringLiteral("work-%1").arg(format));
    const auto session = createSession(&workspace, fixture.quality.repairManifestArtifactId, work, &error);
    QVERIFY2(session.sessionArtifactId.isValid(), qPrintable(error));
    QVERIFY(applyValidEdit(fixture, work));
    const auto sync = syncSession(&workspace, session.sessionArtifactId, work, &error);
    QVERIFY2(sync.workflowRunId.isValid(), qPrintable(error));
    QCOMPARE(sync.terminalState, aitrain::TaskState::Succeeded);
    QCOMPARE(sync.status, aitrain::AnnotationSyncStatus::ChangesDetected);
    QVERIFY(sync.datasetSnapshot.id.isValid());
    QVERIFY(sync.datasetSnapshot.datasetVersionId.isValid());
    QVERIFY(sync.datasetSnapshot.artifactId.isValid());
    QVERIFY(sync.evidenceArtifactId.isValid());
    QVERIFY(QFileInfo::exists(artifactFile(workspace, sync.datasetSnapshot.artifactId,
        fixture.editableFile)));
    QFile sourceAfter(QDir(fixture.root).filePath(fixture.editableFile));
    QVERIFY(sourceAfter.open(QIODevice::ReadOnly));
    QCOMPARE(sourceAfter.readAll(), originalBytes);
}

void AnnotationSessionTests::remainingEditableFormatsEnforceSafety_data()
{
    QTest::addColumn<QString>("format");
    QTest::addColumn<QString>("scenario");
    const QStringList formats{QStringLiteral("yolo_segmentation"), QStringLiteral("yolo_obb"),
        QStringLiteral("semantic_segmentation_mask"), QStringLiteral("paddleocr_det")};
    const QStringList scenarios{QStringLiteral("conflict"), QStringLiteral("out_of_scope"),
        QStringLiteral("cancel")};
    for (const QString& format : formats) {
        for (const QString& scenario : scenarios) {
            const QByteArray row = QStringLiteral("%1-%2").arg(format, scenario).toUtf8();
            QTest::newRow(row.constData()) << format << scenario;
        }
    }
}

void AnnotationSessionTests::remainingEditableFormatsEnforceSafety()
{
    QFETCH(QString, format);
    QFETCH(QString, scenario);
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY2(workspace.createProject(directory.filePath(QStringLiteral("project")), &error), qPrintable(error));
    const FormatFixture fixture = prepareRemainingFormat(format, &directory, &workspace, &error);
    QVERIFY2(fixture.quality.repairManifestArtifactId.isValid(), qPrintable(error));
    const QString work = directory.filePath(QStringLiteral("safe-%1-%2").arg(format, scenario));
    const auto session = createSession(&workspace, fixture.quality.repairManifestArtifactId, work, &error);
    QVERIFY2(session.sessionArtifactId.isValid(), qPrintable(error));

    aitrain::AnnotationSessionSyncResult sync;
    if (scenario == QStringLiteral("conflict")) {
        // Snapshot 已自包含；冲突测试应篡改会话记录的 committed 基线，而不是
        // 修改已经脱离业务边界的外部导入根。
        QVERIFY(applyValidEdit(fixture, fixture.snapshot.artifactPath));
        sync = syncSession(&workspace, session.sessionArtifactId, work, &error);
        QCOMPARE(sync.terminalState, aitrain::TaskState::Failed);
        QCOMPARE(sync.status, aitrain::AnnotationSyncStatus::Conflict);
    } else if (scenario == QStringLiteral("out_of_scope")) {
        QVERIFY(applyOutOfScopeEdit(fixture, work));
        sync = syncSession(&workspace, session.sessionArtifactId, work, &error);
        QCOMPARE(sync.terminalState, aitrain::TaskState::Failed);
        QCOMPARE(sync.status, aitrain::AnnotationSyncStatus::InvalidSession);
        const QJsonObject report = readJson(artifactFile(workspace, sync.changesArtifactId,
            QStringLiteral("change_report.json")));
        QCOMPARE(report.value(QStringLiteral("code")).toString(),
            QStringLiteral("annotation.session.change_out_of_scope"));
    } else {
        sync = syncSession(&workspace, session.sessionArtifactId, work, &error, [] { return true; });
        QCOMPARE(sync.terminalState, aitrain::TaskState::Canceled);
        QCOMPARE(sync.status, aitrain::AnnotationSyncStatus::Canceled);
    }
    QVERIFY2(sync.workflowRunId.isValid(), qPrintable(error));
    QVERIFY(!sync.datasetSnapshot.id.isValid());
    QVERIFY(sync.evidenceArtifactId.isValid());
}

void AnnotationSessionTests::anomalyFolderReviewOnlyActionIsUnsupported()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString root = directory.filePath(QStringLiteral("异常检测数据"));
    QVERIFY(writeImage(QDir(root).filePath(QStringLiteral("train/good/a.png"))));
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY2(workspace.createProject(directory.filePath(QStringLiteral("project")), &error), qPrintable(error));
    const auto snapshot = commitSnapshot(&workspace, root, QStringLiteral("anomaly_folder"), &error);
    const auto quality = qualityRepair(&workspace, snapshot.snapshot.id, {}, &error);
    QVERIFY2(quality.repairManifestArtifactId.isValid(), qPrintable(error));
    const auto session = createSession(&workspace, quality.repairManifestArtifactId,
        directory.filePath(QStringLiteral("anomaly-work")), &error);
    QCOMPARE(session.terminalState, aitrain::TaskState::Failed);
    QVERIFY(!session.sessionArtifactId.isValid());
    QVERIFY(session.evidenceArtifactId.isValid());
    QFile evidence(artifactFile(workspace, session.evidenceArtifactId, QStringLiteral("evidence.json")));
    QVERIFY(evidence.open(QIODevice::ReadOnly));
    const QByteArray bytes = evidence.readAll();
    QVERIFY(bytes.contains("backend_unsupported"));
    QVERIFY(bytes.contains("review_only_add_samples"));
}

QTEST_MAIN(AnnotationSessionTests)
#include "tst_annotation_session.moc"
