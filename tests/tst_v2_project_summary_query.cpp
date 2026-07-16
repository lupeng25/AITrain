#include "aitrain/v2/ProjectQueryServiceV2.h"
#include "aitrain/v2/StorageV2.h"

#include <QDir>
#include <QFile>
#include <QTemporaryDir>
#include <QTest>

namespace {

QString v2DatabasePath(const QString& projectPath)
{
    return QDir(projectPath).filePath(QStringLiteral(".aitrain-v2/project-v2.sqlite"));
}

aitrain::v2::TaskSnapshot createTask(aitrain::v2::StorageV2& storage, QString* error)
{
    aitrain::v2::TaskSnapshot task;
    task.id = aitrain::v2::TaskId::create();
    task.requestId = aitrain::v2::RequestId::create();
    task.capabilityId = QStringLiteral("summary.test");
    task.taskType = QStringLiteral("summary_test");
    if (!storage.createTask(task, error)) return {};
    return task;
}

bool startTask(aitrain::v2::StorageV2& storage,
    const aitrain::v2::TaskId& taskId,
    QString* error)
{
    using aitrain::v2::TaskState;
    return storage.transitionTask(taskId, TaskState::Created, TaskState::Queued, {}, error)
        && storage.transitionTask(taskId, TaskState::Queued, TaskState::Starting, {}, error)
        && storage.transitionTask(taskId, TaskState::Starting, TaskState::Running, {}, error);
}

aitrain::v2::Failure failure(aitrain::v2::FailureCode code, const QString& message)
{
    return {code, message, QStringLiteral("修复测试输入后重试。"), QDateTime::currentDateTimeUtc()};
}

aitrain::v2::ModelManifestV2 modelManifest(const aitrain::v2::TaskId& taskId,
    const QString& sourceSha256)
{
    aitrain::v2::ModelManifestV2 manifest;
    manifest.modelPackageId = aitrain::v2::ModelPackageId::create();
    manifest.modelFamily = QStringLiteral("yolo_detection");
    manifest.taskType = QStringLiteral("detection");
    manifest.sourceBackend = QStringLiteral("ultralytics_yolo_detect");
    manifest.sourceTaskId = taskId;
    manifest.sourceSnapshotId = aitrain::v2::SnapshotId::create();
    manifest.sourceArtifactSha256 = sourceSha256;
    manifest.artifactEntryPath = QStringLiteral("model.onnx");
    manifest.inputs = {{QStringLiteral("images"), QStringLiteral("NCHW"), {1, 3, 640, 640}}};
    manifest.outputs = {{QStringLiteral("output0"), QStringLiteral("NCN"), {1, 84, -1}}};
    manifest.preprocessing = {{QStringLiteral("id"), QStringLiteral("letterbox_rgb_0_1")}};
    manifest.postprocessing = {{QStringLiteral("id"), QStringLiteral("yolo_detection_nms")}};
    manifest.decoder = QStringLiteral("yolo_detection_v8");
    manifest.classNames = QStringList{QStringLiteral("part")};
    manifest.opset = 17;
    manifest.exporterVersion = QStringLiteral("test-exporter");
    manifest.runtimeRoutes = QStringList{QStringLiteral("aitrain_onnxruntime")};
    manifest.verified = true;
    return manifest;
}

bool registerSnapshot(aitrain::v2::StorageV2& storage,
    const aitrain::v2::TaskId& taskId,
    const QString& root,
    const QString& rootHash,
    const aitrain::v2::DatasetId& datasetId,
    QString* error)
{
    const QString manifestSha = QString(64, QLatin1Char('d'));
    const aitrain::v2::ArtifactId artifactId = aitrain::v2::ArtifactId::create();
    if (!storage.recordArtifactWithFiles(artifactId, taskId,
            QStringLiteral("dataset_snapshot_v2"),
            {{QStringLiteral("dataset_snapshot.json"), manifestSha, 64}},
            QDateTime::currentDateTimeUtc(), error)) {
        return false;
    }
    aitrain::v2::DatasetSnapshotRecordV2 snapshot;
    snapshot.id = aitrain::v2::SnapshotId::create();
    snapshot.datasetId = datasetId;
    snapshot.taskId = taskId;
    snapshot.artifactId = artifactId;
    snapshot.rootPath = QDir(root).filePath(QStringLiteral("artifacts/%1").arg(artifactId.toString()));
    snapshot.datasetFormat = QStringLiteral("yolo_detection");
    snapshot.driverId = QStringLiteral("yolo_detection_v2");
    snapshot.driverVersion = QStringLiteral("2.0");
    snapshot.rootHash = rootHash;
    snapshot.manifestSha256 = manifestSha;
    snapshot.fileCount = 1;
    snapshot.totalBytes = 64;
    return storage.registerDatasetSnapshot(&snapshot, error);
}

} // namespace

class V2ProjectSummaryQueryTests : public QObject {
    Q_OBJECT

private slots:
    void readsEmptyProject();
    void countsMixedTaskStates();
    void countsOnlyCommittedArtifacts();
    void reportsDatasetModelWorkflowAndEvidenceFacts();
    void rejectsInvalidInput();
};

void V2ProjectSummaryQueryTests::readsEmptyProject()
{
    QTemporaryDir project;
    QVERIFY(project.isValid());
    aitrain::v2::ProjectWorkspaceV2 workspace;
    QString error;
    QVERIFY2(workspace.open(project.path(), &error), qPrintable(error));

    aitrain::v2::ProjectQueryServiceV2 query(&workspace);
    aitrain::v2::ProjectSummaryReadModelV2 summary;
    QVERIFY2(query.projectSummary(&summary, &error), qPrintable(error));
    QCOMPARE(summary.tasks.total(), qint64(0));
    QCOMPARE(summary.committedArtifactCount, qint64(0));
    QCOMPARE(summary.datasetCount, qint64(0));
    QCOMPARE(summary.datasetVersionCount, qint64(0));
    QCOMPARE(summary.datasetSnapshotCount, qint64(0));
    QCOMPARE(summary.modelPackageCount, qint64(0));
    QCOMPARE(summary.workflowRunCount, qint64(0));
    QCOMPARE(summary.evidenceAvailableWorkflowCount, qint64(0));
}

void V2ProjectSummaryQueryTests::countsMixedTaskStates()
{
    QTemporaryDir project;
    QVERIFY(project.isValid());
    aitrain::v2::ProjectWorkspaceV2 workspace;
    QString error;
    QVERIFY2(workspace.open(project.path(), &error), qPrintable(error));
    aitrain::v2::StorageV2 storage;
    QVERIFY2(storage.open(v2DatabasePath(project.path()), &error), qPrintable(error));

    const auto created = createTask(storage, &error);
    const auto succeeded = createTask(storage, &error);
    const auto failed = createTask(storage, &error);
    const auto canceled = createTask(storage, &error);
    QVERIFY2(created.id.isValid() && succeeded.id.isValid() && failed.id.isValid() && canceled.id.isValid(), qPrintable(error));
    QVERIFY2(startTask(storage, succeeded.id, &error), qPrintable(error));
    QVERIFY2(storage.transitionTask(succeeded.id, aitrain::v2::TaskState::Running,
        aitrain::v2::TaskState::Succeeded, {}, &error), qPrintable(error));
    QVERIFY2(startTask(storage, failed.id, &error), qPrintable(error));
    QVERIFY2(storage.transitionTask(failed.id, aitrain::v2::TaskState::Running,
        aitrain::v2::TaskState::Failed,
        failure(aitrain::v2::FailureCode::InvalidRequest, QStringLiteral("测试失败")), &error), qPrintable(error));
    QVERIFY2(startTask(storage, canceled.id, &error), qPrintable(error));
    QVERIFY2(storage.transitionTask(canceled.id, aitrain::v2::TaskState::Running,
        aitrain::v2::TaskState::CancelRequested, {}, &error), qPrintable(error));
    QVERIFY2(storage.transitionTask(canceled.id, aitrain::v2::TaskState::CancelRequested,
        aitrain::v2::TaskState::Canceled,
        failure(aitrain::v2::FailureCode::Canceled, QStringLiteral("测试取消")), &error), qPrintable(error));

    aitrain::v2::ProjectQueryServiceV2 query(&workspace);
    aitrain::v2::ProjectSummaryReadModelV2 summary;
    QVERIFY2(query.projectSummary(&summary, &error), qPrintable(error));
    QCOMPARE(summary.tasks.total(), qint64(4));
    QCOMPARE(summary.tasks.created, qint64(1));
    QCOMPARE(summary.tasks.succeeded, qint64(1));
    QCOMPARE(summary.tasks.failed, qint64(1));
    QCOMPARE(summary.tasks.canceled, qint64(1));
    QCOMPARE(summary.tasks.running, qint64(0));
}

void V2ProjectSummaryQueryTests::countsOnlyCommittedArtifacts()
{
    QTemporaryDir project;
    QVERIFY(project.isValid());
    aitrain::v2::ProjectWorkspaceV2 workspace;
    QString error;
    QVERIFY2(workspace.open(project.path(), &error), qPrintable(error));
    aitrain::v2::StorageV2 storage;
    QVERIFY2(storage.open(v2DatabasePath(project.path()), &error), qPrintable(error));
    const auto task = createTask(storage, &error);
    QVERIFY2(task.id.isValid(), qPrintable(error));

    const QString staging = QDir(project.path()).filePath(QStringLiteral("artifacts/.staging/uncommitted"));
    QVERIFY(QDir().mkpath(staging));
    QFile looseFile(QDir(staging).filePath(QStringLiteral("partial.bin")));
    QVERIFY(looseFile.open(QIODevice::WriteOnly));
    QCOMPARE(looseFile.write("partial"), qint64(7));
    looseFile.close();

    aitrain::v2::ProjectQueryServiceV2 query(&workspace);
    aitrain::v2::ProjectSummaryReadModelV2 summary;
    QVERIFY2(query.projectSummary(&summary, &error), qPrintable(error));
    QCOMPARE(summary.committedArtifactCount, qint64(0));

    QVERIFY2(storage.recordArtifact(aitrain::v2::ArtifactId::create(), task.id,
        QStringLiteral("committed_test"), QDateTime::currentDateTimeUtc(), &error), qPrintable(error));
    QVERIFY2(query.projectSummary(&summary, &error), qPrintable(error));
    QCOMPARE(summary.committedArtifactCount, qint64(1));
}

void V2ProjectSummaryQueryTests::reportsDatasetModelWorkflowAndEvidenceFacts()
{
    QTemporaryDir project;
    QVERIFY(project.isValid());
    aitrain::v2::ProjectWorkspaceV2 workspace;
    QString error;
    QVERIFY2(workspace.open(project.path(), &error), qPrintable(error));
    aitrain::v2::StorageV2 storage;
    QVERIFY2(storage.open(v2DatabasePath(project.path()), &error), qPrintable(error));
    const auto task = createTask(storage, &error);
    QVERIFY2(task.id.isValid(), qPrintable(error));
    QVERIFY2(startTask(storage, task.id, &error), qPrintable(error));

    const QString datasetRoot = QDir(project.path()).filePath(QStringLiteral("source-dataset"));
    const aitrain::v2::DatasetId datasetId = aitrain::v2::DatasetId::create();
    QVERIFY2(registerSnapshot(storage, task.id, datasetRoot, QString(64, QLatin1Char('1')), datasetId, &error), qPrintable(error));
    QVERIFY2(registerSnapshot(storage, task.id, datasetRoot, QString(64, QLatin1Char('2')), datasetId, &error), qPrintable(error));

    const QString modelSha = QString(64, QLatin1Char('a'));
    const auto modelArtifactId = aitrain::v2::ArtifactId::create();
    QVERIFY2(storage.recordArtifactWithFiles(modelArtifactId, task.id,
        QStringLiteral("model_package_v2"),
        {{QStringLiteral("model.onnx"), modelSha, 128}},
        QDateTime::currentDateTimeUtc(), &error), qPrintable(error));
    const auto manifest = modelManifest(task.id, modelSha);
    QVERIFY2(storage.registerModelPackage({manifest, modelArtifactId, QDateTime::currentDateTimeUtc()}, &error), qPrintable(error));

    aitrain::v2::WorkflowRunSnapshotV2 workflow;
    workflow.id = aitrain::v2::WorkflowRunId::create();
    workflow.taskId = task.id;
    workflow.templateId = QStringLiteral("summary-evidence-workflow");
    workflow.terminalPolicy = aitrain::v2::WorkflowTerminalPolicyV2::EvidenceRequired;
    aitrain::v2::WorkflowStepSnapshotV2 step;
    step.id = aitrain::v2::WorkflowStepId::create();
    step.workflowRunId = workflow.id;
    step.ordinal = 0;
    step.kind = QStringLiteral("RenderDeliveryReport");
    step.backend = QStringLiteral("aitrain_core");
    QVERIFY2(storage.createWorkflowRun(workflow, {step}, &error), qPrintable(error));
    QVERIFY2(storage.transitionWorkflowStep(step.id, aitrain::v2::WorkflowStepState::Pending,
        aitrain::v2::WorkflowStepState::Running, {}, {}, &error), qPrintable(error));
    QVERIFY2(storage.transitionWorkflowStep(step.id, aitrain::v2::WorkflowStepState::Running,
        aitrain::v2::WorkflowStepState::Succeeded, modelArtifactId, {}, &error), qPrintable(error));
    QVERIFY2(storage.sealWorkflowTerminalization(workflow.id, aitrain::v2::TaskState::Succeeded,
        {}, QDateTime::currentDateTimeUtc(), &error), qPrintable(error));

    aitrain::v2::ProjectQueryServiceV2 query(&workspace);
    aitrain::v2::ProjectSummaryReadModelV2 summary;
    QVERIFY2(query.projectSummary(&summary, &error), qPrintable(error));
    QCOMPARE(summary.datasetCount, qint64(1));
    QCOMPARE(summary.datasetVersionCount, qint64(2));
    QCOMPARE(summary.datasetSnapshotCount, qint64(2));
    QCOMPARE(summary.modelPackageCount, qint64(1));
    QCOMPARE(summary.verifiedModelPackageCount, qint64(1));
    QCOMPARE(summary.workflowRunCount, qint64(1));
    QCOMPARE(summary.evidenceRequiredWorkflowCount, qint64(1));
    QCOMPARE(summary.evidencePendingWorkflowCount, qint64(1));
    QCOMPARE(summary.evidenceAvailableWorkflowCount, qint64(0));

    const auto evidenceId = aitrain::v2::ArtifactId::create();
    QVERIFY2(storage.recordEvidenceArtifactWithFilesAndAttachTerminalization(
        evidenceId, task.id, workflow.id,
        {{QStringLiteral("evidence.json"), QString(64, QLatin1Char('e')), 32}},
        QDateTime::currentDateTimeUtc(), &error), qPrintable(error));
    QVERIFY2(storage.closeWorkflowTerminalization(workflow.id,
        aitrain::v2::TaskState::Running, &error), qPrintable(error));
    QVERIFY2(query.projectSummary(&summary, &error), qPrintable(error));
    QCOMPARE(summary.evidencePendingWorkflowCount, qint64(0));
    QCOMPARE(summary.evidenceAvailableWorkflowCount, qint64(1));
}

void V2ProjectSummaryQueryTests::rejectsInvalidInput()
{
    QString error;
    aitrain::v2::ProjectSummaryReadModelV2 summary;
    aitrain::v2::ProjectQueryServiceV2 missingWorkspace(nullptr);
    QVERIFY(!missingWorkspace.projectSummary(&summary, &error));
    QVERIFY(!error.isEmpty());

    QTemporaryDir project;
    QVERIFY(project.isValid());
    aitrain::v2::ProjectWorkspaceV2 workspace;
    error.clear();
    QVERIFY2(workspace.open(project.path(), &error), qPrintable(error));
    aitrain::v2::ProjectQueryServiceV2 query(&workspace);
    QVERIFY(!query.projectSummary(nullptr, &error));
    QVERIFY(!error.isEmpty());
    workspace.close();
    QVERIFY(!query.projectSummary(&summary, &error));
    QVERIFY(!error.isEmpty());
}

QTEST_GUILESS_MAIN(V2ProjectSummaryQueryTests)
#include "tst_v2_project_summary_query.moc"
