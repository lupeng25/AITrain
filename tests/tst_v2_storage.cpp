#include "aitrain/v2/StorageV2.h"
#include "aitrain/v2/ProtocolV2.h"

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QSqlDatabase>
#include <QSqlError>
#include <QSqlQuery>
#include <QTemporaryDir>
#include <QTest>

namespace {

aitrain::v2::TaskSnapshot makeTask()
{
    aitrain::v2::TaskSnapshot task;
    task.id = aitrain::v2::TaskId::create();
    task.requestId = aitrain::v2::RequestId::create();
    task.capabilityId = QStringLiteral("yolo.detect");
    task.taskType = QStringLiteral("training");
    return task;
}

aitrain::v2::ModelManifestV2 makeManifest(const aitrain::v2::TaskId& taskId)
{
    aitrain::v2::ModelManifestV2 manifest;
    manifest.modelPackageId = aitrain::v2::ModelPackageId::create();
    manifest.modelFamily = QStringLiteral("yolo_detection");
    manifest.taskType = QStringLiteral("detection");
    manifest.sourceBackend = QStringLiteral("ultralytics_yolo_detect");
    manifest.sourceTaskId = taskId;
    manifest.sourceSnapshotId = aitrain::v2::SnapshotId::create();
    manifest.sourceArtifactSha256 = QString(64, QLatin1Char('a'));
    manifest.artifactEntryPath = QStringLiteral("model.onnx");
    manifest.inputs = {{QStringLiteral("images"), QStringLiteral("NCHW"), {1, 3, 640, 640}}};
    manifest.outputs = {{QStringLiteral("output0"), QStringLiteral("NCN"), {1, 84, -1}}};
    manifest.preprocessing = {{QStringLiteral("id"), QStringLiteral("letterbox_rgb_0_1")}};
    manifest.postprocessing = {{QStringLiteral("id"), QStringLiteral("yolo_detection_nms")}};
    manifest.decoder = QStringLiteral("yolo_detection_v8");
    manifest.classNames = QStringList{QStringLiteral("part")};
    manifest.opset = 17;
    manifest.exporterVersion = QStringLiteral("ultralytics-8.4.45");
    manifest.runtimeRoutes = QStringList{QStringLiteral("aitrain_onnxruntime")};
    manifest.verified = true;
    return manifest;
}

bool startTask(aitrain::v2::StorageV2& storage,
    const aitrain::v2::TaskSnapshot& task,
    QString* error)
{
    return storage.createTask(task, error)
        && storage.transitionTask(task.id, aitrain::v2::TaskState::Created,
            aitrain::v2::TaskState::Queued, {}, error)
        && storage.transitionTask(task.id, aitrain::v2::TaskState::Queued,
            aitrain::v2::TaskState::Starting, {}, error)
        && storage.transitionTask(task.id, aitrain::v2::TaskState::Starting,
            aitrain::v2::TaskState::Running, {}, error);
}

aitrain::v2::WorkflowRunSnapshotV2 createWorkflow(aitrain::v2::StorageV2& storage,
    const aitrain::v2::TaskId& taskId,
    aitrain::v2::WorkflowStepSnapshotV2* step,
    QString* error)
{
    aitrain::v2::WorkflowRunSnapshotV2 workflow;
    workflow.id = aitrain::v2::WorkflowRunId::create();
    workflow.taskId = taskId;
    workflow.templateId = QStringLiteral("terminalization-test");
    workflow.terminalPolicy = aitrain::v2::WorkflowTerminalPolicyV2::EvidenceRequired;
    step->id = aitrain::v2::WorkflowStepId::create();
    step->workflowRunId = workflow.id;
    step->ordinal = 0;
    step->kind = QStringLiteral("RenderDeliveryReport");
    step->backend = QStringLiteral("aitrain_core");
    if (!storage.createWorkflowRun(workflow, {*step}, error)) return {};
    return workflow;
}

QVector<aitrain::v2::ArtifactFileSnapshot> evidenceFiles()
{
    return {{QStringLiteral("evidence.json"), QString(64, QLatin1Char('e')), 42}};
}

} // namespace

class V2StorageTests : public QObject {
    Q_OBJECT

private slots:
    void createAndTransitionTaskAtomically();
    void listsTasksAndWorkflowRunsForReadModels();
    void persistsCompleteTaskFailure();
    void rejectsInvalidCasAndRecoversInterruptedTask();
    void rejectsLegacyDatabase();
    void enforcesForeignKeys();
    void hostStateEventsDoNotConsumeAdapterProtocolSequence();
    void registersModelPackageOnlyForMatchingArtifactProvenance();
    void listsRegisteredModelPackagesNewestFirst();
    void persistsWorkflowStepsWithArtifactAndRetryGuards();
    void persistsCrossTaskWorkflowInputAndEnforcesOwnership();
    void rejectsSchema7AndPersistsTerminalPolicy();
    void enforcesTerminalizationSchemaConstraints();
    void evidenceGatedSuccessSurvivesReopen();
    void persistsFailedAndCanceledTerminalFacts();
    void terminalizationWritesAreIdempotentAndRejectConflicts();
    void listsUnsealedGatedWorkflowsAndProtectsThemFromInterruption();
};

void V2StorageTests::createAndTransitionTaskAtomically()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::StorageV2 storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));

    const aitrain::v2::TaskSnapshot source = makeTask();
    QVERIFY2(storage.createTask(source, &error), qPrintable(error));
    QCOMPARE(storage.eventCount(source.id, &error), 1);
    QVERIFY2(storage.transitionTask(source.id, aitrain::v2::TaskState::Created, aitrain::v2::TaskState::Queued, {}, &error), qPrintable(error));
    QVERIFY2(storage.transitionTask(source.id, aitrain::v2::TaskState::Queued, aitrain::v2::TaskState::Starting, {}, &error), qPrintable(error));
    QVERIFY2(storage.transitionTask(source.id, aitrain::v2::TaskState::Starting, aitrain::v2::TaskState::Running, {}, &error), qPrintable(error));
    QVERIFY2(storage.transitionTask(source.id, aitrain::v2::TaskState::Running, aitrain::v2::TaskState::Succeeded, {}, &error), qPrintable(error));

    aitrain::v2::TaskSnapshot stored;
    QVERIFY2(storage.task(source.id, &stored, &error), qPrintable(error));
    QCOMPARE(stored.state, aitrain::v2::TaskState::Succeeded);
    QCOMPARE(storage.eventCount(source.id, &error), 5);
}

void V2StorageTests::listsTasksAndWorkflowRunsForReadModels()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::StorageV2 storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));

    aitrain::v2::TaskSnapshot older = makeTask();
    older.createdAt = QDateTime::fromString(QStringLiteral("2026-07-15T01:00:00.000Z"), Qt::ISODateWithMs);
    older.updatedAt = older.createdAt;
    aitrain::v2::TaskSnapshot newer = makeTask();
    newer.createdAt = older.createdAt.addSecs(1);
    newer.updatedAt = newer.createdAt;
    QVERIFY2(storage.createTask(older, &error), qPrintable(error));
    QVERIFY2(storage.createTask(newer, &error), qPrintable(error));

    aitrain::v2::WorkflowStepSnapshotV2 step;
    const aitrain::v2::WorkflowRunSnapshotV2 workflow = createWorkflow(storage, older.id, &step, &error);
    QVERIFY2(workflow.id.isValid(), qPrintable(error));

    const QVector<aitrain::v2::TaskSnapshot> tasks = storage.tasks(10, &error);
    QCOMPARE(tasks.size(), 2);
    QCOMPARE(tasks.at(0).id, newer.id);
    QCOMPARE(tasks.at(1).id, older.id);
    const QVector<aitrain::v2::WorkflowRunSnapshotV2> workflows = storage.workflowRunsForTask(older.id, &error);
    QCOMPARE(workflows.size(), 1);
    QCOMPARE(workflows.first().id, workflow.id);
    QCOMPARE(workflows.first().taskId, older.id);
    QVERIFY(storage.workflowRunsForTask(newer.id, &error).isEmpty());
    QVERIFY(error.isEmpty());
}

void V2StorageTests::persistsCompleteTaskFailure()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::StorageV2 storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));

    const aitrain::v2::TaskSnapshot source = makeTask();
    QVERIFY2(storage.createTask(source, &error), qPrintable(error));
    QVERIFY2(storage.transitionTask(source.id, aitrain::v2::TaskState::Created,
        aitrain::v2::TaskState::Queued, {}, &error), qPrintable(error));
    const QDateTime occurredAt = QDateTime::currentDateTimeUtc().addSecs(-2);
    const aitrain::v2::Failure failure{aitrain::v2::FailureCode::DependencyMissing,
        QStringLiteral("缺少训练依赖"), QStringLiteral("安装依赖后重试"), occurredAt};
    QVERIFY2(storage.transitionTask(source.id, aitrain::v2::TaskState::Queued,
        aitrain::v2::TaskState::Failed, failure, &error), qPrintable(error));

    aitrain::v2::TaskSnapshot stored;
    QVERIFY2(storage.task(source.id, &stored, &error), qPrintable(error));
    QCOMPARE(stored.failure.code, failure.code);
    QCOMPARE(stored.failure.message, failure.message);
    QCOMPARE(stored.failure.suggestedAction, failure.suggestedAction);
    QCOMPARE(stored.failure.occurredAt, failure.occurredAt);
}

void V2StorageTests::rejectsInvalidCasAndRecoversInterruptedTask()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::StorageV2 storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    const aitrain::v2::TaskSnapshot source = makeTask();
    QVERIFY2(storage.createTask(source, &error), qPrintable(error));
    QVERIFY(!storage.transitionTask(source.id, aitrain::v2::TaskState::Running, aitrain::v2::TaskState::Succeeded, {}, &error));
    QVERIFY(error.contains(QStringLiteral("并发更新")));

    QVERIFY2(storage.transitionTask(source.id, aitrain::v2::TaskState::Created, aitrain::v2::TaskState::Queued, {}, &error), qPrintable(error));
    QVERIFY2(storage.markInterruptedTasksFailed(&error), qPrintable(error));
    aitrain::v2::TaskSnapshot stored;
    QVERIFY2(storage.task(source.id, &stored, &error), qPrintable(error));
    QCOMPARE(stored.state, aitrain::v2::TaskState::Failed);
    QCOMPARE(stored.failure.code, aitrain::v2::FailureCode::ProcessCrashed);
}

void V2StorageTests::rejectsLegacyDatabase()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString path = directory.filePath(QStringLiteral("legacy.sqlite"));
    const QString connectionName = QStringLiteral("legacy_setup");
    QSqlDatabase database = QSqlDatabase::addDatabase(QStringLiteral("QSQLITE"), connectionName);
    database.setDatabaseName(path);
    QVERIFY(database.open());
    QSqlQuery query(database);
    QVERIFY(query.exec(QStringLiteral("create table tasks(id text primary key)")));
    database.close();
    database = QSqlDatabase();
    QSqlDatabase::removeDatabase(connectionName);

    aitrain::v2::StorageV2 storage;
    QString error;
    QVERIFY(!storage.open(path, &error));
    QVERIFY(error.contains(QStringLiteral("V1")));
}

void V2StorageTests::enforcesForeignKeys()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString path = directory.filePath(QStringLiteral("project.sqlite"));
    aitrain::v2::StorageV2 storage;
    QString error;
    QVERIFY2(storage.open(path, &error), qPrintable(error));

    const QString connectionName = QStringLiteral("foreign_key_probe");
    QSqlDatabase database = QSqlDatabase::addDatabase(QStringLiteral("QSQLITE"), connectionName);
    database.setDatabaseName(path);
    QVERIFY(database.open());
    QSqlQuery query(database);
    QVERIFY(query.exec(QStringLiteral("pragma foreign_keys = on")));
    QVERIFY(!query.exec(QStringLiteral("insert into task_metrics(id, task_id, name, value, occurred_at) values('metric', 'missing', 'loss', 1.0, '2026-01-01T00:00:00.000Z')")));
    database.close();
    database = QSqlDatabase();
    QSqlDatabase::removeDatabase(connectionName);
}

void V2StorageTests::hostStateEventsDoNotConsumeAdapterProtocolSequence()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::StorageV2 storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    const aitrain::v2::TaskSnapshot task = makeTask();
    QVERIFY2(storage.createTask(task, &error), qPrintable(error));
    quint64 lastSequence = 99;
    QVERIFY2(storage.lastProtocolSequence(task.id, &lastSequence, &error), qPrintable(error));
    QCOMPARE(lastSequence, quint64(0));

    aitrain::v2::ProtocolEnvelope event;
    event.messageId = aitrain::v2::MessageId::create();
    event.requestId = task.requestId;
    event.taskId = task.id;
    event.sequence = 1;
    event.kind = QStringLiteral("event.progress");
    event.timestamp = QDateTime::currentDateTimeUtc();
    event.payload = QJsonObject{{QStringLiteral("percent"), 1}};
    QVERIFY2(storage.recordProtocolEvent(event.taskId, event.requestId, event.messageId,
        event.sequence, event.kind, event.payload, event.timestamp, &error), qPrintable(error));
    QVERIFY2(storage.lastProtocolSequence(task.id, &lastSequence, &error), qPrintable(error));
    QCOMPARE(lastSequence, quint64(1));
    QCOMPARE(storage.eventCount(task.id, &error), 2);
}

void V2StorageTests::registersModelPackageOnlyForMatchingArtifactProvenance()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::StorageV2 storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    const aitrain::v2::TaskSnapshot task = makeTask();
    QVERIFY2(storage.createTask(task, &error), qPrintable(error));
    const aitrain::v2::ArtifactId artifactId = aitrain::v2::ArtifactId::create();
    const QVector<aitrain::v2::ArtifactFileSnapshot> files = {
        {QStringLiteral("model.onnx"), QString(64, QLatin1Char('a')), 42}};
    QVERIFY2(storage.recordArtifactWithFiles(artifactId, task.id, QStringLiteral("export_bundle"), files, QDateTime::currentDateTimeUtc(), &error), qPrintable(error));

    aitrain::v2::ModelPackageSnapshotV2 source;
    source.manifest = makeManifest(task.id);
    source.sourceArtifactId = artifactId;
    QVERIFY2(storage.registerModelPackage(source, &error), qPrintable(error));
    aitrain::v2::ModelPackageSnapshotV2 loaded;
    QVERIFY2(storage.modelPackage(source.manifest.modelPackageId, &loaded, &error), qPrintable(error));
    QCOMPARE(loaded.sourceArtifactId, artifactId);
    QCOMPARE(loaded.manifest.sourceArtifactSha256, source.manifest.sourceArtifactSha256);

    aitrain::v2::ModelPackageSnapshotV2 invalid;
    invalid.manifest = makeManifest(task.id);
    invalid.manifest.sourceArtifactSha256 = QString(64, QLatin1Char('b'));
    invalid.sourceArtifactId = artifactId;
    QVERIFY(!storage.registerModelPackage(invalid, &error));
    QVERIFY(error.contains(QStringLiteral("哈希")));
}

void V2StorageTests::listsRegisteredModelPackagesNewestFirst()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::StorageV2 storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    const aitrain::v2::TaskSnapshot task = makeTask();
    QVERIFY2(storage.createTask(task, &error), qPrintable(error));
    const aitrain::v2::ArtifactId artifactId = aitrain::v2::ArtifactId::create();
    QVERIFY2(storage.recordArtifactWithFiles(artifactId, task.id, QStringLiteral("export_bundle"),
        {{QStringLiteral("model.onnx"), QString(64, QLatin1Char('a')), 42}}, QDateTime::currentDateTimeUtc(), &error), qPrintable(error));

    aitrain::v2::ModelPackageSnapshotV2 older;
    older.manifest = makeManifest(task.id);
    older.sourceArtifactId = artifactId;
    older.createdAt = QDateTime::currentDateTimeUtc().addSecs(-1);
    QVERIFY2(storage.registerModelPackage(older, &error), qPrintable(error));
    aitrain::v2::ModelPackageSnapshotV2 newer;
    newer.manifest = makeManifest(task.id);
    newer.sourceArtifactId = artifactId;
    newer.createdAt = QDateTime::currentDateTimeUtc();
    QVERIFY2(storage.registerModelPackage(newer, &error), qPrintable(error));

    const QVector<aitrain::v2::ModelPackageSnapshotV2> listed = storage.modelPackages(1, &error);
    QCOMPARE(listed.size(), 1);
    QCOMPARE(listed.first().manifest.modelPackageId, newer.manifest.modelPackageId);
    QVERIFY(storage.modelPackages(0, &error).isEmpty());
    QVERIFY(error.contains(QStringLiteral("limit")));
}

void V2StorageTests::persistsWorkflowStepsWithArtifactAndRetryGuards()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::StorageV2 storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    const aitrain::v2::TaskSnapshot task = makeTask();
    QVERIFY2(storage.createTask(task, &error), qPrintable(error));

    const aitrain::v2::ArtifactId inputArtifact = aitrain::v2::ArtifactId::create();
    const aitrain::v2::ArtifactId outputArtifact = aitrain::v2::ArtifactId::create();
    const QVector<aitrain::v2::ArtifactFileSnapshot> files = {
        {QStringLiteral("payload.json"), QString(64, QLatin1Char('a')), 12}};
    QVERIFY2(storage.recordArtifactWithFiles(inputArtifact, task.id, QStringLiteral("source"), files, QDateTime::currentDateTimeUtc(), &error), qPrintable(error));
    QVERIFY2(storage.recordArtifactWithFiles(outputArtifact, task.id, QStringLiteral("result"), files, QDateTime::currentDateTimeUtc(), &error), qPrintable(error));

    aitrain::v2::WorkflowRunSnapshotV2 workflow;
    workflow.id = aitrain::v2::WorkflowRunId::create();
    workflow.taskId = task.id;
    workflow.templateId = QStringLiteral("inference-deployment");
    aitrain::v2::WorkflowStepSnapshotV2 infer;
    infer.id = aitrain::v2::WorkflowStepId::create();
    infer.workflowRunId = workflow.id;
    infer.ordinal = 0;
    infer.kind = QStringLiteral("RunInferenceSmoke");
    infer.inputArtifactId = inputArtifact;
    infer.backend = QStringLiteral("aitrain_onnxruntime");
    infer.parameterSummary = QJsonObject{{QStringLiteral("imageCount"), 1}};
    aitrain::v2::WorkflowStepSnapshotV2 deploy;
    deploy.id = aitrain::v2::WorkflowStepId::create();
    deploy.workflowRunId = workflow.id;
    deploy.ordinal = 1;
    deploy.kind = QStringLiteral("DeploymentValidate");
    deploy.inputArtifactId = outputArtifact;
    deploy.backend = QStringLiteral("aitrain_onnxruntime");
    QVERIFY2(storage.createWorkflowRun(workflow, {infer, deploy}, &error), qPrintable(error));

    aitrain::v2::WorkflowRunSnapshotV2 loadedRun;
    QVERIFY2(storage.workflowRun(workflow.id, &loadedRun, &error), qPrintable(error));
    QCOMPARE(loadedRun.templateId, workflow.templateId);
    QCOMPARE(storage.workflowSteps(workflow.id, &error).size(), 2);

    QVERIFY2(storage.transitionWorkflowStep(infer.id, aitrain::v2::WorkflowStepState::Pending,
        aitrain::v2::WorkflowStepState::Running, {}, {}, &error), qPrintable(error));
    QVERIFY(!storage.transitionWorkflowStep(infer.id, aitrain::v2::WorkflowStepState::Running,
        aitrain::v2::WorkflowStepState::Succeeded, {}, {}, &error));
    QVERIFY(error.contains(QStringLiteral("Artifact")));
    QVERIFY2(storage.transitionWorkflowStep(infer.id, aitrain::v2::WorkflowStepState::Running,
        aitrain::v2::WorkflowStepState::Succeeded, outputArtifact, {}, &error), qPrintable(error));

    QVERIFY2(storage.transitionWorkflowStep(deploy.id, aitrain::v2::WorkflowStepState::Pending,
        aitrain::v2::WorkflowStepState::Running, {}, {}, &error), qPrintable(error));
    const aitrain::v2::Failure failure{aitrain::v2::FailureCode::DependencyMissing,
        QStringLiteral("缺少部署依赖"), QStringLiteral("安装部署依赖后重试"), QDateTime::currentDateTimeUtc()};
    QVERIFY2(storage.transitionWorkflowStep(deploy.id, aitrain::v2::WorkflowStepState::Running,
        aitrain::v2::WorkflowStepState::Failed, {}, failure, &error), qPrintable(error));
    const QVector<aitrain::v2::WorkflowStepSnapshotV2> failedSteps = storage.workflowSteps(workflow.id, &error);
    QCOMPARE(failedSteps.at(1).failure.code, failure.code);
    QCOMPARE(failedSteps.at(1).failure.message, failure.message);
    QCOMPARE(failedSteps.at(1).failure.suggestedAction, failure.suggestedAction);
    QCOMPARE(failedSteps.at(1).failure.occurredAt, failure.occurredAt);
    QVERIFY2(storage.retryWorkflowStep(deploy.id, &error), qPrintable(error));

    const QVector<aitrain::v2::WorkflowStepSnapshotV2> steps = storage.workflowSteps(workflow.id, &error);
    QCOMPARE(steps.at(0).state, aitrain::v2::WorkflowStepState::Succeeded);
    QCOMPARE(steps.at(0).outputArtifactId, outputArtifact);
    QVERIFY(steps.at(0).startedAt.isValid());
    QVERIFY(steps.at(0).finishedAt.isValid());
    QCOMPARE(steps.at(1).state, aitrain::v2::WorkflowStepState::Pending);
    QCOMPARE(steps.at(1).retryCount, 1);
    QVERIFY(!steps.at(1).failure.isFailure());
}

void V2StorageTests::persistsCrossTaskWorkflowInputAndEnforcesOwnership()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::StorageV2 storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    storage.setArtifactStoreRoot(directory.filePath(QStringLiteral("artifact-store")));

    const auto producer = makeTask();
    const auto consumer = makeTask();
    QVERIFY2(startTask(storage, producer, &error), qPrintable(error));
    QVERIFY2(startTask(storage, consumer, &error), qPrintable(error));

    const auto snapshotArtifactId = aitrain::v2::ArtifactId::create();
    const QString manifestSha256(64, QLatin1Char('a'));
    const QString rootHash(64, QLatin1Char('b'));
    QVERIFY2(storage.recordArtifactWithFiles(snapshotArtifactId, producer.id,
        QStringLiteral("dataset_snapshot_v2"),
        {{QStringLiteral("dataset_snapshot.json"), manifestSha256, 64}},
        QDateTime::currentDateTimeUtc(), &error), qPrintable(error));
    aitrain::v2::DatasetSnapshotRecordV2 snapshot;
    snapshot.id = aitrain::v2::SnapshotId::create();
    snapshot.taskId = producer.id;
    snapshot.artifactId = snapshotArtifactId;
    snapshot.rootPath = QDir(directory.path()).filePath(
        QStringLiteral("foreign/artifacts/%1").arg(snapshotArtifactId.toString()));
    snapshot.datasetFormat = QStringLiteral("yolo_detection");
    snapshot.driverId = QStringLiteral("yolo_detection");
    snapshot.driverVersion = QStringLiteral("2.0");
    snapshot.rootHash = rootHash;
    snapshot.manifestSha256 = manifestSha256;
    snapshot.fileCount = 1;
    snapshot.totalBytes = 64;
    QVERIFY(!storage.registerDatasetSnapshot(&snapshot, &error));
    QVERIFY(error.contains(QStringLiteral("committed Snapshot Artifact")));
    snapshot.rootPath = QDir(directory.path()).filePath(
        QStringLiteral("artifact-store/artifacts/%1").arg(snapshotArtifactId.toString()));
    QVERIFY2(storage.registerDatasetSnapshot(&snapshot, &error), qPrintable(error));

    aitrain::v2::WorkflowRunSnapshotV2 workflow;
    workflow.id = aitrain::v2::WorkflowRunId::create();
    workflow.taskId = consumer.id;
    workflow.templateId = QStringLiteral("cross-task-snapshot-test");
    aitrain::v2::WorkflowStepSnapshotV2 snapshotStep;
    snapshotStep.id = aitrain::v2::WorkflowStepId::create();
    snapshotStep.workflowRunId = workflow.id;
    snapshotStep.ordinal = 0;
    snapshotStep.kind = QStringLiteral("CreateSnapshot");
    snapshotStep.backend = QStringLiteral("aitrain_core");
    snapshotStep.inputArtifactId = snapshotArtifactId;
    aitrain::v2::WorkflowStepSnapshotV2 trainStep;
    trainStep.id = aitrain::v2::WorkflowStepId::create();
    trainStep.workflowRunId = workflow.id;
    trainStep.ordinal = 1;
    trainStep.kind = QStringLiteral("Train");
    trainStep.backend = QStringLiteral("official_backend");
    aitrain::v2::WorkflowInputBindingV2 binding;
    binding.workflowRunId = workflow.id;
    binding.role = QStringLiteral("dataset_snapshot");
    binding.sourceArtifactId = snapshot.artifactId;
    binding.sourceTaskId = snapshot.taskId;
    binding.sourceArtifactKind = QStringLiteral("dataset_snapshot_v2");
    binding.datasetId = snapshot.datasetId;
    binding.datasetSnapshotId = snapshot.id;
    binding.datasetVersionId = snapshot.datasetVersionId;
    binding.manifestSha256 = snapshot.manifestSha256;
    binding.rootHash = snapshot.rootHash;
    QVERIFY2(storage.createWorkflowRunWithInput(workflow, {snapshotStep, trainStep}, binding, &error),
        qPrintable(error));

    aitrain::v2::WorkflowInputBindingV2 storedBinding;
    QVERIFY2(storage.workflowInput(workflow.id, QStringLiteral("dataset_snapshot"),
        &storedBinding, &error), qPrintable(error));
    QCOMPARE(storedBinding.sourceTaskId, producer.id);
    QCOMPARE(storedBinding.sourceArtifactId, snapshot.artifactId);
    QCOMPARE(storedBinding.datasetId, snapshot.datasetId);
    QCOMPARE(storedBinding.datasetSnapshotId, snapshot.id);
    QCOMPARE(storedBinding.datasetVersionId, snapshot.datasetVersionId);

    QVERIFY2(storage.transitionWorkflowStep(snapshotStep.id,
        aitrain::v2::WorkflowStepState::Pending, aitrain::v2::WorkflowStepState::Running,
        {}, {}, &error), qPrintable(error));
    QVERIFY2(storage.transitionWorkflowStep(snapshotStep.id,
        aitrain::v2::WorkflowStepState::Running, aitrain::v2::WorkflowStepState::Succeeded,
        snapshot.artifactId, {}, &error), qPrintable(error));
    QVERIFY2(storage.bindWorkflowStepInput(trainStep.id, snapshot.artifactId, &error), qPrintable(error));
    QVERIFY2(storage.transitionWorkflowStep(trainStep.id,
        aitrain::v2::WorkflowStepState::Pending, aitrain::v2::WorkflowStepState::Running,
        {}, {}, &error), qPrintable(error));
    QVERIFY(!storage.transitionWorkflowStep(trainStep.id,
        aitrain::v2::WorkflowStepState::Running, aitrain::v2::WorkflowStepState::Succeeded,
        snapshot.artifactId, {}, &error));
    QVERIFY(error.contains(QStringLiteral("根任务")));

    const auto consumerOutput = aitrain::v2::ArtifactId::create();
    QVERIFY2(storage.recordArtifactWithFiles(consumerOutput, consumer.id, QStringLiteral("training_output"),
        {{QStringLiteral("result.json"), QString(64, QLatin1Char('c')), 32}},
        QDateTime::currentDateTimeUtc(), &error), qPrintable(error));
    error.clear();
    QVERIFY2(storage.transitionWorkflowStep(trainStep.id,
        aitrain::v2::WorkflowStepState::Running, aitrain::v2::WorkflowStepState::Succeeded,
        consumerOutput, {}, &error), qPrintable(error));

    aitrain::v2::WorkflowRunSnapshotV2 rejectedWorkflow = workflow;
    rejectedWorkflow.id = aitrain::v2::WorkflowRunId::create();
    rejectedWorkflow.templateId = QStringLiteral("mismatched-lineage-test");
    aitrain::v2::WorkflowStepSnapshotV2 rejectedStep = snapshotStep;
    rejectedStep.id = aitrain::v2::WorkflowStepId::create();
    rejectedStep.workflowRunId = rejectedWorkflow.id;
    rejectedStep.state = aitrain::v2::WorkflowStepState::Pending;
    rejectedStep.outputArtifactId = {};
    aitrain::v2::WorkflowInputBindingV2 rejectedBinding = binding;
    rejectedBinding.workflowRunId = rejectedWorkflow.id;
    rejectedBinding.datasetId = aitrain::v2::DatasetId::create();
    error.clear();
    QVERIFY(!storage.createWorkflowRunWithInput(rejectedWorkflow, {rejectedStep}, rejectedBinding, &error));
    QVERIFY(error.contains(QStringLiteral("lineage")));
    aitrain::v2::WorkflowRunSnapshotV2 notCreated;
    error.clear();
    QVERIFY(!storage.workflowRun(rejectedWorkflow.id, &notCreated, &error));
}

void V2StorageTests::rejectsSchema7AndPersistsTerminalPolicy()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString oldPath = directory.filePath(QStringLiteral("schema7.sqlite"));
    const QString connectionName = QStringLiteral("schema7_setup");
    QSqlDatabase database = QSqlDatabase::addDatabase(QStringLiteral("QSQLITE"), connectionName);
    database.setDatabaseName(oldPath);
    QVERIFY(database.open());
    QSqlQuery query(database);
    QVERIFY(query.exec(QStringLiteral("create table v2_schema(version integer not null check(version = 7))")));
    QVERIFY(query.exec(QStringLiteral("insert into v2_schema values(7)")));
    database.close();
    database = QSqlDatabase();
    QSqlDatabase::removeDatabase(connectionName);
    aitrain::v2::StorageV2 rejected;
    QString error;
    QVERIFY(!rejected.open(oldPath, &error));
    QVERIFY(error.contains(QStringLiteral("schema")));

    aitrain::v2::StorageV2 storage;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("schema8.sqlite")), &error), qPrintable(error));
    const aitrain::v2::TaskSnapshot task = makeTask();
    QVERIFY2(startTask(storage, task, &error), qPrintable(error));
    aitrain::v2::WorkflowStepSnapshotV2 step;
    const auto workflow = createWorkflow(storage, task.id, &step, &error);
    QVERIFY2(workflow.id.isValid(), qPrintable(error));
    aitrain::v2::WorkflowRunSnapshotV2 loaded;
    QVERIFY2(storage.workflowRun(workflow.id, &loaded, &error), qPrintable(error));
    QCOMPARE(loaded.terminalPolicy, aitrain::v2::WorkflowTerminalPolicyV2::EvidenceRequired);
}

void V2StorageTests::enforcesTerminalizationSchemaConstraints()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString databasePath = directory.filePath(QStringLiteral("project.sqlite"));
    aitrain::v2::StorageV2 storage;
    QString error;
    QVERIFY2(storage.open(databasePath, &error), qPrintable(error));
    const auto task = makeTask();
    QVERIFY2(startTask(storage, task, &error), qPrintable(error));
    aitrain::v2::WorkflowStepSnapshotV2 step;
    const auto workflow = createWorkflow(storage, task.id, &step, &error);
    QVERIFY2(workflow.id.isValid(), qPrintable(error));

    const QString connectionName = QStringLiteral("terminalization-constraints-%1").arg(task.id.toString());
    {
        QSqlDatabase raw = QSqlDatabase::addDatabase(QStringLiteral("QSQLITE"), connectionName);
        raw.setDatabaseName(databasePath);
        QVERIFY2(raw.open(), qPrintable(raw.lastError().text()));
        QSqlQuery pragma(raw);
        QVERIFY(pragma.exec(QStringLiteral("pragma foreign_keys = on")));
        QSqlQuery invalid(raw);
        invalid.prepare(QStringLiteral("insert into workflow_terminalizations(workflow_run_id, task_id, state, terminal_state, failure_code, failure_details, failure_suggested_action, failure_occurred_at, terminal_at, sealed_at) values(:workflow, :task, 'sealed', 'succeeded', 'invalid_dataset', 'invalid success failure', 'fix input', :now, :now, :now)"));
        const QString now = QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs);
        invalid.bindValue(QStringLiteral(":workflow"), workflow.id.toString());
        invalid.bindValue(QStringLiteral(":task"), task.id.toString());
        invalid.bindValue(QStringLiteral(":now"), now);
        QVERIFY(!invalid.exec());
        QVERIFY(invalid.lastError().text().contains(QStringLiteral("CHECK"), Qt::CaseInsensitive));
        raw.close();
    }
    QSqlDatabase::removeDatabase(connectionName);
}

void V2StorageTests::evidenceGatedSuccessSurvivesReopen()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString path = directory.filePath(QStringLiteral("project.sqlite"));
    const aitrain::v2::TaskSnapshot task = makeTask();
    aitrain::v2::WorkflowRunId workflowId;
    aitrain::v2::ArtifactId evidenceId;
    const QDateTime terminalAt = QDateTime::currentDateTimeUtc().addSecs(-3);
    QString error;
    {
        aitrain::v2::StorageV2 storage;
        QVERIFY2(storage.open(path, &error), qPrintable(error));
        QVERIFY2(startTask(storage, task, &error), qPrintable(error));
        aitrain::v2::WorkflowStepSnapshotV2 step;
        const auto workflow = createWorkflow(storage, task.id, &step, &error);
        workflowId = workflow.id;
        const aitrain::v2::ArtifactId output = aitrain::v2::ArtifactId::create();
        QVERIFY2(storage.recordArtifactWithFiles(output, task.id, QStringLiteral("delivery_report"),
            {{QStringLiteral("report.json"), QString(64, QLatin1Char('a')), 12}},
            QDateTime::currentDateTimeUtc(), &error), qPrintable(error));
        QVERIFY2(storage.transitionWorkflowStep(step.id, aitrain::v2::WorkflowStepState::Pending,
            aitrain::v2::WorkflowStepState::Running, {}, {}, &error), qPrintable(error));
        QVERIFY2(storage.transitionWorkflowStep(step.id, aitrain::v2::WorkflowStepState::Running,
            aitrain::v2::WorkflowStepState::Succeeded, output, {}, &error), qPrintable(error));
        QVERIFY2(storage.sealWorkflowTerminalization(workflow.id,
            aitrain::v2::TaskState::Succeeded, {}, terminalAt, &error), qPrintable(error));
        QCOMPARE(storage.pendingWorkflowTerminalizations(10, &error).size(), 1);
        QVERIFY2(storage.markInterruptedTasksFailed(&error), qPrintable(error));
        aitrain::v2::TaskSnapshot stillRunning;
        QVERIFY2(storage.task(task.id, &stillRunning, &error), qPrintable(error));
        QCOMPARE(stillRunning.state, aitrain::v2::TaskState::Running);
        evidenceId = aitrain::v2::ArtifactId::create();
        QVERIFY2(storage.recordEvidenceArtifactWithFilesAndAttachTerminalization(evidenceId,
            task.id, workflow.id, evidenceFiles(), terminalAt.addMSecs(1), &error), qPrintable(error));
    }
    {
        aitrain::v2::StorageV2 storage;
        QVERIFY2(storage.open(path, &error), qPrintable(error));
        const auto pending = storage.pendingWorkflowTerminalizations(10, &error);
        QCOMPARE(pending.size(), 1);
        QCOMPARE(pending.first().state, aitrain::v2::WorkflowTerminalizationStateV2::EvidenceAttached);
        QCOMPARE(pending.first().evidenceArtifactId, evidenceId);
        QVERIFY2(storage.closeWorkflowTerminalization(workflowId,
            aitrain::v2::TaskState::Running, &error), qPrintable(error));
        aitrain::v2::TaskSnapshot terminal;
        QVERIFY2(storage.task(task.id, &terminal, &error), qPrintable(error));
        QCOMPARE(terminal.state, aitrain::v2::TaskState::Succeeded);
        QCOMPARE(terminal.updatedAt, terminalAt);
        QVERIFY(storage.pendingWorkflowTerminalizations(10, &error).isEmpty());
        QVERIFY(storage.pendingEvidenceRequiredWorkflows(10, &error).isEmpty());
    }
}

void V2StorageTests::persistsFailedAndCanceledTerminalFacts()
{
    const QVector<aitrain::v2::TaskState> terminalStates = {
        aitrain::v2::TaskState::Failed, aitrain::v2::TaskState::Canceled};
    for (const auto terminalState : terminalStates) {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        aitrain::v2::StorageV2 storage;
        QString error;
        QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
        const auto task = makeTask();
        QVERIFY2(startTask(storage, task, &error), qPrintable(error));
        aitrain::v2::WorkflowStepSnapshotV2 step;
        const auto workflow = createWorkflow(storage, task.id, &step, &error);
        const QDateTime occurredAt = QDateTime::currentDateTimeUtc().addSecs(-2);
        const aitrain::v2::Failure failure{
            terminalState == aitrain::v2::TaskState::Canceled
                ? aitrain::v2::FailureCode::Canceled : aitrain::v2::FailureCode::DependencyMissing,
            terminalState == aitrain::v2::TaskState::Canceled
                ? QStringLiteral("用户取消") : QStringLiteral("依赖缺失"),
            terminalState == aitrain::v2::TaskState::Canceled
                ? QStringLiteral("重新启动任务") : QStringLiteral("安装依赖后重试"), occurredAt};
        QVERIFY2(storage.transitionWorkflowStep(step.id, aitrain::v2::WorkflowStepState::Pending,
            aitrain::v2::WorkflowStepState::Running, {}, {}, &error), qPrintable(error));
        QVERIFY2(storage.transitionWorkflowStep(step.id, aitrain::v2::WorkflowStepState::Running,
            terminalState == aitrain::v2::TaskState::Canceled
                ? aitrain::v2::WorkflowStepState::Canceled : aitrain::v2::WorkflowStepState::Failed,
            {}, failure, &error), qPrintable(error));
        if (terminalState == aitrain::v2::TaskState::Canceled) {
            QVERIFY2(storage.transitionTask(task.id, aitrain::v2::TaskState::Running,
                aitrain::v2::TaskState::CancelRequested, {}, &error), qPrintable(error));
        }
        QVERIFY2(storage.sealWorkflowTerminalization(workflow.id, terminalState,
            failure, occurredAt.addMSecs(10), &error), qPrintable(error));
        const auto evidence = aitrain::v2::ArtifactId::create();
        QVERIFY2(storage.recordEvidenceArtifactWithFilesAndAttachTerminalization(evidence,
            task.id, workflow.id, evidenceFiles(), QDateTime::currentDateTimeUtc(), &error), qPrintable(error));
        QVERIFY2(storage.closeWorkflowTerminalization(workflow.id,
            terminalState == aitrain::v2::TaskState::Canceled
                ? aitrain::v2::TaskState::CancelRequested : aitrain::v2::TaskState::Running,
            &error), qPrintable(error));
        aitrain::v2::TaskSnapshot loaded;
        QVERIFY2(storage.task(task.id, &loaded, &error), qPrintable(error));
        QCOMPARE(loaded.state, terminalState);
        QCOMPARE(loaded.failure.code, failure.code);
        QCOMPARE(loaded.failure.message, failure.message);
        QCOMPARE(loaded.failure.suggestedAction, failure.suggestedAction);
        QCOMPARE(loaded.failure.occurredAt, failure.occurredAt);
    }
}

void V2StorageTests::terminalizationWritesAreIdempotentAndRejectConflicts()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::StorageV2 storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    const auto task = makeTask();
    QVERIFY2(startTask(storage, task, &error), qPrintable(error));
    aitrain::v2::WorkflowStepSnapshotV2 step;
    const auto workflow = createWorkflow(storage, task.id, &step, &error);
    const QDateTime occurredAt = QDateTime::currentDateTimeUtc().addSecs(-4);
    const aitrain::v2::Failure terminalFailure{aitrain::v2::FailureCode::InvalidDataset,
        QStringLiteral("数据集无效"), QStringLiteral("修复数据集后重试"), occurredAt};
    QVERIFY2(storage.transitionWorkflowStep(step.id, aitrain::v2::WorkflowStepState::Pending,
        aitrain::v2::WorkflowStepState::Running, {}, {}, &error), qPrintable(error));
    QVERIFY2(storage.transitionWorkflowStep(step.id, aitrain::v2::WorkflowStepState::Running,
        aitrain::v2::WorkflowStepState::Failed, {}, terminalFailure, &error), qPrintable(error));
    const QDateTime terminalAt = occurredAt.addMSecs(20);
    QVERIFY2(storage.sealWorkflowTerminalization(workflow.id,
        aitrain::v2::TaskState::Failed, terminalFailure, terminalAt, &error), qPrintable(error));
    QVERIFY2(storage.sealWorkflowTerminalization(workflow.id,
        aitrain::v2::TaskState::Failed, terminalFailure, terminalAt, &error), qPrintable(error));
    QVERIFY(!storage.sealWorkflowTerminalization(workflow.id,
        aitrain::v2::TaskState::Failed, terminalFailure, terminalAt.addMSecs(1), &error));
    error.clear();
    QVERIFY(!storage.transitionTask(task.id, aitrain::v2::TaskState::Running,
        aitrain::v2::TaskState::Failed, terminalFailure, &error));
    QVERIFY(error.contains(QStringLiteral("evidence_required")));

    const aitrain::v2::Failure evidenceFailure{aitrain::v2::FailureCode::ArtifactIncomplete,
        QStringLiteral("Evidence 写入失败"), QStringLiteral("检查磁盘后重试"), occurredAt.addMSecs(30)};
    QVERIFY2(storage.recordWorkflowTerminalizationEvidenceFailure(workflow.id,
        evidenceFailure, &error), qPrintable(error));
    QVERIFY2(storage.recordWorkflowTerminalizationEvidenceFailure(workflow.id,
        evidenceFailure, &error), qPrintable(error));
    aitrain::v2::WorkflowTerminalizationSnapshotV2 snapshot;
    QVERIFY2(storage.workflowTerminalization(workflow.id, &snapshot, &error), qPrintable(error));
    QCOMPARE(snapshot.evidenceAttemptCount, 1);
    QCOMPARE(snapshot.lastEvidenceFailure.message, evidenceFailure.message);

    const auto evidence = aitrain::v2::ArtifactId::create();
    const QDateTime evidenceAt = QDateTime::currentDateTimeUtc();
    QVERIFY2(storage.recordEvidenceArtifactWithFilesAndAttachTerminalization(evidence,
        task.id, workflow.id, evidenceFiles(), evidenceAt, &error), qPrintable(error));
    QVERIFY2(storage.recordEvidenceArtifactWithFilesAndAttachTerminalization(evidence,
        task.id, workflow.id, evidenceFiles(), evidenceAt, &error), qPrintable(error));
    bool exists = false;
    QVERIFY2(storage.artifactExists(evidence, &exists, &error), qPrintable(error));
    QVERIFY(exists);
    QVector<aitrain::v2::ArtifactFileSnapshot> conflictingFiles = evidenceFiles();
    conflictingFiles[0].sha256 = QString(64, QLatin1Char('f'));
    QVERIFY(!storage.recordEvidenceArtifactWithFilesAndAttachTerminalization(evidence,
        task.id, workflow.id, conflictingFiles, evidenceAt, &error));
    const auto conflictingArtifact = aitrain::v2::ArtifactId::create();
    QVERIFY(!storage.recordEvidenceArtifactWithFilesAndAttachTerminalization(conflictingArtifact,
        task.id, workflow.id, evidenceFiles(), evidenceAt, &error));
    QVERIFY2(storage.artifactExists(conflictingArtifact, &exists, &error), qPrintable(error));
    QVERIFY(!exists);
    QVERIFY2(storage.closeWorkflowTerminalization(workflow.id,
        aitrain::v2::TaskState::Running, &error), qPrintable(error));
    QVERIFY2(storage.closeWorkflowTerminalization(workflow.id,
        aitrain::v2::TaskState::Running, &error), qPrintable(error));
}

void V2StorageTests::listsUnsealedGatedWorkflowsAndProtectsThemFromInterruption()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::StorageV2 storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    const auto task = makeTask();
    QVERIFY2(startTask(storage, task, &error), qPrintable(error));
    aitrain::v2::WorkflowStepSnapshotV2 step;
    const auto workflow = createWorkflow(storage, task.id, &step, &error);
    const auto pending = storage.pendingEvidenceRequiredWorkflows(10, &error);
    QCOMPARE(pending.size(), 1);
    QCOMPARE(pending.first().id, workflow.id);
    QVERIFY(storage.pendingWorkflowTerminalizations(10, &error).isEmpty());
    QVERIFY2(storage.markInterruptedTasksFailed(&error), qPrintable(error));
    aitrain::v2::TaskSnapshot loaded;
    QVERIFY2(storage.task(task.id, &loaded, &error), qPrintable(error));
    QCOMPARE(loaded.state, aitrain::v2::TaskState::Running);
    QVERIFY(storage.pendingEvidenceRequiredWorkflows(0, &error).isEmpty());
    QVERIFY(error.contains(QStringLiteral("limit")));
}

QTEST_MAIN(V2StorageTests)
#include "tst_v2_storage.moc"
