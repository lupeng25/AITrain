#include "aitrain/storage/ProjectStore.h"
#include "aitrain/protocol/Protocol.h"

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QSqlDatabase>
#include <QSqlError>
#include <QSqlQuery>
#include <QTemporaryDir>
#include <QTest>

#include <algorithm>

namespace {

aitrain::TaskSnapshot makeTask()
{
    aitrain::TaskSnapshot task;
    task.id = aitrain::TaskId::create();
    task.requestId = aitrain::RequestId::create();
    task.capabilityId = QStringLiteral("yolo.detect");
    task.taskType = QStringLiteral("training");
    return task;
}

aitrain::ModelManifest makeManifest(const aitrain::TaskId& taskId)
{
    aitrain::ModelManifest manifest;
    manifest.modelPackageId = aitrain::ModelPackageId::create();
    manifest.modelFamily = QStringLiteral("yolo_detection");
    manifest.taskType = QStringLiteral("detection");
    manifest.sourceBackend = QStringLiteral("ultralytics_yolo_detect");
    manifest.sourceTaskId = taskId;
    manifest.sourceSnapshotId = aitrain::SnapshotId::create();
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

bool startTask(aitrain::ProjectStore& storage,
    const aitrain::TaskSnapshot& task,
    QString* error)
{
    return storage.createTask(task, error)
        && storage.transitionTask(task.id, aitrain::TaskState::Created,
            aitrain::TaskState::Queued, {}, error)
        && storage.transitionTask(task.id, aitrain::TaskState::Queued,
            aitrain::TaskState::Starting, {}, error)
        && storage.transitionTask(task.id, aitrain::TaskState::Starting,
            aitrain::TaskState::Running, {}, error);
}

aitrain::WorkflowRunSnapshot createWorkflow(aitrain::ProjectStore& storage,
    const aitrain::TaskId& taskId,
    aitrain::WorkflowStepSnapshot* step,
    QString* error)
{
    aitrain::WorkflowRunSnapshot workflow;
    workflow.id = aitrain::WorkflowRunId::create();
    workflow.taskId = taskId;
    workflow.templateId = QStringLiteral("terminalization-test");
    workflow.terminalPolicy = aitrain::WorkflowTerminalPolicy::EvidenceRequired;
    step->id = aitrain::WorkflowStepId::create();
    step->workflowRunId = workflow.id;
    step->ordinal = 0;
    step->kind = QStringLiteral("RenderDeliveryReport");
    step->backend = QStringLiteral("application");
    if (!storage.createWorkflowRun(workflow, {*step}, error)) return {};
    return workflow;
}

QVector<aitrain::ArtifactFileSnapshot> evidenceFiles()
{
    return {{QStringLiteral("evidence.json"), QString(64, QLatin1Char('e')), 42}};
}

} // namespace

class StorageTests : public QObject {
    Q_OBJECT

private slots:
    void createAndTransitionTaskAtomically();
    void listsTasksAndWorkflowRunsForReadModels();
    void paginatesTasksWithOpaqueKeysetCursor();
    void paginatesTaskHistoryWithTypedKeysetCursors();
    void persistsCompleteTaskFailure();
    void rejectsInvalidCasAndRecoversInterruptedTask();
    void recoversCancelRequestedInterruptionAsCanceled();
    void rejectsLegacyDatabase();
    void enforcesForeignKeys();
    void hostStateEventsDoNotConsumeAdapterProtocolSequence();
    void rejectsNonCanonicalArtifactMemberPaths();
    void rejectsNonHexArtifactSha256();
    void registersModelPackageOnlyForMatchingArtifactProvenance();
    void listsRegisteredModelPackagesNewestFirst();
    void persistsWorkflowStepsWithArtifactAndRetryGuards();
    void terminalizesWorkflowAndSkipsSuccessorsAtomically();
    void persistsCrossTaskWorkflowInputAndEnforcesOwnership();
    void listsDatasetCatalogThroughVersionJoin();
    void rejectsSchema7AndPersistsTerminalPolicy();
    void createsCanonicalSchemaWithoutLegacyProjectTable();
    void enforcesTerminalizationSchemaConstraints();
    void evidenceGatedSuccessSurvivesReopen();
    void persistsFailedAndCanceledTerminalFacts();
    void terminalizationWritesAreIdempotentAndRejectConflicts();
    void listsUnsealedGatedWorkflowsAndProtectsThemFromInterruption();
    void durableWorkflowTerminalOutboxIsIdempotent();
};

void StorageTests::createAndTransitionTaskAtomically()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));

    const aitrain::TaskSnapshot source = makeTask();
    QVERIFY2(storage.createTask(source, &error), qPrintable(error));
    QCOMPARE(storage.eventCount(source.id, &error), 1);
    QVERIFY2(storage.transitionTask(source.id, aitrain::TaskState::Created, aitrain::TaskState::Queued, {}, &error), qPrintable(error));
    QVERIFY2(storage.transitionTask(source.id, aitrain::TaskState::Queued, aitrain::TaskState::Starting, {}, &error), qPrintable(error));
    QVERIFY2(storage.transitionTask(source.id, aitrain::TaskState::Starting, aitrain::TaskState::Running, {}, &error), qPrintable(error));
    QVERIFY2(storage.transitionTask(source.id, aitrain::TaskState::Running, aitrain::TaskState::Succeeded, {}, &error), qPrintable(error));

    aitrain::TaskSnapshot stored;
    QVERIFY2(storage.task(source.id, &stored, &error), qPrintable(error));
    QCOMPARE(stored.state, aitrain::TaskState::Succeeded);
    QCOMPARE(storage.eventCount(source.id, &error), 5);
}

void StorageTests::listsTasksAndWorkflowRunsForReadModels()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));

    aitrain::TaskSnapshot older = makeTask();
    older.createdAt = QDateTime::fromString(QStringLiteral("2026-07-15T01:00:00.000Z"), Qt::ISODateWithMs);
    older.updatedAt = older.createdAt;
    aitrain::TaskSnapshot newer = makeTask();
    newer.createdAt = older.createdAt.addSecs(1);
    newer.updatedAt = newer.createdAt;
    QVERIFY2(storage.createTask(older, &error), qPrintable(error));
    QVERIFY2(storage.createTask(newer, &error), qPrintable(error));

    aitrain::WorkflowStepSnapshot step;
    const aitrain::WorkflowRunSnapshot workflow = createWorkflow(storage, older.id, &step, &error);
    QVERIFY2(workflow.id.isValid(), qPrintable(error));

    const QVector<aitrain::TaskSnapshot> tasks =
        storage.tasks(aitrain::PageRequest{10, {}}, &error).items;
    QCOMPARE(tasks.size(), 2);
    QCOMPARE(tasks.at(0).id, newer.id);
    QCOMPARE(tasks.at(1).id, older.id);
    const QVector<aitrain::WorkflowRunSnapshot> workflows =
        storage.workflowRunsForTask(older.id, {50, {}}, &error).items;
    QCOMPARE(workflows.size(), 1);
    QCOMPARE(workflows.first().id, workflow.id);
    QCOMPARE(workflows.first().taskId, older.id);
    QVERIFY(storage.workflowRunsForTask(newer.id, {50, {}}, &error).items.isEmpty());
    QVERIFY(error.isEmpty());
}

void StorageTests::paginatesTasksWithOpaqueKeysetCursor()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error),
        qPrintable(error));

    const QDateTime timestamp =
        QDateTime::fromString(QStringLiteral("2026-07-26T08:00:00.000Z"), Qt::ISODateWithMs);
    QVector<aitrain::TaskId> expected;
    for (int index = 0; index < 5; ++index) {
        aitrain::TaskSnapshot task = makeTask();
        task.createdAt = timestamp;
        task.updatedAt = timestamp;
        QVERIFY2(storage.createTask(task, &error), qPrintable(error));
        expected.append(task.id);
    }
    std::sort(expected.begin(), expected.end(), [](const auto& left, const auto& right) {
        return left.toString() > right.toString();
    });

    const auto first = storage.tasks({2, {}}, &error);
    QCOMPARE(first.items.size(), 2);
    QVERIFY(first.hasMore);
    QVERIFY(!first.nextCursor.isEmpty());
    const auto second = storage.tasks({2, first.nextCursor}, &error);
    QCOMPARE(second.items.size(), 2);
    QVERIFY(second.hasMore);
    const auto third = storage.tasks({2, second.nextCursor}, &error);
    QCOMPARE(third.items.size(), 1);
    QVERIFY(!third.hasMore);

    QVector<aitrain::TaskId> actual;
    for (const auto& task : first.items) actual.append(task.id);
    for (const auto& task : second.items) actual.append(task.id);
    for (const auto& task : third.items) actual.append(task.id);
    QCOMPARE(actual, expected);

    error.clear();
    QVERIFY(storage.tasks({0, {}}, &error).items.isEmpty());
    QCOMPARE(storage.lastErrorCode(), aitrain::ProjectErrorCode::InvalidPageCursor);
    error.clear();
    QVERIFY(storage.tasks({2, QStringLiteral("not-a-cursor")}, &error).items.isEmpty());
    QCOMPARE(storage.lastErrorCode(), aitrain::ProjectErrorCode::InvalidPageCursor);
}

void StorageTests::paginatesTaskHistoryWithTypedKeysetCursors()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error),
        qPrintable(error));

    const aitrain::TaskSnapshot task = makeTask();
    QVERIFY2(storage.createTask(task, &error), qPrintable(error));
    const QDateTime base =
        QDateTime::fromString(QStringLiteral("2026-07-26T09:00:00.000Z"), Qt::ISODateWithMs);

    QVector<aitrain::ArtifactId> expectedArtifacts;
    for (int index = 0; index < 5; ++index) {
        const aitrain::ArtifactId artifactId = aitrain::ArtifactId::create();
        QVERIFY2(storage.recordArtifact(artifactId, task.id, QStringLiteral("report"),
            base.addMSecs(index), &error), qPrintable(error));
        expectedArtifacts.append(artifactId);
        QVERIFY2(storage.recordMetric(task.id, QStringLiteral("metric_%1").arg(index),
            index, base.addMSecs(index), &error), qPrintable(error));

        aitrain::WorkflowRunSnapshot workflow;
        workflow.id = aitrain::WorkflowRunId::create();
        workflow.taskId = task.id;
        workflow.templateId = QStringLiteral("history_%1").arg(index);
        workflow.terminalPolicy = aitrain::WorkflowTerminalPolicy::Immediate;
        workflow.createdAt = base.addMSecs(index);
        aitrain::WorkflowStepSnapshot step;
        step.id = aitrain::WorkflowStepId::create();
        step.workflowRunId = workflow.id;
        step.ordinal = 0;
        step.kind = QStringLiteral("Inspect");
        step.backend = QStringLiteral("application");
        QVERIFY2(storage.createWorkflowRun(workflow, {step}, &error), qPrintable(error));
    }

    const auto artifactFirst = storage.artifactsForTask(task.id, {2, {}}, &error);
    const auto artifactSecond =
        storage.artifactsForTask(task.id, {2, artifactFirst.nextCursor}, &error);
    const auto artifactThird =
        storage.artifactsForTask(task.id, {2, artifactSecond.nextCursor}, &error);
    QVERIFY(artifactFirst.hasMore);
    QVERIFY(artifactSecond.hasMore);
    QVERIFY(!artifactThird.hasMore);
    QVector<aitrain::ArtifactId> actualArtifacts;
    for (const auto& item : artifactFirst.items) actualArtifacts.append(item.id);
    for (const auto& item : artifactSecond.items) actualArtifacts.append(item.id);
    for (const auto& item : artifactThird.items) actualArtifacts.append(item.id);
    QCOMPARE(actualArtifacts, expectedArtifacts);

    const auto metricFirst = storage.metricsForTask(task.id, {2, {}}, &error);
    const auto metricSecond =
        storage.metricsForTask(task.id, {2, metricFirst.nextCursor}, &error);
    const auto metricThird =
        storage.metricsForTask(task.id, {2, metricSecond.nextCursor}, &error);
    QCOMPARE(metricFirst.items.size() + metricSecond.items.size() + metricThird.items.size(), 5);
    QCOMPARE(metricFirst.items.first().name, QStringLiteral("metric_0"));
    QCOMPARE(metricThird.items.last().name, QStringLiteral("metric_4"));

    const auto workflowFirst = storage.workflowRunsForTask(task.id, {2, {}}, &error);
    const auto workflowSecond =
        storage.workflowRunsForTask(task.id, {2, workflowFirst.nextCursor}, &error);
    const auto workflowThird =
        storage.workflowRunsForTask(task.id, {2, workflowSecond.nextCursor}, &error);
    QCOMPARE(workflowFirst.items.size() + workflowSecond.items.size()
        + workflowThird.items.size(), 5);
    QCOMPARE(workflowFirst.items.first().templateId, QStringLiteral("history_0"));
    QCOMPARE(workflowThird.items.last().templateId, QStringLiteral("history_4"));

    error.clear();
    QVERIFY(storage.metricsForTask(task.id, {2, artifactFirst.nextCursor}, &error)
        .items.isEmpty());
    QCOMPARE(storage.lastErrorCode(), aitrain::ProjectErrorCode::InvalidPageCursor);
}

void StorageTests::persistsCompleteTaskFailure()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));

    const aitrain::TaskSnapshot source = makeTask();
    QVERIFY2(storage.createTask(source, &error), qPrintable(error));
    QVERIFY2(storage.transitionTask(source.id, aitrain::TaskState::Created,
        aitrain::TaskState::Queued, {}, &error), qPrintable(error));
    const QDateTime occurredAt = QDateTime::currentDateTimeUtc().addSecs(-2);
    const aitrain::Failure failure{aitrain::FailureCode::DependencyMissing,
        QStringLiteral("缺少训练依赖"), QStringLiteral("安装依赖后重试"), occurredAt};
    QVERIFY2(storage.transitionTask(source.id, aitrain::TaskState::Queued,
        aitrain::TaskState::Failed, failure, &error), qPrintable(error));

    aitrain::TaskSnapshot stored;
    QVERIFY2(storage.task(source.id, &stored, &error), qPrintable(error));
    QCOMPARE(stored.failure.code, failure.code);
    QCOMPARE(stored.failure.message, failure.message);
    QCOMPARE(stored.failure.suggestedAction, failure.suggestedAction);
    QCOMPARE(stored.failure.occurredAt, failure.occurredAt);
}

void StorageTests::rejectsInvalidCasAndRecoversInterruptedTask()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    const aitrain::TaskSnapshot source = makeTask();
    QVERIFY2(storage.createTask(source, &error), qPrintable(error));
    QVERIFY(!storage.transitionTask(source.id, aitrain::TaskState::Running, aitrain::TaskState::Succeeded, {}, &error));
    QVERIFY(error.contains(QStringLiteral("并发更新")));

    QVERIFY2(storage.transitionTask(source.id, aitrain::TaskState::Created, aitrain::TaskState::Queued, {}, &error), qPrintable(error));
    QVERIFY2(storage.markInterruptedTasksFailed(&error), qPrintable(error));
    aitrain::TaskSnapshot stored;
    QVERIFY2(storage.task(source.id, &stored, &error), qPrintable(error));
    QCOMPARE(stored.state, aitrain::TaskState::Failed);
    QCOMPARE(stored.failure.code, aitrain::FailureCode::ProcessCrashed);
}

void StorageTests::recoversCancelRequestedInterruptionAsCanceled()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));

    const aitrain::TaskSnapshot source = makeTask();
    QVERIFY2(startTask(storage, source, &error), qPrintable(error));
    QVERIFY2(storage.transitionTask(source.id, aitrain::TaskState::Running,
        aitrain::TaskState::CancelRequested, {}, &error), qPrintable(error));
    QVERIFY2(storage.markInterruptedTasksFailed(&error), qPrintable(error));

    aitrain::TaskSnapshot stored;
    QVERIFY2(storage.task(source.id, &stored, &error), qPrintable(error));
    QCOMPARE(stored.state, aitrain::TaskState::Canceled);
    QCOMPARE(stored.failure.code, aitrain::FailureCode::Canceled);
}

void StorageTests::rejectsLegacyDatabase()
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

    aitrain::ProjectStore storage;
    QString error;
    QVERIFY(!storage.open(path, &error));
    QVERIFY(error.contains(QStringLiteral("旧项目数据库")));
    QVERIFY(!storage.isOpen());
}

void StorageTests::enforcesForeignKeys()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString path = directory.filePath(QStringLiteral("project.sqlite"));
    aitrain::ProjectStore storage;
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

void StorageTests::hostStateEventsDoNotConsumeAdapterProtocolSequence()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    const aitrain::TaskSnapshot task = makeTask();
    QVERIFY2(storage.createTask(task, &error), qPrintable(error));
    quint64 lastSequence = 99;
    QVERIFY2(storage.lastProtocolSequence(task.id, &lastSequence, &error), qPrintable(error));
    QCOMPARE(lastSequence, quint64(0));

    aitrain::ProtocolEnvelope event;
    event.messageId = aitrain::MessageId::create();
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

void StorageTests::rejectsNonCanonicalArtifactMemberPaths()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    const aitrain::TaskSnapshot task = makeTask();
    QVERIFY2(storage.createTask(task, &error), qPrintable(error));
    const auto sha = QString(64, QLatin1Char('a'));

    const QStringList invalidPaths = {
        QStringLiteral("../escape.bin"),
        QStringLiteral("/absolute.bin"),
        QStringLiteral("C:/absolute.bin"),
        QStringLiteral("nested/../model.onnx")};
    for (const QString& path : invalidPaths) {
        error.clear();
        QVERIFY(!storage.recordArtifactWithFiles(aitrain::ArtifactId::create(), task.id,
            QStringLiteral("fixture"), {{path, sha, 1}}, QDateTime::currentDateTimeUtc(), &error));
        QVERIFY2(!error.isEmpty(), qPrintable(path));
    }

    error.clear();
    QVERIFY(!storage.recordArtifactWithFiles(aitrain::ArtifactId::create(), task.id,
        QStringLiteral("fixture"),
        {{QStringLiteral("Model.onnx"), sha, 1}, {QStringLiteral("model.onnx"), sha, 1}},
        QDateTime::currentDateTimeUtc(), &error));
    QVERIFY(!error.isEmpty());
}

void StorageTests::rejectsNonHexArtifactSha256()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    const aitrain::TaskSnapshot task = makeTask();
    QVERIFY2(storage.createTask(task, &error), qPrintable(error));

    const QVector<QString> invalidDigests{
        QString(63, QLatin1Char('a')) + QStringLiteral("g"),
        QString(64, QLatin1Char('A')),
        QString(64, QLatin1Char('0')) + QStringLiteral(" ")};
    for (const QString& digest : invalidDigests) {
        error.clear();
        QVERIFY(!storage.recordArtifactWithFiles(aitrain::ArtifactId::create(), task.id,
            QStringLiteral("fixture"), {{QStringLiteral("model.bin"), digest, 1}},
            QDateTime::currentDateTimeUtc(), &error));
        QVERIFY2(!error.isEmpty(), qPrintable(digest));
    }

    QVERIFY2(storage.recordArtifactWithFiles(aitrain::ArtifactId::create(), task.id,
        QStringLiteral("fixture"), {{QStringLiteral("model.bin"), QString(64, QLatin1Char('a')), 1}},
        QDateTime::currentDateTimeUtc(), &error), qPrintable(error));
}

void StorageTests::registersModelPackageOnlyForMatchingArtifactProvenance()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    const aitrain::TaskSnapshot task = makeTask();
    QVERIFY2(storage.createTask(task, &error), qPrintable(error));
    const aitrain::ArtifactId artifactId = aitrain::ArtifactId::create();
    const QVector<aitrain::ArtifactFileSnapshot> files = {
        {QStringLiteral("model.onnx"), QString(64, QLatin1Char('a')), 42},
        {QStringLiteral("unrelated.bin"), QString(64, QLatin1Char('b')), 7}};
    QVERIFY2(storage.recordArtifactWithFiles(artifactId, task.id, QStringLiteral("export_bundle"), files, QDateTime::currentDateTimeUtc(), &error), qPrintable(error));

    aitrain::ModelPackageSnapshot source;
    source.manifest = makeManifest(task.id);
    source.sourceArtifactId = artifactId;
    QVERIFY2(storage.registerModelPackage(source, &error), qPrintable(error));
    aitrain::ModelPackageSnapshot loaded;
    QVERIFY2(storage.modelPackage(source.manifest.modelPackageId, &loaded, &error), qPrintable(error));
    QCOMPARE(loaded.sourceArtifactId, artifactId);
    QCOMPARE(loaded.manifest.sourceArtifactSha256, source.manifest.sourceArtifactSha256);

    aitrain::ModelPackageSnapshot invalid;
    invalid.manifest = makeManifest(task.id);
    // 同一制品中的其他文件即使哈希匹配，也不能冒充清单指定的入口文件。
    invalid.manifest.sourceArtifactSha256 = QString(64, QLatin1Char('b'));
    invalid.sourceArtifactId = artifactId;
    QVERIFY(!storage.registerModelPackage(invalid, &error));
    QVERIFY(error.contains(QStringLiteral("哈希")));
}

void StorageTests::listsRegisteredModelPackagesNewestFirst()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    const aitrain::TaskSnapshot task = makeTask();
    QVERIFY2(storage.createTask(task, &error), qPrintable(error));
    const aitrain::ArtifactId artifactId = aitrain::ArtifactId::create();
    QVERIFY2(storage.recordArtifactWithFiles(artifactId, task.id, QStringLiteral("export_bundle"),
        {{QStringLiteral("model.onnx"), QString(64, QLatin1Char('a')), 42}}, QDateTime::currentDateTimeUtc(), &error), qPrintable(error));

    aitrain::ModelPackageSnapshot older;
    older.manifest = makeManifest(task.id);
    older.sourceArtifactId = artifactId;
    older.createdAt = QDateTime::currentDateTimeUtc().addSecs(-1);
    QVERIFY2(storage.registerModelPackage(older, &error), qPrintable(error));
    aitrain::ModelPackageSnapshot newer;
    newer.manifest = makeManifest(task.id);
    newer.sourceArtifactId = artifactId;
    newer.createdAt = QDateTime::currentDateTimeUtc();
    QVERIFY2(storage.registerModelPackage(newer, &error), qPrintable(error));

    const QVector<aitrain::ModelPackageSnapshot> listed =
        storage.modelPackages({1, {}}, &error).items;
    QCOMPARE(listed.size(), 1);
    QCOMPARE(listed.first().manifest.modelPackageId, newer.manifest.modelPackageId);
    QVERIFY(storage.modelPackages({0, {}}, &error).items.isEmpty());
    QVERIFY(error.contains(QStringLiteral("InvalidPageCursor")));
}

void StorageTests::persistsWorkflowStepsWithArtifactAndRetryGuards()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    const aitrain::TaskSnapshot task = makeTask();
    QVERIFY2(storage.createTask(task, &error), qPrintable(error));

    const aitrain::ArtifactId inputArtifact = aitrain::ArtifactId::create();
    const aitrain::ArtifactId outputArtifact = aitrain::ArtifactId::create();
    const QVector<aitrain::ArtifactFileSnapshot> files = {
        {QStringLiteral("payload.json"), QString(64, QLatin1Char('a')), 12}};
    QVERIFY2(storage.recordArtifactWithFiles(inputArtifact, task.id, QStringLiteral("source"), files, QDateTime::currentDateTimeUtc(), &error), qPrintable(error));
    QVERIFY2(storage.recordArtifactWithFiles(outputArtifact, task.id, QStringLiteral("result"), files, QDateTime::currentDateTimeUtc(), &error), qPrintable(error));

    aitrain::WorkflowRunSnapshot workflow;
    workflow.id = aitrain::WorkflowRunId::create();
    workflow.taskId = task.id;
    workflow.templateId = QStringLiteral("inference-deployment");
    aitrain::WorkflowStepSnapshot infer;
    infer.id = aitrain::WorkflowStepId::create();
    infer.workflowRunId = workflow.id;
    infer.ordinal = 0;
    infer.kind = QStringLiteral("RunInferenceSmoke");
    infer.inputArtifactId = inputArtifact;
    infer.backend = QStringLiteral("aitrain_onnxruntime");
    infer.parameterSummary = QJsonObject{{QStringLiteral("imageCount"), 1}};
    aitrain::WorkflowStepSnapshot deploy;
    deploy.id = aitrain::WorkflowStepId::create();
    deploy.workflowRunId = workflow.id;
    deploy.ordinal = 1;
    deploy.kind = QStringLiteral("DeploymentValidate");
    deploy.inputArtifactId = outputArtifact;
    deploy.backend = QStringLiteral("aitrain_onnxruntime");
    aitrain::WorkflowStepSnapshot report;
    report.id = aitrain::WorkflowStepId::create();
    report.workflowRunId = workflow.id;
    report.ordinal = 2;
    report.kind = QStringLiteral("RenderDeliveryReport");
    report.inputArtifactId = outputArtifact;
    report.backend = QStringLiteral("evidence_renderer");
    QVERIFY2(storage.createWorkflowRun(workflow, {infer, deploy, report}, &error), qPrintable(error));

    aitrain::WorkflowRunSnapshot loadedRun;
    QVERIFY2(storage.workflowRun(workflow.id, &loadedRun, &error), qPrintable(error));
    QCOMPARE(loadedRun.templateId, workflow.templateId);
    QCOMPARE(storage.workflowSteps(workflow.id, &error).size(), 3);

    QVERIFY2(storage.transitionWorkflowStep(infer.id, aitrain::WorkflowStepState::Pending,
        aitrain::WorkflowStepState::Running, {}, {}, &error), qPrintable(error));
    QVERIFY(!storage.transitionWorkflowStep(infer.id, aitrain::WorkflowStepState::Running,
        aitrain::WorkflowStepState::Succeeded, {}, {}, &error));
    QVERIFY(error.contains(QStringLiteral("Artifact")));
    QVERIFY2(storage.transitionWorkflowStep(infer.id, aitrain::WorkflowStepState::Running,
        aitrain::WorkflowStepState::Succeeded, outputArtifact, {}, &error), qPrintable(error));

    QVERIFY2(storage.transitionWorkflowStep(deploy.id, aitrain::WorkflowStepState::Pending,
        aitrain::WorkflowStepState::Running, {}, {}, &error), qPrintable(error));
    const aitrain::Failure failure{aitrain::FailureCode::DependencyMissing,
        QStringLiteral("缺少部署依赖"), QStringLiteral("安装部署依赖后重试"), QDateTime::currentDateTimeUtc()};
    QVERIFY2(storage.transitionWorkflowStep(deploy.id, aitrain::WorkflowStepState::Running,
        aitrain::WorkflowStepState::Failed, {}, failure, &error), qPrintable(error));
    QVERIFY2(storage.transitionWorkflowStep(report.id, aitrain::WorkflowStepState::Pending,
        aitrain::WorkflowStepState::Skipped, {}, {}, &error), qPrintable(error));
    const QVector<aitrain::WorkflowStepSnapshot> failedSteps = storage.workflowSteps(workflow.id, &error);
    QCOMPARE(failedSteps.at(1).failure.code, failure.code);
    QCOMPARE(failedSteps.at(2).state, aitrain::WorkflowStepState::Skipped);
    QCOMPARE(failedSteps.at(1).failure.message, failure.message);
    QCOMPARE(failedSteps.at(1).failure.suggestedAction, failure.suggestedAction);
    QCOMPARE(failedSteps.at(1).failure.occurredAt, failure.occurredAt);
    QVERIFY2(storage.retryWorkflowStep(deploy.id, &error), qPrintable(error));

    const QVector<aitrain::WorkflowStepSnapshot> steps = storage.workflowSteps(workflow.id, &error);
    QCOMPARE(steps.at(0).state, aitrain::WorkflowStepState::Succeeded);
    QCOMPARE(steps.at(0).outputArtifactId, outputArtifact);
    QVERIFY(steps.at(0).startedAt.isValid());
    QVERIFY(steps.at(0).finishedAt.isValid());
    QCOMPARE(steps.at(1).state, aitrain::WorkflowStepState::Pending);
    QCOMPARE(steps.at(1).retryCount, 1);
    QVERIFY(!steps.at(1).failure.isFailure());
    QCOMPARE(steps.at(2).state, aitrain::WorkflowStepState::Pending);
    QVERIFY(!steps.at(2).inputArtifactId.isValid());
    QVERIFY(!steps.at(2).outputArtifactId.isValid());
    QVERIFY(!steps.at(2).failure.isFailure());
}

void StorageTests::terminalizesWorkflowAndSkipsSuccessorsAtomically()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    const auto task = makeTask();
    QVERIFY2(startTask(storage, task, &error), qPrintable(error));

    aitrain::WorkflowRunSnapshot workflow;
    workflow.id = aitrain::WorkflowRunId::create();
    workflow.taskId = task.id;
    workflow.templateId = QStringLiteral("atomic-terminalization-test");
    QVector<aitrain::WorkflowStepSnapshot> steps;
    for (int ordinal = 0; ordinal < 3; ++ordinal) {
        aitrain::WorkflowStepSnapshot step;
        step.id = aitrain::WorkflowStepId::create();
        step.workflowRunId = workflow.id;
        step.ordinal = ordinal;
        step.kind = QStringLiteral("Step%1").arg(ordinal);
        step.backend = QStringLiteral("fixture");
        steps.push_back(step);
    }
    QVERIFY2(storage.createWorkflowRun(workflow, steps, &error), qPrintable(error));
    QVERIFY2(storage.transitionWorkflowStep(steps.at(0).id, aitrain::WorkflowStepState::Pending,
        aitrain::WorkflowStepState::Running, {}, {}, &error), qPrintable(error));
    const aitrain::Failure failure{aitrain::FailureCode::ProcessCrashed,
        QStringLiteral("fixture failure"), {}, QDateTime::currentDateTimeUtc()};
    QVERIFY2(storage.terminalizeWorkflowStepAndSkipSuccessors(steps.at(0).id,
        aitrain::WorkflowStepState::Running, aitrain::WorkflowStepState::Failed,
        failure, &error), qPrintable(error));
    auto loaded = storage.workflowSteps(workflow.id, &error);
    QCOMPARE(loaded.size(), 3);
    QCOMPARE(loaded.at(0).state, aitrain::WorkflowStepState::Failed);
    QCOMPARE(loaded.at(1).state, aitrain::WorkflowStepState::Skipped);
    QCOMPARE(loaded.at(2).state, aitrain::WorkflowStepState::Skipped);

    // 幂等恢复：终态步骤再次收口时仍应保持后继步骤全部跳过。
    QVERIFY2(storage.terminalizeWorkflowStepAndSkipSuccessors(steps.at(0).id,
        aitrain::WorkflowStepState::Failed, aitrain::WorkflowStepState::Failed,
        failure, &error), qPrintable(error));
}

void StorageTests::persistsCrossTaskWorkflowInputAndEnforcesOwnership()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    storage.setArtifactStoreRoot(directory.filePath(QStringLiteral("artifact-store")));

    const auto producer = makeTask();
    const auto consumer = makeTask();
    QVERIFY2(startTask(storage, producer, &error), qPrintable(error));
    QVERIFY2(startTask(storage, consumer, &error), qPrintable(error));

    const auto snapshotArtifactId = aitrain::ArtifactId::create();
    const QString manifestSha256(64, QLatin1Char('a'));
    const QString rootHash(64, QLatin1Char('b'));
    QVERIFY2(storage.recordArtifactWithFiles(snapshotArtifactId, producer.id,
        QStringLiteral("dataset_snapshot"),
        {{QStringLiteral("dataset_snapshot.json"), manifestSha256, 64}},
        QDateTime::currentDateTimeUtc(), &error), qPrintable(error));
    aitrain::DatasetSnapshotRecord snapshot;
    snapshot.id = aitrain::SnapshotId::create();
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
        QStringLiteral("artifact-store/committed/%1").arg(snapshotArtifactId.toString()));
    QVERIFY2(storage.registerDatasetSnapshot(&snapshot, &error), qPrintable(error));

    aitrain::WorkflowRunSnapshot workflow;
    workflow.id = aitrain::WorkflowRunId::create();
    workflow.taskId = consumer.id;
    workflow.templateId = QStringLiteral("cross-task-snapshot-test");
    aitrain::WorkflowStepSnapshot snapshotStep;
    snapshotStep.id = aitrain::WorkflowStepId::create();
    snapshotStep.workflowRunId = workflow.id;
    snapshotStep.ordinal = 0;
    snapshotStep.kind = QStringLiteral("CreateSnapshot");
    snapshotStep.backend = QStringLiteral("application");
    snapshotStep.inputArtifactId = snapshotArtifactId;
    aitrain::WorkflowStepSnapshot trainStep;
    trainStep.id = aitrain::WorkflowStepId::create();
    trainStep.workflowRunId = workflow.id;
    trainStep.ordinal = 1;
    trainStep.kind = QStringLiteral("Train");
    trainStep.backend = QStringLiteral("official_backend");
    aitrain::WorkflowInputBinding binding;
    binding.workflowRunId = workflow.id;
    binding.role = QStringLiteral("dataset_snapshot");
    binding.sourceArtifactId = snapshot.artifactId;
    binding.sourceTaskId = snapshot.taskId;
    binding.sourceArtifactKind = QStringLiteral("dataset_snapshot");
    binding.datasetId = snapshot.datasetId;
    binding.datasetSnapshotId = snapshot.id;
    binding.datasetVersionId = snapshot.datasetVersionId;
    binding.manifestSha256 = snapshot.manifestSha256;
    binding.rootHash = snapshot.rootHash;
    QVERIFY2(storage.createWorkflowRunWithInput(workflow, {snapshotStep, trainStep}, binding, &error),
        qPrintable(error));

    aitrain::WorkflowInputBinding storedBinding;
    QVERIFY2(storage.workflowInput(workflow.id, QStringLiteral("dataset_snapshot"),
        &storedBinding, &error), qPrintable(error));
    QCOMPARE(storedBinding.sourceTaskId, producer.id);
    QCOMPARE(storedBinding.sourceArtifactId, snapshot.artifactId);
    QCOMPARE(storedBinding.datasetId, snapshot.datasetId);
    QCOMPARE(storedBinding.datasetSnapshotId, snapshot.id);
    QCOMPARE(storedBinding.datasetVersionId, snapshot.datasetVersionId);

    QVERIFY2(storage.transitionWorkflowStep(snapshotStep.id,
        aitrain::WorkflowStepState::Pending, aitrain::WorkflowStepState::Running,
        {}, {}, &error), qPrintable(error));
    QVERIFY2(storage.transitionWorkflowStep(snapshotStep.id,
        aitrain::WorkflowStepState::Running, aitrain::WorkflowStepState::Succeeded,
        snapshot.artifactId, {}, &error), qPrintable(error));
    QVERIFY2(storage.bindWorkflowStepInput(trainStep.id, snapshot.artifactId, &error), qPrintable(error));
    QVERIFY2(storage.transitionWorkflowStep(trainStep.id,
        aitrain::WorkflowStepState::Pending, aitrain::WorkflowStepState::Running,
        {}, {}, &error), qPrintable(error));
    QVERIFY(!storage.transitionWorkflowStep(trainStep.id,
        aitrain::WorkflowStepState::Running, aitrain::WorkflowStepState::Succeeded,
        snapshot.artifactId, {}, &error));
    QVERIFY(error.contains(QStringLiteral("根任务")));

    const auto consumerOutput = aitrain::ArtifactId::create();
    QVERIFY2(storage.recordArtifactWithFiles(consumerOutput, consumer.id, QStringLiteral("training_output"),
        {{QStringLiteral("result.json"), QString(64, QLatin1Char('c')), 32}},
        QDateTime::currentDateTimeUtc(), &error), qPrintable(error));
    error.clear();
    QVERIFY2(storage.transitionWorkflowStep(trainStep.id,
        aitrain::WorkflowStepState::Running, aitrain::WorkflowStepState::Succeeded,
        consumerOutput, {}, &error), qPrintable(error));

    aitrain::WorkflowRunSnapshot rejectedWorkflow = workflow;
    rejectedWorkflow.id = aitrain::WorkflowRunId::create();
    rejectedWorkflow.templateId = QStringLiteral("mismatched-lineage-test");
    aitrain::WorkflowStepSnapshot rejectedStep = snapshotStep;
    rejectedStep.id = aitrain::WorkflowStepId::create();
    rejectedStep.workflowRunId = rejectedWorkflow.id;
    rejectedStep.state = aitrain::WorkflowStepState::Pending;
    rejectedStep.outputArtifactId = {};
    aitrain::WorkflowInputBinding rejectedBinding = binding;
    rejectedBinding.workflowRunId = rejectedWorkflow.id;
    rejectedBinding.datasetId = aitrain::DatasetId::create();
    error.clear();
    QVERIFY(!storage.createWorkflowRunWithInput(rejectedWorkflow, {rejectedStep}, rejectedBinding, &error));
    QVERIFY(error.contains(QStringLiteral("lineage")));
    aitrain::WorkflowRunSnapshot notCreated;
    error.clear();
    QVERIFY(!storage.workflowRun(rejectedWorkflow.id, &notCreated, &error));
}

void StorageTests::listsDatasetCatalogThroughVersionJoin()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    storage.setArtifactStoreRoot(directory.filePath(QStringLiteral("artifact-store")));

    const auto producer = makeTask();
    QVERIFY2(startTask(storage, producer, &error), qPrintable(error));
    const aitrain::DatasetId datasetId = aitrain::DatasetId::create();
    const auto registerSnapshot = [&](QChar hashChar, int fileCount) {
        const aitrain::ArtifactId artifactId = aitrain::ArtifactId::create();
        const QString manifestSha256(64, hashChar);
        const QString rootHash(64, QChar(hashChar.unicode() + 1));
        if (!storage.recordArtifactWithFiles(artifactId, producer.id,
                QStringLiteral("dataset_snapshot"),
                {{QStringLiteral("dataset_snapshot.json"), manifestSha256, 64}},
                QDateTime::currentDateTimeUtc(), &error)) {
            return aitrain::DatasetSnapshotRecord{};
        }
        aitrain::DatasetSnapshotRecord snapshot;
        snapshot.datasetId = datasetId;
        snapshot.id = aitrain::SnapshotId::create();
        snapshot.taskId = producer.id;
        snapshot.artifactId = artifactId;
        snapshot.rootPath = QDir(directory.path()).filePath(
            QStringLiteral("artifact-store/committed/%1").arg(artifactId.toString()));
        snapshot.datasetFormat = QStringLiteral("yolo_detection");
        snapshot.driverId = QStringLiteral("yolo_detection");
        snapshot.driverVersion = QStringLiteral("2.0");
        snapshot.rootHash = rootHash;
        snapshot.manifestSha256 = manifestSha256;
        snapshot.fileCount = fileCount;
        snapshot.totalBytes = fileCount * 10;
        snapshot.createdAt = QDateTime::currentDateTimeUtc().addSecs(fileCount);
        if (!storage.registerDatasetSnapshot(&snapshot, &error)) {
            return aitrain::DatasetSnapshotRecord{};
        }
        return snapshot;
    };

    const auto older = registerSnapshot(QLatin1Char('1'), 1);
    QVERIFY2(older.id.isValid(), qPrintable(error));
    const auto newer = registerSnapshot(QLatin1Char('3'), 2);
    QVERIFY2(newer.id.isValid(), qPrintable(error));

    const auto items = storage.datasets({10, {}}, &error).items;
    QVERIFY2(error.isEmpty(), qPrintable(error));
    QCOMPARE(items.size(), 1);
    QCOMPARE(items.first().datasetId, datasetId);
    QCOMPARE(items.first().versionCount, qint64(2));
    QCOMPARE(items.first().snapshotCount, qint64(2));
    QCOMPARE(items.first().latestVersionId, newer.datasetVersionId);
    QCOMPARE(items.first().latestSnapshotId, newer.id);
    QCOMPARE(items.first().latestArtifactId, newer.artifactId);
    QCOMPARE(items.first().latestRootHash, newer.rootHash);
    QCOMPARE(items.first().latestFileCount, newer.fileCount);
}

void StorageTests::rejectsSchema7AndPersistsTerminalPolicy()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString oldPath = directory.filePath(QStringLiteral("schema7.sqlite"));
    const QString connectionName = QStringLiteral("schema7_setup");
    QSqlDatabase database = QSqlDatabase::addDatabase(QStringLiteral("QSQLITE"), connectionName);
    database.setDatabaseName(oldPath);
    QVERIFY(database.open());
    QSqlQuery query(database);
    QVERIFY(query.exec(QStringLiteral("create table schema_info(version integer not null check(version = 7))")));
    QVERIFY(query.exec(QStringLiteral("insert into schema_info values(7)")));
    database.close();
    database = QSqlDatabase();
    QSqlDatabase::removeDatabase(connectionName);
    aitrain::ProjectStore rejected;
    QString error;
    QVERIFY(!rejected.open(oldPath, &error));
    QCOMPARE(rejected.lastErrorCode(), aitrain::ProjectErrorCode::SchemaRebuildRequired);
    QVERIFY(error.contains(QStringLiteral("SchemaRebuildRequired")));

    aitrain::ProjectStore storage;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("schema8.sqlite")), &error), qPrintable(error));
    const aitrain::TaskSnapshot task = makeTask();
    QVERIFY2(startTask(storage, task, &error), qPrintable(error));
    aitrain::WorkflowStepSnapshot step;
    const auto workflow = createWorkflow(storage, task.id, &step, &error);
    QVERIFY2(workflow.id.isValid(), qPrintable(error));
    aitrain::WorkflowRunSnapshot loaded;
    QVERIFY2(storage.workflowRun(workflow.id, &loaded, &error), qPrintable(error));
    QCOMPARE(loaded.terminalPolicy, aitrain::WorkflowTerminalPolicy::EvidenceRequired);
}

void StorageTests::createsCanonicalSchemaWithoutLegacyProjectTable()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString databasePath = directory.filePath(QStringLiteral("project.sqlite"));
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(databasePath, &error), qPrintable(error));
    QCOMPARE(aitrain::ProjectStore::schemaVersion(), 13);
    aitrain::ProjectMetaSnapshot meta;
    QVERIFY2(storage.projectMeta(&meta, &error), qPrintable(error));
    QVERIFY(meta.projectId.isValid());
    QCOMPARE(meta.schemaVersion, 13);
    QCOMPARE(meta.openGeneration, qint64(0));
    storage.close();

    const QString connectionName = QStringLiteral("canonical_schema_check");
    QSqlDatabase database = QSqlDatabase::addDatabase(QStringLiteral("QSQLITE"), connectionName);
    database.setDatabaseName(databasePath);
    QVERIFY(database.open());
    QSqlQuery query(database);
    QVERIFY(query.exec(QStringLiteral(
        "select project_id, schema_version, open_generation from project_meta "
        "where singleton = 1")));
    QVERIFY(query.next());
    QCOMPARE(query.value(0).toString(), meta.projectId.toString());
    QCOMPARE(query.value(1).toInt(), 13);
    QCOMPARE(query.value(2).toLongLong(), qint64(0));
    QVERIFY(!query.next());
    QVERIFY(query.exec(QStringLiteral(
        "select 1 from sqlite_master where type = 'table' and name = 'schema_info'")));
    QVERIFY(!query.next());
    QVERIFY(query.exec(QStringLiteral(
        "select 1 from sqlite_master where type = 'table' and name = 'projects'")));
    QVERIFY(!query.next());
    database.close();
    database = QSqlDatabase();
    QSqlDatabase::removeDatabase(connectionName);
}

void StorageTests::enforcesTerminalizationSchemaConstraints()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString databasePath = directory.filePath(QStringLiteral("project.sqlite"));
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(databasePath, &error), qPrintable(error));
    const auto task = makeTask();
    QVERIFY2(startTask(storage, task, &error), qPrintable(error));
    aitrain::WorkflowStepSnapshot step;
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

void StorageTests::evidenceGatedSuccessSurvivesReopen()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString path = directory.filePath(QStringLiteral("project.sqlite"));
    const aitrain::TaskSnapshot task = makeTask();
    aitrain::WorkflowRunId workflowId;
    aitrain::ArtifactId evidenceId;
    const QDateTime terminalAt = QDateTime::currentDateTimeUtc().addSecs(-3);
    QString error;
    {
        aitrain::ProjectStore storage;
        QVERIFY2(storage.open(path, &error), qPrintable(error));
        QVERIFY2(startTask(storage, task, &error), qPrintable(error));
        aitrain::WorkflowStepSnapshot step;
        const auto workflow = createWorkflow(storage, task.id, &step, &error);
        workflowId = workflow.id;
        const aitrain::ArtifactId output = aitrain::ArtifactId::create();
        QVERIFY2(storage.recordArtifactWithFiles(output, task.id, QStringLiteral("delivery_report"),
            {{QStringLiteral("report.json"), QString(64, QLatin1Char('a')), 12}},
            QDateTime::currentDateTimeUtc(), &error), qPrintable(error));
        QVERIFY2(storage.transitionWorkflowStep(step.id, aitrain::WorkflowStepState::Pending,
            aitrain::WorkflowStepState::Running, {}, {}, &error), qPrintable(error));
        QVERIFY2(storage.transitionWorkflowStep(step.id, aitrain::WorkflowStepState::Running,
            aitrain::WorkflowStepState::Succeeded, output, {}, &error), qPrintable(error));
        QVERIFY2(storage.sealWorkflowTerminalization(workflow.id,
            aitrain::TaskState::Succeeded, {}, terminalAt, &error), qPrintable(error));
        QCOMPARE(storage.pendingWorkflowTerminalizations(10, &error).size(), 1);
        QVERIFY2(storage.markInterruptedTasksFailed(&error), qPrintable(error));
        aitrain::TaskSnapshot stillRunning;
        QVERIFY2(storage.task(task.id, &stillRunning, &error), qPrintable(error));
        QCOMPARE(stillRunning.state, aitrain::TaskState::Running);
        evidenceId = aitrain::ArtifactId::create();
        QVERIFY2(storage.recordEvidenceArtifactWithFilesAndAttachTerminalization(evidenceId,
            task.id, workflow.id, evidenceFiles(), terminalAt.addMSecs(1), &error), qPrintable(error));
    }
    {
        aitrain::ProjectStore storage;
        QVERIFY2(storage.open(path, &error), qPrintable(error));
        const auto pending = storage.pendingWorkflowTerminalizations(10, &error);
        QCOMPARE(pending.size(), 1);
        QCOMPARE(pending.first().state, aitrain::WorkflowTerminalizationState::EvidenceAttached);
        QCOMPARE(pending.first().evidenceArtifactId, evidenceId);
        QVERIFY2(storage.closeWorkflowTerminalization(workflowId, &error), qPrintable(error));
        aitrain::TaskSnapshot terminal;
        QVERIFY2(storage.task(task.id, &terminal, &error), qPrintable(error));
        QCOMPARE(terminal.state, aitrain::TaskState::Succeeded);
        QCOMPARE(terminal.updatedAt, terminalAt);
        QVERIFY(storage.pendingWorkflowTerminalizations(10, &error).isEmpty());
        QVERIFY(storage.pendingEvidenceRequiredWorkflows(10, &error).isEmpty());
    }
}

void StorageTests::persistsFailedAndCanceledTerminalFacts()
{
    const QVector<aitrain::TaskState> terminalStates = {
        aitrain::TaskState::Failed, aitrain::TaskState::Canceled};
    for (const auto terminalState : terminalStates) {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        aitrain::ProjectStore storage;
        QString error;
        QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
        const auto task = makeTask();
        QVERIFY2(startTask(storage, task, &error), qPrintable(error));
        aitrain::WorkflowStepSnapshot step;
        const auto workflow = createWorkflow(storage, task.id, &step, &error);
        const QDateTime occurredAt = QDateTime::currentDateTimeUtc().addSecs(-2);
        const aitrain::Failure failure{
            terminalState == aitrain::TaskState::Canceled
                ? aitrain::FailureCode::Canceled : aitrain::FailureCode::DependencyMissing,
            terminalState == aitrain::TaskState::Canceled
                ? QStringLiteral("用户取消") : QStringLiteral("依赖缺失"),
            terminalState == aitrain::TaskState::Canceled
                ? QStringLiteral("重新启动任务") : QStringLiteral("安装依赖后重试"), occurredAt};
        QVERIFY2(storage.transitionWorkflowStep(step.id, aitrain::WorkflowStepState::Pending,
            aitrain::WorkflowStepState::Running, {}, {}, &error), qPrintable(error));
        QVERIFY2(storage.transitionWorkflowStep(step.id, aitrain::WorkflowStepState::Running,
            terminalState == aitrain::TaskState::Canceled
                ? aitrain::WorkflowStepState::Canceled : aitrain::WorkflowStepState::Failed,
            {}, failure, &error), qPrintable(error));
        if (terminalState == aitrain::TaskState::Canceled) {
            QVERIFY2(storage.transitionTask(task.id, aitrain::TaskState::Running,
                aitrain::TaskState::CancelRequested, {}, &error), qPrintable(error));
        }
        QVERIFY2(storage.sealWorkflowTerminalization(workflow.id, terminalState,
            failure, occurredAt.addMSecs(10), &error), qPrintable(error));
        const auto evidence = aitrain::ArtifactId::create();
        QVERIFY2(storage.recordEvidenceArtifactWithFilesAndAttachTerminalization(evidence,
            task.id, workflow.id, evidenceFiles(), QDateTime::currentDateTimeUtc(), &error), qPrintable(error));
        QVERIFY2(storage.closeWorkflowTerminalization(workflow.id, &error), qPrintable(error));
        aitrain::TaskSnapshot loaded;
        QVERIFY2(storage.task(task.id, &loaded, &error), qPrintable(error));
        QCOMPARE(loaded.state, terminalState);
        QCOMPARE(loaded.failure.code, failure.code);
        QCOMPARE(loaded.failure.message, failure.message);
        QCOMPARE(loaded.failure.suggestedAction, failure.suggestedAction);
        QCOMPARE(loaded.failure.occurredAt, failure.occurredAt);
    }
}

void StorageTests::terminalizationWritesAreIdempotentAndRejectConflicts()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    const auto task = makeTask();
    QVERIFY2(startTask(storage, task, &error), qPrintable(error));
    aitrain::WorkflowStepSnapshot step;
    const auto workflow = createWorkflow(storage, task.id, &step, &error);
    const QDateTime occurredAt = QDateTime::currentDateTimeUtc().addSecs(-4);
    const aitrain::Failure terminalFailure{aitrain::FailureCode::InvalidDataset,
        QStringLiteral("数据集无效"), QStringLiteral("修复数据集后重试"), occurredAt};
    QVERIFY2(storage.transitionWorkflowStep(step.id, aitrain::WorkflowStepState::Pending,
        aitrain::WorkflowStepState::Running, {}, {}, &error), qPrintable(error));
    QVERIFY2(storage.transitionWorkflowStep(step.id, aitrain::WorkflowStepState::Running,
        aitrain::WorkflowStepState::Failed, {}, terminalFailure, &error), qPrintable(error));
    const QDateTime terminalAt = occurredAt.addMSecs(20);
    QVERIFY2(storage.sealWorkflowTerminalization(workflow.id,
        aitrain::TaskState::Failed, terminalFailure, terminalAt, &error), qPrintable(error));
    QVERIFY2(storage.sealWorkflowTerminalization(workflow.id,
        aitrain::TaskState::Failed, terminalFailure, terminalAt, &error), qPrintable(error));
    QVERIFY(!storage.sealWorkflowTerminalization(workflow.id,
        aitrain::TaskState::Failed, terminalFailure, terminalAt.addMSecs(1), &error));
    error.clear();
    QVERIFY(!storage.transitionTask(task.id, aitrain::TaskState::Running,
        aitrain::TaskState::Failed, terminalFailure, &error));
    QVERIFY(error.contains(QStringLiteral("evidence_required")));

    const aitrain::Failure evidenceFailure{aitrain::FailureCode::ArtifactIncomplete,
        QStringLiteral("Evidence 写入失败"), QStringLiteral("检查磁盘后重试"), occurredAt.addMSecs(30)};
    QVERIFY2(storage.recordWorkflowTerminalizationEvidenceFailure(workflow.id,
        evidenceFailure, &error), qPrintable(error));
    QVERIFY2(storage.recordWorkflowTerminalizationEvidenceFailure(workflow.id,
        evidenceFailure, &error), qPrintable(error));
    aitrain::WorkflowTerminalizationSnapshot snapshot;
    QVERIFY2(storage.workflowTerminalization(workflow.id, &snapshot, &error), qPrintable(error));
    QCOMPARE(snapshot.evidenceAttemptCount, 1);
    QCOMPARE(snapshot.lastEvidenceFailure.message, evidenceFailure.message);

    const auto evidence = aitrain::ArtifactId::create();
    const QDateTime evidenceAt = QDateTime::currentDateTimeUtc();
    QVERIFY2(storage.recordEvidenceArtifactWithFilesAndAttachTerminalization(evidence,
        task.id, workflow.id, evidenceFiles(), evidenceAt, &error), qPrintable(error));
    QVERIFY2(storage.recordEvidenceArtifactWithFilesAndAttachTerminalization(evidence,
        task.id, workflow.id, evidenceFiles(), evidenceAt, &error), qPrintable(error));
    bool exists = false;
    QVERIFY2(storage.artifactExists(evidence, &exists, &error), qPrintable(error));
    QVERIFY(exists);
    QVector<aitrain::ArtifactFileSnapshot> conflictingFiles = evidenceFiles();
    conflictingFiles[0].sha256 = QString(64, QLatin1Char('f'));
    QVERIFY(!storage.recordEvidenceArtifactWithFilesAndAttachTerminalization(evidence,
        task.id, workflow.id, conflictingFiles, evidenceAt, &error));
    const auto conflictingArtifact = aitrain::ArtifactId::create();
    QVERIFY(!storage.recordEvidenceArtifactWithFilesAndAttachTerminalization(conflictingArtifact,
        task.id, workflow.id, evidenceFiles(), evidenceAt, &error));
    QVERIFY2(storage.artifactExists(conflictingArtifact, &exists, &error), qPrintable(error));
    QVERIFY(!exists);
    QVERIFY2(storage.closeWorkflowTerminalization(workflow.id, &error), qPrintable(error));
    QVERIFY2(storage.closeWorkflowTerminalization(workflow.id, &error), qPrintable(error));
}

void StorageTests::listsUnsealedGatedWorkflowsAndProtectsThemFromInterruption()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    const auto task = makeTask();
    QVERIFY2(startTask(storage, task, &error), qPrintable(error));
    aitrain::WorkflowStepSnapshot step;
    const auto workflow = createWorkflow(storage, task.id, &step, &error);
    const auto pending = storage.pendingEvidenceRequiredWorkflows(10, &error);
    QCOMPARE(pending.size(), 1);
    QCOMPARE(pending.first().id, workflow.id);
    QVERIFY(storage.pendingWorkflowTerminalizations(10, &error).isEmpty());
    QVERIFY2(storage.markInterruptedTasksFailed(&error), qPrintable(error));
    aitrain::TaskSnapshot loaded;
    QVERIFY2(storage.task(task.id, &loaded, &error), qPrintable(error));
    QCOMPARE(loaded.state, aitrain::TaskState::Running);
    QVERIFY(storage.pendingEvidenceRequiredWorkflows(0, &error).isEmpty());
    QVERIFY(error.contains(QStringLiteral("limit")));
}

void StorageTests::durableWorkflowTerminalOutboxIsIdempotent()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    const aitrain::TaskSnapshot task = makeTask();
    QVERIFY2(startTask(storage, task, &error), qPrintable(error));
    aitrain::WorkflowStepSnapshot step;
    const aitrain::WorkflowRunSnapshot workflow = createWorkflow(storage, task.id, &step, &error);
    QVERIFY2(workflow.id.isValid(), qPrintable(error));
    QVERIFY2(storage.transitionWorkflowStep(step.id, aitrain::WorkflowStepState::Pending,
        aitrain::WorkflowStepState::Running, {}, {}, &error), qPrintable(error));

    aitrain::ProtocolEnvelope event;
    event.messageId = aitrain::MessageId::create();
    event.requestId = task.requestId;
    event.taskId = task.id;
    event.sequence = 1;
    event.kind = QStringLiteral("event.failed");
    event.timestamp = QDateTime::currentDateTimeUtc();
    event.payload = {{QStringLiteral("message"), QStringLiteral("adapter failed")},
        {QStringLiteral("failureCode"), aitrain::failureCodeToString(aitrain::FailureCode::ProcessCrashed)},
        {QStringLiteral("suggestedAction"), QStringLiteral("retry")}};
    bool idempotent = false;
    QVERIFY2(storage.recordWorkflowTerminalEvent(event, {}, &idempotent, &error), qPrintable(error));
    QVERIFY(!idempotent);
    QVector<aitrain::WorkflowTerminalEventSnapshot> pending =
        storage.pendingWorkflowTerminalEvents(16, &error);
    QVERIFY2(error.isEmpty(), qPrintable(error));
    QCOMPARE(pending.size(), 1);
    QCOMPARE(pending.first().workflowRunId, workflow.id);
    QCOMPARE(pending.first().workflowStepId, step.id);
    QCOMPARE(pending.first().kind, event.kind);
    QVERIFY(!pending.first().applied);

    QVERIFY2(storage.recordWorkflowTerminalEvent(event, {}, &idempotent, &error), qPrintable(error));
    QVERIFY(idempotent);
    QVERIFY2(storage.markWorkflowTerminalEventApplied(event.messageId, &error), qPrintable(error));
    QVERIFY2(storage.markWorkflowTerminalEventApplied(event.messageId, &error), qPrintable(error));
    pending = storage.pendingWorkflowTerminalEvents(16, &error);
    QVERIFY2(error.isEmpty(), qPrintable(error));
    QVERIFY(pending.isEmpty());
}

QTEST_MAIN(StorageTests)
#include "tst_storage.moc"
