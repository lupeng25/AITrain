#include "aitrain/workflow/TaskCoordinator.h"
#include "aitrain/workflow/CapabilityPlanner.h"
#include "aitrain/workflow/EvidenceBundle.h"
#include "aitrain/workflow/TaskExecutionHost.h"
#include "aitrain/model/ModelImportService.h"
#include "aitrain/runtime/ModelPackageRuntimeService.h"
#include "aitrain/workflow/ProjectWorkspace.h"
#include "aitrain/workflow/ProjectQueryService.h"
#include "aitrain/workflow/WorkflowRunner.h"

#include <QDateTime>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QImage>
#include <QSqlDatabase>
#include <QSqlError>
#include <QSqlQuery>
#include <QTemporaryDir>
#include <QTcpSocket>
#include <QTest>
#include <QUuid>

#include <algorithm>

namespace {

bool writeFile(const QString& path, const QByteArray& bytes)
{
    if (!QDir().mkpath(QFileInfo(path).absolutePath())) return false;
    QFile file(path);
    return file.open(QIODevice::WriteOnly) && file.write(bytes) == bytes.size();
}

bool createYoloImportFixture(const QString& root, const QByteArray& label)
{
    QImage image(8, 8, QImage::Format_RGB32);
    image.fill(Qt::white);
    return writeFile(QDir(root).filePath(QStringLiteral("data.yaml")),
               QByteArray("path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: [item]\n"))
        && QDir().mkpath(QDir(root).filePath(QStringLiteral("images/train")))
        && QDir().mkpath(QDir(root).filePath(QStringLiteral("images/val")))
        && image.save(QDir(root).filePath(QStringLiteral("images/train/a.png")))
        && image.save(QDir(root).filePath(QStringLiteral("images/val/b.png")))
        && writeFile(QDir(root).filePath(QStringLiteral("labels/train/a.txt")), label)
        && writeFile(QDir(root).filePath(QStringLiteral("labels/val/b.txt")), label);
}

bool attachTrainingSnapshotFixture(
    aitrain::ProjectWorkspace& workspace,
    const aitrain::TaskId& taskId,
    const QString& root,
    const QString& datasetFormat,
    aitrain::TrainingWorkflowRequest* request,
    QString* error)
{
    if (!request || !QDir().mkpath(root)
        || !writeFile(QDir(root).filePath(QStringLiteral("fixture.bin")), QByteArrayLiteral("snapshot"))) {
        if (error) *error = QStringLiteral("无法创建训练 Snapshot 测试夹具。");
        return false;
    }
    aitrain::DatasetSnapshotCommitRequest snapshotRequest;
    snapshotRequest.datasetRoot = root;
    snapshotRequest.datasetFormat = datasetFormat;
    snapshotRequest.driverId = datasetFormat;
    snapshotRequest.driverVersion = QStringLiteral("2.0");
    aitrain::DatasetSnapshotArtifactBundle snapshot;
    if (!workspace.commitDatasetSnapshot(taskId, snapshotRequest, &snapshot, error)) return false;
    request->datasetId = snapshot.snapshot.datasetId;
    request->datasetVersionId = snapshot.snapshot.datasetVersionId;
    request->snapshotId = snapshot.snapshot.id;
    request->snapshotArtifactId = snapshot.snapshot.artifactId;
    return true;
}

} // namespace

class ApplicationTests : public QObject {
    Q_OBJECT

private slots:
    void fakeWorkerSuccessfulRunPersistsTaskMetricsAndArtifacts();
    void duplicateWorkerMessageIsRejectedWithoutDuplicateData();
    void invalidWorkerPayloadIsRejectedBeforePersistence();
    void capabilityPlannerProducesVerifiableImmutableSummary();
    void capabilityPlannerAcceptsOfficialWorkflowAdapterProfiles();
    void cancellationTransitionsThroughCancelRequestedAndTerminalCanceled();
    void adapterHostSynthesizesCanceledTerminalAfterForcedCancellation();
    void adapterArtifactCandidatesCommitAsOneBundleBeforeSuccess();
    void adapterArtifactCandidateRejectsOutsideRoot();
    void adapterHostDelegatesExistingTaskTerminalToWorkflowOwner();
    void adapterHostSyntheticTerminalUsesPersistedSequenceOffset();
    void importCreatesVerifiedModelPackageWithoutGuessingType();
    void importCancellationLeavesNoRegisteredPackageOrArtifact();
    void runtimeResolutionAcceptsOnlyRegisteredUntamperedModelPackages();
    void projectWorkspaceFirstStartCreatesStableLayout();
    void projectWorkspaceOwnsRuntimeTaskLifecycle();
    void projectQueryServiceReadsOnlyPersistedTaskState();
    void externalAcceptanceEvidenceRequiresStrictSchemaAndStaysUnverified();
    void deliveryEvidenceLimitCountsEvidenceArtifacts();
    void deliveryEvidenceKeepsInvalidArtifactAsRow();
    void projectWorkspaceCommitsRuntimeArtifactsBeforeSuccess();
    void projectWorkspaceRegistersDatasetSnapshotAndSequencesTrainingWorkflow();
    void datasetSnapshotImportRegistersNewAndExistingDatasetVersions();
    void datasetSnapshotImportRejectsSourceChangedAfterPlanWithEvidence();
    void datasetSplitWorkflowRegistersSelfContainedSnapshotAndRejectsIdentityMismatch();
    void datasetSplitWorkflowCancellationAfterMaterializeDoesNotRegisterSnapshot();
    void trainingWorkflowRejectsCrossProfileBackendMix();
    void trainingWorkflowEvidenceGatePersistsEvidenceBeforeTerminalTask();
    void trainingWorkflowEvidenceGateRecoversAcrossReopen();
    void officialYoloWorkflowPreservesVariantTaskType_data();
    void officialYoloWorkflowPreservesVariantTaskType();
    void projectWorkspaceDispatchesOfficialAdapterStepThroughTrainingWorkflow();
    void workflowRunnerSequencesCommittedArtifactsAndStopsOnCancellation();
};

void ApplicationTests::fakeWorkerSuccessfulRunPersistsTaskMetricsAndArtifacts()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    aitrain::TaskCoordinator coordinator(&storage);
    aitrain::TaskSnapshot task;
    QVERIFY2(coordinator.createAndStartTask(QStringLiteral("yolo.detect"), QStringLiteral("training"), &task, &error), qPrintable(error));
    QCOMPARE(task.state, aitrain::TaskState::Running);

    const QVector<aitrain::ProtocolEnvelope> events = aitrain::FakeWorker::successfulRun(task, 5);
    for (const aitrain::ProtocolEnvelope& event : events) {
        QVERIFY2(coordinator.consumeWorkerEvent(event, &error), qPrintable(error));
    }

    aitrain::TaskSnapshot stored;
    QVERIFY2(storage.task(task.id, &stored, &error), qPrintable(error));
    QCOMPARE(stored.state, aitrain::TaskState::Succeeded);
    QCOMPARE(storage.metricCount(task.id, &error), 1);
    QCOMPARE(storage.artifactCount(task.id, &error), 1);
    QCOMPARE(storage.eventCount(task.id, &error), 9);
}

void ApplicationTests::duplicateWorkerMessageIsRejectedWithoutDuplicateData()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    aitrain::TaskCoordinator coordinator(&storage);
    aitrain::TaskSnapshot task;
    QVERIFY2(coordinator.createAndStartTask(QStringLiteral("yolo.detect"), QStringLiteral("training"), &task, &error), qPrintable(error));
    const aitrain::ProtocolEnvelope event = aitrain::FakeWorker::successfulRun(task, 5).first();
    QVERIFY2(coordinator.consumeWorkerEvent(event, &error), qPrintable(error));
    QVERIFY(!coordinator.consumeWorkerEvent(event, &error));
    QVERIFY(error.contains(QStringLiteral("duplicate"), Qt::CaseInsensitive));
    QCOMPARE(storage.eventCount(task.id, &error), 5);
}

void ApplicationTests::invalidWorkerPayloadIsRejectedBeforePersistence()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    aitrain::TaskCoordinator coordinator(&storage);
    aitrain::TaskSnapshot task;
    QVERIFY2(coordinator.createAndStartTask(QStringLiteral("yolo.detect"), QStringLiteral("training"), &task, &error), qPrintable(error));
    aitrain::ProtocolEnvelope invalid = aitrain::FakeWorker::successfulRun(task, 5).at(1);
    invalid.payload.remove(QStringLiteral("name"));
    QVERIFY(!coordinator.consumeWorkerEvent(invalid, &error));
    QVERIFY(error.contains(QStringLiteral("name")));
    QCOMPARE(storage.eventCount(task.id, &error), 4);
    QCOMPARE(storage.metricCount(task.id, &error), 0);
}

void ApplicationTests::capabilityPlannerProducesVerifiableImmutableSummary()
{
    aitrain::ExecutionRequest request;
    request.capabilityId = QStringLiteral("semantic_segmentation");
    request.taskType = QStringLiteral("semantic_segmentation");
    request.datasetFormat = QStringLiteral("semantic_segmentation_mask");
    request.trainingBackend = QStringLiteral("smp_semantic_segmentation");
    request.evaluationBackend = QStringLiteral("smp_semantic_segmentation");
    request.exportFormat = QStringLiteral("onnx");
    request.runtimeRoute = QStringLiteral("aitrain_onnxruntime");

    aitrain::CapabilityPlanner planner;
    aitrain::ExecutionPlan plan;
    QString error;
    QVERIFY2(planner.plan(request, &plan, &error), qPrintable(error));
    QCOMPARE(plan.runtimeRoute, QStringLiteral("aitrain_onnxruntime"));
    QVERIFY(plan.summaryHash.size() == 64);
    QVERIFY2(planner.verify(request, plan.summaryHash, nullptr, &error), qPrintable(error));

    request.exportFormat = QStringLiteral("tensorrt");
    QVERIFY(!planner.plan(request, &plan, &error));
    QVERIFY(error.contains(QStringLiteral("导出格式")));
}

void ApplicationTests::capabilityPlannerAcceptsOfficialWorkflowAdapterProfiles()
{
    const QList<aitrain::ExecutionRequest> requests = {
        {QStringLiteral("yolo"), QStringLiteral("detection"), QStringLiteral("yolo_detection"),
            QStringLiteral("ultralytics_yolo_detect"), QStringLiteral("ultralytics_yolo_eval"),
            QStringLiteral("onnx"), QStringLiteral("aitrain_onnxruntime")},
        {QStringLiteral("anomaly_detection"), QStringLiteral("anomaly_detection"), QStringLiteral("anomaly_folder"),
            QStringLiteral("anomalib_patchcore"), QStringLiteral("anomalib_python_eval"),
            QStringLiteral("anomalib_bundle"), QStringLiteral("anomalib_python")}};
    aitrain::CapabilityPlanner planner;
    for (const aitrain::ExecutionRequest& request : requests) {
        aitrain::ExecutionPlan plan;
        QString error;
        QVERIFY2(planner.plan(request, &plan, &error), qPrintable(error));
        QCOMPARE(plan.evaluationBackend, request.evaluationBackend);
        QVERIFY2(planner.verify(request, plan.summaryHash, nullptr, &error), qPrintable(error));
    }
}

void ApplicationTests::cancellationTransitionsThroughCancelRequestedAndTerminalCanceled()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    aitrain::TaskCoordinator coordinator(&storage);
    aitrain::TaskSnapshot task;
    QVERIFY2(coordinator.createAndStartTask(QStringLiteral("yolo.detect"), QStringLiteral("training"), &task, &error), qPrintable(error));
    QVERIFY2(coordinator.requestCancellation(task.id, &error), qPrintable(error));
    QVERIFY2(coordinator.requestCancellation(task.id, &error), qPrintable(error));

    aitrain::TaskSnapshot cancelRequested;
    QVERIFY2(storage.task(task.id, &cancelRequested, &error), qPrintable(error));
    QCOMPARE(cancelRequested.state, aitrain::TaskState::CancelRequested);
    const QVector<aitrain::ProtocolEnvelope> events = aitrain::FakeWorker::canceledRun(task, 6, QStringLiteral("用户取消"));
    for (const aitrain::ProtocolEnvelope& event : events) {
        QVERIFY2(coordinator.consumeWorkerEvent(event, &error), qPrintable(error));
    }
    aitrain::TaskSnapshot canceled;
    QVERIFY2(storage.task(task.id, &canceled, &error), qPrintable(error));
    QCOMPARE(canceled.state, aitrain::TaskState::Canceled);
    QCOMPARE(canceled.failure.code, aitrain::FailureCode::Canceled);
    QVERIFY(!coordinator.requestCancellation(task.id, &error));
}

void ApplicationTests::adapterHostSynthesizesCanceledTerminalAfterForcedCancellation()
{
#ifdef Q_OS_WIN
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    aitrain::TaskCoordinator coordinator(&storage);
    aitrain::ArtifactStore artifacts(directory.filePath(QStringLiteral("store")));
    aitrain::TaskExecutionHost host(&coordinator, &artifacts);
    aitrain::PythonAdapterLaunch launch;
    launch.program = QStringLiteral("cmd.exe");
    launch.arguments.append(QStringLiteral("/c"));
    launch.arguments.append(QStringLiteral("ping 127.0.0.1 -n 20 > nul"));
    launch.cancellationGraceMs = 50;
    aitrain::TaskSnapshot task;
    QVERIFY2(host.start(QStringLiteral("yolo.detect"), QStringLiteral("training"), launch, &task, &error), qPrintable(error));
    QVERIFY2(host.requestCancellation(task.id, &error), qPrintable(error));
    QTRY_VERIFY_WITH_TIMEOUT(!host.isRunning(), 15000);

    aitrain::TaskSnapshot stored;
    QVERIFY2(storage.task(task.id, &stored, &error), qPrintable(error));
    QVERIFY2(stored.state == aitrain::TaskState::Canceled,
        qPrintable(QStringLiteral("actual state=%1, host error=%2").arg(aitrain::taskStateToString(stored.state), host.lastError())));
    QCOMPARE(stored.failure.code, aitrain::FailureCode::Canceled);
    QVERIFY(host.lastError().isEmpty());
    QCOMPARE(storage.artifactCount(task.id, &error), 0);
    // Created/queued/starting/running, cancel-requested, one protocol terminal,
    // and one terminal state transition. A second synthesized terminal would
    // make this count larger and violate -307's single-terminal invariant.
    QCOMPARE(storage.eventCount(task.id, &error), 7);
    const QDir stagingRoot(directory.filePath(QStringLiteral("store/.staging")));
    QCOMPARE(stagingRoot.entryList(QDir::Dirs | QDir::NoDotAndDotDot).size(), 0);
#else
    QSKIP(" Adapter Host integration uses Windows Job Object.");
#endif
}

void ApplicationTests::adapterArtifactCandidatesCommitAsOneBundleBeforeSuccess()
{
#ifdef Q_OS_WIN
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString candidatePath = directory.filePath(QStringLiteral("adapter-report.json"));
    QFile candidate(candidatePath);
    QVERIFY(candidate.open(QIODevice::WriteOnly));
    QCOMPARE(candidate.write("{\"ok\":true}\n"), qint64(12));
    candidate.close();

    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    aitrain::TaskCoordinator coordinator(&storage);
    aitrain::ArtifactStore artifacts(directory.filePath(QStringLiteral("store")));
    aitrain::TaskExecutionHost host(&coordinator, &artifacts);
    aitrain::PythonAdapterLaunch launch;
    launch.program = QStringLiteral("cmd.exe");
    launch.arguments.append(QStringLiteral("/c"));
    launch.arguments.append(QStringLiteral("ping 127.0.0.1 -n 2 > nul"));
    launch.artifactCandidateRoots = QStringList{directory.path()};
    aitrain::TaskSnapshot task;
    QVERIFY2(host.start(QStringLiteral("yolo.detect"), QStringLiteral("training"), launch, &task, &error), qPrintable(error));

    const aitrain::AdapterEventEndpoint endpoint = host.adapterEndpoint();
    QTcpSocket socket;
    socket.connectToHost(endpoint.host, endpoint.port);
    QVERIFY(socket.waitForConnected(3000));
    socket.write(QByteArrayLiteral("{\"channel\":\"aitrain.adapter\",\"token\":\"") + endpoint.token.toUtf8() + QByteArrayLiteral("\"}\n"));
    QTRY_VERIFY(socket.bytesAvailable() > 0);
    QCOMPARE(QJsonDocument::fromJson(socket.readLine()).object().value(QStringLiteral("status")).toString(), QStringLiteral("accepted"));

    aitrain::ProtocolEnvelope candidateEvent;
    candidateEvent.messageId = aitrain::MessageId::create();
    candidateEvent.requestId = task.requestId;
    candidateEvent.taskId = task.id;
    candidateEvent.sequence = 1;
    candidateEvent.kind = QStringLiteral("event.artifact_candidate");
    candidateEvent.timestamp = QDateTime::currentDateTimeUtc();
    candidateEvent.payload = QJsonObject{{QStringLiteral("kind"), QStringLiteral("report")}, {QStringLiteral("path"), candidatePath}};
    const QByteArray candidateWire = aitrain::encodeProtocolMessage(candidateEvent, &error);
    QVERIFY2(!candidateWire.isEmpty(), qPrintable(error));
    socket.write(candidateWire);

    aitrain::ProtocolEnvelope successEvent = candidateEvent;
    successEvent.messageId = aitrain::MessageId::create();
    successEvent.sequence = 2;
    successEvent.kind = QStringLiteral("event.succeeded");
    successEvent.payload = {};
    const QByteArray successWire = aitrain::encodeProtocolMessage(successEvent, &error);
    QVERIFY2(!successWire.isEmpty(), qPrintable(error));
    socket.write(successWire);
    QVERIFY(socket.waitForBytesWritten(3000));

    QTRY_COMPARE(storage.artifactCount(task.id, &error), 1);
    aitrain::TaskSnapshot stored;
    QVERIFY2(storage.task(task.id, &stored, &error), qPrintable(error));
    QCOMPARE(stored.state, aitrain::TaskState::Succeeded);
    const QDir bundleRoot(directory.filePath(QStringLiteral("store/committed")));
    const QStringList bundles = bundleRoot.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
    QCOMPARE(bundles.size(), 1);
    const QDir bundle(bundleRoot.filePath(bundles.constFirst()));
    QVERIFY(QFileInfo::exists(bundle.filePath(QStringLiteral("report/adapter-report.json"))));
    QVERIFY(QFileInfo::exists(bundle.filePath(QStringLiteral("candidates.json"))));
#else
    QSKIP(" Adapter Host integration uses Windows Job Object.");
#endif
}

void ApplicationTests::adapterArtifactCandidateRejectsOutsideRoot()
{
#ifdef Q_OS_WIN
    QTemporaryDir directory;
    QTemporaryDir outside;
    QVERIFY(directory.isValid());
    QVERIFY(outside.isValid());
    const QString candidatePath = outside.filePath(QStringLiteral("outside.json"));
    QVERIFY(writeFile(candidatePath, QByteArrayLiteral("{}")));

    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    aitrain::TaskCoordinator coordinator(&storage);
    aitrain::ArtifactStore artifacts(directory.filePath(QStringLiteral("store")));
    aitrain::TaskExecutionHost host(&coordinator, &artifacts);
    aitrain::PythonAdapterLaunch launch;
    launch.program = QStringLiteral("cmd.exe");
    launch.arguments = QStringList{QStringLiteral("/c"), QStringLiteral("ping 127.0.0.1 -n 2 > nul")};
    launch.artifactCandidateRoots = QStringList{directory.path()};
    aitrain::TaskSnapshot task;
    QVERIFY2(host.start(QStringLiteral("yolo.detect"), QStringLiteral("training"), launch, &task, &error), qPrintable(error));

    const aitrain::AdapterEventEndpoint endpoint = host.adapterEndpoint();
    QTcpSocket socket;
    socket.connectToHost(endpoint.host, endpoint.port);
    QVERIFY(socket.waitForConnected(3000));
    socket.write(QByteArrayLiteral("{\"channel\":\"aitrain.adapter\",\"token\":\"")
        + endpoint.token.toUtf8() + QByteArrayLiteral("\"}\n"));
    QTRY_VERIFY(socket.bytesAvailable() > 0);
    QCOMPARE(QJsonDocument::fromJson(socket.readLine()).object().value(QStringLiteral("status")).toString(), QStringLiteral("accepted"));

    aitrain::ProtocolEnvelope candidate;
    candidate.messageId = aitrain::MessageId::create();
    candidate.requestId = task.requestId;
    candidate.taskId = task.id;
    candidate.sequence = 1;
    candidate.kind = QStringLiteral("event.artifact_candidate");
    candidate.timestamp = QDateTime::currentDateTimeUtc();
    candidate.payload = QJsonObject{{QStringLiteral("kind"), QStringLiteral("report")}, {QStringLiteral("path"), candidatePath}};
    const QByteArray wire = aitrain::encodeProtocolMessage(candidate, &error);
    QVERIFY2(!wire.isEmpty(), qPrintable(error));
    socket.write(wire);
    QVERIFY(socket.waitForBytesWritten(3000));

    QTRY_VERIFY(!host.isRunning());
    aitrain::TaskSnapshot stored;
    QVERIFY2(storage.task(task.id, &stored, &error), qPrintable(error));
    QCOMPARE(stored.state, aitrain::TaskState::Failed);
    QCOMPARE(storage.artifactCount(task.id, &error), 0);
#else
    QSKIP(" Adapter Host integration uses Windows Job Object.");
#endif
}

void ApplicationTests::adapterHostDelegatesExistingTaskTerminalToWorkflowOwner()
{
#ifdef Q_OS_WIN
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString candidatePath = directory.filePath(QStringLiteral("workflow-result.json"));
    QFile candidate(candidatePath);
    QVERIFY(candidate.open(QIODevice::WriteOnly));
    QCOMPARE(candidate.write("{}"), qint64(2));
    candidate.close();

    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    aitrain::TaskCoordinator coordinator(&storage);
    aitrain::TaskSnapshot task;
    QVERIFY2(coordinator.createAndStartTask(QStringLiteral("yolo.detect"), QStringLiteral("training"), &task, &error), qPrintable(error));
    aitrain::ArtifactStore artifacts(directory.filePath(QStringLiteral("store")));
    aitrain::TaskExecutionHost host(&coordinator, &artifacts);
    aitrain::PythonAdapterLaunch launch;
    launch.program = QStringLiteral("cmd.exe");
    launch.arguments = QStringList{QStringLiteral("/c"), QStringLiteral("ping 127.0.0.1 -n 3 > nul")};
    launch.artifactCandidateRoots = QStringList{directory.path()};

    bool callbackCalled = false;
    aitrain::ArtifactId callbackArtifact;
    QStringList forwardedKinds;
    QVERIFY2(host.startExistingTask(task, launch,
        [&callbackCalled, &callbackArtifact](const aitrain::ProtocolEnvelope& terminal,
            const aitrain::ArtifactId& outputArtifactId, QString* callbackError) {
            if (terminal.kind != QStringLiteral("event.succeeded") || !outputArtifactId.isValid()) {
                if (callbackError) *callbackError = QStringLiteral("托管终态不包含已提交输出 Artifact。");
                return false;
            }
            callbackCalled = true;
            callbackArtifact = outputArtifactId;
            return true;
        }, &error, {}, [&forwardedKinds](const aitrain::ProtocolEnvelope& event) {
            forwardedKinds.append(event.kind);
        }), qPrintable(error));

    const aitrain::AdapterEventEndpoint endpoint = host.adapterEndpoint();
    QTcpSocket socket;
    socket.connectToHost(endpoint.host, endpoint.port);
    QVERIFY(socket.waitForConnected(3000));
    socket.write(QByteArrayLiteral("{\"channel\":\"aitrain.adapter\",\"token\":\"") + endpoint.token.toUtf8() + QByteArrayLiteral("\"}\n"));
    QTRY_VERIFY(socket.bytesAvailable() > 0);
    QCOMPARE(QJsonDocument::fromJson(socket.readLine()).object().value(QStringLiteral("status")).toString(), QStringLiteral("accepted"));

    aitrain::ProtocolEnvelope artifact;
    artifact.messageId = aitrain::MessageId::create();
    artifact.requestId = task.requestId;
    artifact.taskId = task.id;
    artifact.sequence = 1;
    artifact.kind = QStringLiteral("event.artifact_candidate");
    artifact.timestamp = QDateTime::currentDateTimeUtc();
    artifact.payload = QJsonObject{{QStringLiteral("kind"), QStringLiteral("training_result")}, {QStringLiteral("path"), candidatePath}};
    const QByteArray artifactWire = aitrain::encodeProtocolMessage(artifact, &error);
    QVERIFY2(!artifactWire.isEmpty(), qPrintable(error));
    socket.write(artifactWire);
    aitrain::ProtocolEnvelope succeeded = artifact;
    succeeded.messageId = aitrain::MessageId::create();
    succeeded.sequence = 2;
    succeeded.kind = QStringLiteral("event.succeeded");
    succeeded.payload = {};
    const QByteArray successWire = aitrain::encodeProtocolMessage(succeeded, &error);
    QVERIFY2(!successWire.isEmpty(), qPrintable(error));
    socket.write(successWire);
    QVERIFY(socket.waitForBytesWritten(3000));
    QTRY_VERIFY(callbackCalled);
    QVERIFY(callbackArtifact.isValid());
    QTRY_COMPARE(forwardedKinds, QStringList({QStringLiteral("event.artifact_candidate"), QStringLiteral("event.succeeded")}));

    aitrain::TaskSnapshot stillRunning;
    QVERIFY2(storage.task(task.id, &stillRunning, &error), qPrintable(error));
    QCOMPARE(stillRunning.state, aitrain::TaskState::Running);
    QCOMPARE(storage.artifactCount(task.id, &error), 1);
    QVERIFY2(coordinator.finalizeTask(task.id, aitrain::TaskState::Succeeded, {}, &error), qPrintable(error));
    QTRY_VERIFY(!host.isRunning());
#else
    QSKIP(" Adapter Host integration uses Windows Job Object.");
#endif
}

void ApplicationTests::adapterHostSyntheticTerminalUsesPersistedSequenceOffset()
{
#ifdef Q_OS_WIN
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    aitrain::TaskCoordinator coordinator(&storage);
    aitrain::TaskSnapshot task;
    QVERIFY2(coordinator.createAndStartTask(QStringLiteral("yolo.detect"), QStringLiteral("training"), &task, &error), qPrintable(error));

    aitrain::ProtocolEnvelope progress;
    progress.messageId = aitrain::MessageId::create();
    progress.requestId = task.requestId;
    progress.taskId = task.id;
    progress.sequence = 7;
    progress.kind = QStringLiteral("event.progress");
    progress.timestamp = QDateTime::currentDateTimeUtc();
    progress.payload = QJsonObject{{QStringLiteral("percent"), 40}};
    QVERIFY2(coordinator.consumeWorkerEvent(progress, &error), qPrintable(error));
    quint64 lastSequence = 0;
    QVERIFY2(storage.lastProtocolSequence(task.id, &lastSequence, &error), qPrintable(error));
    QCOMPARE(lastSequence, quint64(7));

    aitrain::ArtifactStore artifacts(directory.filePath(QStringLiteral("store")));
    aitrain::TaskExecutionHost host(&coordinator, &artifacts);
    aitrain::PythonAdapterLaunch launch;
    launch.program = directory.filePath(QStringLiteral("missing-adapter.exe"));

    bool terminalCalled = false;
    quint64 terminalSequence = 0;
    QString terminalKind;
    QVERIFY2(host.startExistingTask(task, launch,
        [&terminalCalled, &terminalSequence, &terminalKind](const aitrain::ProtocolEnvelope& terminal,
            const aitrain::ArtifactId& outputArtifactId, QString*) {
            terminalCalled = true;
            terminalSequence = terminal.sequence;
            terminalKind = terminal.kind;
            return !outputArtifactId.isValid();
        }, &error), qPrintable(error));

    QTRY_VERIFY_WITH_TIMEOUT(terminalCalled, 5000);
    QCOMPARE(terminalKind, QStringLiteral("event.failed"));
    QCOMPARE(terminalSequence, quint64(8));
    QVERIFY2(storage.lastProtocolSequence(task.id, &lastSequence, &error), qPrintable(error));
    QCOMPARE(lastSequence, quint64(8));
#else
    QSKIP(" Adapter Host integration uses Windows Job Object.");
#endif
}

void ApplicationTests::importCreatesVerifiedModelPackageWithoutGuessingType()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString sourcePath = directory.filePath(QStringLiteral("external.onnx"));
    QFile source(sourcePath);
    QVERIFY(source.open(QIODevice::WriteOnly));
    QVERIFY(source.write("externally-confirmed-model") > 0);
    source.close();
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    aitrain::TaskCoordinator coordinator(&storage);
    aitrain::ArtifactStore artifacts(directory.filePath(QStringLiteral("store")));
    aitrain::ModelImportService importer(&coordinator, &artifacts);
    aitrain::ModelImportRequest request;
    request.taskId = aitrain::TaskId::create();
    request.sourceFilePath = sourcePath;
    request.manifest.modelPackageId = aitrain::ModelPackageId::create();
    request.manifest.modelFamily = QStringLiteral("yolo_detection");
    request.manifest.taskType = QStringLiteral("detection");
    request.manifest.sourceBackend = QStringLiteral("external_confirmed_import");
    request.manifest.sourceSnapshotId = aitrain::SnapshotId::create();
    request.manifest.artifactEntryPath = QStringLiteral("model/external.onnx");
    request.manifest.inputs = {{QStringLiteral("images"), QStringLiteral("NCHW"), {1, 3, 640, 640}}};
    request.manifest.outputs = {{QStringLiteral("output0"), QStringLiteral("NCN"), {1, 84, -1}}};
    request.manifest.preprocessing = {{QStringLiteral("id"), QStringLiteral("letterbox_rgb_0_1")}};
    request.manifest.postprocessing = {{QStringLiteral("id"), QStringLiteral("yolo_detection_nms")}};
    request.manifest.decoder = QStringLiteral("yolo_detection_v8");
    request.manifest.classNames.append(QStringLiteral("part"));
    request.manifest.opset = 17;
    request.manifest.exporterVersion = QStringLiteral("external-confirmed");
    request.manifest.runtimeRoutes.append(QStringLiteral("aitrain_onnxruntime"));
    request.manifest.verified = true;
    aitrain::ModelImportResult imported;
    QVERIFY2(importer.importModel(request, &imported, &error), qPrintable(error));
    QCOMPARE(imported.task.state, aitrain::TaskState::Succeeded);
    QCOMPARE(imported.task.id, request.taskId);
    QCOMPARE(imported.modelPackage.manifest.sourceTaskId, imported.task.id);
    QVERIFY(QFileInfo::exists(QDir(imported.artifactPath).filePath(QStringLiteral("model/external.onnx"))));
    aitrain::ModelPackageSnapshot loaded;
    QVERIFY2(storage.modelPackage(request.manifest.modelPackageId, &loaded, &error), qPrintable(error));
    QCOMPARE(loaded.manifest.modelFamily, QStringLiteral("yolo_detection"));
}

void ApplicationTests::importCancellationLeavesNoRegisteredPackageOrArtifact()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString sourcePath = directory.filePath(QStringLiteral("large-external.onnx"));
    QFile source(sourcePath);
    QVERIFY(source.open(QIODevice::WriteOnly));
    QVERIFY(source.write(QByteArray(3 * 1024 * 1024, 'm')) == 3 * 1024 * 1024);
    source.close();

    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    aitrain::TaskCoordinator coordinator(&storage);
    aitrain::ArtifactStore artifacts(directory.filePath(QStringLiteral("store")));
    aitrain::ModelImportService importer(&coordinator, &artifacts);
    aitrain::ModelImportRequest request;
    request.taskId = aitrain::TaskId::create();
    request.sourceFilePath = sourcePath;
    request.manifest.modelPackageId = aitrain::ModelPackageId::create();
    request.manifest.modelFamily = QStringLiteral("yolo_detection");
    request.manifest.taskType = QStringLiteral("detection");
    request.manifest.sourceBackend = QStringLiteral("external_confirmed_import");
    request.manifest.sourceSnapshotId = aitrain::SnapshotId::create();
    request.manifest.artifactEntryPath = QStringLiteral("model/large-external.onnx");
    request.manifest.inputs = {{QStringLiteral("images"), QStringLiteral("NCHW"), {1, 3, 640, 640}}};
    request.manifest.outputs = {{QStringLiteral("output0"), QStringLiteral("NCN"), {1, 84, -1}}};
    request.manifest.preprocessing = {{QStringLiteral("id"), QStringLiteral("letterbox_rgb_0_1")}};
    request.manifest.postprocessing = {{QStringLiteral("id"), QStringLiteral("yolo_detection_nms")}};
    request.manifest.decoder = QStringLiteral("yolo_detection_v8");
    request.manifest.classNames.append(QStringLiteral("part"));
    request.manifest.opset = 17;
    request.manifest.exporterVersion = QStringLiteral("external-confirmed");
    request.manifest.runtimeRoutes.append(QStringLiteral("aitrain_onnxruntime"));
    request.manifest.verified = true;

    int cancellationChecks = 0;
    aitrain::ModelImportResult imported;
    QVERIFY(!importer.importModel(request, &imported, &error, [&cancellationChecks]() {
        // 前五次检查覆盖开始/复制，随后在 Artifact 二次哈希阶段取消。
        return ++cancellationChecks >= 7;
    }));
    QVERIFY(error.contains(QStringLiteral("取消")));
    QCOMPARE(imported.task.state, aitrain::TaskState::Canceled);
    QCOMPARE(imported.task.id, request.taskId);
    aitrain::TaskSnapshot stored;
    QVERIFY2(storage.task(imported.task.id, &stored, &error), qPrintable(error));
    QCOMPARE(stored.state, aitrain::TaskState::Canceled);
    QCOMPARE(stored.failure.code, aitrain::FailureCode::Canceled);
    QCOMPARE(storage.artifactCount(imported.task.id, &error), 0);
    QCOMPARE(storage.modelPackages(10, &error).size(), 0);
    const QDir stagingRoot(directory.filePath(QStringLiteral("store/.staging")));
    QCOMPARE(stagingRoot.entryList(QDir::Dirs | QDir::NoDotAndDotDot).size(), 0);
}

void ApplicationTests::runtimeResolutionAcceptsOnlyRegisteredUntamperedModelPackages()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString sourcePath = directory.filePath(QStringLiteral("external.onnx"));
    QFile source(sourcePath);
    QVERIFY(source.open(QIODevice::WriteOnly));
    QVERIFY(source.write("model-package-runtime") > 0);
    source.close();
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    aitrain::TaskCoordinator coordinator(&storage);
    aitrain::ArtifactStore artifacts(directory.filePath(QStringLiteral("store")));
    aitrain::ModelImportService importer(&coordinator, &artifacts);
    aitrain::ModelImportRequest request;
    request.sourceFilePath = sourcePath;
    request.manifest.modelPackageId = aitrain::ModelPackageId::create();
    request.manifest.modelFamily = QStringLiteral("yolo_detection");
    request.manifest.taskType = QStringLiteral("detection");
    request.manifest.sourceBackend = QStringLiteral("external_confirmed_import");
    request.manifest.sourceSnapshotId = aitrain::SnapshotId::create();
    request.manifest.artifactEntryPath = QStringLiteral("model/model.onnx");
    request.manifest.inputs = {{QStringLiteral("images"), QStringLiteral("NCHW"), {1, 3, 640, 640}}};
    request.manifest.outputs = {{QStringLiteral("output0"), QStringLiteral("NCN"), {1, 84, -1}}};
    request.manifest.preprocessing = {{QStringLiteral("id"), QStringLiteral("letterbox_rgb_0_1")}};
    request.manifest.postprocessing = {{QStringLiteral("id"), QStringLiteral("yolo_detection_nms")}};
    request.manifest.decoder = QStringLiteral("yolo_detection_v8");
    request.manifest.classNames.append(QStringLiteral("part"));
    request.manifest.opset = 17;
    request.manifest.exporterVersion = QStringLiteral("external-confirmed");
    request.manifest.runtimeRoutes.append(QStringLiteral("aitrain_onnxruntime"));
    request.manifest.verified = true;
    aitrain::ModelImportResult imported;
    QVERIFY2(importer.importModel(request, &imported, &error), qPrintable(error));
    aitrain::ModelPackageRuntimeService service(&storage, artifacts.rootPath());
    aitrain::RuntimeModelLocation location;
    aitrain::RuntimeCapability capability;
    QVERIFY2(service.resolve(request.manifest.modelPackageId, QStringLiteral("aitrain_onnxruntime"), &location, &capability, &error), qPrintable(error));
    QCOMPARE(capability.status, aitrain::RuntimeCapabilityStatus::Supported);
    QVERIFY(QFile::remove(QDir(location.artifactDirectory).filePath(QStringLiteral("model/model.onnx"))));
    QVERIFY(!service.resolve(request.manifest.modelPackageId, QStringLiteral("aitrain_onnxruntime"), &location, &capability, &error));
    QVERIFY(error.contains(QStringLiteral("入口")));
}

void ApplicationTests::projectWorkspaceOwnsRuntimeTaskLifecycle()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY2(workspace.open(directory.filePath(QStringLiteral("project")), &error), qPrintable(error));

    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::TaskSnapshot started;
    QVERIFY2(workspace.startTask(taskId,
        QStringLiteral("runtime.aitrain_onnxruntime"),
        QStringLiteral("deployment_validation"),
        &started,
        &error), qPrintable(error));
    QCOMPARE(started.id, taskId);
    QCOMPARE(started.state, aitrain::TaskState::Running);

    QVERIFY2(workspace.requestTaskCancellation(taskId, &error), qPrintable(error));
    const aitrain::Failure cancellation{
        aitrain::FailureCode::Canceled,
        QStringLiteral("用户取消"),
        QStringLiteral("确认任务已停止后重新运行。"),
        QDateTime::currentDateTimeUtc()};
    QVERIFY2(workspace.finalizeTask(taskId, aitrain::TaskState::Canceled, cancellation, &error), qPrintable(error));

    aitrain::ProjectStore storage;
    QVERIFY2(storage.open(QDir(workspace.workspacePath()).filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    aitrain::TaskSnapshot persisted;
    QVERIFY2(storage.task(taskId, &persisted, &error), qPrintable(error));
    QCOMPARE(persisted.state, aitrain::TaskState::Canceled);
    QCOMPARE(persisted.failure.code, aitrain::FailureCode::Canceled);
}

void ApplicationTests::projectWorkspaceFirstStartCreatesStableLayout()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString projectRoot = QDir(directory.path()).filePath(QStringLiteral("first-start"));
    QVERIFY(!QFileInfo::exists(projectRoot));

    QString error;
    aitrain::ProjectWorkspace firstWorkspace;
    QVERIFY2(firstWorkspace.open(projectRoot, &error), qPrintable(error));
    QCOMPARE(firstWorkspace.workspacePath(), QDir(projectRoot).filePath(QStringLiteral(".aitrain")));
    firstWorkspace.close();

    const QString metadataRoot = QDir(projectRoot).filePath(QStringLiteral(".aitrain"));
    const QString artifactRoot = QDir(metadataRoot).filePath(QStringLiteral("artifacts"));
    QVERIFY(QFileInfo::exists(QDir(metadataRoot).filePath(QStringLiteral("project.sqlite"))));
    QVERIFY(QDir(artifactRoot).exists());
    QVERIFY(QDir(QDir(artifactRoot).filePath(QStringLiteral(".staging"))).exists());
    QVERIFY(QDir(QDir(artifactRoot).filePath(QStringLiteral(".staging-meta"))).exists());
    QVERIFY(QDir(QDir(artifactRoot).filePath(QStringLiteral("committed"))).exists());
    QVERIFY(QDir(QDir(metadataRoot).filePath(QStringLiteral(".runtime-staging"))).exists());

    const QString connectionName = QStringLiteral("application_first_start_%1")
        .arg(QUuid::createUuid().toString(QUuid::Id128));
    {
        QSqlDatabase database = QSqlDatabase::addDatabase(QStringLiteral("QSQLITE"), connectionName);
        database.setDatabaseName(QDir(metadataRoot).filePath(QStringLiteral("project.sqlite")));
        QVERIFY2(database.open(), qPrintable(database.lastError().text()));
        QSqlQuery schema(database);
        QVERIFY2(schema.exec(QStringLiteral("select version from schema_info limit 1")),
            qPrintable(schema.lastError().text()));
        QVERIFY(schema.next());
        QCOMPARE(schema.value(0).toInt(), aitrain::ProjectStore::schemaVersion());
        QSqlQuery projects(database);
        QVERIFY2(projects.exec(QStringLiteral(
            "select 1 from sqlite_master where type = 'table' and name = 'projects'")),
            qPrintable(projects.lastError().text()));
        QVERIFY(!projects.next());
        database.close();
    }
    QSqlDatabase::removeDatabase(connectionName);

    aitrain::ProjectWorkspace secondWorkspace;
    QVERIFY2(secondWorkspace.open(projectRoot, &error), qPrintable(error));
    secondWorkspace.close();
    QVERIFY(QDir(QDir(artifactRoot).filePath(QStringLiteral(".staging")))
        .entryInfoList(QDir::NoDotAndDotDot | QDir::AllEntries).isEmpty());
    QVERIFY(QDir(QDir(artifactRoot).filePath(QStringLiteral(".staging-meta")))
        .entryInfoList(QDir::NoDotAndDotDot | QDir::AllEntries).isEmpty());
    QVERIFY(QDir(QDir(metadataRoot).filePath(QStringLiteral(".runtime-staging")))
        .entryInfoList(QDir::NoDotAndDotDot | QDir::AllEntries).isEmpty());
}

void ApplicationTests::projectQueryServiceReadsOnlyPersistedTaskState()
{
    QTemporaryDir project;
    QVERIFY(project.isValid());
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY2(workspace.open(project.path(), &error), qPrintable(error));

    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::TaskSnapshot started;
    QVERIFY2(workspace.startTask(taskId, QStringLiteral("yolo.detect"),
        QStringLiteral("training"), &started, &error), qPrintable(error));

    aitrain::ProjectQueryService queries(&workspace);
    const QVector<aitrain::TaskSnapshot> tasks = queries.recentTasks(20, &error);
    QCOMPARE(tasks.size(), 1);
    QCOMPARE(tasks.first().id, taskId);
    QCOMPARE(tasks.first().state, aitrain::TaskState::Running);

    aitrain::TaskReadModel details;
    QVERIFY2(queries.taskDetails(taskId, &details, &error), qPrintable(error));
    QCOMPARE(details.task.id, taskId);
    QVERIFY(details.artifacts.isEmpty());
    QVERIFY(details.metrics.isEmpty());
    QVERIFY(details.workflows.isEmpty());

    QVERIFY2(workspace.requestTaskCancellation(taskId, &error), qPrintable(error));
    QVERIFY2(workspace.finalizeTask(taskId, aitrain::TaskState::Canceled,
        aitrain::Failure{aitrain::FailureCode::Canceled,
            QStringLiteral("测试取消"), QStringLiteral("无需操作"), QDateTime::currentDateTimeUtc()},
        &error), qPrintable(error));
    QVERIFY2(queries.taskDetails(taskId, &details, &error), qPrintable(error));
    QCOMPARE(details.task.state, aitrain::TaskState::Canceled);
    QCOMPARE(details.task.failure.code, aitrain::FailureCode::Canceled);
}

void ApplicationTests::externalAcceptanceEvidenceRequiresStrictSchemaAndStaysUnverified()
{
    QTemporaryDir project;
    QTemporaryDir external;
    QVERIFY(project.isValid());
    QVERIFY(external.isValid());
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY2(workspace.open(project.path(), &error), qPrintable(error));
    const QString sourcePath = QDir(external.path()).filePath(QStringLiteral("acceptance.json"));
    QVERIFY(writeFile(sourcePath, QJsonDocument(QJsonObject{
        {QStringLiteral("schemaVersion"), 1},
        {QStringLiteral("kind"), QStringLiteral("aitrain_external_acceptance_evidence")},
        {QStringLiteral("evidenceKind"), QStringLiteral("clean_windows")},
        {QStringLiteral("status"), QStringLiteral("passed")},
        {QStringLiteral("producer"), QStringLiteral("qa-lab")},
        {QStringLiteral("observedAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs)},
        {QStringLiteral("limitations"), QJsonArray{QStringLiteral("外部证据")}}}).toJson(QJsonDocument::Compact)));

    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::TaskSnapshot task;
    QVERIFY2(workspace.startTask(taskId, QStringLiteral("delivery.external_acceptance"),
        QStringLiteral("external_acceptance_evidence"), &task, &error), qPrintable(error));
    aitrain::ExternalAcceptanceEvidenceImportResult imported;
    QVERIFY2(workspace.importExternalAcceptanceEvidence(taskId,
        aitrain::ExternalAcceptanceEvidenceImportRequest{sourcePath}, &imported, &error), qPrintable(error));
    QVERIFY(imported.evidenceArtifactId.isValid());
    QCOMPARE(imported.status, QStringLiteral("passed"));
    QVERIFY2(workspace.finalizeTask(taskId, aitrain::TaskState::Succeeded, {}, &error), qPrintable(error));

    aitrain::ProjectQueryService queries(&workspace);
    const QVector<aitrain::DeliveryEvidenceReadModel> evidence = queries.deliveryEvidence(20, &error);
    QVERIFY2(error.isEmpty(), qPrintable(error));
    QCOMPARE(evidence.size(), 1);
    QCOMPARE(evidence.first().evidenceKind, QStringLiteral("clean_windows"));
    QCOMPARE(evidence.first().runtimeStatus, QStringLiteral("passed"));
    QVERIFY(!evidence.first().verified);

    QVERIFY(writeFile(sourcePath, QJsonDocument(QJsonObject{
        {QStringLiteral("schemaVersion"), 1},
        {QStringLiteral("kind"), QStringLiteral("aitrain_external_acceptance_evidence")},
        {QStringLiteral("evidenceKind"), QStringLiteral("tampered")},
        {QStringLiteral("status"), QStringLiteral("passed")},
        {QStringLiteral("producer"), QStringLiteral("qa-lab")},
        {QStringLiteral("observedAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs)},
        {QStringLiteral("unexpected"), true}}).toJson(QJsonDocument::Compact)));
    const aitrain::TaskId rejectedTaskId = aitrain::TaskId::create();
    QVERIFY2(workspace.startTask(rejectedTaskId, QStringLiteral("delivery.external_acceptance"),
        QStringLiteral("external_acceptance_evidence"), &task, &error), qPrintable(error));
    aitrain::ExternalAcceptanceEvidenceImportResult rejected;
    QVERIFY(!workspace.importExternalAcceptanceEvidence(rejectedTaskId,
        aitrain::ExternalAcceptanceEvidenceImportRequest{sourcePath}, &rejected, &error));
    QVERIFY(error.contains(QStringLiteral("未知字段")));
    const aitrain::Failure rejectedFailure{
        aitrain::FailureCode::InvalidRequest, error, QStringLiteral("修正 schema"),
        QDateTime::currentDateTimeUtc()};
    QVERIFY2(workspace.finalizeTask(rejectedTaskId, aitrain::TaskState::Failed,
        rejectedFailure, &error), qPrintable(error));
}

void ApplicationTests::deliveryEvidenceLimitCountsEvidenceArtifacts()
{
    QTemporaryDir project;
    QTemporaryDir external;
    QVERIFY(project.isValid());
    QVERIFY(external.isValid());
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY2(workspace.open(project.path(), &error), qPrintable(error));

    const QString sourcePath = QDir(external.path()).filePath(QStringLiteral("acceptance.json"));
    QVERIFY(writeFile(sourcePath, QJsonDocument(QJsonObject{
        {QStringLiteral("schemaVersion"), 1},
        {QStringLiteral("kind"), QStringLiteral("aitrain_external_acceptance_evidence")},
        {QStringLiteral("evidenceKind"), QStringLiteral("clean_windows")},
        {QStringLiteral("status"), QStringLiteral("passed")},
        {QStringLiteral("producer"), QStringLiteral("qa-lab")},
        {QStringLiteral("observedAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs)}})
        .toJson(QJsonDocument::Compact)));

    const aitrain::TaskId evidenceTaskId = aitrain::TaskId::create();
    aitrain::TaskSnapshot task;
    QVERIFY2(workspace.startTask(evidenceTaskId, QStringLiteral("delivery.external_acceptance"),
        QStringLiteral("external_acceptance_evidence"), &task, &error), qPrintable(error));
    aitrain::ExternalAcceptanceEvidenceImportResult imported;
    QVERIFY2(workspace.importExternalAcceptanceEvidence(evidenceTaskId,
        aitrain::ExternalAcceptanceEvidenceImportRequest{sourcePath}, &imported, &error),
        qPrintable(error));
    QVERIFY2(workspace.finalizeTask(evidenceTaskId, aitrain::TaskState::Succeeded, {}, &error),
        qPrintable(error));

    // 确保后续无 Artifact 任务在任务时间上更新，旧实现会错误地只扫描这些任务。
    QTest::qWait(5);
    for (int index = 0; index < 3; ++index) {
        const aitrain::TaskId noiseTaskId = aitrain::TaskId::create();
        QVERIFY2(workspace.startTask(noiseTaskId, QStringLiteral("diagnostic.noise"),
            QStringLiteral("diagnostics"), &task, &error), qPrintable(error));
        QVERIFY2(workspace.requestTaskCancellation(noiseTaskId, &error), qPrintable(error));
        QVERIFY2(workspace.finalizeTask(noiseTaskId, aitrain::TaskState::Canceled,
            aitrain::Failure{aitrain::FailureCode::Canceled, QStringLiteral("测试取消"),
                QStringLiteral("无需操作"), QDateTime::currentDateTimeUtc()}, &error),
            qPrintable(error));
    }

    aitrain::ProjectQueryService queries(&workspace);
    const QVector<aitrain::DeliveryEvidenceReadModel> evidence = queries.deliveryEvidence(1, &error);
    QVERIFY2(error.isEmpty(), qPrintable(error));
    QCOMPARE(evidence.size(), 1);
    QCOMPARE(evidence.first().evidenceArtifactId, imported.evidenceArtifactId);
}

void ApplicationTests::deliveryEvidenceKeepsInvalidArtifactAsRow()
{
    QTemporaryDir project;
    QVERIFY(project.isValid());
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY2(workspace.open(project.path(), &error), qPrintable(error));

    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::TaskSnapshot task;
    QVERIFY2(workspace.startTask(taskId, QStringLiteral("delivery.external_acceptance"),
        QStringLiteral("external_acceptance_evidence"), &task, &error), qPrintable(error));
    const QString staging = workspace.runtimeStagingPath(taskId);
    QVERIFY(QDir().mkpath(staging));
    const QString sourcePath = QDir(staging).filePath(QStringLiteral("acceptance.json"));
    QVERIFY(writeFile(sourcePath, QByteArrayLiteral("{}")));
    aitrain::RuntimeArtifactBundle committed;
    QVERIFY2(workspace.commitRuntimeArtifacts(taskId,
        QStringLiteral("external_acceptance_evidence"),
        {{QStringLiteral("acceptance"), sourcePath}}, &committed, &error), qPrintable(error));
    QVERIFY2(workspace.finalizeTask(taskId, aitrain::TaskState::Succeeded, {}, &error),
        qPrintable(error));

    aitrain::ProjectQueryService queries(&workspace);
    const QVector<aitrain::DeliveryEvidenceReadModel> evidence = queries.deliveryEvidence(10, &error);
    QVERIFY2(error.isEmpty(), qPrintable(error));
    QCOMPARE(evidence.size(), 1);
    QVERIFY(!evidence.first().valid);
    QCOMPARE(evidence.first().validationFailure.code, aitrain::FailureCode::ArtifactIncomplete);
    QCOMPARE(evidence.first().runtimeStatus, QStringLiteral("invalid"));
}

void ApplicationTests::projectWorkspaceCommitsRuntimeArtifactsBeforeSuccess()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY2(workspace.open(directory.filePath(QStringLiteral("project")), &error), qPrintable(error));

    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::TaskSnapshot task;
    QVERIFY2(workspace.startTask(taskId,
        QStringLiteral("runtime.aitrain_onnxruntime"), QStringLiteral("inference"), &task, &error), qPrintable(error));
    const QString runtimeStaging = workspace.runtimeStagingPath(taskId);
    QVERIFY(QDir().mkpath(runtimeStaging));
    const QString predictionPath = QDir(runtimeStaging).filePath(QStringLiteral("predictions.json"));
    QFile prediction(predictionPath);
    QVERIFY(prediction.open(QIODevice::WriteOnly));
    QVERIFY(prediction.write("{\"detections\":[]}" ) > 0);
    prediction.close();

    aitrain::RuntimeArtifactBundle bundle;
    QVERIFY2(workspace.commitRuntimeArtifacts(taskId,
        QStringLiteral("runtime_output_bundle"),
        {{QStringLiteral("inference_predictions"), predictionPath}},
        &bundle,
        &error), qPrintable(error));
    QVERIFY(bundle.artifactId.isValid());
    QVERIFY(QFileInfo::exists(bundle.pathsByKind.value(QStringLiteral("inference_predictions"))));
    QVERIFY2(workspace.cleanupRuntimeStaging(taskId, &error), qPrintable(error));
    QVERIFY(!QFileInfo::exists(runtimeStaging));
    QVERIFY2(workspace.finalizeTask(taskId, aitrain::TaskState::Succeeded, {}, &error), qPrintable(error));

    aitrain::ProjectStore storage;
    QVERIFY2(storage.open(QDir(workspace.workspacePath()).filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    QCOMPARE(storage.artifactCount(taskId, &error), 1);
    QCOMPARE(storage.artifactFileCount(bundle.artifactId, &error), 1);
}

void ApplicationTests::datasetSnapshotImportRegistersNewAndExistingDatasetVersions()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString sourceRoot = directory.filePath(QStringLiteral("外部 数据集"));
    QVERIFY(createYoloImportFixture(sourceRoot, QByteArray("0 0.5 0.5 0.25 0.25\n")));
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY2(workspace.open(directory.filePath(QStringLiteral("project")), &error), qPrintable(error));
    const aitrain::DatasetId datasetId = aitrain::DatasetId::create();

    const auto runImport = [&](aitrain::DatasetSnapshotImportWorkflowResult* result) {
        const aitrain::TaskId taskId = aitrain::TaskId::create();
        aitrain::TaskSnapshot task;
        if (!workspace.startTask(taskId, QStringLiteral("dataset.snapshot.import"),
                QStringLiteral("dataset_snapshot_import"), &task, &error)) return false;
        aitrain::DatasetSnapshotImportWorkflowRequest request;
        request.sourcePath = sourceRoot;
        request.sourceFormat = QStringLiteral("yolo_detection");
        request.targetDatasetId = datasetId;
        request.targetDatasetName = QStringLiteral("审计名称");
        return workspace.runDatasetSnapshotImportWorkflow(taskId, request, result, &error);
    };

    aitrain::DatasetSnapshotImportWorkflowResult first;
    QVERIFY2(runImport(&first), qPrintable(error));
    QCOMPARE(first.terminalState, aitrain::TaskState::Succeeded);
    QCOMPARE(first.datasetSnapshot.datasetId, datasetId);
    QVERIFY(first.datasetSnapshot.datasetVersionId.isValid());
    QVERIFY(first.evidenceArtifactId.isValid());
    QVERIFY(QFileInfo(QDir(first.datasetSnapshot.rootPath).filePath(
        QStringLiteral("images/train/a.png"))).isFile());
    QVERIFY(QFileInfo(QDir(first.datasetSnapshot.rootPath).filePath(
        QStringLiteral("dataset_snapshot.json"))).isFile());
    QVERIFY(first.datasetSnapshot.rootPath.contains(first.datasetSnapshot.artifactId.toString()));

    QVERIFY(writeFile(QDir(sourceRoot).filePath(QStringLiteral("labels/train/a.txt")),
        QByteArray("0 0.4 0.4 0.20 0.20\n")));
    aitrain::DatasetSnapshotImportWorkflowResult second;
    QVERIFY2(runImport(&second), qPrintable(error));
    QCOMPARE(second.terminalState, aitrain::TaskState::Succeeded);
    QCOMPARE(second.datasetSnapshot.datasetId, datasetId);
    QVERIFY(second.datasetSnapshot.datasetVersionId != first.datasetSnapshot.datasetVersionId);
    QVERIFY(second.datasetSnapshot.id != first.datasetSnapshot.id);
    QVERIFY(second.datasetSnapshot.rootPath != first.datasetSnapshot.rootPath);

    aitrain::ProjectStore storage;
    QVERIFY2(storage.open(QDir(workspace.workspacePath()).filePath(
        QStringLiteral("project.sqlite")), &error), qPrintable(error));
    aitrain::DatasetSnapshotRecord loadedFirst;
    aitrain::DatasetSnapshotRecord loadedSecond;
    QVERIFY2(storage.datasetSnapshot(first.datasetSnapshot.id, &loadedFirst, &error), qPrintable(error));
    QVERIFY2(storage.datasetSnapshot(second.datasetSnapshot.id, &loadedSecond, &error), qPrintable(error));
    QCOMPARE(loadedFirst.rootPath, first.datasetSnapshot.rootPath);
    QCOMPARE(loadedSecond.rootPath, second.datasetSnapshot.rootPath);
}

void ApplicationTests::datasetSnapshotImportRejectsSourceChangedAfterPlanWithEvidence()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString sourceRoot = directory.filePath(QStringLiteral("source"));
    QVERIFY(createYoloImportFixture(sourceRoot, QByteArray("0 0.5 0.5 0.25 0.25\n")));
    const QString projectRoot = directory.filePath(QStringLiteral("project"));
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY2(workspace.open(projectRoot, &error), qPrintable(error));
    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::TaskSnapshot task;
    QVERIFY2(workspace.startTask(taskId, QStringLiteral("dataset.snapshot.import"),
        QStringLiteral("dataset_snapshot_import"), &task, &error), qPrintable(error));
    aitrain::DatasetSnapshotImportWorkflowRequest request;
    request.sourcePath = sourceRoot;
    request.sourceFormat = QStringLiteral("yolo_detection");
    request.targetDatasetId = aitrain::DatasetId::create();
    request.targetDatasetName = QStringLiteral("source-change");
    bool mutated = false;
    const auto cancellation = [&]() {
        if (!mutated) {
            QDirIterator iterator(QDir(projectRoot).filePath(
                QStringLiteral(".aitrain/artifacts/committed")),
                QStringList{QStringLiteral("snapshot_import_plan.json")}, QDir::Files,
                QDirIterator::Subdirectories);
            if (iterator.hasNext()) {
                iterator.next();
                mutated = writeFile(QDir(sourceRoot).filePath(
                    QStringLiteral("labels/train/a.txt")),
                    QByteArray("0 0.3 0.3 0.10 0.10\n"));
            }
        }
        return false;
    };
    aitrain::DatasetSnapshotImportWorkflowResult result;
    QVERIFY2(workspace.runDatasetSnapshotImportWorkflow(
        taskId, request, &result, &error, cancellation), qPrintable(error));
    QVERIFY(mutated);
    QCOMPARE(result.terminalState, aitrain::TaskState::Failed);
    QVERIFY(result.importPlanArtifactId.isValid());
    QVERIFY(!result.datasetSnapshot.id.isValid());
    QVERIFY(result.evidenceArtifactId.isValid());
    QCOMPARE(result.failure.code, aitrain::FailureCode::ArtifactIncompatible);
}

void ApplicationTests::datasetSplitWorkflowRegistersSelfContainedSnapshotAndRejectsIdentityMismatch()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString sourceRoot = directory.filePath(QStringLiteral("source"));
    QVERIFY(createYoloImportFixture(sourceRoot, QByteArray("0 0.5 0.5 0.25 0.25\n")));
    const QString projectRoot = directory.filePath(QStringLiteral("project"));
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY2(workspace.open(projectRoot, &error), qPrintable(error));

    const aitrain::TaskId importTaskId = aitrain::TaskId::create();
    aitrain::TaskSnapshot task;
    QVERIFY2(workspace.startTask(importTaskId, QStringLiteral("dataset.snapshot.import"),
        QStringLiteral("dataset_snapshot_import"), &task, &error), qPrintable(error));
    aitrain::DatasetSnapshotImportWorkflowRequest importRequest;
    importRequest.sourcePath = sourceRoot;
    importRequest.sourceFormat = QStringLiteral("yolo_detection");
    importRequest.targetDatasetId = aitrain::DatasetId::create();
    importRequest.targetDatasetName = QStringLiteral("源数据集");
    aitrain::DatasetSnapshotImportWorkflowResult imported;
    QVERIFY2(workspace.runDatasetSnapshotImportWorkflow(importTaskId, importRequest,
        &imported, &error), qPrintable(error));
    QCOMPARE(imported.terminalState, aitrain::TaskState::Succeeded);

    aitrain::DatasetSplitWorkflowRequest request;
    request.sourceDatasetId = imported.datasetSnapshot.datasetId;
    request.sourceDatasetVersionId = imported.datasetSnapshot.datasetVersionId;
    request.sourceSnapshotId = imported.datasetSnapshot.id;
    request.sourceSnapshotArtifactId = imported.datasetSnapshot.artifactId;
    request.targetDatasetId = aitrain::DatasetId::create();
    request.targetDatasetName = QStringLiteral("划分目标");
    request.options = QJsonObject{{QStringLiteral("trainRatio"), 0.5},
        {QStringLiteral("valRatio"), 0.5}, {QStringLiteral("testRatio"), 0.0},
        {QStringLiteral("seed"), 42}};
    const aitrain::TaskId splitTaskId = aitrain::TaskId::create();
    QVERIFY2(workspace.startTask(splitTaskId, QStringLiteral("dataset.split"),
        QStringLiteral("dataset_split"), &task, &error), qPrintable(error));
    aitrain::DatasetSplitWorkflowResult split;
    QVERIFY2(workspace.runDatasetSplitWorkflow(splitTaskId, request, &split, &error), qPrintable(error));
    QCOMPARE(split.terminalState, aitrain::TaskState::Succeeded);
    QVERIFY(split.splitPlanArtifactId.isValid());
    QVERIFY(split.splitArtifactId.isValid());
    QVERIFY(split.datasetSnapshot.id.isValid());
    QCOMPARE(split.datasetSnapshot.datasetId, request.targetDatasetId);
    QVERIFY(QFileInfo(QDir(split.datasetSnapshot.rootPath).filePath(
        QStringLiteral("images/train/a.png"))).isFile()
        || QFileInfo(QDir(split.datasetSnapshot.rootPath).filePath(
            QStringLiteral("images/val/a.png"))).isFile());
    QVERIFY(QFileInfo(QDir(split.datasetSnapshot.rootPath).filePath(
        QStringLiteral("dataset_snapshot.json"))).isFile());
    QVERIFY(!QFileInfo(QDir(split.datasetSnapshot.rootPath).filePath(
        QStringLiteral("split_plan.json"))).exists());

    QDirIterator planIterator(QDir(projectRoot).filePath(
        QStringLiteral(".aitrain/artifacts/committed/%1")
            .arg(split.splitPlanArtifactId.toString())),
        QStringList{QStringLiteral("dataset_split_plan.json")}, QDir::Files);
    QVERIFY(planIterator.hasNext());
    QFile planFile(planIterator.next());
    QVERIFY(planFile.open(QIODevice::ReadOnly));
    const QByteArray planBytes = planFile.readAll();
    QVERIFY(!planBytes.contains(sourceRoot.toUtf8()));
    QVERIFY(!planBytes.contains("sourcePath"));
    QVERIFY(!planBytes.contains("outputPath"));

    const aitrain::TaskId mismatchTaskId = aitrain::TaskId::create();
    QVERIFY2(workspace.startTask(mismatchTaskId, QStringLiteral("dataset.split"),
        QStringLiteral("dataset_split"), &task, &error), qPrintable(error));
    request.sourceDatasetId = aitrain::DatasetId::create();
    aitrain::DatasetSplitWorkflowResult mismatch;
    QVERIFY2(workspace.runDatasetSplitWorkflow(mismatchTaskId, request, &mismatch, &error), qPrintable(error));
    QCOMPARE(mismatch.terminalState, aitrain::TaskState::Failed);
    QVERIFY(!mismatch.datasetSnapshot.id.isValid());
    QVERIFY(mismatch.evidenceArtifactId.isValid());
    QCOMPARE(mismatch.failure.code, aitrain::FailureCode::ArtifactIncompatible);

    request.sourceDatasetId = imported.datasetSnapshot.datasetId;
    QVERIFY(writeFile(QDir(imported.datasetSnapshot.rootPath).filePath(
        QStringLiteral("labels/train/a.txt")), QByteArray("0 0.4 0.4 0.2 0.2\n")));
    const aitrain::TaskId tamperTaskId = aitrain::TaskId::create();
    QVERIFY2(workspace.startTask(tamperTaskId, QStringLiteral("dataset.split"),
        QStringLiteral("dataset_split"), &task, &error), qPrintable(error));
    aitrain::DatasetSplitWorkflowResult tampered;
    QVERIFY2(workspace.runDatasetSplitWorkflow(tamperTaskId, request, &tampered, &error), qPrintable(error));
    QCOMPARE(tampered.terminalState, aitrain::TaskState::Failed);
    QVERIFY(!tampered.datasetSnapshot.id.isValid());
    QVERIFY(tampered.evidenceArtifactId.isValid());
    QCOMPARE(tampered.failure.code, aitrain::FailureCode::ArtifactIncompatible);
}

void ApplicationTests::datasetSplitWorkflowCancellationAfterMaterializeDoesNotRegisterSnapshot()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString sourceRoot = directory.filePath(QStringLiteral("source"));
    QVERIFY(createYoloImportFixture(sourceRoot, QByteArray("0 0.5 0.5 0.25 0.25\n")));
    const QString projectRoot = directory.filePath(QStringLiteral("project"));
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY2(workspace.open(projectRoot, &error), qPrintable(error));
    aitrain::TaskSnapshot task;
    const aitrain::TaskId importTaskId = aitrain::TaskId::create();
    QVERIFY2(workspace.startTask(importTaskId, QStringLiteral("dataset.snapshot.import"),
        QStringLiteral("dataset_snapshot_import"), &task, &error), qPrintable(error));
    aitrain::DatasetSnapshotImportWorkflowRequest importRequest;
    importRequest.sourcePath = sourceRoot;
    importRequest.sourceFormat = QStringLiteral("yolo_detection");
    importRequest.targetDatasetId = aitrain::DatasetId::create();
    importRequest.targetDatasetName = QStringLiteral("源数据集");
    aitrain::DatasetSnapshotImportWorkflowResult imported;
    QVERIFY2(workspace.runDatasetSnapshotImportWorkflow(importTaskId, importRequest,
        &imported, &error), qPrintable(error));

    aitrain::DatasetSplitWorkflowRequest request;
    request.sourceDatasetId = imported.datasetSnapshot.datasetId;
    request.sourceDatasetVersionId = imported.datasetSnapshot.datasetVersionId;
    request.sourceSnapshotId = imported.datasetSnapshot.id;
    request.sourceSnapshotArtifactId = imported.datasetSnapshot.artifactId;
    request.targetDatasetId = aitrain::DatasetId::create();
    request.targetDatasetName = QStringLiteral("取消目标");
    request.options = QJsonObject{{QStringLiteral("trainRatio"), 0.5},
        {QStringLiteral("valRatio"), 0.5}, {QStringLiteral("testRatio"), 0.0},
        {QStringLiteral("seed"), 7}};
    const aitrain::TaskId splitTaskId = aitrain::TaskId::create();
    QVERIFY2(workspace.startTask(splitTaskId, QStringLiteral("dataset.split"),
        QStringLiteral("dataset_split"), &task, &error), qPrintable(error));
    bool materialized = false;
    const auto cancellation = [&]() {
        QDirIterator iterator(QDir(projectRoot).filePath(
            QStringLiteral(".aitrain/artifacts/committed")),
            QStringList{QStringLiteral("split_plan.json")}, QDir::Files,
            QDirIterator::Subdirectories);
        materialized = iterator.hasNext();
        return materialized;
    };
    aitrain::DatasetSplitWorkflowResult result;
    QVERIFY2(workspace.runDatasetSplitWorkflow(splitTaskId, request, &result,
        &error, cancellation), qPrintable(error));
    QVERIFY(materialized);
    QCOMPARE(result.terminalState, aitrain::TaskState::Canceled);
    QVERIFY(result.splitPlanArtifactId.isValid());
    QVERIFY(result.splitArtifactId.isValid());
    QVERIFY(!result.datasetSnapshot.id.isValid());
    QVERIFY(result.evidenceArtifactId.isValid());
}

void ApplicationTests::projectWorkspaceRegistersDatasetSnapshotAndSequencesTrainingWorkflow()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString datasetRoot = directory.filePath(QStringLiteral("dataset"));
    QVERIFY(QDir().mkpath(QDir(datasetRoot).filePath(QStringLiteral("images"))));
    QFile image(QDir(datasetRoot).filePath(QStringLiteral("images/sample.jpg")));
    QVERIFY(image.open(QIODevice::WriteOnly));
    QVERIFY(image.write("not-an-image-but-a-snapshot-fixture") > 0);
    image.close();

    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY2(workspace.open(directory.filePath(QStringLiteral("project")), &error), qPrintable(error));
    const aitrain::TaskId snapshotTaskId = aitrain::TaskId::create();
    aitrain::TaskSnapshot snapshotTask;
    QVERIFY2(workspace.startTask(snapshotTaskId, QStringLiteral("dataset.snapshot"),
        QStringLiteral("detection"), &snapshotTask, &error), qPrintable(error));
    aitrain::DatasetSnapshotCommitRequest snapshotRequest;
    snapshotRequest.datasetRoot = datasetRoot;
    snapshotRequest.datasetFormat = QStringLiteral("yolo_detection");
    snapshotRequest.driverId = QStringLiteral("yolo_detection");
    snapshotRequest.driverVersion = QStringLiteral("2.0");
    snapshotRequest.options.classDefinitions.append(QJsonObject{{QStringLiteral("id"), 0}, {QStringLiteral("name"), QStringLiteral("part")}});
    aitrain::DatasetSnapshotArtifactBundle snapshot;
    QVERIFY2(workspace.commitDatasetSnapshot(snapshotTaskId, snapshotRequest, &snapshot, &error), qPrintable(error));
    QVERIFY2(workspace.finalizeTask(snapshotTaskId, aitrain::TaskState::Succeeded, {}, &error), qPrintable(error));

    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::TaskSnapshot task;
    QVERIFY2(workspace.startTask(taskId, QStringLiteral("yolo.detect"), QStringLiteral("detection"), &task, &error), qPrintable(error));
    QVERIFY(snapshot.snapshot.id.isValid());
    QVERIFY(snapshot.snapshot.datasetId.isValid());
    QVERIFY(snapshot.snapshot.datasetVersionId.isValid());
    QVERIFY(snapshot.snapshot.artifactId.isValid());
    QVERIFY(QFileInfo::exists(snapshot.manifestPath));
    QCOMPARE(snapshot.manifest.value(QStringLiteral("complete")).toBool(), true);
    QCOMPARE(snapshot.manifest.value(QStringLiteral("snapshotId")).toString(), snapshot.snapshot.id.toString());
    const QJsonArray snapshotFiles = snapshot.manifest.value(QStringLiteral("files")).toArray();
    QVERIFY(!snapshotFiles.isEmpty());
    QVERIFY(!snapshotFiles.at(0).toObject().value(QStringLiteral("relativePath")).toString().isEmpty());

    aitrain::TrainingWorkflowRequest workflowRequest;
    workflowRequest.datasetId = snapshot.snapshot.datasetId;
    workflowRequest.datasetVersionId = snapshot.snapshot.datasetVersionId;
    workflowRequest.snapshotId = snapshot.snapshot.id;
    workflowRequest.snapshotArtifactId = snapshot.snapshot.artifactId;
    workflowRequest.templateId = QStringLiteral("official_yolo_training_delivery");
    workflowRequest.trainingBackend = QStringLiteral("ultralytics_yolo_detect");
    workflowRequest.evaluationBackend = QStringLiteral("ultralytics_yolo_eval");
    workflowRequest.exportBackend = QStringLiteral("ultralytics_yolo_export");
    workflowRequest.deploymentBackend = QStringLiteral("aitrain_onnxruntime");
    workflowRequest.parameterSummary = QJsonObject{{QStringLiteral("epochs"), 1}};
    aitrain::ProjectStore rejectionStorage;
    QVERIFY2(rejectionStorage.open(QDir(workspace.workspacePath()).filePath(QStringLiteral("project.sqlite")),
        &error), qPrintable(error));
    const qsizetype artifactsBeforeMismatch = rejectionStorage.artifactsForTask(taskId, &error).size();
    QVERIFY2(error.isEmpty(), qPrintable(error));
    QCOMPARE(artifactsBeforeMismatch, qsizetype(0));
    const auto rejectsMismatchWithoutArtifacts = [&](const aitrain::TrainingWorkflowRequest& rejectedRequest) {
        aitrain::TrainingWorkflowDispatch rejected;
        QString rejectionError;
        if (workspace.beginTrainingWorkflow(taskId, rejectedRequest, &rejected, &rejectionError)
            || rejectionError.isEmpty()) {
            error = QStringLiteral("错配的 snapshot 绑定未被拒绝");
            return false;
        }
        QString storageError;
        const qsizetype artifactCount = rejectionStorage.artifactsForTask(taskId, &storageError).size();
        if (!storageError.isEmpty() || artifactCount != artifactsBeforeMismatch) {
            error = storageError.isEmpty()
                ? QStringLiteral("拒绝错配输入后仍为 Task B 创建了 artifact")
                : storageError;
            return false;
        }
        return true;
    };
    aitrain::TrainingWorkflowRequest mismatched = workflowRequest;
    mismatched.datasetId = aitrain::DatasetId::create();
    QVERIFY2(rejectsMismatchWithoutArtifacts(mismatched), qPrintable(error));
    mismatched = workflowRequest;
    mismatched.datasetVersionId = aitrain::DatasetVersionId::create();
    QVERIFY2(rejectsMismatchWithoutArtifacts(mismatched), qPrintable(error));
    mismatched = workflowRequest;
    mismatched.snapshotId = aitrain::SnapshotId::create();
    QVERIFY2(rejectsMismatchWithoutArtifacts(mismatched), qPrintable(error));
    mismatched = workflowRequest;
    mismatched.snapshotArtifactId = aitrain::ArtifactId::create();
    QVERIFY2(rejectsMismatchWithoutArtifacts(mismatched), qPrintable(error));
    error.clear();
    aitrain::TrainingWorkflowDispatch workflow;
    QVERIFY2(workspace.beginTrainingWorkflow(taskId, workflowRequest, &workflow, &error), qPrintable(error));
    QVERIFY(workflow.dispatch.hasStep);
    QCOMPARE(workflow.dispatch.step.kind, QStringLiteral("Train"));
    QCOMPARE(workflow.dispatch.step.inputArtifactId, snapshot.snapshot.artifactId);
    aitrain::ProjectStore lineageStorage;
    QVERIFY2(lineageStorage.open(QDir(workspace.workspacePath()).filePath(QStringLiteral("project.sqlite")),
        &error), qPrintable(error));
    aitrain::WorkflowInputBinding inputBinding;
    QVERIFY2(lineageStorage.workflowInput(workflow.workflowRunId, QStringLiteral("dataset_snapshot"),
        &inputBinding, &error), qPrintable(error));
    QCOMPARE(inputBinding.sourceTaskId, snapshotTaskId);
    QCOMPARE(inputBinding.sourceArtifactId, snapshot.snapshot.artifactId);
    QCOMPARE(inputBinding.datasetId, snapshot.snapshot.datasetId);
    QCOMPARE(inputBinding.datasetVersionId, snapshot.snapshot.datasetVersionId);
    QCOMPARE(inputBinding.datasetSnapshotId, snapshot.snapshot.id);
    QCOMPARE(inputBinding.manifestSha256, snapshot.snapshot.manifestSha256);
    QCOMPARE(inputBinding.rootHash, snapshot.snapshot.rootHash);
    aitrain::VerifiedTrainingWorkflowInput trainInput;
    QVERIFY2(workspace.resolveTrainingWorkflowStepInput(workflow.workflowRunId, workflow.dispatch.step.id,
        &trainInput, &error), qPrintable(error));
    QCOMPARE(trainInput.artifactId, snapshot.snapshot.artifactId);
    QVERIFY(trainInput.files.size() >= 2);
    const auto manifestFile = std::find_if(trainInput.files.cbegin(), trainInput.files.cend(),
        [](const aitrain::VerifiedWorkflowArtifactFile& file) {
            return file.relativePath == QStringLiteral("dataset_snapshot.json");
        });
    QVERIFY(manifestFile != trainInput.files.cend());
    aitrain::TrainingWorkflowAdapterConfig adapterConfig;
    adapterConfig.pythonProgram = QStringLiteral("cmd.exe");
    adapterConfig.trainersRoot = QDir::current().filePath(QStringLiteral("python_trainers"));
    aitrain::TrainingWorkflowAdapterLaunch adapterLaunch;
    QVERIFY2(workspace.prepareTrainingWorkflowAdapterLaunch(workflow.workflowRunId, workflow.dispatch.step.id,
        adapterConfig, &adapterLaunch, &error), qPrintable(error));
    QCOMPARE(adapterLaunch.launch.arguments.at(0), QDir(adapterConfig.trainersRoot).filePath(QStringLiteral("detection/ultralytics_trainer.py")));
    QCOMPARE(adapterLaunch.request.value(QStringLiteral("datasetPath")).toString(), snapshot.artifactPath);
    QCOMPARE(adapterLaunch.request.value(QStringLiteral("parameters")).toObject().value(QStringLiteral("exportOnnx")).toBool(), false);
    QCOMPARE(adapterLaunch.request.value(QStringLiteral("datasetSnapshotManifest")).toString(), manifestFile->absolutePath);

    const QString stagingRoot = workspace.runtimeStagingPath(taskId);
    QVERIFY(QDir().mkpath(stagingRoot));
    for (int index = 0; workflow.dispatch.hasStep; ++index) {
        if (workflow.dispatch.step.kind == QStringLiteral("RegisterModel")) {
            aitrain::TrainingModelRegistration registration;
            QVERIFY2(workspace.registerTrainingWorkflowModel(workflow.workflowRunId, workflow.dispatch.step.id,
                &registration, &error), qPrintable(error));
            QVERIFY(registration.modelPackage.manifest.modelPackageId.isValid());
            QCOMPARE(registration.modelPackage.manifest.sourceTaskId, taskId);
            QCOMPARE(registration.modelPackage.manifest.sourceSnapshotId, snapshot.snapshot.id);
            QCOMPARE(registration.modelPackage.manifest.classNames, QStringList{QStringLiteral("part")});
            QVERIFY(registration.registrationArtifact.artifactId.isValid());
            aitrain::WorkflowStepExecutionResult execution;
            execution.state = aitrain::WorkflowStepState::Succeeded;
            execution.outputArtifactId = registration.registrationArtifact.artifactId;
            QVERIFY2(workspace.completeTrainingWorkflowStep(workflow.workflowRunId, workflow.dispatch.step.id,
                execution, &workflow, &error), qPrintable(error));
            continue;
        }
        if (workflow.dispatch.step.kind == QStringLiteral("RenderDeliveryReport")) {
            aitrain::RuntimeArtifactBundle report;
            QVERIFY2(workspace.renderTrainingWorkflowDeliveryReport(workflow.workflowRunId, workflow.dispatch.step.id,
                &report, &error), qPrintable(error));
            const QString deliveryReportPath = report.pathsByKind.value(QStringLiteral("delivery_report_json"));
            QVERIFY(QFileInfo(deliveryReportPath).isFile());
            QVERIFY(QFileInfo(report.pathsByKind.value(QStringLiteral("delivery_report_markdown"))).isFile());
            QFile deliveryReportFile(deliveryReportPath);
            QVERIFY(deliveryReportFile.open(QIODevice::ReadOnly));
            const QJsonObject deliveryReport = QJsonDocument::fromJson(
                deliveryReportFile.readAll()).object();
            const QJsonObject deliveryInput = deliveryReport.value(QStringLiteral("externalInput")).toObject();
            QCOMPARE(deliveryInput.value(QStringLiteral("producerTaskId")).toString(), snapshotTaskId.toString());
            QCOMPARE(deliveryInput.value(QStringLiteral("artifactId")).toString(), snapshot.snapshot.artifactId.toString());
            QVERIFY(!QString::fromUtf8(QJsonDocument(deliveryReport).toJson(QJsonDocument::Compact))
                .contains(QDir::fromNativeSeparators(datasetRoot)));
            aitrain::WorkflowStepExecutionResult execution;
            execution.state = aitrain::WorkflowStepState::Succeeded;
            execution.outputArtifactId = report.artifactId;
            QVERIFY2(workspace.completeTrainingWorkflowStep(workflow.workflowRunId, workflow.dispatch.step.id,
                execution, &workflow, &error), qPrintable(error));
            continue;
        }
        if (workflow.dispatch.step.kind == QStringLiteral("DeploymentValidate")) {
            aitrain::TrainingDeploymentInvocation deployment;
            QVERIFY2(workspace.prepareTrainingWorkflowDeploymentInvocation(workflow.workflowRunId, workflow.dispatch.step.id,
                QStringLiteral("images/sample.jpg"), &deployment, &error), qPrintable(error));
            QCOMPARE(deployment.sourceArtifactId, workflow.dispatch.step.inputArtifactId);
            QCOMPARE(deployment.invocation.value(QStringLiteral("runtimeRoute")).toString(), QStringLiteral("aitrain_onnxruntime"));
            QCOMPARE(deployment.invocation.value(QStringLiteral("imagePath")).toString(),
                QDir(snapshot.artifactPath).filePath(QStringLiteral("images/sample.jpg")));
        }
        const bool requiresCheckpoint = index == 0 || index == 1;
        const bool exportStep = workflow.dispatch.step.kind == QStringLiteral("Export");
        const QString candidatePath = QDir(stagingRoot).filePath(requiresCheckpoint
            ? QStringLiteral("best.pt")
            : (exportStep ? QStringLiteral("model.onnx") : QStringLiteral("step-%1.json").arg(index)));
        QFile candidate(candidatePath);
        QVERIFY(candidate.open(QIODevice::WriteOnly));
        QVERIFY(candidate.write("{}") == qint64(2));
        candidate.close();
        QVector<aitrain::RuntimeArtifactCandidate> candidates{{requiresCheckpoint ? QStringLiteral("checkpoint") : (exportStep ? QStringLiteral("export") : QStringLiteral("result")), candidatePath}};
        if (index == 1) {
            const QString evaluationPath = QDir(stagingRoot).filePath(QStringLiteral("evaluation_report.json"));
            QFile evaluation(evaluationPath);
            QVERIFY(evaluation.open(QIODevice::WriteOnly));
            QVERIFY(evaluation.write("{\"runtime\":\"ultralytics_official_val\",\"perClass\":[{\"classId\":0,\"className\":\"part\"}]}") > 0);
            evaluation.close();
            candidates.append({QStringLiteral("evaluation_report"), evaluationPath});
        }
        if (exportStep) {
            const QString sidecarPath = QDir(stagingRoot).filePath(QStringLiteral("model.aitrain-export.json"));
            QFile sidecar(sidecarPath);
            QVERIFY(sidecar.open(QIODevice::WriteOnly));
            const QJsonObject contract{{QStringLiteral("modelFamily"), QStringLiteral("yolo_detection")},
                {QStringLiteral("taskType"), QStringLiteral("detection")},
                {QStringLiteral("inputs"), QJsonArray{QJsonObject{{QStringLiteral("name"), QStringLiteral("images")}, {QStringLiteral("layout"), QStringLiteral("NCHW")}, {QStringLiteral("shape"), QJsonArray{1, 3, 640, 640}}}}},
                {QStringLiteral("outputs"), QJsonArray{QJsonObject{{QStringLiteral("name"), QStringLiteral("output0")}, {QStringLiteral("layout"), QStringLiteral("NCN")}, {QStringLiteral("shape"), QJsonArray{1, 84, -1}}}}},
                {QStringLiteral("preprocessing"), QJsonObject{{QStringLiteral("id"), QStringLiteral("letterbox_rgb_0_1")}}},
                {QStringLiteral("postprocessing"), QJsonObject{{QStringLiteral("id"), QStringLiteral("yolo_detection_nms")}}},
                {QStringLiteral("decoder"), QStringLiteral("yolo_detection_v8")},
                {QStringLiteral("classNames"), QJsonArray{QStringLiteral("part")}},
                {QStringLiteral("runtimeRoutes"), QJsonArray{QStringLiteral("aitrain_onnxruntime")}}};
            const QJsonObject sidecarPayload{{QStringLiteral("backend"), QStringLiteral("ultralytics_yolo_export")},
                {QStringLiteral("format"), QStringLiteral("onnx")}, {QStringLiteral("ultralyticsVersion"), QStringLiteral("test")},
                {QStringLiteral("modelContract"), contract}};
            QVERIFY(sidecar.write(QJsonDocument(sidecarPayload).toJson(QJsonDocument::Compact)) > 0);
            sidecar.close();
            candidates.append({QStringLiteral("export_sidecar"), sidecarPath});
        }
        aitrain::RuntimeArtifactBundle output;
        QVERIFY2(workspace.commitRuntimeArtifacts(taskId, QStringLiteral("training_step_output"),
            candidates, &output, &error), qPrintable(error));
        aitrain::WorkflowStepExecutionResult execution;
        execution.state = aitrain::WorkflowStepState::Succeeded;
        execution.outputArtifactId = output.artifactId;
        QVERIFY2(workspace.completeTrainingWorkflowStep(workflow.workflowRunId, workflow.dispatch.step.id,
            execution, &workflow, &error), qPrintable(error));
        if (workflow.dispatch.hasStep) {
            aitrain::VerifiedTrainingWorkflowInput nextInput;
            QVERIFY2(workspace.resolveTrainingWorkflowStepInput(workflow.workflowRunId, workflow.dispatch.step.id,
                &nextInput, &error), qPrintable(error));
            QCOMPARE(nextInput.artifactId, output.artifactId);
            if (workflow.dispatch.step.kind == QStringLiteral("Evaluate") || workflow.dispatch.step.kind == QStringLiteral("Export")) {
                aitrain::TrainingWorkflowAdapterLaunch downstreamLaunch;
                QVERIFY2(workspace.prepareTrainingWorkflowAdapterLaunch(workflow.workflowRunId, workflow.dispatch.step.id,
                    adapterConfig, &downstreamLaunch, &error), qPrintable(error));
                const QString modelPath = downstreamLaunch.request.value(QStringLiteral("modelPath")).toString();
                QVERIFY(QFileInfo(modelPath).isFile());
                QVERIFY(modelPath.endsWith(QStringLiteral("checkpoint/best.pt"))
                    || modelPath.endsWith(QStringLiteral("checkpoint\\best.pt")));
                if (workflow.dispatch.step.kind == QStringLiteral("Export")) {
                    const QString evaluationReportPath = downstreamLaunch.request.value(QStringLiteral("evaluationReportPath")).toString();
                    QVERIFY(QFileInfo(evaluationReportPath).isFile());
                    QVERIFY(evaluationReportPath.endsWith(QStringLiteral("evaluation_report/evaluation_report.json"))
                        || evaluationReportPath.endsWith(QStringLiteral("evaluation_report\\evaluation_report.json")));
                }
            }
        }
    }
    QCOMPARE(workflow.dispatch.result.state, aitrain::WorkflowStepState::Succeeded);
    QVERIFY2(workspace.cleanupRuntimeStaging(taskId, &error), qPrintable(error));

    aitrain::ProjectStore storage;
    QVERIFY2(storage.open(QDir(workspace.workspacePath()).filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    aitrain::DatasetSnapshotRecord persistedSnapshot;
    QVERIFY2(storage.datasetSnapshot(snapshot.snapshot.id, &persistedSnapshot, &error), qPrintable(error));
    QCOMPARE(persistedSnapshot.artifactId, snapshot.snapshot.artifactId);
    QCOMPARE(persistedSnapshot.rootHash, snapshot.snapshot.rootHash);
    const QVector<aitrain::WorkflowStepSnapshot> steps = storage.workflowSteps(workflow.workflowRunId, &error);
    QCOMPARE(steps.size(), 8);
    QCOMPARE(steps.constFirst().kind, QStringLiteral("ValidateDataset"));
    QCOMPARE(steps.at(1).kind, QStringLiteral("CreateSnapshot"));
    QCOMPARE(steps.at(2).kind, QStringLiteral("Train"));
    QCOMPARE(steps.constLast().kind, QStringLiteral("RenderDeliveryReport"));
    for (int index = 0; index < steps.size(); ++index) {
        const aitrain::WorkflowStepSnapshot& step = steps.at(index);
        QCOMPARE(step.state, aitrain::WorkflowStepState::Succeeded);
        if (index == 0) {
            QVERIFY(!step.inputArtifactId.isValid());
        } else {
            QVERIFY(step.inputArtifactId.isValid());
        }
        QVERIFY(step.outputArtifactId.isValid());
    }
    aitrain::TaskSnapshot completed;
    QVERIFY2(storage.task(taskId, &completed, &error), qPrintable(error));
    QCOMPARE(completed.state, aitrain::TaskState::Succeeded);
    aitrain::EvidenceBundle evidence;
    QVERIFY2(workspace.buildWorkflowEvidenceBundle(workflow.workflowRunId, &evidence, &error), qPrintable(error));
    QCOMPARE(evidence.externalInputs.size(), 1);
    const aitrain::EvidenceExternalInput& external = evidence.externalInputs.constFirst();
    QCOMPARE(external.role, QStringLiteral("dataset_snapshot"));
    QCOMPARE(external.producerTaskId, snapshotTaskId);
    QCOMPARE(external.artifactId, snapshot.snapshot.artifactId);
    QCOMPARE(external.datasetId, snapshot.snapshot.datasetId);
    QCOMPARE(external.datasetVersionId, snapshot.snapshot.datasetVersionId);
    QCOMPARE(external.datasetSnapshotId, snapshot.snapshot.id);
    QCOMPARE(external.manifestSha256, snapshot.snapshot.manifestSha256);
    QCOMPARE(external.rootHash, snapshot.snapshot.rootHash);
    const QJsonObject encodedEvidence = aitrain::encodeEvidenceBundle(evidence, &error);
    QVERIFY2(!encodedEvidence.isEmpty(), qPrintable(error));
    const QJsonArray encodedInputs = encodedEvidence.value(QStringLiteral("externalInputs")).toArray();
    QCOMPARE(encodedInputs.size(), 1);
    const QJsonObject encodedInput = encodedInputs.at(0).toObject();
    QVERIFY(!encodedInput.contains(QStringLiteral("path")));
    QVERIFY(!encodedInput.contains(QStringLiteral("artifactPath")));
    QVERIFY(!encodedInput.contains(QStringLiteral("datasetRoot")));
    const QByteArray encodedInputBytes = QJsonDocument(encodedInput).toJson(QJsonDocument::Compact);
    QVERIFY(!encodedInputBytes.contains(datasetRoot.toUtf8()));
    QVERIFY(!encodedInputBytes.contains(snapshot.artifactPath.toUtf8()));
    aitrain::EvidenceBundle decodedEvidence;
    QVERIFY2(aitrain::decodeEvidenceBundle(encodedEvidence, &decodedEvidence, &error), qPrintable(error));
    QCOMPARE(decodedEvidence.externalInputs.size(), 1);
    QCOMPARE(decodedEvidence.externalInputs.constFirst().datasetId, snapshot.snapshot.datasetId);
}

void ApplicationTests::trainingWorkflowRejectsCrossProfileBackendMix()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY2(workspace.open(directory.filePath(QStringLiteral("project")), &error), qPrintable(error));
    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::TaskSnapshot task;
    QVERIFY2(workspace.startTask(taskId, QStringLiteral("yolo"), QStringLiteral("detection"),
        &task, &error), qPrintable(error));

    aitrain::TrainingWorkflowRequest request;
    request.templateId = QStringLiteral("official_yolo_training_delivery");
    request.trainingBackend = QStringLiteral("ultralytics_yolo_detect");
    request.evaluationBackend = QStringLiteral("smp_semantic_segmentation_eval");
    request.exportBackend = QStringLiteral("ultralytics_yolo_export");
    request.deploymentBackend = QStringLiteral("aitrain_onnxruntime");
    aitrain::TrainingWorkflowDispatch workflow;
    QVERIFY(!workspace.beginTrainingWorkflow(taskId, request, &workflow, &error));
    QVERIFY(error.contains(QStringLiteral("Profile")));
}

void ApplicationTests::trainingWorkflowEvidenceGatePersistsEvidenceBeforeTerminalTask()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());

    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY2(workspace.open(directory.filePath(QStringLiteral("project")), &error), qPrintable(error));

    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::TaskSnapshot started;
    QVERIFY2(workspace.startTask(taskId, QStringLiteral("yolo.detect"), QStringLiteral("detection"),
        &started, &error), qPrintable(error));

    aitrain::TrainingWorkflowRequest request;
    request.templateId = QStringLiteral("official_yolo_training_delivery");
    request.trainingBackend = QStringLiteral("ultralytics_yolo_detect");
    request.evaluationBackend = QStringLiteral("ultralytics_yolo_eval");
    request.exportBackend = QStringLiteral("ultralytics_yolo_export");
    request.deploymentBackend = QStringLiteral("aitrain_onnxruntime");
    request.requireEvidenceBeforeTerminal = true;
    QVERIFY2(attachTrainingSnapshotFixture(workspace, taskId,
        directory.filePath(QStringLiteral("evidence-dataset")), QStringLiteral("yolo_detection"),
        &request, &error), qPrintable(error));

    aitrain::TrainingWorkflowDispatch workflow;
    QVERIFY2(workspace.beginTrainingWorkflow(taskId, request, &workflow, &error), qPrintable(error));
    QCOMPARE(workflow.dispatch.step.kind, QStringLiteral("Train"));

    aitrain::WorkflowStepExecutionResult failed;
    failed.state = aitrain::WorkflowStepState::Failed;
    failed.failure = {aitrain::FailureCode::InvalidDataset,
        QStringLiteral("测试数据集预检失败。"), QStringLiteral("修复数据集后重试。"),
        QDateTime::currentDateTimeUtc()};
    QVERIFY2(workspace.completeTrainingWorkflowStep(workflow.workflowRunId, workflow.dispatch.step.id,
        failed, &workflow, &error), qPrintable(error));
    QVERIFY(!workflow.dispatch.hasStep);
    QCOMPARE(workflow.dispatch.result.state, aitrain::WorkflowStepState::Failed);

    aitrain::ProjectStore storage;
    QVERIFY2(storage.open(QDir(workspace.workspacePath()).filePath(QStringLiteral("project.sqlite")),
        &error), qPrintable(error));
    aitrain::TaskSnapshot persisted;
    QVERIFY2(storage.task(taskId, &persisted, &error), qPrintable(error));
    QCOMPARE(persisted.state, aitrain::TaskState::Running);
    QVERIFY(!workspace.finalizeTask(taskId, aitrain::TaskState::Failed, failed.failure, &error));
    QVERIFY(error.contains(QStringLiteral("evidence_required")));
    error.clear();

    aitrain::EvidenceBundle evidence;
    aitrain::EvidenceArtifactBundle committed;
    QVERIFY2(workspace.buildWorkflowEvidenceBundle(workflow.workflowRunId, &evidence, &error), qPrintable(error));
    QCOMPARE(evidence.task.state, aitrain::TaskState::Failed);
    QVERIFY2(workspace.commitEvidenceBundle(evidence, &committed, &error), qPrintable(error));
    QVERIFY(committed.artifactId.isValid());

    QVERIFY2(storage.task(taskId, &persisted, &error), qPrintable(error));
    QCOMPARE(persisted.state, aitrain::TaskState::Running);
    QVERIFY2(workspace.closeWorkflowTerminalization(workflow.workflowRunId, &error), qPrintable(error));
    QVERIFY2(storage.task(taskId, &persisted, &error), qPrintable(error));
    QCOMPARE(persisted.state, aitrain::TaskState::Failed);
    QCOMPARE(persisted.failure.code, aitrain::FailureCode::InvalidDataset);
}

void ApplicationTests::trainingWorkflowEvidenceGateRecoversAcrossReopen()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString projectRoot = directory.filePath(QStringLiteral("project"));
    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::WorkflowRunId workflowRunId;
    QString error;
    {
        aitrain::ProjectWorkspace workspace;
        QVERIFY2(workspace.open(projectRoot, &error), qPrintable(error));
        aitrain::TaskSnapshot started;
        QVERIFY2(workspace.startTask(taskId, QStringLiteral("yolo.detect"), QStringLiteral("detection"),
            &started, &error), qPrintable(error));
        aitrain::TrainingWorkflowRequest request;
        request.templateId = QStringLiteral("official_yolo_training_delivery");
        request.trainingBackend = QStringLiteral("ultralytics_yolo_detect");
        request.evaluationBackend = QStringLiteral("ultralytics_yolo_eval");
        request.exportBackend = QStringLiteral("ultralytics_yolo_export");
        request.deploymentBackend = QStringLiteral("aitrain_onnxruntime");
        request.requireEvidenceBeforeTerminal = true;
        QVERIFY2(attachTrainingSnapshotFixture(workspace, taskId,
            directory.filePath(QStringLiteral("recovery-dataset")), QStringLiteral("yolo_detection"),
            &request, &error), qPrintable(error));
        aitrain::TrainingWorkflowDispatch workflow;
        QVERIFY2(workspace.beginTrainingWorkflow(taskId, request, &workflow, &error), qPrintable(error));
        workflowRunId = workflow.workflowRunId;
        aitrain::WorkflowStepExecutionResult failed;
        failed.state = aitrain::WorkflowStepState::Failed;
        failed.failure = {aitrain::FailureCode::InvalidDataset,
            QStringLiteral("重启恢复夹具中的数据集预检失败。"), QStringLiteral("修复数据集后重试。"),
            QDateTime::currentDateTimeUtc()};
        QVERIFY2(workspace.completeTrainingWorkflowStep(workflow.workflowRunId, workflow.dispatch.step.id,
            failed, &workflow, &error), qPrintable(error));
        workspace.close();
    }
    {
        aitrain::ProjectWorkspace recovered;
        QVERIFY2(recovered.open(projectRoot, &error), qPrintable(error));
        aitrain::ProjectStore storage;
        QVERIFY2(storage.open(QDir(recovered.workspacePath()).filePath(QStringLiteral("project.sqlite")),
            &error), qPrintable(error));
        aitrain::TaskSnapshot task;
        QVERIFY2(storage.task(taskId, &task, &error), qPrintable(error));
        QCOMPARE(task.state, aitrain::TaskState::Failed);
        aitrain::WorkflowTerminalizationSnapshot terminalization;
        QVERIFY2(storage.workflowTerminalization(workflowRunId, &terminalization, &error), qPrintable(error));
        QCOMPARE(terminalization.state, aitrain::WorkflowTerminalizationState::Closed);
        QVERIFY(terminalization.evidenceArtifactId.isValid());
        aitrain::ArtifactSnapshot evidence;
        QVERIFY2(storage.artifact(terminalization.evidenceArtifactId, &evidence, &error), qPrintable(error));
        QCOMPARE(evidence.kind, QStringLiteral("evidence_bundle"));
        QCOMPARE(evidence.files.size(), 4);
    }
}

void ApplicationTests::officialYoloWorkflowPreservesVariantTaskType_data()
{
    QTest::addColumn<QString>("trainingBackend");
    QTest::addColumn<QString>("datasetFormat");
    QTest::addColumn<QString>("rootTaskType");
    QTest::addColumn<QString>("adapterTaskType");
    QTest::addColumn<QString>("trainerScript");

    QTest::newRow("segmentation")
        << QStringLiteral("ultralytics_yolo_segment")
        << QStringLiteral("yolo_segmentation")
        << QStringLiteral("segmentation")
        << QStringLiteral("segmentation")
        << QStringLiteral("segmentation/ultralytics_trainer.py");
    QTest::newRow("obb")
        << QStringLiteral("ultralytics_yolo_obb")
        << QStringLiteral("yolo_obb")
        << QStringLiteral("obb_detection")
        << QStringLiteral("obb_detection")
        << QStringLiteral("obb/ultralytics_trainer.py");
}

void ApplicationTests::officialYoloWorkflowPreservesVariantTaskType()
{
    QFETCH(QString, trainingBackend);
    QFETCH(QString, datasetFormat);
    QFETCH(QString, rootTaskType);
    QFETCH(QString, adapterTaskType);
    QFETCH(QString, trainerScript);

    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString datasetRoot = directory.filePath(QStringLiteral("dataset"));
    QVERIFY(QDir().mkpath(datasetRoot));
    QFile sample(QDir(datasetRoot).filePath(QStringLiteral("sample.txt")));
    QVERIFY(sample.open(QIODevice::WriteOnly));
    QVERIFY(sample.write("variant-workflow-snapshot") > 0);
    sample.close();

    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY2(workspace.open(directory.filePath(QStringLiteral("project")), &error), qPrintable(error));
    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::TaskSnapshot task;
    QVERIFY2(workspace.startTask(taskId, QStringLiteral("yolo"), rootTaskType, &task, &error), qPrintable(error));

    aitrain::DatasetSnapshotCommitRequest snapshotRequest;
    snapshotRequest.datasetRoot = datasetRoot;
    snapshotRequest.datasetFormat = datasetFormat;
    snapshotRequest.driverId = datasetFormat;
    snapshotRequest.driverVersion = QStringLiteral("2.0");
    aitrain::DatasetSnapshotArtifactBundle snapshot;
    QVERIFY2(workspace.commitDatasetSnapshot(taskId, snapshotRequest, &snapshot, &error), qPrintable(error));

    aitrain::TrainingWorkflowRequest request;
    request.datasetId = snapshot.snapshot.datasetId;
    request.datasetVersionId = snapshot.snapshot.datasetVersionId;
    request.snapshotId = snapshot.snapshot.id;
    request.snapshotArtifactId = snapshot.snapshot.artifactId;
    request.templateId = QStringLiteral("official_yolo_training_delivery");
    request.trainingBackend = trainingBackend;
    request.evaluationBackend = QStringLiteral("ultralytics_yolo_eval");
    request.exportBackend = QStringLiteral("ultralytics_yolo_export");
    request.deploymentBackend = QStringLiteral("aitrain_onnxruntime");
    request.parameterSummary = QJsonObject{{QStringLiteral("trainingBackend"), trainingBackend}};
    aitrain::TrainingWorkflowDispatch workflow;
    QVERIFY2(workspace.beginTrainingWorkflow(taskId, request, &workflow, &error), qPrintable(error));

    aitrain::TrainingWorkflowAdapterConfig adapterConfig;
    adapterConfig.pythonProgram = QStringLiteral("cmd.exe");
    adapterConfig.trainersRoot = QDir::current().filePath(QStringLiteral("python_trainers"));
    aitrain::TrainingWorkflowAdapterLaunch launch;
    QVERIFY2(workspace.prepareTrainingWorkflowAdapterLaunch(workflow.workflowRunId, workflow.dispatch.step.id,
        adapterConfig, &launch, &error), qPrintable(error));
    QCOMPARE(launch.request.value(QStringLiteral("taskType")).toString(), adapterTaskType);
    QCOMPARE(launch.launch.arguments.constFirst(), QDir(adapterConfig.trainersRoot).filePath(trainerScript));

    const QString stagingRoot = workspace.runtimeStagingPath(taskId);
    QVERIFY(QDir().mkpath(stagingRoot));
    const auto commitStepOutput = [&](const QString& suffix, bool evaluation,
                                      aitrain::RuntimeArtifactBundle* output) {
        const QString sourceRoot = QDir(stagingRoot).filePath(suffix);
        if (!QDir().mkpath(sourceRoot)) return false;
        const QString checkpointPath = QDir(sourceRoot).filePath(QStringLiteral("best.pt"));
        QFile checkpoint(checkpointPath);
        if (!checkpoint.open(QIODevice::WriteOnly) || checkpoint.write("checkpoint") <= 0) return false;
        checkpoint.close();
        QVector<aitrain::RuntimeArtifactCandidate> candidates{
            {QStringLiteral("checkpoint"), checkpointPath}};
        if (evaluation) {
            const QString reportPath = QDir(sourceRoot).filePath(QStringLiteral("evaluation_report.json"));
            QFile report(reportPath);
            if (!report.open(QIODevice::WriteOnly) || report.write("{\"runtime\":\"ultralytics_official_val\"}") <= 0) return false;
            report.close();
            candidates.append({QStringLiteral("evaluation_report"), reportPath});
        }
        return workspace.commitRuntimeArtifacts(taskId, QStringLiteral("training_step_output"),
            candidates, output, &error);
    };

    aitrain::RuntimeArtifactBundle trainOutput;
    QVERIFY2(commitStepOutput(QStringLiteral("train"), false, &trainOutput), qPrintable(error));
    aitrain::WorkflowStepExecutionResult execution;
    execution.state = aitrain::WorkflowStepState::Succeeded;
    execution.outputArtifactId = trainOutput.artifactId;
    QVERIFY2(workspace.completeTrainingWorkflowStep(workflow.workflowRunId, workflow.dispatch.step.id,
        execution, &workflow, &error), qPrintable(error));
    QCOMPARE(workflow.dispatch.step.kind, QStringLiteral("Evaluate"));
    QVERIFY2(workspace.prepareTrainingWorkflowAdapterLaunch(workflow.workflowRunId, workflow.dispatch.step.id,
        adapterConfig, &launch, &error), qPrintable(error));
    QCOMPARE(launch.request.value(QStringLiteral("taskType")).toString(), adapterTaskType);

    aitrain::RuntimeArtifactBundle evaluationOutput;
    QVERIFY2(commitStepOutput(QStringLiteral("evaluation"), true, &evaluationOutput), qPrintable(error));
    execution.outputArtifactId = evaluationOutput.artifactId;
    QVERIFY2(workspace.completeTrainingWorkflowStep(workflow.workflowRunId, workflow.dispatch.step.id,
        execution, &workflow, &error), qPrintable(error));
    QCOMPARE(workflow.dispatch.step.kind, QStringLiteral("Export"));
    QVERIFY2(workspace.prepareTrainingWorkflowAdapterLaunch(workflow.workflowRunId, workflow.dispatch.step.id,
        adapterConfig, &launch, &error), qPrintable(error));
    QCOMPARE(launch.request.value(QStringLiteral("taskType")).toString(), adapterTaskType);
    QCOMPARE(launch.request.value(QStringLiteral("parameters")).toObject()
        .value(QStringLiteral("trainingBackend")).toString(), trainingBackend);
}

void ApplicationTests::projectWorkspaceDispatchesOfficialAdapterStepThroughTrainingWorkflow()
{
#ifdef Q_OS_WIN
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString datasetRoot = directory.filePath(QStringLiteral("dataset"));
    QVERIFY(QDir().mkpath(QDir(datasetRoot).filePath(QStringLiteral("images"))));
    QFile image(QDir(datasetRoot).filePath(QStringLiteral("images/sample.jpg")));
    QVERIFY(image.open(QIODevice::WriteOnly));
    QVERIFY(image.write("snapshot-fixture") > 0);
    image.close();
    const QString candidatePath = directory.filePath(QStringLiteral("official-train-report.json"));
    QFile candidate(candidatePath);
    QVERIFY(candidate.open(QIODevice::WriteOnly));
    QCOMPARE(candidate.write("{}"), qint64(2));
    candidate.close();

    aitrain::ProjectWorkspace workspace;
    QString error;
    QVERIFY2(workspace.open(directory.filePath(QStringLiteral("project")), &error), qPrintable(error));
    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::TaskSnapshot task;
    QVERIFY2(workspace.startTask(taskId, QStringLiteral("yolo.detect"), QStringLiteral("detection"), &task, &error), qPrintable(error));
    aitrain::DatasetSnapshotCommitRequest snapshotRequest;
    snapshotRequest.datasetRoot = datasetRoot;
    snapshotRequest.datasetFormat = QStringLiteral("yolo_detection");
    snapshotRequest.driverId = QStringLiteral("yolo_detection");
    snapshotRequest.driverVersion = QStringLiteral("2.0");
    snapshotRequest.options.classDefinitions.append(QJsonObject{{QStringLiteral("id"), 0}, {QStringLiteral("name"), QStringLiteral("part")}});
    aitrain::DatasetSnapshotArtifactBundle snapshot;
    QVERIFY2(workspace.commitDatasetSnapshot(taskId, snapshotRequest, &snapshot, &error), qPrintable(error));
    aitrain::TrainingWorkflowRequest request;
    request.datasetId = snapshot.snapshot.datasetId;
    request.datasetVersionId = snapshot.snapshot.datasetVersionId;
    request.snapshotId = snapshot.snapshot.id;
    request.snapshotArtifactId = snapshot.snapshot.artifactId;
    request.templateId = QStringLiteral("official_yolo_training_delivery");
    request.trainingBackend = QStringLiteral("ultralytics_yolo_detect");
    request.evaluationBackend = QStringLiteral("ultralytics_yolo_eval");
    request.exportBackend = QStringLiteral("ultralytics_yolo_export");
    request.deploymentBackend = QStringLiteral("aitrain_onnxruntime");
    aitrain::TrainingWorkflowDispatch workflow;
    QVERIFY2(workspace.beginTrainingWorkflow(taskId, request, &workflow, &error), qPrintable(error));

    aitrain::PythonAdapterLaunch launch;
    launch.program = QStringLiteral("cmd.exe");
    launch.arguments = QStringList{QStringLiteral("/c"), QStringLiteral("ping 127.0.0.1 -n 3 > nul")};
    launch.artifactCandidateRoots = QStringList{directory.path()};
    bool dispatchedNextStep = false;
    aitrain::TrainingWorkflowDispatch next;
    QVERIFY2(workspace.startTrainingWorkflowAdapterStep(workflow.workflowRunId, workflow.dispatch.step.id, launch,
        [&dispatchedNextStep, &next](const aitrain::TrainingWorkflowDispatch& dispatch) {
            dispatchedNextStep = true;
            next = dispatch;
        }, &error), qPrintable(error));

    const aitrain::AdapterEventEndpoint endpoint = workspace.trainingWorkflowAdapterEndpoint();
    QVERIFY(!endpoint.host.isEmpty());
    QVERIFY(endpoint.port > 0);
    QVERIFY(!endpoint.token.isEmpty());
    QTcpSocket socket;
    socket.connectToHost(endpoint.host, endpoint.port);
    QVERIFY(socket.waitForConnected(3000));
    socket.write(QByteArrayLiteral("{\"channel\":\"aitrain.adapter\",\"token\":\"") + endpoint.token.toUtf8() + QByteArrayLiteral("\"}\n"));
    QTRY_VERIFY(socket.bytesAvailable() > 0);
    QCOMPARE(QJsonDocument::fromJson(socket.readLine()).object().value(QStringLiteral("status")).toString(), QStringLiteral("accepted"));
    aitrain::ProtocolEnvelope artifact;
    artifact.messageId = aitrain::MessageId::create();
    artifact.requestId = task.requestId;
    artifact.taskId = task.id;
    artifact.sequence = 1;
    artifact.kind = QStringLiteral("event.artifact_candidate");
    artifact.timestamp = QDateTime::currentDateTimeUtc();
    artifact.payload = QJsonObject{{QStringLiteral("kind"), QStringLiteral("training_report")}, {QStringLiteral("path"), candidatePath}};
    const QByteArray artifactWire = aitrain::encodeProtocolMessage(artifact, &error);
    QVERIFY2(!artifactWire.isEmpty(), qPrintable(error));
    socket.write(artifactWire);
    aitrain::ProtocolEnvelope succeeded = artifact;
    succeeded.messageId = aitrain::MessageId::create();
    succeeded.sequence = 2;
    succeeded.kind = QStringLiteral("event.succeeded");
    succeeded.payload = {};
    const QByteArray succeededWire = aitrain::encodeProtocolMessage(succeeded, &error);
    QVERIFY2(!succeededWire.isEmpty(), qPrintable(error));
    socket.write(succeededWire);
    QVERIFY(socket.waitForBytesWritten(3000));

    QTRY_VERIFY(dispatchedNextStep);
    QVERIFY(!workspace.isTrainingWorkflowAdapterRunning());
    QVERIFY(next.dispatch.hasStep);
    QCOMPARE(next.dispatch.step.kind, QStringLiteral("Evaluate"));
    QVERIFY(next.dispatch.step.inputArtifactId.isValid());

    aitrain::ProjectStore storage;
    QVERIFY2(storage.open(QDir(workspace.workspacePath()).filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    const QVector<aitrain::WorkflowStepSnapshot> steps = storage.workflowSteps(workflow.workflowRunId, &error);
    QCOMPARE(steps.at(0).state, aitrain::WorkflowStepState::Succeeded);
    QVERIFY(steps.at(0).outputArtifactId.isValid());
    QCOMPARE(steps.at(1).state, aitrain::WorkflowStepState::Succeeded);
    QCOMPARE(steps.at(2).state, aitrain::WorkflowStepState::Succeeded);
    QVERIFY(steps.at(2).outputArtifactId.isValid());
    QCOMPARE(steps.at(3).state, aitrain::WorkflowStepState::Running);
    aitrain::TaskSnapshot persisted;
    QVERIFY2(storage.task(taskId, &persisted, &error), qPrintable(error));
    QCOMPARE(persisted.state, aitrain::TaskState::Running);
#else
    QSKIP(" Adapter Host integration uses Windows Job Object.");
#endif
}

void ApplicationTests::workflowRunnerSequencesCommittedArtifactsAndStopsOnCancellation()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::ProjectStore storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    aitrain::TaskCoordinator coordinator(&storage);
    aitrain::TaskSnapshot rootTask;
    QVERIFY2(coordinator.createAndStartTask(QStringLiteral("workflow"), QStringLiteral("workflow"), &rootTask, &error), qPrintable(error));
    const QVector<aitrain::ArtifactFileSnapshot> files = {
        {QStringLiteral("payload.json"), QString(64, QLatin1Char('a')), 2}};
    const aitrain::ArtifactId firstOutput = aitrain::ArtifactId::create();
    const aitrain::ArtifactId secondOutput = aitrain::ArtifactId::create();
    QVERIFY2(storage.recordArtifactWithFiles(firstOutput, rootTask.id, QStringLiteral("first"), files, QDateTime::currentDateTimeUtc(), &error), qPrintable(error));
    QVERIFY2(storage.recordArtifactWithFiles(secondOutput, rootTask.id, QStringLiteral("second"), files, QDateTime::currentDateTimeUtc(), &error), qPrintable(error));

    aitrain::WorkflowRunSnapshot workflow;
    workflow.id = aitrain::WorkflowRunId::create();
    workflow.taskId = rootTask.id;
    workflow.templateId = QStringLiteral("inference-deployment");
    aitrain::WorkflowStepSnapshot first;
    first.id = aitrain::WorkflowStepId::create();
    first.workflowRunId = workflow.id;
    first.ordinal = 0;
    first.kind = QStringLiteral("RunInferenceSmoke");
    first.backend = QStringLiteral("aitrain_onnxruntime");
    aitrain::WorkflowStepSnapshot second;
    second.id = aitrain::WorkflowStepId::create();
    second.workflowRunId = workflow.id;
    second.ordinal = 1;
    second.kind = QStringLiteral("DeploymentValidate");
    second.backend = QStringLiteral("aitrain_onnxruntime");
    QVERIFY2(storage.createWorkflowRun(workflow, {first, second}, &error), qPrintable(error));

    aitrain::WorkflowRunner runner(&storage);
    aitrain::WorkflowRunExecutionResult result;
    QVERIFY2(runner.run(workflow.id,
        [firstOutput, secondOutput](const aitrain::WorkflowStepSnapshot& step, const aitrain::CancellationCallback&) {
            return aitrain::WorkflowStepExecutionResult{
                aitrain::WorkflowStepState::Succeeded,
                step.ordinal == 0 ? firstOutput : secondOutput,
                {}};
        },
        &result,
        &error), qPrintable(error));
    QCOMPARE(result.state, aitrain::WorkflowStepState::Succeeded);
    QCOMPARE(result.finalOutputArtifactId, secondOutput);
    const QVector<aitrain::WorkflowStepSnapshot> completed = storage.workflowSteps(workflow.id, &error);
    QCOMPARE(completed.at(0).state, aitrain::WorkflowStepState::Succeeded);
    QCOMPARE(completed.at(1).inputArtifactId, firstOutput);
    QCOMPARE(completed.at(1).outputArtifactId, secondOutput);

    aitrain::WorkflowRunSnapshot canceledWorkflow;
    canceledWorkflow.id = aitrain::WorkflowRunId::create();
    canceledWorkflow.taskId = rootTask.id;
    canceledWorkflow.templateId = QStringLiteral("canceled-workflow");
    first.id = aitrain::WorkflowStepId::create();
    first.workflowRunId = canceledWorkflow.id;
    first.ordinal = 0;
    second.id = aitrain::WorkflowStepId::create();
    second.workflowRunId = canceledWorkflow.id;
    second.ordinal = 1;
    QVERIFY2(storage.createWorkflowRun(canceledWorkflow, {first, second}, &error), qPrintable(error));
    int executions = 0;
    QVERIFY2(runner.run(canceledWorkflow.id,
        [&executions](const aitrain::WorkflowStepSnapshot&, const aitrain::CancellationCallback&) {
            ++executions;
            return aitrain::WorkflowStepExecutionResult{};
        },
        &result,
        &error,
        []() { return true; }), qPrintable(error));
    QCOMPARE(executions, 0);
    QCOMPARE(result.state, aitrain::WorkflowStepState::Canceled);
    const QVector<aitrain::WorkflowStepSnapshot> canceled = storage.workflowSteps(canceledWorkflow.id, &error);
    QCOMPARE(canceled.at(0).state, aitrain::WorkflowStepState::Canceled);
    QCOMPARE(canceled.at(1).state, aitrain::WorkflowStepState::Skipped);
}

QTEST_MAIN(ApplicationTests)
#include "tst_application.moc"
