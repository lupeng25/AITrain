#include "aitrain/v2/TaskCoordinator.h"
#include "aitrain/v2/CapabilityPlannerV2.h"
#include "aitrain/v2/EvidenceBundleV2.h"
#include "aitrain/v2/TaskExecutionHostV2.h"
#include "aitrain/v2/ModelImportServiceV2.h"
#include "aitrain/v2/ModelPackageRuntimeServiceV2.h"
#include "aitrain/v2/ProjectWorkspaceV2.h"
#include "aitrain/v2/ProjectQueryServiceV2.h"
#include "aitrain/v2/WorkflowRunnerV2.h"

#include <QDateTime>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QImage>
#include <QTemporaryDir>
#include <QTcpSocket>
#include <QTest>

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
    aitrain::v2::ProjectWorkspaceV2& workspace,
    const aitrain::v2::TaskId& taskId,
    const QString& root,
    const QString& datasetFormat,
    aitrain::v2::TrainingWorkflowRequestV2* request,
    QString* error)
{
    if (!request || !QDir().mkpath(root)
        || !writeFile(QDir(root).filePath(QStringLiteral("fixture.bin")), QByteArrayLiteral("snapshot"))) {
        if (error) *error = QStringLiteral("无法创建训练 Snapshot 测试夹具。");
        return false;
    }
    aitrain::v2::DatasetSnapshotCommitRequestV2 snapshotRequest;
    snapshotRequest.datasetRoot = root;
    snapshotRequest.datasetFormat = datasetFormat;
    snapshotRequest.driverId = datasetFormat;
    snapshotRequest.driverVersion = QStringLiteral("2.0");
    aitrain::v2::DatasetSnapshotArtifactBundleV2 snapshot;
    if (!workspace.commitDatasetSnapshot(taskId, snapshotRequest, &snapshot, error)) return false;
    request->datasetId = snapshot.snapshot.datasetId;
    request->datasetVersionId = snapshot.snapshot.datasetVersionId;
    request->snapshotId = snapshot.snapshot.id;
    request->snapshotArtifactId = snapshot.snapshot.artifactId;
    return true;
}

} // namespace

class V2ApplicationTests : public QObject {
    Q_OBJECT

private slots:
    void fakeWorkerSuccessfulRunPersistsTaskMetricsAndArtifacts();
    void duplicateWorkerMessageIsRejectedWithoutDuplicateData();
    void invalidWorkerPayloadIsRejectedBeforePersistence();
    void capabilityPlannerProducesVerifiableImmutableSummary();
    void cancellationTransitionsThroughCancelRequestedAndTerminalCanceled();
    void adapterHostSynthesizesCanceledTerminalAfterForcedCancellation();
    void adapterArtifactCandidatesCommitAsOneBundleBeforeSuccess();
    void adapterHostDelegatesExistingTaskTerminalToWorkflowOwner();
    void importCreatesVerifiedModelPackageWithoutGuessingType();
    void importCancellationLeavesNoRegisteredPackageOrArtifact();
    void runtimeResolutionAcceptsOnlyRegisteredUntamperedModelPackages();
    void projectWorkspaceOwnsRuntimeTaskLifecycle();
    void projectQueryServiceReadsOnlyPersistedV2TaskState();
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

void V2ApplicationTests::fakeWorkerSuccessfulRunPersistsTaskMetricsAndArtifacts()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::StorageV2 storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    aitrain::v2::TaskCoordinator coordinator(&storage);
    aitrain::v2::TaskSnapshot task;
    QVERIFY2(coordinator.createAndStartTask(QStringLiteral("yolo.detect"), QStringLiteral("training"), &task, &error), qPrintable(error));
    QCOMPARE(task.state, aitrain::v2::TaskState::Running);

    const QVector<aitrain::v2::ProtocolEnvelope> events = aitrain::v2::FakeWorkerV2::successfulRun(task, 5);
    for (const aitrain::v2::ProtocolEnvelope& event : events) {
        QVERIFY2(coordinator.consumeWorkerEvent(event, &error), qPrintable(error));
    }

    aitrain::v2::TaskSnapshot stored;
    QVERIFY2(storage.task(task.id, &stored, &error), qPrintable(error));
    QCOMPARE(stored.state, aitrain::v2::TaskState::Succeeded);
    QCOMPARE(storage.metricCount(task.id, &error), 1);
    QCOMPARE(storage.artifactCount(task.id, &error), 1);
    QCOMPARE(storage.eventCount(task.id, &error), 9);
}

void V2ApplicationTests::duplicateWorkerMessageIsRejectedWithoutDuplicateData()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::StorageV2 storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    aitrain::v2::TaskCoordinator coordinator(&storage);
    aitrain::v2::TaskSnapshot task;
    QVERIFY2(coordinator.createAndStartTask(QStringLiteral("yolo.detect"), QStringLiteral("training"), &task, &error), qPrintable(error));
    const aitrain::v2::ProtocolEnvelope event = aitrain::v2::FakeWorkerV2::successfulRun(task, 5).first();
    QVERIFY2(coordinator.consumeWorkerEvent(event, &error), qPrintable(error));
    QVERIFY(!coordinator.consumeWorkerEvent(event, &error));
    QVERIFY(error.contains(QStringLiteral("duplicate")));
    QCOMPARE(storage.eventCount(task.id, &error), 5);
}

void V2ApplicationTests::invalidWorkerPayloadIsRejectedBeforePersistence()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::StorageV2 storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    aitrain::v2::TaskCoordinator coordinator(&storage);
    aitrain::v2::TaskSnapshot task;
    QVERIFY2(coordinator.createAndStartTask(QStringLiteral("yolo.detect"), QStringLiteral("training"), &task, &error), qPrintable(error));
    aitrain::v2::ProtocolEnvelope invalid = aitrain::v2::FakeWorkerV2::successfulRun(task, 5).at(1);
    invalid.payload.remove(QStringLiteral("name"));
    QVERIFY(!coordinator.consumeWorkerEvent(invalid, &error));
    QVERIFY(error.contains(QStringLiteral("name")));
    QCOMPARE(storage.eventCount(task.id, &error), 4);
    QCOMPARE(storage.metricCount(task.id, &error), 0);
}

void V2ApplicationTests::capabilityPlannerProducesVerifiableImmutableSummary()
{
    aitrain::v2::ExecutionRequestV2 request;
    request.capabilityId = QStringLiteral("semantic_segmentation");
    request.taskType = QStringLiteral("semantic_segmentation");
    request.datasetFormat = QStringLiteral("semantic_segmentation_mask");
    request.trainingBackend = QStringLiteral("smp_semantic_segmentation");
    request.evaluationBackend = QStringLiteral("smp_semantic_segmentation");
    request.exportFormat = QStringLiteral("onnx");
    request.runtimeRoute = QStringLiteral("aitrain_onnxruntime");

    aitrain::v2::CapabilityPlannerV2 planner;
    aitrain::v2::ExecutionPlanV2 plan;
    QString error;
    QVERIFY2(planner.plan(request, &plan, &error), qPrintable(error));
    QCOMPARE(plan.runtimeRoute, QStringLiteral("aitrain_onnxruntime"));
    QVERIFY(plan.summaryHash.size() == 64);
    QVERIFY2(planner.verify(request, plan.summaryHash, nullptr, &error), qPrintable(error));

    request.exportFormat = QStringLiteral("tensorrt");
    QVERIFY(!planner.plan(request, &plan, &error));
    QVERIFY(error.contains(QStringLiteral("导出格式")));
}

void V2ApplicationTests::cancellationTransitionsThroughCancelRequestedAndTerminalCanceled()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::StorageV2 storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    aitrain::v2::TaskCoordinator coordinator(&storage);
    aitrain::v2::TaskSnapshot task;
    QVERIFY2(coordinator.createAndStartTask(QStringLiteral("yolo.detect"), QStringLiteral("training"), &task, &error), qPrintable(error));
    QVERIFY2(coordinator.requestCancellation(task.id, &error), qPrintable(error));
    QVERIFY2(coordinator.requestCancellation(task.id, &error), qPrintable(error));

    aitrain::v2::TaskSnapshot cancelRequested;
    QVERIFY2(storage.task(task.id, &cancelRequested, &error), qPrintable(error));
    QCOMPARE(cancelRequested.state, aitrain::v2::TaskState::CancelRequested);
    const QVector<aitrain::v2::ProtocolEnvelope> events = aitrain::v2::FakeWorkerV2::canceledRun(task, 6, QStringLiteral("用户取消"));
    for (const aitrain::v2::ProtocolEnvelope& event : events) {
        QVERIFY2(coordinator.consumeWorkerEvent(event, &error), qPrintable(error));
    }
    aitrain::v2::TaskSnapshot canceled;
    QVERIFY2(storage.task(task.id, &canceled, &error), qPrintable(error));
    QCOMPARE(canceled.state, aitrain::v2::TaskState::Canceled);
    QCOMPARE(canceled.failure.code, aitrain::v2::FailureCode::Canceled);
    QVERIFY(!coordinator.requestCancellation(task.id, &error));
}

void V2ApplicationTests::adapterHostSynthesizesCanceledTerminalAfterForcedCancellation()
{
#ifdef Q_OS_WIN
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::StorageV2 storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    aitrain::v2::TaskCoordinator coordinator(&storage);
    aitrain::v2::ArtifactStoreV2 artifacts(directory.filePath(QStringLiteral("store")));
    aitrain::v2::TaskExecutionHostV2 host(&coordinator, &artifacts);
    aitrain::v2::PythonAdapterLaunchV2 launch;
    launch.program = QStringLiteral("cmd.exe");
    launch.arguments.append(QStringLiteral("/c"));
    launch.arguments.append(QStringLiteral("ping 127.0.0.1 -n 20 > nul"));
    launch.cancellationGraceMs = 50;
    aitrain::v2::TaskSnapshot task;
    QVERIFY2(host.start(QStringLiteral("yolo.detect"), QStringLiteral("training"), launch, &task, &error), qPrintable(error));
    QVERIFY2(host.requestCancellation(task.id, &error), qPrintable(error));
    QTRY_VERIFY_WITH_TIMEOUT(!host.isRunning(), 15000);

    aitrain::v2::TaskSnapshot stored;
    QVERIFY2(storage.task(task.id, &stored, &error), qPrintable(error));
    QVERIFY2(stored.state == aitrain::v2::TaskState::Canceled,
        qPrintable(QStringLiteral("actual state=%1, host error=%2").arg(aitrain::v2::taskStateToString(stored.state), host.lastError())));
    QCOMPARE(stored.failure.code, aitrain::v2::FailureCode::Canceled);
    QVERIFY(host.lastError().isEmpty());
    QCOMPARE(storage.artifactCount(task.id, &error), 0);
    // Created/queued/starting/running, cancel-requested, one protocol terminal,
    // and one terminal state transition. A second synthesized terminal would
    // make this count larger and violate V2-307's single-terminal invariant.
    QCOMPARE(storage.eventCount(task.id, &error), 7);
    const QDir stagingRoot(directory.filePath(QStringLiteral("store/staging")));
    QCOMPARE(stagingRoot.entryList(QDir::Dirs | QDir::NoDotAndDotDot).size(), 0);
#else
    QSKIP("V2 Adapter Host integration uses Windows Job Object.");
#endif
}

void V2ApplicationTests::adapterArtifactCandidatesCommitAsOneBundleBeforeSuccess()
{
#ifdef Q_OS_WIN
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString candidatePath = directory.filePath(QStringLiteral("adapter-report.json"));
    QFile candidate(candidatePath);
    QVERIFY(candidate.open(QIODevice::WriteOnly));
    QCOMPARE(candidate.write("{\"ok\":true}\n"), qint64(12));
    candidate.close();

    aitrain::v2::StorageV2 storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    aitrain::v2::TaskCoordinator coordinator(&storage);
    aitrain::v2::ArtifactStoreV2 artifacts(directory.filePath(QStringLiteral("store")));
    aitrain::v2::TaskExecutionHostV2 host(&coordinator, &artifacts);
    aitrain::v2::PythonAdapterLaunchV2 launch;
    launch.program = QStringLiteral("cmd.exe");
    launch.arguments.append(QStringLiteral("/c"));
    launch.arguments.append(QStringLiteral("ping 127.0.0.1 -n 2 > nul"));
    aitrain::v2::TaskSnapshot task;
    QVERIFY2(host.start(QStringLiteral("yolo.detect"), QStringLiteral("training"), launch, &task, &error), qPrintable(error));

    const aitrain::v2::AdapterEventEndpointV2 endpoint = host.adapterEndpoint();
    QTcpSocket socket;
    socket.connectToHost(endpoint.host, endpoint.port);
    QVERIFY(socket.waitForConnected(3000));
    socket.write(QByteArrayLiteral("{\"channel\":\"aitrain.adapter.v2\",\"token\":\"") + endpoint.token.toUtf8() + QByteArrayLiteral("\"}\n"));
    QTRY_VERIFY(socket.bytesAvailable() > 0);
    QCOMPARE(QJsonDocument::fromJson(socket.readLine()).object().value(QStringLiteral("status")).toString(), QStringLiteral("accepted"));

    aitrain::v2::ProtocolEnvelope candidateEvent;
    candidateEvent.messageId = aitrain::v2::MessageId::create();
    candidateEvent.requestId = task.requestId;
    candidateEvent.taskId = task.id;
    candidateEvent.sequence = 1;
    candidateEvent.kind = QStringLiteral("event.artifact_candidate");
    candidateEvent.timestamp = QDateTime::currentDateTimeUtc();
    candidateEvent.payload = QJsonObject{{QStringLiteral("kind"), QStringLiteral("report")}, {QStringLiteral("path"), candidatePath}};
    const QByteArray candidateWire = aitrain::v2::encodeProtocolV2Message(candidateEvent, &error);
    QVERIFY2(!candidateWire.isEmpty(), qPrintable(error));
    socket.write(candidateWire);

    aitrain::v2::ProtocolEnvelope successEvent = candidateEvent;
    successEvent.messageId = aitrain::v2::MessageId::create();
    successEvent.sequence = 2;
    successEvent.kind = QStringLiteral("event.succeeded");
    successEvent.payload = {};
    const QByteArray successWire = aitrain::v2::encodeProtocolV2Message(successEvent, &error);
    QVERIFY2(!successWire.isEmpty(), qPrintable(error));
    socket.write(successWire);
    QVERIFY(socket.waitForBytesWritten(3000));

    QTRY_COMPARE(storage.artifactCount(task.id, &error), 1);
    aitrain::v2::TaskSnapshot stored;
    QVERIFY2(storage.task(task.id, &stored, &error), qPrintable(error));
    QCOMPARE(stored.state, aitrain::v2::TaskState::Succeeded);
    const QDir bundleRoot(directory.filePath(QStringLiteral("store/artifacts")));
    const QStringList bundles = bundleRoot.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
    QCOMPARE(bundles.size(), 1);
    const QDir bundle(bundleRoot.filePath(bundles.constFirst()));
    QVERIFY(QFileInfo::exists(bundle.filePath(QStringLiteral("report/adapter-report.json"))));
    QVERIFY(QFileInfo::exists(bundle.filePath(QStringLiteral("candidates.json"))));
#else
    QSKIP("V2 Adapter Host integration uses Windows Job Object.");
#endif
}

void V2ApplicationTests::adapterHostDelegatesExistingTaskTerminalToWorkflowOwner()
{
#ifdef Q_OS_WIN
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString candidatePath = directory.filePath(QStringLiteral("workflow-result.json"));
    QFile candidate(candidatePath);
    QVERIFY(candidate.open(QIODevice::WriteOnly));
    QCOMPARE(candidate.write("{}"), qint64(2));
    candidate.close();

    aitrain::v2::StorageV2 storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    aitrain::v2::TaskCoordinator coordinator(&storage);
    aitrain::v2::TaskSnapshot task;
    QVERIFY2(coordinator.createAndStartTask(QStringLiteral("yolo.detect"), QStringLiteral("training"), &task, &error), qPrintable(error));
    aitrain::v2::ArtifactStoreV2 artifacts(directory.filePath(QStringLiteral("store")));
    aitrain::v2::TaskExecutionHostV2 host(&coordinator, &artifacts);
    aitrain::v2::PythonAdapterLaunchV2 launch;
    launch.program = QStringLiteral("cmd.exe");
    launch.arguments = QStringList{QStringLiteral("/c"), QStringLiteral("ping 127.0.0.1 -n 3 > nul")};

    bool callbackCalled = false;
    aitrain::v2::ArtifactId callbackArtifact;
    QStringList forwardedKinds;
    QVERIFY2(host.startExistingTask(task, launch,
        [&callbackCalled, &callbackArtifact](const aitrain::v2::ProtocolEnvelope& terminal,
            const aitrain::v2::ArtifactId& outputArtifactId, QString* callbackError) {
            if (terminal.kind != QStringLiteral("event.succeeded") || !outputArtifactId.isValid()) {
                if (callbackError) *callbackError = QStringLiteral("托管终态不包含已提交输出 Artifact。");
                return false;
            }
            callbackCalled = true;
            callbackArtifact = outputArtifactId;
            return true;
        }, &error, {}, [&forwardedKinds](const aitrain::v2::ProtocolEnvelope& event) {
            forwardedKinds.append(event.kind);
        }), qPrintable(error));

    const aitrain::v2::AdapterEventEndpointV2 endpoint = host.adapterEndpoint();
    QTcpSocket socket;
    socket.connectToHost(endpoint.host, endpoint.port);
    QVERIFY(socket.waitForConnected(3000));
    socket.write(QByteArrayLiteral("{\"channel\":\"aitrain.adapter.v2\",\"token\":\"") + endpoint.token.toUtf8() + QByteArrayLiteral("\"}\n"));
    QTRY_VERIFY(socket.bytesAvailable() > 0);
    QCOMPARE(QJsonDocument::fromJson(socket.readLine()).object().value(QStringLiteral("status")).toString(), QStringLiteral("accepted"));

    aitrain::v2::ProtocolEnvelope artifact;
    artifact.messageId = aitrain::v2::MessageId::create();
    artifact.requestId = task.requestId;
    artifact.taskId = task.id;
    artifact.sequence = 1;
    artifact.kind = QStringLiteral("event.artifact_candidate");
    artifact.timestamp = QDateTime::currentDateTimeUtc();
    artifact.payload = QJsonObject{{QStringLiteral("kind"), QStringLiteral("training_result")}, {QStringLiteral("path"), candidatePath}};
    const QByteArray artifactWire = aitrain::v2::encodeProtocolV2Message(artifact, &error);
    QVERIFY2(!artifactWire.isEmpty(), qPrintable(error));
    socket.write(artifactWire);
    aitrain::v2::ProtocolEnvelope succeeded = artifact;
    succeeded.messageId = aitrain::v2::MessageId::create();
    succeeded.sequence = 2;
    succeeded.kind = QStringLiteral("event.succeeded");
    succeeded.payload = {};
    const QByteArray successWire = aitrain::v2::encodeProtocolV2Message(succeeded, &error);
    QVERIFY2(!successWire.isEmpty(), qPrintable(error));
    socket.write(successWire);
    QVERIFY(socket.waitForBytesWritten(3000));
    QTRY_VERIFY(callbackCalled);
    QVERIFY(callbackArtifact.isValid());
    QTRY_COMPARE(forwardedKinds, QStringList({QStringLiteral("event.artifact_candidate"), QStringLiteral("event.succeeded")}));

    aitrain::v2::TaskSnapshot stillRunning;
    QVERIFY2(storage.task(task.id, &stillRunning, &error), qPrintable(error));
    QCOMPARE(stillRunning.state, aitrain::v2::TaskState::Running);
    QCOMPARE(storage.artifactCount(task.id, &error), 1);
    QVERIFY2(coordinator.finalizeTask(task.id, aitrain::v2::TaskState::Succeeded, {}, &error), qPrintable(error));
    QTRY_VERIFY(!host.isRunning());
#else
    QSKIP("V2 Adapter Host integration uses Windows Job Object.");
#endif
}

void V2ApplicationTests::importCreatesVerifiedModelPackageWithoutGuessingType()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString sourcePath = directory.filePath(QStringLiteral("external.onnx"));
    QFile source(sourcePath);
    QVERIFY(source.open(QIODevice::WriteOnly));
    QVERIFY(source.write("externally-confirmed-model") > 0);
    source.close();
    aitrain::v2::StorageV2 storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    aitrain::v2::TaskCoordinator coordinator(&storage);
    aitrain::v2::ArtifactStoreV2 artifacts(directory.filePath(QStringLiteral("store")));
    aitrain::v2::ModelImportServiceV2 importer(&coordinator, &artifacts);
    aitrain::v2::ModelImportRequestV2 request;
    request.taskId = aitrain::v2::TaskId::create();
    request.sourceFilePath = sourcePath;
    request.manifest.modelPackageId = aitrain::v2::ModelPackageId::create();
    request.manifest.modelFamily = QStringLiteral("yolo_detection");
    request.manifest.taskType = QStringLiteral("detection");
    request.manifest.sourceBackend = QStringLiteral("external_confirmed_import");
    request.manifest.sourceSnapshotId = aitrain::v2::SnapshotId::create();
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
    aitrain::v2::ModelImportResultV2 imported;
    QVERIFY2(importer.importModel(request, &imported, &error), qPrintable(error));
    QCOMPARE(imported.task.state, aitrain::v2::TaskState::Succeeded);
    QCOMPARE(imported.task.id, request.taskId);
    QCOMPARE(imported.modelPackage.manifest.sourceTaskId, imported.task.id);
    QVERIFY(QFileInfo::exists(QDir(imported.artifactPath).filePath(QStringLiteral("model/external.onnx"))));
    aitrain::v2::ModelPackageSnapshotV2 loaded;
    QVERIFY2(storage.modelPackage(request.manifest.modelPackageId, &loaded, &error), qPrintable(error));
    QCOMPARE(loaded.manifest.modelFamily, QStringLiteral("yolo_detection"));
}

void V2ApplicationTests::importCancellationLeavesNoRegisteredPackageOrArtifact()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString sourcePath = directory.filePath(QStringLiteral("large-external.onnx"));
    QFile source(sourcePath);
    QVERIFY(source.open(QIODevice::WriteOnly));
    QVERIFY(source.write(QByteArray(3 * 1024 * 1024, 'm')) == 3 * 1024 * 1024);
    source.close();

    aitrain::v2::StorageV2 storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    aitrain::v2::TaskCoordinator coordinator(&storage);
    aitrain::v2::ArtifactStoreV2 artifacts(directory.filePath(QStringLiteral("store")));
    aitrain::v2::ModelImportServiceV2 importer(&coordinator, &artifacts);
    aitrain::v2::ModelImportRequestV2 request;
    request.taskId = aitrain::v2::TaskId::create();
    request.sourceFilePath = sourcePath;
    request.manifest.modelPackageId = aitrain::v2::ModelPackageId::create();
    request.manifest.modelFamily = QStringLiteral("yolo_detection");
    request.manifest.taskType = QStringLiteral("detection");
    request.manifest.sourceBackend = QStringLiteral("external_confirmed_import");
    request.manifest.sourceSnapshotId = aitrain::v2::SnapshotId::create();
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
    aitrain::v2::ModelImportResultV2 imported;
    QVERIFY(!importer.importModel(request, &imported, &error, [&cancellationChecks]() {
        // 前五次检查覆盖开始/复制，随后在 Artifact 二次哈希阶段取消。
        return ++cancellationChecks >= 7;
    }));
    QVERIFY(error.contains(QStringLiteral("取消")));
    QCOMPARE(imported.task.state, aitrain::v2::TaskState::Canceled);
    QCOMPARE(imported.task.id, request.taskId);
    aitrain::v2::TaskSnapshot stored;
    QVERIFY2(storage.task(imported.task.id, &stored, &error), qPrintable(error));
    QCOMPARE(stored.state, aitrain::v2::TaskState::Canceled);
    QCOMPARE(stored.failure.code, aitrain::v2::FailureCode::Canceled);
    QCOMPARE(storage.artifactCount(imported.task.id, &error), 0);
    QCOMPARE(storage.modelPackages(10, &error).size(), 0);
    const QDir stagingRoot(directory.filePath(QStringLiteral("store/staging")));
    QCOMPARE(stagingRoot.entryList(QDir::Dirs | QDir::NoDotAndDotDot).size(), 0);
}

void V2ApplicationTests::runtimeResolutionAcceptsOnlyRegisteredUntamperedModelPackages()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString sourcePath = directory.filePath(QStringLiteral("external.onnx"));
    QFile source(sourcePath);
    QVERIFY(source.open(QIODevice::WriteOnly));
    QVERIFY(source.write("model-package-runtime") > 0);
    source.close();
    aitrain::v2::StorageV2 storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    aitrain::v2::TaskCoordinator coordinator(&storage);
    aitrain::v2::ArtifactStoreV2 artifacts(directory.filePath(QStringLiteral("store")));
    aitrain::v2::ModelImportServiceV2 importer(&coordinator, &artifacts);
    aitrain::v2::ModelImportRequestV2 request;
    request.sourceFilePath = sourcePath;
    request.manifest.modelPackageId = aitrain::v2::ModelPackageId::create();
    request.manifest.modelFamily = QStringLiteral("yolo_detection");
    request.manifest.taskType = QStringLiteral("detection");
    request.manifest.sourceBackend = QStringLiteral("external_confirmed_import");
    request.manifest.sourceSnapshotId = aitrain::v2::SnapshotId::create();
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
    aitrain::v2::ModelImportResultV2 imported;
    QVERIFY2(importer.importModel(request, &imported, &error), qPrintable(error));
    aitrain::v2::ModelPackageRuntimeServiceV2 service(&storage, artifacts.rootPath());
    aitrain::v2::RuntimeModelLocationV2 location;
    aitrain::v2::RuntimeCapabilityV2 capability;
    QVERIFY2(service.resolve(request.manifest.modelPackageId, QStringLiteral("aitrain_onnxruntime"), &location, &capability, &error), qPrintable(error));
    QCOMPARE(capability.status, aitrain::v2::RuntimeCapabilityStatusV2::Supported);
    QVERIFY(QFile::remove(QDir(location.artifactDirectory).filePath(QStringLiteral("model/model.onnx"))));
    QVERIFY(!service.resolve(request.manifest.modelPackageId, QStringLiteral("aitrain_onnxruntime"), &location, &capability, &error));
    QVERIFY(error.contains(QStringLiteral("入口")));
}

void V2ApplicationTests::projectWorkspaceOwnsRuntimeTaskLifecycle()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::ProjectWorkspaceV2 workspace;
    QString error;
    QVERIFY2(workspace.open(directory.filePath(QStringLiteral("project")), &error), qPrintable(error));

    const aitrain::v2::TaskId taskId = aitrain::v2::TaskId::create();
    aitrain::v2::TaskSnapshot started;
    QVERIFY2(workspace.startTask(taskId,
        QStringLiteral("runtime.aitrain_onnxruntime"),
        QStringLiteral("deployment_validation"),
        &started,
        &error), qPrintable(error));
    QCOMPARE(started.id, taskId);
    QCOMPARE(started.state, aitrain::v2::TaskState::Running);

    QVERIFY2(workspace.requestTaskCancellation(taskId, &error), qPrintable(error));
    const aitrain::v2::Failure cancellation{
        aitrain::v2::FailureCode::Canceled,
        QStringLiteral("用户取消"),
        QString(),
        QDateTime::currentDateTimeUtc()};
    QVERIFY2(workspace.finalizeTask(taskId, aitrain::v2::TaskState::Canceled, cancellation, &error), qPrintable(error));

    aitrain::v2::StorageV2 storage;
    QVERIFY2(storage.open(QDir(workspace.workspacePath()).filePath(QStringLiteral("project-v2.sqlite")), &error), qPrintable(error));
    aitrain::v2::TaskSnapshot persisted;
    QVERIFY2(storage.task(taskId, &persisted, &error), qPrintable(error));
    QCOMPARE(persisted.state, aitrain::v2::TaskState::Canceled);
    QCOMPARE(persisted.failure.code, aitrain::v2::FailureCode::Canceled);
}

void V2ApplicationTests::projectQueryServiceReadsOnlyPersistedV2TaskState()
{
    QTemporaryDir project;
    QVERIFY(project.isValid());
    aitrain::v2::ProjectWorkspaceV2 workspace;
    QString error;
    QVERIFY2(workspace.open(project.path(), &error), qPrintable(error));

    const aitrain::v2::TaskId taskId = aitrain::v2::TaskId::create();
    aitrain::v2::TaskSnapshot started;
    QVERIFY2(workspace.startTask(taskId, QStringLiteral("yolo.detect"),
        QStringLiteral("training"), &started, &error), qPrintable(error));

    aitrain::v2::ProjectQueryServiceV2 queries(&workspace);
    const QVector<aitrain::v2::TaskSnapshot> tasks = queries.recentTasks(20, &error);
    QCOMPARE(tasks.size(), 1);
    QCOMPARE(tasks.first().id, taskId);
    QCOMPARE(tasks.first().state, aitrain::v2::TaskState::Running);

    aitrain::v2::TaskReadModelV2 details;
    QVERIFY2(queries.taskDetails(taskId, &details, &error), qPrintable(error));
    QCOMPARE(details.task.id, taskId);
    QVERIFY(details.artifacts.isEmpty());
    QVERIFY(details.metrics.isEmpty());
    QVERIFY(details.workflows.isEmpty());

    QVERIFY2(workspace.requestTaskCancellation(taskId, &error), qPrintable(error));
    QVERIFY2(workspace.finalizeTask(taskId, aitrain::v2::TaskState::Canceled,
        aitrain::v2::Failure{aitrain::v2::FailureCode::Canceled,
            QStringLiteral("测试取消"), QStringLiteral("无需操作"), QDateTime::currentDateTimeUtc()},
        &error), qPrintable(error));
    QVERIFY2(queries.taskDetails(taskId, &details, &error), qPrintable(error));
    QCOMPARE(details.task.state, aitrain::v2::TaskState::Canceled);
    QCOMPARE(details.task.failure.code, aitrain::v2::FailureCode::Canceled);
}

void V2ApplicationTests::projectWorkspaceCommitsRuntimeArtifactsBeforeSuccess()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::ProjectWorkspaceV2 workspace;
    QString error;
    QVERIFY2(workspace.open(directory.filePath(QStringLiteral("project")), &error), qPrintable(error));

    const aitrain::v2::TaskId taskId = aitrain::v2::TaskId::create();
    aitrain::v2::TaskSnapshot task;
    QVERIFY2(workspace.startTask(taskId,
        QStringLiteral("runtime.aitrain_onnxruntime"), QStringLiteral("inference"), &task, &error), qPrintable(error));
    const QString runtimeStaging = workspace.runtimeStagingPath(taskId);
    QVERIFY(QDir().mkpath(runtimeStaging));
    const QString predictionPath = QDir(runtimeStaging).filePath(QStringLiteral("predictions.json"));
    QFile prediction(predictionPath);
    QVERIFY(prediction.open(QIODevice::WriteOnly));
    QVERIFY(prediction.write("{\"detections\":[]}" ) > 0);
    prediction.close();

    aitrain::v2::RuntimeArtifactBundleV2 bundle;
    QVERIFY2(workspace.commitRuntimeArtifacts(taskId,
        QStringLiteral("runtime_output_bundle"),
        {{QStringLiteral("inference_predictions"), predictionPath}},
        &bundle,
        &error), qPrintable(error));
    QVERIFY(bundle.artifactId.isValid());
    QVERIFY(QFileInfo::exists(bundle.pathsByKind.value(QStringLiteral("inference_predictions"))));
    QVERIFY2(workspace.cleanupRuntimeStaging(taskId, &error), qPrintable(error));
    QVERIFY(!QFileInfo::exists(runtimeStaging));
    QVERIFY2(workspace.finalizeTask(taskId, aitrain::v2::TaskState::Succeeded, {}, &error), qPrintable(error));

    aitrain::v2::StorageV2 storage;
    QVERIFY2(storage.open(QDir(workspace.workspacePath()).filePath(QStringLiteral("project-v2.sqlite")), &error), qPrintable(error));
    QCOMPARE(storage.artifactCount(taskId, &error), 1);
    QCOMPARE(storage.artifactFileCount(bundle.artifactId, &error), 1);
}

void V2ApplicationTests::datasetSnapshotImportRegistersNewAndExistingDatasetVersions()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString sourceRoot = directory.filePath(QStringLiteral("外部 数据集"));
    QVERIFY(createYoloImportFixture(sourceRoot, QByteArray("0 0.5 0.5 0.25 0.25\n")));
    aitrain::v2::ProjectWorkspaceV2 workspace;
    QString error;
    QVERIFY2(workspace.open(directory.filePath(QStringLiteral("project")), &error), qPrintable(error));
    const aitrain::v2::DatasetId datasetId = aitrain::v2::DatasetId::create();

    const auto runImport = [&](aitrain::v2::DatasetSnapshotImportWorkflowResultV2* result) {
        const aitrain::v2::TaskId taskId = aitrain::v2::TaskId::create();
        aitrain::v2::TaskSnapshot task;
        if (!workspace.startTask(taskId, QStringLiteral("dataset.snapshot.import.v2"),
                QStringLiteral("dataset_snapshot_import"), &task, &error)) return false;
        aitrain::v2::DatasetSnapshotImportWorkflowRequestV2 request;
        request.sourcePath = sourceRoot;
        request.sourceFormat = QStringLiteral("yolo_detection");
        request.targetDatasetId = datasetId;
        request.targetDatasetName = QStringLiteral("审计名称");
        return workspace.runDatasetSnapshotImportWorkflow(taskId, request, result, &error);
    };

    aitrain::v2::DatasetSnapshotImportWorkflowResultV2 first;
    QVERIFY2(runImport(&first), qPrintable(error));
    QCOMPARE(first.terminalState, aitrain::v2::TaskState::Succeeded);
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
    aitrain::v2::DatasetSnapshotImportWorkflowResultV2 second;
    QVERIFY2(runImport(&second), qPrintable(error));
    QCOMPARE(second.terminalState, aitrain::v2::TaskState::Succeeded);
    QCOMPARE(second.datasetSnapshot.datasetId, datasetId);
    QVERIFY(second.datasetSnapshot.datasetVersionId != first.datasetSnapshot.datasetVersionId);
    QVERIFY(second.datasetSnapshot.id != first.datasetSnapshot.id);
    QVERIFY(second.datasetSnapshot.rootPath != first.datasetSnapshot.rootPath);

    aitrain::v2::StorageV2 storage;
    QVERIFY2(storage.open(QDir(workspace.workspacePath()).filePath(
        QStringLiteral("project-v2.sqlite")), &error), qPrintable(error));
    aitrain::v2::DatasetSnapshotRecordV2 loadedFirst;
    aitrain::v2::DatasetSnapshotRecordV2 loadedSecond;
    QVERIFY2(storage.datasetSnapshot(first.datasetSnapshot.id, &loadedFirst, &error), qPrintable(error));
    QVERIFY2(storage.datasetSnapshot(second.datasetSnapshot.id, &loadedSecond, &error), qPrintable(error));
    QCOMPARE(loadedFirst.rootPath, first.datasetSnapshot.rootPath);
    QCOMPARE(loadedSecond.rootPath, second.datasetSnapshot.rootPath);
}

void V2ApplicationTests::datasetSnapshotImportRejectsSourceChangedAfterPlanWithEvidence()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString sourceRoot = directory.filePath(QStringLiteral("source"));
    QVERIFY(createYoloImportFixture(sourceRoot, QByteArray("0 0.5 0.5 0.25 0.25\n")));
    const QString projectRoot = directory.filePath(QStringLiteral("project"));
    aitrain::v2::ProjectWorkspaceV2 workspace;
    QString error;
    QVERIFY2(workspace.open(projectRoot, &error), qPrintable(error));
    const aitrain::v2::TaskId taskId = aitrain::v2::TaskId::create();
    aitrain::v2::TaskSnapshot task;
    QVERIFY2(workspace.startTask(taskId, QStringLiteral("dataset.snapshot.import.v2"),
        QStringLiteral("dataset_snapshot_import"), &task, &error), qPrintable(error));
    aitrain::v2::DatasetSnapshotImportWorkflowRequestV2 request;
    request.sourcePath = sourceRoot;
    request.sourceFormat = QStringLiteral("yolo_detection");
    request.targetDatasetId = aitrain::v2::DatasetId::create();
    request.targetDatasetName = QStringLiteral("source-change");
    bool mutated = false;
    const auto cancellation = [&]() {
        if (!mutated) {
            QDirIterator iterator(QDir(projectRoot).filePath(
                QStringLiteral(".aitrain-v2/artifact-store/artifacts")),
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
    aitrain::v2::DatasetSnapshotImportWorkflowResultV2 result;
    QVERIFY2(workspace.runDatasetSnapshotImportWorkflow(
        taskId, request, &result, &error, cancellation), qPrintable(error));
    QVERIFY(mutated);
    QCOMPARE(result.terminalState, aitrain::v2::TaskState::Failed);
    QVERIFY(result.importPlanArtifactId.isValid());
    QVERIFY(!result.datasetSnapshot.id.isValid());
    QVERIFY(result.evidenceArtifactId.isValid());
    QCOMPARE(result.failure.code, aitrain::v2::FailureCode::ArtifactIncompatible);
}

void V2ApplicationTests::datasetSplitWorkflowRegistersSelfContainedSnapshotAndRejectsIdentityMismatch()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString sourceRoot = directory.filePath(QStringLiteral("source"));
    QVERIFY(createYoloImportFixture(sourceRoot, QByteArray("0 0.5 0.5 0.25 0.25\n")));
    const QString projectRoot = directory.filePath(QStringLiteral("project"));
    aitrain::v2::ProjectWorkspaceV2 workspace;
    QString error;
    QVERIFY2(workspace.open(projectRoot, &error), qPrintable(error));

    const aitrain::v2::TaskId importTaskId = aitrain::v2::TaskId::create();
    aitrain::v2::TaskSnapshot task;
    QVERIFY2(workspace.startTask(importTaskId, QStringLiteral("dataset.snapshot.import.v2"),
        QStringLiteral("dataset_snapshot_import"), &task, &error), qPrintable(error));
    aitrain::v2::DatasetSnapshotImportWorkflowRequestV2 importRequest;
    importRequest.sourcePath = sourceRoot;
    importRequest.sourceFormat = QStringLiteral("yolo_detection");
    importRequest.targetDatasetId = aitrain::v2::DatasetId::create();
    importRequest.targetDatasetName = QStringLiteral("源数据集");
    aitrain::v2::DatasetSnapshotImportWorkflowResultV2 imported;
    QVERIFY2(workspace.runDatasetSnapshotImportWorkflow(importTaskId, importRequest,
        &imported, &error), qPrintable(error));
    QCOMPARE(imported.terminalState, aitrain::v2::TaskState::Succeeded);

    aitrain::v2::DatasetSplitWorkflowRequestV2 request;
    request.sourceDatasetId = imported.datasetSnapshot.datasetId;
    request.sourceDatasetVersionId = imported.datasetSnapshot.datasetVersionId;
    request.sourceSnapshotId = imported.datasetSnapshot.id;
    request.sourceSnapshotArtifactId = imported.datasetSnapshot.artifactId;
    request.targetDatasetId = aitrain::v2::DatasetId::create();
    request.targetDatasetName = QStringLiteral("划分目标");
    request.options = QJsonObject{{QStringLiteral("trainRatio"), 0.5},
        {QStringLiteral("valRatio"), 0.5}, {QStringLiteral("testRatio"), 0.0},
        {QStringLiteral("seed"), 42}};
    const aitrain::v2::TaskId splitTaskId = aitrain::v2::TaskId::create();
    QVERIFY2(workspace.startTask(splitTaskId, QStringLiteral("dataset.split.v2"),
        QStringLiteral("dataset_split"), &task, &error), qPrintable(error));
    aitrain::v2::DatasetSplitWorkflowResultV2 split;
    QVERIFY2(workspace.runDatasetSplitWorkflow(splitTaskId, request, &split, &error), qPrintable(error));
    QCOMPARE(split.terminalState, aitrain::v2::TaskState::Succeeded);
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
        QStringLiteral("split_plan_v2.json"))).exists());

    QDirIterator planIterator(QDir(projectRoot).filePath(
        QStringLiteral(".aitrain-v2/artifact-store/artifacts/%1")
            .arg(split.splitPlanArtifactId.toString())),
        QStringList{QStringLiteral("dataset_split_plan_v2.json")}, QDir::Files);
    QVERIFY(planIterator.hasNext());
    QFile planFile(planIterator.next());
    QVERIFY(planFile.open(QIODevice::ReadOnly));
    const QByteArray planBytes = planFile.readAll();
    QVERIFY(!planBytes.contains(sourceRoot.toUtf8()));
    QVERIFY(!planBytes.contains("sourcePath"));
    QVERIFY(!planBytes.contains("outputPath"));

    const aitrain::v2::TaskId mismatchTaskId = aitrain::v2::TaskId::create();
    QVERIFY2(workspace.startTask(mismatchTaskId, QStringLiteral("dataset.split.v2"),
        QStringLiteral("dataset_split"), &task, &error), qPrintable(error));
    request.sourceDatasetId = aitrain::v2::DatasetId::create();
    aitrain::v2::DatasetSplitWorkflowResultV2 mismatch;
    QVERIFY2(workspace.runDatasetSplitWorkflow(mismatchTaskId, request, &mismatch, &error), qPrintable(error));
    QCOMPARE(mismatch.terminalState, aitrain::v2::TaskState::Failed);
    QVERIFY(!mismatch.datasetSnapshot.id.isValid());
    QVERIFY(mismatch.evidenceArtifactId.isValid());
    QCOMPARE(mismatch.failure.code, aitrain::v2::FailureCode::ArtifactIncompatible);

    request.sourceDatasetId = imported.datasetSnapshot.datasetId;
    QVERIFY(writeFile(QDir(imported.datasetSnapshot.rootPath).filePath(
        QStringLiteral("labels/train/a.txt")), QByteArray("0 0.4 0.4 0.2 0.2\n")));
    const aitrain::v2::TaskId tamperTaskId = aitrain::v2::TaskId::create();
    QVERIFY2(workspace.startTask(tamperTaskId, QStringLiteral("dataset.split.v2"),
        QStringLiteral("dataset_split"), &task, &error), qPrintable(error));
    aitrain::v2::DatasetSplitWorkflowResultV2 tampered;
    QVERIFY2(workspace.runDatasetSplitWorkflow(tamperTaskId, request, &tampered, &error), qPrintable(error));
    QCOMPARE(tampered.terminalState, aitrain::v2::TaskState::Failed);
    QVERIFY(!tampered.datasetSnapshot.id.isValid());
    QVERIFY(tampered.evidenceArtifactId.isValid());
    QCOMPARE(tampered.failure.code, aitrain::v2::FailureCode::ArtifactIncompatible);
}

void V2ApplicationTests::datasetSplitWorkflowCancellationAfterMaterializeDoesNotRegisterSnapshot()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString sourceRoot = directory.filePath(QStringLiteral("source"));
    QVERIFY(createYoloImportFixture(sourceRoot, QByteArray("0 0.5 0.5 0.25 0.25\n")));
    const QString projectRoot = directory.filePath(QStringLiteral("project"));
    aitrain::v2::ProjectWorkspaceV2 workspace;
    QString error;
    QVERIFY2(workspace.open(projectRoot, &error), qPrintable(error));
    aitrain::v2::TaskSnapshot task;
    const aitrain::v2::TaskId importTaskId = aitrain::v2::TaskId::create();
    QVERIFY2(workspace.startTask(importTaskId, QStringLiteral("dataset.snapshot.import.v2"),
        QStringLiteral("dataset_snapshot_import"), &task, &error), qPrintable(error));
    aitrain::v2::DatasetSnapshotImportWorkflowRequestV2 importRequest;
    importRequest.sourcePath = sourceRoot;
    importRequest.sourceFormat = QStringLiteral("yolo_detection");
    importRequest.targetDatasetId = aitrain::v2::DatasetId::create();
    importRequest.targetDatasetName = QStringLiteral("源数据集");
    aitrain::v2::DatasetSnapshotImportWorkflowResultV2 imported;
    QVERIFY2(workspace.runDatasetSnapshotImportWorkflow(importTaskId, importRequest,
        &imported, &error), qPrintable(error));

    aitrain::v2::DatasetSplitWorkflowRequestV2 request;
    request.sourceDatasetId = imported.datasetSnapshot.datasetId;
    request.sourceDatasetVersionId = imported.datasetSnapshot.datasetVersionId;
    request.sourceSnapshotId = imported.datasetSnapshot.id;
    request.sourceSnapshotArtifactId = imported.datasetSnapshot.artifactId;
    request.targetDatasetId = aitrain::v2::DatasetId::create();
    request.targetDatasetName = QStringLiteral("取消目标");
    request.options = QJsonObject{{QStringLiteral("trainRatio"), 0.5},
        {QStringLiteral("valRatio"), 0.5}, {QStringLiteral("testRatio"), 0.0},
        {QStringLiteral("seed"), 7}};
    const aitrain::v2::TaskId splitTaskId = aitrain::v2::TaskId::create();
    QVERIFY2(workspace.startTask(splitTaskId, QStringLiteral("dataset.split.v2"),
        QStringLiteral("dataset_split"), &task, &error), qPrintable(error));
    bool materialized = false;
    const auto cancellation = [&]() {
        QDirIterator iterator(QDir(projectRoot).filePath(
            QStringLiteral(".aitrain-v2/artifact-store/artifacts")),
            QStringList{QStringLiteral("split_plan_v2.json")}, QDir::Files,
            QDirIterator::Subdirectories);
        materialized = iterator.hasNext();
        return materialized;
    };
    aitrain::v2::DatasetSplitWorkflowResultV2 result;
    QVERIFY2(workspace.runDatasetSplitWorkflow(splitTaskId, request, &result,
        &error, cancellation), qPrintable(error));
    QVERIFY(materialized);
    QCOMPARE(result.terminalState, aitrain::v2::TaskState::Canceled);
    QVERIFY(result.splitPlanArtifactId.isValid());
    QVERIFY(result.splitArtifactId.isValid());
    QVERIFY(!result.datasetSnapshot.id.isValid());
    QVERIFY(result.evidenceArtifactId.isValid());
}

void V2ApplicationTests::projectWorkspaceRegistersDatasetSnapshotAndSequencesTrainingWorkflow()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString datasetRoot = directory.filePath(QStringLiteral("dataset"));
    QVERIFY(QDir().mkpath(QDir(datasetRoot).filePath(QStringLiteral("images"))));
    QFile image(QDir(datasetRoot).filePath(QStringLiteral("images/sample.jpg")));
    QVERIFY(image.open(QIODevice::WriteOnly));
    QVERIFY(image.write("not-an-image-but-a-snapshot-fixture") > 0);
    image.close();

    aitrain::v2::ProjectWorkspaceV2 workspace;
    QString error;
    QVERIFY2(workspace.open(directory.filePath(QStringLiteral("project")), &error), qPrintable(error));
    const aitrain::v2::TaskId snapshotTaskId = aitrain::v2::TaskId::create();
    aitrain::v2::TaskSnapshot snapshotTask;
    QVERIFY2(workspace.startTask(snapshotTaskId, QStringLiteral("dataset.snapshot"),
        QStringLiteral("detection"), &snapshotTask, &error), qPrintable(error));
    aitrain::v2::DatasetSnapshotCommitRequestV2 snapshotRequest;
    snapshotRequest.datasetRoot = datasetRoot;
    snapshotRequest.datasetFormat = QStringLiteral("yolo_detection");
    snapshotRequest.driverId = QStringLiteral("yolo_detection");
    snapshotRequest.driverVersion = QStringLiteral("2.0");
    snapshotRequest.options.classDefinitions.append(QJsonObject{{QStringLiteral("id"), 0}, {QStringLiteral("name"), QStringLiteral("part")}});
    aitrain::v2::DatasetSnapshotArtifactBundleV2 snapshot;
    QVERIFY2(workspace.commitDatasetSnapshot(snapshotTaskId, snapshotRequest, &snapshot, &error), qPrintable(error));
    QVERIFY2(workspace.finalizeTask(snapshotTaskId, aitrain::v2::TaskState::Succeeded, {}, &error), qPrintable(error));

    const aitrain::v2::TaskId taskId = aitrain::v2::TaskId::create();
    aitrain::v2::TaskSnapshot task;
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

    aitrain::v2::TrainingWorkflowRequestV2 workflowRequest;
    workflowRequest.datasetId = snapshot.snapshot.datasetId;
    workflowRequest.datasetVersionId = snapshot.snapshot.datasetVersionId;
    workflowRequest.snapshotId = snapshot.snapshot.id;
    workflowRequest.snapshotArtifactId = snapshot.snapshot.artifactId;
    workflowRequest.templateId = QStringLiteral("official_yolo_training_delivery_v2");
    workflowRequest.trainingBackend = QStringLiteral("ultralytics_yolo_detect");
    workflowRequest.evaluationBackend = QStringLiteral("ultralytics_yolo_eval");
    workflowRequest.exportBackend = QStringLiteral("ultralytics_yolo_export");
    workflowRequest.deploymentBackend = QStringLiteral("aitrain_onnxruntime");
    workflowRequest.parameterSummary = QJsonObject{{QStringLiteral("epochs"), 1}};
    aitrain::v2::StorageV2 rejectionStorage;
    QVERIFY2(rejectionStorage.open(QDir(workspace.workspacePath()).filePath(QStringLiteral("project-v2.sqlite")),
        &error), qPrintable(error));
    const qsizetype artifactsBeforeMismatch = rejectionStorage.artifactsForTask(taskId, &error).size();
    QVERIFY2(error.isEmpty(), qPrintable(error));
    QCOMPARE(artifactsBeforeMismatch, qsizetype(0));
    const auto rejectsMismatchWithoutArtifacts = [&](const aitrain::v2::TrainingWorkflowRequestV2& rejectedRequest) {
        aitrain::v2::TrainingWorkflowDispatchV2 rejected;
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
    aitrain::v2::TrainingWorkflowRequestV2 mismatched = workflowRequest;
    mismatched.datasetId = aitrain::v2::DatasetId::create();
    QVERIFY2(rejectsMismatchWithoutArtifacts(mismatched), qPrintable(error));
    mismatched = workflowRequest;
    mismatched.datasetVersionId = aitrain::v2::DatasetVersionId::create();
    QVERIFY2(rejectsMismatchWithoutArtifacts(mismatched), qPrintable(error));
    mismatched = workflowRequest;
    mismatched.snapshotId = aitrain::v2::SnapshotId::create();
    QVERIFY2(rejectsMismatchWithoutArtifacts(mismatched), qPrintable(error));
    mismatched = workflowRequest;
    mismatched.snapshotArtifactId = aitrain::v2::ArtifactId::create();
    QVERIFY2(rejectsMismatchWithoutArtifacts(mismatched), qPrintable(error));
    error.clear();
    aitrain::v2::TrainingWorkflowDispatchV2 workflow;
    QVERIFY2(workspace.beginTrainingWorkflow(taskId, workflowRequest, &workflow, &error), qPrintable(error));
    QVERIFY(workflow.dispatch.hasStep);
    QCOMPARE(workflow.dispatch.step.kind, QStringLiteral("Train"));
    QCOMPARE(workflow.dispatch.step.inputArtifactId, snapshot.snapshot.artifactId);
    aitrain::v2::StorageV2 lineageStorage;
    QVERIFY2(lineageStorage.open(QDir(workspace.workspacePath()).filePath(QStringLiteral("project-v2.sqlite")),
        &error), qPrintable(error));
    aitrain::v2::WorkflowInputBindingV2 inputBinding;
    QVERIFY2(lineageStorage.workflowInput(workflow.workflowRunId, QStringLiteral("dataset_snapshot"),
        &inputBinding, &error), qPrintable(error));
    QCOMPARE(inputBinding.sourceTaskId, snapshotTaskId);
    QCOMPARE(inputBinding.sourceArtifactId, snapshot.snapshot.artifactId);
    QCOMPARE(inputBinding.datasetId, snapshot.snapshot.datasetId);
    QCOMPARE(inputBinding.datasetVersionId, snapshot.snapshot.datasetVersionId);
    QCOMPARE(inputBinding.datasetSnapshotId, snapshot.snapshot.id);
    QCOMPARE(inputBinding.manifestSha256, snapshot.snapshot.manifestSha256);
    QCOMPARE(inputBinding.rootHash, snapshot.snapshot.rootHash);
    aitrain::v2::VerifiedTrainingWorkflowInputV2 trainInput;
    QVERIFY2(workspace.resolveTrainingWorkflowStepInput(workflow.workflowRunId, workflow.dispatch.step.id,
        &trainInput, &error), qPrintable(error));
    QCOMPARE(trainInput.artifactId, snapshot.snapshot.artifactId);
    QVERIFY(trainInput.files.size() >= 2);
    const auto manifestFile = std::find_if(trainInput.files.cbegin(), trainInput.files.cend(),
        [](const aitrain::v2::VerifiedWorkflowArtifactFileV2& file) {
            return file.relativePath == QStringLiteral("dataset_snapshot.json");
        });
    QVERIFY(manifestFile != trainInput.files.cend());
    aitrain::v2::TrainingWorkflowAdapterConfigV2 adapterConfig;
    adapterConfig.pythonProgram = QStringLiteral("cmd.exe");
    adapterConfig.trainersRoot = QDir::current().filePath(QStringLiteral("python_trainers"));
    aitrain::v2::TrainingWorkflowAdapterLaunchV2 adapterLaunch;
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
            aitrain::v2::TrainingModelRegistrationV2 registration;
            QVERIFY2(workspace.registerTrainingWorkflowModel(workflow.workflowRunId, workflow.dispatch.step.id,
                &registration, &error), qPrintable(error));
            QVERIFY(registration.modelPackage.manifest.modelPackageId.isValid());
            QCOMPARE(registration.modelPackage.manifest.sourceTaskId, taskId);
            QCOMPARE(registration.modelPackage.manifest.sourceSnapshotId, snapshot.snapshot.id);
            QCOMPARE(registration.modelPackage.manifest.classNames, QStringList{QStringLiteral("part")});
            QVERIFY(registration.registrationArtifact.artifactId.isValid());
            aitrain::v2::WorkflowStepExecutionResultV2 execution;
            execution.state = aitrain::v2::WorkflowStepState::Succeeded;
            execution.outputArtifactId = registration.registrationArtifact.artifactId;
            QVERIFY2(workspace.completeTrainingWorkflowStep(workflow.workflowRunId, workflow.dispatch.step.id,
                execution, &workflow, &error), qPrintable(error));
            continue;
        }
        if (workflow.dispatch.step.kind == QStringLiteral("RenderDeliveryReport")) {
            aitrain::v2::RuntimeArtifactBundleV2 report;
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
            aitrain::v2::WorkflowStepExecutionResultV2 execution;
            execution.state = aitrain::v2::WorkflowStepState::Succeeded;
            execution.outputArtifactId = report.artifactId;
            QVERIFY2(workspace.completeTrainingWorkflowStep(workflow.workflowRunId, workflow.dispatch.step.id,
                execution, &workflow, &error), qPrintable(error));
            continue;
        }
        if (workflow.dispatch.step.kind == QStringLiteral("DeploymentValidate")) {
            aitrain::v2::TrainingDeploymentInvocationV2 deployment;
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
        QVector<aitrain::v2::RuntimeArtifactCandidateV2> candidates{{requiresCheckpoint ? QStringLiteral("checkpoint") : (exportStep ? QStringLiteral("export") : QStringLiteral("result")), candidatePath}};
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
        aitrain::v2::RuntimeArtifactBundleV2 output;
        QVERIFY2(workspace.commitRuntimeArtifacts(taskId, QStringLiteral("training_step_output_v2"),
            candidates, &output, &error), qPrintable(error));
        aitrain::v2::WorkflowStepExecutionResultV2 execution;
        execution.state = aitrain::v2::WorkflowStepState::Succeeded;
        execution.outputArtifactId = output.artifactId;
        QVERIFY2(workspace.completeTrainingWorkflowStep(workflow.workflowRunId, workflow.dispatch.step.id,
            execution, &workflow, &error), qPrintable(error));
        if (workflow.dispatch.hasStep) {
            aitrain::v2::VerifiedTrainingWorkflowInputV2 nextInput;
            QVERIFY2(workspace.resolveTrainingWorkflowStepInput(workflow.workflowRunId, workflow.dispatch.step.id,
                &nextInput, &error), qPrintable(error));
            QCOMPARE(nextInput.artifactId, output.artifactId);
            if (workflow.dispatch.step.kind == QStringLiteral("Evaluate") || workflow.dispatch.step.kind == QStringLiteral("Export")) {
                aitrain::v2::TrainingWorkflowAdapterLaunchV2 downstreamLaunch;
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
    QCOMPARE(workflow.dispatch.result.state, aitrain::v2::WorkflowStepState::Succeeded);
    QVERIFY2(workspace.cleanupRuntimeStaging(taskId, &error), qPrintable(error));

    aitrain::v2::StorageV2 storage;
    QVERIFY2(storage.open(QDir(workspace.workspacePath()).filePath(QStringLiteral("project-v2.sqlite")), &error), qPrintable(error));
    aitrain::v2::DatasetSnapshotRecordV2 persistedSnapshot;
    QVERIFY2(storage.datasetSnapshot(snapshot.snapshot.id, &persistedSnapshot, &error), qPrintable(error));
    QCOMPARE(persistedSnapshot.artifactId, snapshot.snapshot.artifactId);
    QCOMPARE(persistedSnapshot.rootHash, snapshot.snapshot.rootHash);
    const QVector<aitrain::v2::WorkflowStepSnapshotV2> steps = storage.workflowSteps(workflow.workflowRunId, &error);
    QCOMPARE(steps.size(), 8);
    QCOMPARE(steps.constFirst().kind, QStringLiteral("ValidateDataset"));
    QCOMPARE(steps.at(1).kind, QStringLiteral("CreateSnapshot"));
    QCOMPARE(steps.at(2).kind, QStringLiteral("Train"));
    QCOMPARE(steps.constLast().kind, QStringLiteral("RenderDeliveryReport"));
    for (int index = 0; index < steps.size(); ++index) {
        const aitrain::v2::WorkflowStepSnapshotV2& step = steps.at(index);
        QCOMPARE(step.state, aitrain::v2::WorkflowStepState::Succeeded);
        if (index == 0) {
            QVERIFY(!step.inputArtifactId.isValid());
        } else {
            QVERIFY(step.inputArtifactId.isValid());
        }
        QVERIFY(step.outputArtifactId.isValid());
    }
    aitrain::v2::TaskSnapshot completed;
    QVERIFY2(storage.task(taskId, &completed, &error), qPrintable(error));
    QCOMPARE(completed.state, aitrain::v2::TaskState::Succeeded);
    aitrain::v2::EvidenceBundleV2 evidence;
    QVERIFY2(workspace.buildWorkflowEvidenceBundle(workflow.workflowRunId, &evidence, &error), qPrintable(error));
    QCOMPARE(evidence.externalInputs.size(), 1);
    const aitrain::v2::EvidenceExternalInputV2& external = evidence.externalInputs.constFirst();
    QCOMPARE(external.role, QStringLiteral("dataset_snapshot"));
    QCOMPARE(external.producerTaskId, snapshotTaskId);
    QCOMPARE(external.artifactId, snapshot.snapshot.artifactId);
    QCOMPARE(external.datasetId, snapshot.snapshot.datasetId);
    QCOMPARE(external.datasetVersionId, snapshot.snapshot.datasetVersionId);
    QCOMPARE(external.datasetSnapshotId, snapshot.snapshot.id);
    QCOMPARE(external.manifestSha256, snapshot.snapshot.manifestSha256);
    QCOMPARE(external.rootHash, snapshot.snapshot.rootHash);
    const QJsonObject encodedEvidence = aitrain::v2::encodeEvidenceBundleV2(evidence, &error);
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
    aitrain::v2::EvidenceBundleV2 decodedEvidence;
    QVERIFY2(aitrain::v2::decodeEvidenceBundleV2(encodedEvidence, &decodedEvidence, &error), qPrintable(error));
    QCOMPARE(decodedEvidence.externalInputs.size(), 1);
    QCOMPARE(decodedEvidence.externalInputs.constFirst().datasetId, snapshot.snapshot.datasetId);
}

void V2ApplicationTests::trainingWorkflowRejectsCrossProfileBackendMix()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::ProjectWorkspaceV2 workspace;
    QString error;
    QVERIFY2(workspace.open(directory.filePath(QStringLiteral("project")), &error), qPrintable(error));
    const aitrain::v2::TaskId taskId = aitrain::v2::TaskId::create();
    aitrain::v2::TaskSnapshot task;
    QVERIFY2(workspace.startTask(taskId, QStringLiteral("yolo"), QStringLiteral("detection"),
        &task, &error), qPrintable(error));

    aitrain::v2::TrainingWorkflowRequestV2 request;
    request.templateId = QStringLiteral("official_yolo_training_delivery_v2");
    request.trainingBackend = QStringLiteral("ultralytics_yolo_detect");
    request.evaluationBackend = QStringLiteral("smp_semantic_segmentation_eval");
    request.exportBackend = QStringLiteral("ultralytics_yolo_export");
    request.deploymentBackend = QStringLiteral("aitrain_onnxruntime");
    aitrain::v2::TrainingWorkflowDispatchV2 workflow;
    QVERIFY(!workspace.beginTrainingWorkflow(taskId, request, &workflow, &error));
    QVERIFY(error.contains(QStringLiteral("Profile")));
}

void V2ApplicationTests::trainingWorkflowEvidenceGatePersistsEvidenceBeforeTerminalTask()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());

    aitrain::v2::ProjectWorkspaceV2 workspace;
    QString error;
    QVERIFY2(workspace.open(directory.filePath(QStringLiteral("project")), &error), qPrintable(error));

    const aitrain::v2::TaskId taskId = aitrain::v2::TaskId::create();
    aitrain::v2::TaskSnapshot started;
    QVERIFY2(workspace.startTask(taskId, QStringLiteral("yolo.detect"), QStringLiteral("detection"),
        &started, &error), qPrintable(error));

    aitrain::v2::TrainingWorkflowRequestV2 request;
    request.templateId = QStringLiteral("official_yolo_training_delivery_v2");
    request.trainingBackend = QStringLiteral("ultralytics_yolo");
    request.evaluationBackend = QStringLiteral("ultralytics_yolo_eval");
    request.exportBackend = QStringLiteral("ultralytics_yolo_export");
    request.deploymentBackend = QStringLiteral("aitrain_onnxruntime");
    request.requireEvidenceBeforeTerminal = true;
    QVERIFY2(attachTrainingSnapshotFixture(workspace, taskId,
        directory.filePath(QStringLiteral("evidence-dataset")), QStringLiteral("yolo_detection"),
        &request, &error), qPrintable(error));

    aitrain::v2::TrainingWorkflowDispatchV2 workflow;
    QVERIFY2(workspace.beginTrainingWorkflow(taskId, request, &workflow, &error), qPrintable(error));
    QCOMPARE(workflow.dispatch.step.kind, QStringLiteral("Train"));

    aitrain::v2::WorkflowStepExecutionResultV2 failed;
    failed.state = aitrain::v2::WorkflowStepState::Failed;
    failed.failure = {aitrain::v2::FailureCode::InvalidDataset,
        QStringLiteral("测试数据集预检失败。"), QStringLiteral("修复数据集后重试。"),
        QDateTime::currentDateTimeUtc()};
    QVERIFY2(workspace.completeTrainingWorkflowStep(workflow.workflowRunId, workflow.dispatch.step.id,
        failed, &workflow, &error), qPrintable(error));
    QVERIFY(!workflow.dispatch.hasStep);
    QCOMPARE(workflow.dispatch.result.state, aitrain::v2::WorkflowStepState::Failed);

    aitrain::v2::StorageV2 storage;
    QVERIFY2(storage.open(QDir(workspace.workspacePath()).filePath(QStringLiteral("project-v2.sqlite")),
        &error), qPrintable(error));
    aitrain::v2::TaskSnapshot persisted;
    QVERIFY2(storage.task(taskId, &persisted, &error), qPrintable(error));
    QCOMPARE(persisted.state, aitrain::v2::TaskState::Running);
    QVERIFY(!workspace.finalizeTask(taskId, aitrain::v2::TaskState::Failed, failed.failure, &error));
    QVERIFY(error.contains(QStringLiteral("evidence_required")));
    error.clear();

    aitrain::v2::EvidenceBundleV2 evidence;
    aitrain::v2::EvidenceArtifactBundleV2 committed;
    QVERIFY2(workspace.buildWorkflowEvidenceBundle(workflow.workflowRunId, &evidence, &error), qPrintable(error));
    QCOMPARE(evidence.task.state, aitrain::v2::TaskState::Failed);
    QVERIFY2(workspace.commitEvidenceBundle(evidence, &committed, &error), qPrintable(error));
    QVERIFY(committed.artifactId.isValid());

    QVERIFY2(storage.task(taskId, &persisted, &error), qPrintable(error));
    QCOMPARE(persisted.state, aitrain::v2::TaskState::Running);
    QVERIFY2(workspace.closeWorkflowTerminalization(workflow.workflowRunId, &error), qPrintable(error));
    QVERIFY2(storage.task(taskId, &persisted, &error), qPrintable(error));
    QCOMPARE(persisted.state, aitrain::v2::TaskState::Failed);
    QCOMPARE(persisted.failure.code, aitrain::v2::FailureCode::InvalidDataset);
}

void V2ApplicationTests::trainingWorkflowEvidenceGateRecoversAcrossReopen()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const QString projectRoot = directory.filePath(QStringLiteral("project"));
    const aitrain::v2::TaskId taskId = aitrain::v2::TaskId::create();
    aitrain::v2::WorkflowRunId workflowRunId;
    QString error;
    {
        aitrain::v2::ProjectWorkspaceV2 workspace;
        QVERIFY2(workspace.open(projectRoot, &error), qPrintable(error));
        aitrain::v2::TaskSnapshot started;
        QVERIFY2(workspace.startTask(taskId, QStringLiteral("yolo.detect"), QStringLiteral("detection"),
            &started, &error), qPrintable(error));
        aitrain::v2::TrainingWorkflowRequestV2 request;
        request.templateId = QStringLiteral("official_yolo_training_delivery_v2");
        request.trainingBackend = QStringLiteral("ultralytics_yolo");
        request.evaluationBackend = QStringLiteral("ultralytics_yolo_eval");
        request.exportBackend = QStringLiteral("ultralytics_yolo_export");
        request.deploymentBackend = QStringLiteral("aitrain_onnxruntime");
        request.requireEvidenceBeforeTerminal = true;
        QVERIFY2(attachTrainingSnapshotFixture(workspace, taskId,
            directory.filePath(QStringLiteral("recovery-dataset")), QStringLiteral("yolo_detection"),
            &request, &error), qPrintable(error));
        aitrain::v2::TrainingWorkflowDispatchV2 workflow;
        QVERIFY2(workspace.beginTrainingWorkflow(taskId, request, &workflow, &error), qPrintable(error));
        workflowRunId = workflow.workflowRunId;
        aitrain::v2::WorkflowStepExecutionResultV2 failed;
        failed.state = aitrain::v2::WorkflowStepState::Failed;
        failed.failure = {aitrain::v2::FailureCode::InvalidDataset,
            QStringLiteral("重启恢复夹具中的数据集预检失败。"), QStringLiteral("修复数据集后重试。"),
            QDateTime::currentDateTimeUtc()};
        QVERIFY2(workspace.completeTrainingWorkflowStep(workflow.workflowRunId, workflow.dispatch.step.id,
            failed, &workflow, &error), qPrintable(error));
        workspace.close();
    }
    {
        aitrain::v2::ProjectWorkspaceV2 recovered;
        QVERIFY2(recovered.open(projectRoot, &error), qPrintable(error));
        aitrain::v2::StorageV2 storage;
        QVERIFY2(storage.open(QDir(recovered.workspacePath()).filePath(QStringLiteral("project-v2.sqlite")),
            &error), qPrintable(error));
        aitrain::v2::TaskSnapshot task;
        QVERIFY2(storage.task(taskId, &task, &error), qPrintable(error));
        QCOMPARE(task.state, aitrain::v2::TaskState::Failed);
        aitrain::v2::WorkflowTerminalizationSnapshotV2 terminalization;
        QVERIFY2(storage.workflowTerminalization(workflowRunId, &terminalization, &error), qPrintable(error));
        QCOMPARE(terminalization.state, aitrain::v2::WorkflowTerminalizationStateV2::Closed);
        QVERIFY(terminalization.evidenceArtifactId.isValid());
        aitrain::v2::ArtifactSnapshotV2 evidence;
        QVERIFY2(storage.artifact(terminalization.evidenceArtifactId, &evidence, &error), qPrintable(error));
        QCOMPARE(evidence.kind, QStringLiteral("evidence_bundle_v2"));
        QCOMPARE(evidence.files.size(), 4);
    }
}

void V2ApplicationTests::officialYoloWorkflowPreservesVariantTaskType_data()
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
        << QStringLiteral("obb")
        << QStringLiteral("obb/ultralytics_trainer.py");
}

void V2ApplicationTests::officialYoloWorkflowPreservesVariantTaskType()
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

    aitrain::v2::ProjectWorkspaceV2 workspace;
    QString error;
    QVERIFY2(workspace.open(directory.filePath(QStringLiteral("project")), &error), qPrintable(error));
    const aitrain::v2::TaskId taskId = aitrain::v2::TaskId::create();
    aitrain::v2::TaskSnapshot task;
    QVERIFY2(workspace.startTask(taskId, QStringLiteral("yolo"), rootTaskType, &task, &error), qPrintable(error));

    aitrain::v2::DatasetSnapshotCommitRequestV2 snapshotRequest;
    snapshotRequest.datasetRoot = datasetRoot;
    snapshotRequest.datasetFormat = datasetFormat;
    snapshotRequest.driverId = datasetFormat;
    snapshotRequest.driverVersion = QStringLiteral("2.0");
    aitrain::v2::DatasetSnapshotArtifactBundleV2 snapshot;
    QVERIFY2(workspace.commitDatasetSnapshot(taskId, snapshotRequest, &snapshot, &error), qPrintable(error));

    aitrain::v2::TrainingWorkflowRequestV2 request;
    request.datasetId = snapshot.snapshot.datasetId;
    request.datasetVersionId = snapshot.snapshot.datasetVersionId;
    request.snapshotId = snapshot.snapshot.id;
    request.snapshotArtifactId = snapshot.snapshot.artifactId;
    request.templateId = QStringLiteral("official_yolo_training_delivery_v2");
    request.trainingBackend = trainingBackend;
    request.evaluationBackend = QStringLiteral("ultralytics_yolo_eval");
    request.exportBackend = QStringLiteral("ultralytics_yolo_export");
    request.deploymentBackend = QStringLiteral("aitrain_onnxruntime");
    request.parameterSummary = QJsonObject{{QStringLiteral("trainingBackend"), trainingBackend}};
    aitrain::v2::TrainingWorkflowDispatchV2 workflow;
    QVERIFY2(workspace.beginTrainingWorkflow(taskId, request, &workflow, &error), qPrintable(error));

    aitrain::v2::TrainingWorkflowAdapterConfigV2 adapterConfig;
    adapterConfig.pythonProgram = QStringLiteral("cmd.exe");
    adapterConfig.trainersRoot = QDir::current().filePath(QStringLiteral("python_trainers"));
    aitrain::v2::TrainingWorkflowAdapterLaunchV2 launch;
    QVERIFY2(workspace.prepareTrainingWorkflowAdapterLaunch(workflow.workflowRunId, workflow.dispatch.step.id,
        adapterConfig, &launch, &error), qPrintable(error));
    QCOMPARE(launch.request.value(QStringLiteral("taskType")).toString(), adapterTaskType);
    QCOMPARE(launch.launch.arguments.constFirst(), QDir(adapterConfig.trainersRoot).filePath(trainerScript));

    const QString stagingRoot = workspace.runtimeStagingPath(taskId);
    QVERIFY(QDir().mkpath(stagingRoot));
    const auto commitStepOutput = [&](const QString& suffix, bool evaluation,
                                      aitrain::v2::RuntimeArtifactBundleV2* output) {
        const QString sourceRoot = QDir(stagingRoot).filePath(suffix);
        if (!QDir().mkpath(sourceRoot)) return false;
        const QString checkpointPath = QDir(sourceRoot).filePath(QStringLiteral("best.pt"));
        QFile checkpoint(checkpointPath);
        if (!checkpoint.open(QIODevice::WriteOnly) || checkpoint.write("checkpoint") <= 0) return false;
        checkpoint.close();
        QVector<aitrain::v2::RuntimeArtifactCandidateV2> candidates{
            {QStringLiteral("checkpoint"), checkpointPath}};
        if (evaluation) {
            const QString reportPath = QDir(sourceRoot).filePath(QStringLiteral("evaluation_report.json"));
            QFile report(reportPath);
            if (!report.open(QIODevice::WriteOnly) || report.write("{\"runtime\":\"ultralytics_official_val\"}") <= 0) return false;
            report.close();
            candidates.append({QStringLiteral("evaluation_report"), reportPath});
        }
        return workspace.commitRuntimeArtifacts(taskId, QStringLiteral("training_step_output_v2"),
            candidates, output, &error);
    };

    aitrain::v2::RuntimeArtifactBundleV2 trainOutput;
    QVERIFY2(commitStepOutput(QStringLiteral("train"), false, &trainOutput), qPrintable(error));
    aitrain::v2::WorkflowStepExecutionResultV2 execution;
    execution.state = aitrain::v2::WorkflowStepState::Succeeded;
    execution.outputArtifactId = trainOutput.artifactId;
    QVERIFY2(workspace.completeTrainingWorkflowStep(workflow.workflowRunId, workflow.dispatch.step.id,
        execution, &workflow, &error), qPrintable(error));
    QCOMPARE(workflow.dispatch.step.kind, QStringLiteral("Evaluate"));
    QVERIFY2(workspace.prepareTrainingWorkflowAdapterLaunch(workflow.workflowRunId, workflow.dispatch.step.id,
        adapterConfig, &launch, &error), qPrintable(error));
    QCOMPARE(launch.request.value(QStringLiteral("taskType")).toString(), adapterTaskType);

    aitrain::v2::RuntimeArtifactBundleV2 evaluationOutput;
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

void V2ApplicationTests::projectWorkspaceDispatchesOfficialAdapterStepThroughTrainingWorkflow()
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

    aitrain::v2::ProjectWorkspaceV2 workspace;
    QString error;
    QVERIFY2(workspace.open(directory.filePath(QStringLiteral("project")), &error), qPrintable(error));
    const aitrain::v2::TaskId taskId = aitrain::v2::TaskId::create();
    aitrain::v2::TaskSnapshot task;
    QVERIFY2(workspace.startTask(taskId, QStringLiteral("yolo.detect"), QStringLiteral("detection"), &task, &error), qPrintable(error));
    aitrain::v2::DatasetSnapshotCommitRequestV2 snapshotRequest;
    snapshotRequest.datasetRoot = datasetRoot;
    snapshotRequest.datasetFormat = QStringLiteral("yolo_detection");
    snapshotRequest.driverId = QStringLiteral("yolo_detection");
    snapshotRequest.driverVersion = QStringLiteral("2.0");
    snapshotRequest.options.classDefinitions.append(QJsonObject{{QStringLiteral("id"), 0}, {QStringLiteral("name"), QStringLiteral("part")}});
    aitrain::v2::DatasetSnapshotArtifactBundleV2 snapshot;
    QVERIFY2(workspace.commitDatasetSnapshot(taskId, snapshotRequest, &snapshot, &error), qPrintable(error));
    aitrain::v2::TrainingWorkflowRequestV2 request;
    request.datasetId = snapshot.snapshot.datasetId;
    request.datasetVersionId = snapshot.snapshot.datasetVersionId;
    request.snapshotId = snapshot.snapshot.id;
    request.snapshotArtifactId = snapshot.snapshot.artifactId;
    request.templateId = QStringLiteral("official_yolo_training_delivery_v2");
    request.trainingBackend = QStringLiteral("ultralytics_yolo_detect");
    request.evaluationBackend = QStringLiteral("ultralytics_yolo_eval");
    request.exportBackend = QStringLiteral("ultralytics_yolo_export");
    request.deploymentBackend = QStringLiteral("aitrain_onnxruntime");
    aitrain::v2::TrainingWorkflowDispatchV2 workflow;
    QVERIFY2(workspace.beginTrainingWorkflow(taskId, request, &workflow, &error), qPrintable(error));

    aitrain::v2::PythonAdapterLaunchV2 launch;
    launch.program = QStringLiteral("cmd.exe");
    launch.arguments = QStringList{QStringLiteral("/c"), QStringLiteral("ping 127.0.0.1 -n 3 > nul")};
    bool dispatchedNextStep = false;
    aitrain::v2::TrainingWorkflowDispatchV2 next;
    QVERIFY2(workspace.startTrainingWorkflowAdapterStep(workflow.workflowRunId, workflow.dispatch.step.id, launch,
        [&dispatchedNextStep, &next](const aitrain::v2::TrainingWorkflowDispatchV2& dispatch) {
            dispatchedNextStep = true;
            next = dispatch;
        }, &error), qPrintable(error));

    const aitrain::v2::AdapterEventEndpointV2 endpoint = workspace.trainingWorkflowAdapterEndpoint();
    QVERIFY(!endpoint.host.isEmpty());
    QVERIFY(endpoint.port > 0);
    QVERIFY(!endpoint.token.isEmpty());
    QTcpSocket socket;
    socket.connectToHost(endpoint.host, endpoint.port);
    QVERIFY(socket.waitForConnected(3000));
    socket.write(QByteArrayLiteral("{\"channel\":\"aitrain.adapter.v2\",\"token\":\"") + endpoint.token.toUtf8() + QByteArrayLiteral("\"}\n"));
    QTRY_VERIFY(socket.bytesAvailable() > 0);
    QCOMPARE(QJsonDocument::fromJson(socket.readLine()).object().value(QStringLiteral("status")).toString(), QStringLiteral("accepted"));
    aitrain::v2::ProtocolEnvelope artifact;
    artifact.messageId = aitrain::v2::MessageId::create();
    artifact.requestId = task.requestId;
    artifact.taskId = task.id;
    artifact.sequence = 1;
    artifact.kind = QStringLiteral("event.artifact_candidate");
    artifact.timestamp = QDateTime::currentDateTimeUtc();
    artifact.payload = QJsonObject{{QStringLiteral("kind"), QStringLiteral("training_report")}, {QStringLiteral("path"), candidatePath}};
    const QByteArray artifactWire = aitrain::v2::encodeProtocolV2Message(artifact, &error);
    QVERIFY2(!artifactWire.isEmpty(), qPrintable(error));
    socket.write(artifactWire);
    aitrain::v2::ProtocolEnvelope succeeded = artifact;
    succeeded.messageId = aitrain::v2::MessageId::create();
    succeeded.sequence = 2;
    succeeded.kind = QStringLiteral("event.succeeded");
    succeeded.payload = {};
    const QByteArray succeededWire = aitrain::v2::encodeProtocolV2Message(succeeded, &error);
    QVERIFY2(!succeededWire.isEmpty(), qPrintable(error));
    socket.write(succeededWire);
    QVERIFY(socket.waitForBytesWritten(3000));

    QTRY_VERIFY(dispatchedNextStep);
    QVERIFY(!workspace.isTrainingWorkflowAdapterRunning());
    QVERIFY(next.dispatch.hasStep);
    QCOMPARE(next.dispatch.step.kind, QStringLiteral("Evaluate"));
    QVERIFY(next.dispatch.step.inputArtifactId.isValid());

    aitrain::v2::StorageV2 storage;
    QVERIFY2(storage.open(QDir(workspace.workspacePath()).filePath(QStringLiteral("project-v2.sqlite")), &error), qPrintable(error));
    const QVector<aitrain::v2::WorkflowStepSnapshotV2> steps = storage.workflowSteps(workflow.workflowRunId, &error);
    QCOMPARE(steps.at(0).state, aitrain::v2::WorkflowStepState::Succeeded);
    QVERIFY(steps.at(0).outputArtifactId.isValid());
    QCOMPARE(steps.at(1).state, aitrain::v2::WorkflowStepState::Succeeded);
    QCOMPARE(steps.at(2).state, aitrain::v2::WorkflowStepState::Succeeded);
    QVERIFY(steps.at(2).outputArtifactId.isValid());
    QCOMPARE(steps.at(3).state, aitrain::v2::WorkflowStepState::Running);
    aitrain::v2::TaskSnapshot persisted;
    QVERIFY2(storage.task(taskId, &persisted, &error), qPrintable(error));
    QCOMPARE(persisted.state, aitrain::v2::TaskState::Running);
#else
    QSKIP("V2 Adapter Host integration uses Windows Job Object.");
#endif
}

void V2ApplicationTests::workflowRunnerSequencesCommittedArtifactsAndStopsOnCancellation()
{
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    aitrain::v2::StorageV2 storage;
    QString error;
    QVERIFY2(storage.open(directory.filePath(QStringLiteral("project.sqlite")), &error), qPrintable(error));
    aitrain::v2::TaskCoordinator coordinator(&storage);
    aitrain::v2::TaskSnapshot rootTask;
    QVERIFY2(coordinator.createAndStartTask(QStringLiteral("workflow.v2"), QStringLiteral("workflow"), &rootTask, &error), qPrintable(error));
    const QVector<aitrain::v2::ArtifactFileSnapshot> files = {
        {QStringLiteral("payload.json"), QString(64, QLatin1Char('a')), 2}};
    const aitrain::v2::ArtifactId firstOutput = aitrain::v2::ArtifactId::create();
    const aitrain::v2::ArtifactId secondOutput = aitrain::v2::ArtifactId::create();
    QVERIFY2(storage.recordArtifactWithFiles(firstOutput, rootTask.id, QStringLiteral("first"), files, QDateTime::currentDateTimeUtc(), &error), qPrintable(error));
    QVERIFY2(storage.recordArtifactWithFiles(secondOutput, rootTask.id, QStringLiteral("second"), files, QDateTime::currentDateTimeUtc(), &error), qPrintable(error));

    aitrain::v2::WorkflowRunSnapshotV2 workflow;
    workflow.id = aitrain::v2::WorkflowRunId::create();
    workflow.taskId = rootTask.id;
    workflow.templateId = QStringLiteral("inference-deployment");
    aitrain::v2::WorkflowStepSnapshotV2 first;
    first.id = aitrain::v2::WorkflowStepId::create();
    first.workflowRunId = workflow.id;
    first.ordinal = 0;
    first.kind = QStringLiteral("RunInferenceSmoke");
    first.backend = QStringLiteral("aitrain_onnxruntime");
    aitrain::v2::WorkflowStepSnapshotV2 second;
    second.id = aitrain::v2::WorkflowStepId::create();
    second.workflowRunId = workflow.id;
    second.ordinal = 1;
    second.kind = QStringLiteral("DeploymentValidate");
    second.backend = QStringLiteral("aitrain_onnxruntime");
    QVERIFY2(storage.createWorkflowRun(workflow, {first, second}, &error), qPrintable(error));

    aitrain::v2::WorkflowRunnerV2 runner(&storage);
    aitrain::v2::WorkflowRunExecutionResultV2 result;
    QVERIFY2(runner.run(workflow.id,
        [firstOutput, secondOutput](const aitrain::v2::WorkflowStepSnapshotV2& step, const aitrain::CancellationCallback&) {
            return aitrain::v2::WorkflowStepExecutionResultV2{
                aitrain::v2::WorkflowStepState::Succeeded,
                step.ordinal == 0 ? firstOutput : secondOutput,
                {}};
        },
        &result,
        &error), qPrintable(error));
    QCOMPARE(result.state, aitrain::v2::WorkflowStepState::Succeeded);
    QCOMPARE(result.finalOutputArtifactId, secondOutput);
    const QVector<aitrain::v2::WorkflowStepSnapshotV2> completed = storage.workflowSteps(workflow.id, &error);
    QCOMPARE(completed.at(0).state, aitrain::v2::WorkflowStepState::Succeeded);
    QCOMPARE(completed.at(1).inputArtifactId, firstOutput);
    QCOMPARE(completed.at(1).outputArtifactId, secondOutput);

    aitrain::v2::WorkflowRunSnapshotV2 canceledWorkflow;
    canceledWorkflow.id = aitrain::v2::WorkflowRunId::create();
    canceledWorkflow.taskId = rootTask.id;
    canceledWorkflow.templateId = QStringLiteral("canceled-workflow");
    first.id = aitrain::v2::WorkflowStepId::create();
    first.workflowRunId = canceledWorkflow.id;
    first.ordinal = 0;
    second.id = aitrain::v2::WorkflowStepId::create();
    second.workflowRunId = canceledWorkflow.id;
    second.ordinal = 1;
    QVERIFY2(storage.createWorkflowRun(canceledWorkflow, {first, second}, &error), qPrintable(error));
    int executions = 0;
    QVERIFY2(runner.run(canceledWorkflow.id,
        [&executions](const aitrain::v2::WorkflowStepSnapshotV2&, const aitrain::CancellationCallback&) {
            ++executions;
            return aitrain::v2::WorkflowStepExecutionResultV2{};
        },
        &result,
        &error,
        []() { return true; }), qPrintable(error));
    QCOMPARE(executions, 0);
    QCOMPARE(result.state, aitrain::v2::WorkflowStepState::Canceled);
    const QVector<aitrain::v2::WorkflowStepSnapshotV2> canceled = storage.workflowSteps(canceledWorkflow.id, &error);
    QCOMPARE(canceled.at(0).state, aitrain::v2::WorkflowStepState::Canceled);
    QCOMPARE(canceled.at(1).state, aitrain::v2::WorkflowStepState::Skipped);
}

QTEST_MAIN(V2ApplicationTests)
#include "tst_v2_application.moc"
