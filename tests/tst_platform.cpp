#include "TestSupport.h"

#include "aitrain/core/CapabilityRegistry.h"
#include "aitrain/core/Deployment.h"
#include "aitrain/core/WorkerProtocol.h"
#include "aitrain/protocol/Protocol.h"

class PlatformTests final : public QObject {
    Q_OBJECT

private slots:
    void builtinCapabilityRegistryIsAuthoritative()
    {
        const auto& registry = aitrain::BuiltinCapabilityRegistry::instance();
        QVERIFY(!registry.capabilities().isEmpty());
        QVERIFY(!registry.backends().isEmpty());
        QVERIFY(registry.supports(
            QStringLiteral("yolo"),
            QStringLiteral("detection"),
            QStringLiteral("yolo_detection"),
            QStringLiteral("ultralytics_yolo_detect")));
        QVERIFY(registry.supports(
            QStringLiteral("yolo"),
            QStringLiteral("detection"),
            QStringLiteral("yolo_txt"),
            QStringLiteral("ultralytics_yolo")));
        QVERIFY(registry.supports(
            QStringLiteral("paddleocr"),
            QStringLiteral("ocr_recognition"),
            QStringLiteral("paddleocr_rec"),
            QStringLiteral("paddleocr_ppocrv4_rec")));
        QCOMPARE(registry.backendsForTask(QStringLiteral("obb"), QStringLiteral("yolo_obb")),
            QStringList() << QStringLiteral("ultralytics_yolo_obb"));
        QCOMPARE(registry.backend(QStringLiteral("ultralytics_yolo")).id,
            QStringLiteral("ultralytics_yolo_detect"));
        QVERIFY(!registry.supports(
            QStringLiteral("yolo"),
            QStringLiteral("detection"),
            QStringLiteral("yolo_detection"),
            QStringLiteral("paddleocr_rec_official")));
        QVERIFY(registry.capability(QStringLiteral("dataset_interop")).backendIds.isEmpty());
        QCOMPARE(registry.backend(QStringLiteral("ultralytics_yolo_obb")).exportFormats,
            QStringList() << QStringLiteral("onnx"));
        QVERIFY(registry.backend(QStringLiteral("ultralytics_yolo_detect")).modelPresets.contains(
            QStringLiteral("yolov8n.yaml")));
        QVERIFY(registry.backend(QStringLiteral("ultralytics_yolo_detect")).modelPresets.contains(
            QStringLiteral("yolo26x.pt")));
        QCOMPARE(registry.backend(QStringLiteral("ultralytics_yolo_obb")).modelPresets.size(), 10);
        QCOMPARE(registry.backend(QStringLiteral("smp_semantic_segmentation")).modelPresets.size(), 5);
        QCOMPARE(registry.backend(QStringLiteral("anomalib_efficientad")).modelPresets,
            QStringList() << QStringLiteral("anomalib_efficientad_s"));
        QCOMPARE(registry.backend(QStringLiteral("paddleocr_det_official")).modelPresets.size(), 6);
        QCOMPARE(registry.backend(QStringLiteral("paddleocr_rec_official")).modelPresets.size(), 7);

        const QJsonObject json = registry.toJson();
        QCOMPARE(json.value(QStringLiteral("schemaVersion")).toInt(), 1);
        QCOMPARE(json.value(QStringLiteral("aliases")).toObject()
                     .value(QStringLiteral("datasetFormats")).toObject()
                     .value(QStringLiteral("yolo_txt")).toString(),
            QStringLiteral("yolo_detection"));
        QVERIFY(json.value(QStringLiteral("capabilities")).toArray().size() >= 4);
        QVERIFY(json.value(QStringLiteral("backends")).toArray().size() >= 8);
    }

    void workerProtocolBuildersStayTyped()
    {
        namespace wp = aitrain::worker_protocol;
        const QJsonObject options{{QStringLiteral("dryRun"), true}};
        const QJsonObject request = wp::dataQualityWorkflowRequest(
            QStringLiteral("task-1"), QStringLiteral("project"), QStringLiteral("dataset-id"),
            QStringLiteral("version-id"), QStringLiteral("snapshot-id"),
            QStringLiteral("artifact-id"), options);
        QCOMPARE(request.value(QStringLiteral("taskId")).toString(), QStringLiteral("task-1"));
        QCOMPARE(request.value(QStringLiteral("projectRoot")).toString(), QStringLiteral("project"));
        QVERIFY(request.value(QStringLiteral("options")).toObject().value(QStringLiteral("dryRun")).toBool());
        QVERIFY(!request.contains(QStringLiteral("datasetPath")));
        QVERIFY(!request.contains(QStringLiteral("outputPath")));
        QVERIFY(wp::isTerminalEvent(wp::event::completed()));
    }

    void workerControlBridgeUsesProtocolEnvelope()
    {
        namespace wp = aitrain::worker_protocol;
        const aitrain::RequestId requestId = aitrain::RequestId::create();
        const aitrain::TaskId taskId = aitrain::TaskId::create();
        const QJsonObject requestPayload{
            {QStringLiteral("taskId"), taskId.toString()},
            {QStringLiteral("projectRoot"), QStringLiteral("C:/AITrain/project")},
            {QStringLiteral("datasetId"), aitrain::DatasetId::create().toString()},
            {QStringLiteral("datasetVersionId"), aitrain::DatasetVersionId::create().toString()},
            {QStringLiteral("snapshotId"), aitrain::SnapshotId::create().toString()},
            {QStringLiteral("snapshotArtifactId"), aitrain::ArtifactId::create().toString()},
            {QStringLiteral("options"), QJsonObject{}}};
        wp::TaskCommand commandValue;
        QString error;
        QVERIFY2(wp::taskCommandFromPayload(
            wp::command::runDataQualityWorkflow(), requestPayload, &commandValue, &error), qPrintable(error));

        const aitrain::ProtocolEnvelope start = wp::control::startTaskEnvelope(
            requestId, taskId, 1, commandValue);
        QCOMPARE(start.kind, QStringLiteral("command.start_task"));
        wp::TaskCommand decodedCommand;
        QVERIFY2(wp::control::unpackStartTask(start, &decodedCommand, &error), qPrintable(error));
        QCOMPARE(wp::taskCommandType(decodedCommand), wp::command::runDataQualityWorkflow());
        QCOMPARE(wp::taskCommandPayload(decodedCommand), requestPayload);

        const wp::TaskEvent resultEvent = wp::taskEventFromType(
            wp::event::dataQualityWorkflow(), requestPayload);
        const aitrain::ProtocolEnvelope result = wp::control::eventEnvelope(
            requestId, taskId, 2, resultEvent);
        QCOMPARE(result.kind, QStringLiteral("event.result"));
        QVERIFY(aitrain::isKnownProtocolKind(result.kind));
        wp::TaskEvent decodedEvent;
        QVERIFY2(wp::control::unpackTaskEvent(result, &decodedEvent, &error), qPrintable(error));
        QCOMPARE(wp::taskEventType(decodedEvent), wp::event::dataQualityWorkflow());
        QCOMPARE(decodedEvent.details, requestPayload);

        const QJsonObject externalPayload{
            {QStringLiteral("taskId"), taskId.toString()},
            {QStringLiteral("projectRoot"), QStringLiteral("C:/AITrain/project")},
            {QStringLiteral("sourcePath"), QStringLiteral("C:/evidence/acceptance.json")}};
        wp::TaskCommand externalCommand;
        QVERIFY2(wp::taskCommandFromPayload(
            wp::command::importExternalAcceptanceEvidence(), externalPayload,
            &externalCommand, &error), qPrintable(error));
        QCOMPARE(wp::taskCommandType(externalCommand), wp::command::importExternalAcceptanceEvidence());
        QCOMPARE(wp::taskCommandPayload(externalCommand), externalPayload);
    }

    void workerEventCodecRejectsInvalidIdentityAndPayload()
    {
        namespace wp = aitrain::worker_protocol;
        const aitrain::RequestId requestId = aitrain::RequestId::create();
        const aitrain::TaskId taskId = aitrain::TaskId::create();
        const QJsonObject validDetails{
            {QStringLiteral("taskId"), taskId.toString()},
            {QStringLiteral("message"), QStringLiteral("hello")}};
        const wp::TaskEvent validEvent = wp::taskEventFromType(wp::event::log(), validDetails);
        const aitrain::ProtocolEnvelope validEnvelope = wp::control::eventEnvelope(
            requestId, taskId, 1, validEvent);
        QVERIFY(validEnvelope.taskId.isValid());

        wp::TaskEvent decoded;
        QString error;
        QVERIFY2(wp::control::unpackTaskEvent(validEnvelope, &decoded, &error), qPrintable(error));
        QCOMPARE(decoded.taskId, taskId);

        aitrain::ProtocolEnvelope mismatched = validEnvelope;
        mismatched.payload.insert(QStringLiteral("taskId"), aitrain::TaskId::create().toString());
        QVERIFY(!wp::control::unpackTaskEvent(mismatched, &decoded, &error));
        QVERIFY(error.contains(QStringLiteral("taskId")));

        aitrain::ProtocolEnvelope missingMessage = validEnvelope;
        missingMessage.payload.remove(QStringLiteral("message"));
        QVERIFY(!wp::control::unpackTaskEvent(missingMessage, &decoded, &error));
        QVERIFY(error.contains(QStringLiteral("message")));

        const wp::TaskEvent invalidProgress = wp::taskEventFromType(
            wp::event::progress(), QJsonObject{{QStringLiteral("taskId"), taskId.toString()},
                {QStringLiteral("percent"), QStringLiteral("not-a-number")}});
        const aitrain::ProtocolEnvelope invalidProgressEnvelope = wp::control::eventEnvelope(
            requestId, taskId, 2, invalidProgress);
        QVERIFY(invalidProgressEnvelope.kind.isEmpty());

        // 预执行协议拒绝发生在 Worker 尚未绑定 activeTaskId 的窗口，
        // 诊断 payload 会带空 taskId；封包必须以已认证 envelope 身份补齐它。
        const wp::TaskEvent earlyFailure = wp::taskEventFromType(
            wp::event::failed(), QJsonObject{{QStringLiteral("taskId"), QString()},
                {QStringLiteral("message"), QStringLiteral("protocol rejected")},
                {QStringLiteral("errorCode"), QStringLiteral("protocol_rejected")}});
        const aitrain::ProtocolEnvelope earlyFailureEnvelope = wp::control::eventEnvelope(
            requestId, taskId, 3, earlyFailure);
        QCOMPARE(earlyFailureEnvelope.kind, QStringLiteral("event.failed"));
        QCOMPARE(earlyFailureEnvelope.payload.value(QStringLiteral("taskId")).toString(), taskId.toString());
    }

    void packagingLayoutContainsOnlyProductDirectories()
    {
        const aitrain::PackagingLayout layout = aitrain::packagingLayoutForRoot(QStringLiteral("C:/AITrain"));
        QVERIFY(!layout.rootPath.isEmpty());
        QVERIFY(!layout.appExecutablePath.isEmpty());
        QVERIFY(!layout.workerExecutablePath.isEmpty());
        QVERIFY(!layout.runtimesDirectory.isEmpty());
        QVERIFY(!layout.toJson().contains(QStringLiteral("pluginModelsDirectory")));
    }
};

QTEST_MAIN(PlatformTests)
#include "tst_platform.moc"
